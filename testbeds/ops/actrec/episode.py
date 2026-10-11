"""One episode = one task instance x one focal write x one fault world x one condition x one driver.

The driver (agent) runs in a CHILD process so that crash worlds are real process deaths: all memory is lost
and only durable files survive. The parent (runner.py) owns ground truth and grading.

Dispatcher: the single choke point for every write. It injects the episode's fault on the first dispatch of
the focal step (network fault through the worker's fault proxy, transport redelivery, or a process crash
before/after the write), creates the background actor's look-alike or concurrent aggregate change for the
read-model variants, and logs every dispatch and compensation.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .contracts import SPECS, Toolbox
from .durable import Log
from .faults.toxi import NETWORK_MODES, NetFault
from .services import base
from .services.base import CallResult, nohook

CRASH_EXIT = 75
CRASH_MODES = ["crash_after", "crash_before"]
ALL_MODES = NETWORK_MODES + ["late_tail", "partial"] + CRASH_MODES
BATCH_K = 3                     # partial batch: rows before index 3 commit, the rest do not
INTENDED_EXEC = {"none": 1, "lost_ack": 1, "not_exec_err": 0, "held": 1, "not_exec_to": 0, "late": 1,
                 "late_tail": 1, "redelivery": 2, "crash_after": 1, "crash_before": 0, "partial": None,
                 "redelivery_late": 2}
REDELIVER_LATE_S = 8.0          # follow-up: the transport delivers the duplicate this long after the acknowledged request
ALL_MODES = ALL_MODES + ["redelivery_late"]


@dataclass
class EpisodeConfig:
    episode_id: str
    run: str
    task: dict                       # TaskInstance.to_dict()
    focal_step: str
    mode: str
    variant: str = "exact"           # exact | lagged | lookalike | agg_shared | lookalike_other (follow-up)
    condition: str = "C0"
    driver: str = "scripted"         # scripted | llm:<model>
    worker: int = 0
    deadline_s: float = 4.0
    delay_s: float = 12.0            # in-flight delay of the focal request (late / late_tail)
    delta_assumed_s: float = 20.0    # Settle's (and G+w's) assumed in-flight bound
    retries: int = 1                 # scripted driver: re-issues allowed per step after a failure
    repeat: int = 0
    seed: int = 0
    state_dir: str = ""
    gitea_token_file: str = ""
    tb_accounts: list | None = None
    extra: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "EpisodeConfig":
        return EpisodeConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def tail_delay(seed: int, lo: float = 5.0, hi: float = 120.0) -> float:
    """Heavy-tailed in-flight delay: log-uniform on [lo, hi], fixed per world."""
    r = random.Random(f"tail:{seed}")
    return round(math.exp(r.uniform(math.log(lo), math.log(hi))), 2)


class Dispatcher:
    def __init__(self, cfg: EpisodeConfig, tb: Toolbox):
        self.cfg, self.tb = cfg, tb
        self.state = Path(cfg.state_dir)
        self.log = Log(self.state / "dispatch.jsonl")
        self.actor_log = Log(self.state / "actor.jsonl")
        self.attempts: dict[str, int] = {}
        self._timers: list[threading.Timer] = []

    # ------------------------------------------------------------------ focal bookkeeping (durable)
    def _fired(self) -> bool:
        return (self.state / "focal_fired").exists()

    def _mark_fired(self) -> None:
        p = self.state / "focal_fired"
        with p.open("w") as f:
            f.write(str(time.time()))
            f.flush()
            os.fsync(f.fileno())

    # ------------------------------------------------------------------ the background actor
    def _actor(self, tool: str, args: dict) -> None:
        v, tb = self.cfg.variant, self.tb
        rec = None
        try:
            if v == "lookalike_other" and tool in ("issue_create", "issue_comment"):
                # follow-up: the look-alike comes from ANOTHER account (an ordinary second user), same moment
                from .config import stack_secrets
                from .services.gitea import Gitea
                other = Gitea(stack_secrets()["gitea_other_token"], repo=tb.gitea.repo, owner=tb.gitea.owner)
                if tool == "issue_create":
                    r = other.issue_create(args["title"], args["body"], route="direct", deadline_s=30)
                    rec = {"tool": tool, "id": (r.value or {}).get("number"), "account": "other"}
                else:
                    r = other.comment_create(int(args["issue"]), args["body"], route="direct", deadline_s=30)
                    rec = {"tool": tool, "id": (r.value or {}).get("comment_id"), "account": "other"}
                if r.kind != "ok":
                    rec["error"] = f"{r.exc_type}: {r.exc_msg}"
            elif v == "lookalike":
                if tool == "issue_create":
                    r = tb.gitea.issue_create(args["title"], args["body"], route="direct", deadline_s=30)
                    rec = {"tool": tool, "id": (r.value or {}).get("number")}
                elif tool == "issue_comment":
                    r = tb.gitea.comment_create(int(args["issue"]), args["body"], route="direct", deadline_s=30)
                    rec = {"tool": tool, "id": (r.value or {}).get("comment_id")}
                elif tool in ("mail_send", "mail_send_plain"):
                    mid = f"actor-{uuid.uuid4()}@actrec.invalid"
                    tb.mail.actor_send(tb.addr(args["to"]), args["subject"], args["body"], mid)
                    rec = {"tool": tool, "id": mid}
                elif tool == "row_insert":
                    rec = {"tool": tool, "id": tb.pg.actor_insert(args["item"], int(args["amount"]))}
                elif tool == "event_append":
                    rec = {"tool": tool, "id": tb.streams.actor_append(args["kind"], args["data"])}
            elif v == "agg_shared":
                if tool == "stock_adjust":
                    tb.pg.actor_stock(args["sku"], int(args["delta"]))
                    rec = {"tool": tool, "delta": int(args["delta"])}
                elif tool == "counter_incr":
                    tb.streams.actor_counter(args["name"], 1)
                    rec = {"tool": tool, "delta": 1}
        except Exception as e:  # noqa: BLE001
            rec = {"tool": tool, "error": f"{type(e).__name__}: {str(e)[:120]}"}
        if rec:
            rec["t"] = time.time()
            self.actor_log.append(rec)

    # ------------------------------------------------------------------ writes
    def dispatch(self, step: str, tool: str, args: dict, *, tag: str | None = None, key: str | None = None,
                 batch_only: list[int] | None = None, deadline_s: float | None = None) -> CallResult:
        cfg = self.cfg
        n = self.attempts[step] = self.attempts.get(step, 0) + 1
        focal = step == cfg.focal_step and not self._fired()
        mode = cfg.mode if focal else "none"
        route, hook, repeat, fault = "clean", nohook, 1, None
        if focal:
            self._mark_fired()
            self._actor(tool, args)
            if mode in ("lost_ack", "not_exec_err", "held", "not_exec_to", "late", "late_tail", "partial"):
                fault = NetFault(service=SPECS[tool].service, mode=mode, deadline_s=cfg.deadline_s,
                                 delay_s=cfg.delay_s, worker=cfg.worker)
                route, hook = fault.route, fault.hook
            elif mode == "redelivery":
                repeat = 2
            elif mode == "redelivery_late":
                # follow-up: the request is acknowledged normally; the transport delivers an identical copy later
                tm = threading.Timer(REDELIVER_LATE_S, self._redeliver, args=(step, tool, dict(args), tag, key, batch_only))
                tm.daemon = True
                tm.start()
                self._timers.append(tm)
            elif mode == "crash_before":
                self._log(step, tool, n, tag, key, focal, mode, None, 0.0)
                os._exit(CRASH_EXIT)
        t0 = time.time()
        res = self.tb.execute(tool, args, tag=tag, key=key, route=route, hook=hook,
                              deadline_s=deadline_s or cfg.deadline_s, repeat=repeat, batch_only=batch_only,
                              batch_hook_before=BATCH_K if mode == "partial" else 0)
        self._log(step, tool, n, tag, key, focal, mode, res, t0)
        if fault is not None:
            tm = threading.Timer(fault.settle_s(), fault.cleanup)
            tm.daemon = True
            tm.start()
            self._timers.append(tm)
        if focal and mode == "crash_after":
            os._exit(CRASH_EXIT)
        return res

    def _redeliver(self, step, tool, args, tag, key, batch_only) -> None:
        t0 = time.time()
        try:
            res = self.tb.execute(tool, args, tag=tag, key=key, route="clean", hook=nohook, deadline_s=30,
                                  repeat=1, batch_only=batch_only)
        except Exception as e:  # noqa: BLE001
            res = CallResult("error", 0.0, exc_type=type(e).__name__, exc_msg=str(e)[:100])
        self.log.append({"type": "redelivery", "step": step, "tool": tool, "tag": tag, "key": key, "t": t0,
                         "pid": os.getpid(), "result": res.to_dict()})

    def compensate(self, step: str, tool: str, args: dict, effect_id) -> str:
        t0 = time.time()
        try:
            out = self.tb.compensate(tool, args, effect_id)
        except Exception as e:  # noqa: BLE001
            out = f"error: {type(e).__name__}: {str(e)[:100]}"
        self.log.append({"type": "compensate", "step": step, "tool": tool, "effect_id": str(effect_id),
                         "args": args, "result": out, "t": t0, "pid": os.getpid()})
        return out

    def _log(self, step, tool, n, tag, key, focal, mode, res, t0) -> None:
        self.log.append({"type": "dispatch", "step": step, "tool": tool, "attempt": n, "tag": tag, "key": key,
                         "focal": focal, "mode": mode, "t": t0 or time.time(), "pid": os.getpid(),
                         "result": res.to_dict() if res else None})

    def drain(self, max_s: float = 900.0) -> None:
        """Before the process exits: let abandoned in-flight requests finish and faults be cleared."""
        for tm in self._timers:
            tm.join(max_s)
        base.wait_inflight(max_s)


# ---------------------------------------------------------------------- child entry point
def child_main(cfg_path: str) -> None:
    cfg = EpisodeConfig.load(Path(cfg_path))
    token = Path(cfg.gitea_token_file).read_text(encoding="utf-8").strip()
    from .config import register_secret
    register_secret(token)
    tb = Toolbox(cfg.run, token, variant=cfg.variant, tb_accounts=cfg.tb_accounts)
    disp = Dispatcher(cfg, tb)
    state = Path(cfg.state_dir)
    resumed = (state / "focal_fired").exists() and cfg.mode in CRASH_MODES
    t0 = time.time()
    if cfg.condition == "C4":
        from .temporal_wf import run_temporal
        report = run_temporal(cfg, tb, disp, resumed)
    elif cfg.driver == "scripted":
        from .driver_scripted import run_scripted
        report = run_scripted(cfg, tb, disp, resumed)
    else:
        from .driver_llm import run_llm
        report = run_llm(cfg, tb, disp, resumed)
    report["agent_done_t"] = time.time()
    report["agent_elapsed_s"] = round(time.time() - t0, 3)
    report["resumed"] = resumed
    tmp = state / "report.json.tmp"
    tmp.write_text(json.dumps(report, default=str, indent=1), encoding="utf-8")
    os.replace(tmp, state / f"report{'_resumed' if resumed else ''}.json")
    disp.drain()
    sys.stdout.flush()
    os._exit(0)
