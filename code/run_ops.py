"""Run the ops testbed (multi-service operations agent; ported from the action-recovery study) for one or more
models, one temperature and k repeats.

Usage:
  run_ops.py --models gpt-4.1-nano,gpt-5.6-luna --subset pilot --repeats 3 --temperature 0 --workers 4
  run_ops.py --models gpt-4.1-nano --subset pilot --repeats 3 --world lost_ack     # secondary setting
  run_ops.py --check-env [--subset main]   # start/verify only the Docker services the subset needs (free)

Output: runs/ops/<subset>/<label>/T<temp>/r<k>/<task_id>.json (one file per episode) and
        runs/ops/<subset>/episodes.jsonl (one summary row per episode, appended). With --world lost_ack the
        subset folder is runs/ops/<subset>-lost_ack/ (use flips.py --testbed ops --subset <subset>-lost_ack).
Episodes already on disk are skipped, so a job can be re-queued after a quota stop or crash. Episodes that
could not be measured (service/harness error, model API error after retries) are NOT written; they are logged
to runs/ops/<subset>/errors.jsonl and re-run next time (as the source harness re-ran unfinished episodes).
Exit codes: 0 done, 2 environment not ready, 3 budget/quota stop, 4 some episodes errored.

Condition: C0 (plain agent: writes go straight to the service, the model alone decides about retries).
World: "none" (fault-free, main setting) or "lost_ack" (the first non-batch write of the task commits but its
reply is lost: the agent sees a transport error for a write that did take effect).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(os.environ.get("MM_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
TB = ROOT / "testbeds" / "ops"
os.environ["ACTREC_ROOT"] = str(TB)            # actrec.config: ROOT, compose file, secrets folder
os.environ["ACTREC_PROJECT"] = "mmops"         # docker compose project (own volumes, never the source's)
sys.path.insert(0, str(TB))
sys.path.insert(0, str(ROOT / "code"))

from mm import llm as mmllm  # noqa: E402

TESTBED = "ops"
CONDITION = "C0"
WORLDS = ["none", "lost_ack"]
SEED = 20261006
TEMPLATES = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
INSTANCE_RANGE = range(200, 400)   # fresh instances: the source used 0 (dev), 100-102 (test), 900 (harness check)
MAX_WORKERS = 8                    # one fault-proxy slot per worker (actrec.config.MAX_WORKERS)
QUIESCE_S = 3.0                    # as actrec.runner
STACK_ENV = TB / "runs" / ".secrets" / "stack_env.json"
STACK_KEYS = ("POSTGRES_PASSWORD", "MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD", "GITEA_ADMIN_PASSWORD")
_LOCK = threading.Lock()


class HarnessError(RuntimeError):
    """The episode could not be measured (service, grading or model-API failure); it is re-run later."""


# ------------------------------------------------------------------ local stack credentials
def ensure_stack_env(create: bool) -> bool:
    """Credentials of the local Docker stack: generated once (random), kept in runs/.secrets/stack_env.json
    of the testbed, exported to this process (and so to docker compose). Values are never printed."""
    d = {}
    if STACK_ENV.exists():
        d = json.loads(STACK_ENV.read_text(encoding="utf-8"))
    elif create:
        d = {"POSTGRES_PASSWORD": secrets.token_urlsafe(24), "MINIO_ROOT_USER": "actrec",
             "MINIO_ROOT_PASSWORD": secrets.token_urlsafe(24), "GITEA_ADMIN_PASSWORD": secrets.token_urlsafe(24)}
        STACK_ENV.parent.mkdir(parents=True, exist_ok=True)
        STACK_ENV.write_text(json.dumps(d), encoding="utf-8")
    for k in STACK_KEYS:
        if d.get(k):
            os.environ[k] = d[k]
    return bool(d)


ensure_stack_env(create=False)


# ------------------------------------------------------------------ tasks and subsets
def task_of(task_id: str):
    from actrec.tasks import make
    t, i = task_id.split("_")
    return make(t, int(i))


def _draw() -> dict:
    """All three subsets from one seeded draw, so they are disjoint whatever order they are created in."""
    rng = random.Random(SEED)
    pools = {t: rng.sample(list(INSTANCE_RANGE), 40) for t in TEMPLATES}
    smoke_t = sorted(rng.sample(TEMPLATES, 4))
    main_plus = sorted(rng.sample(TEMPLATES, 4))          # 8 x 12 + 4 = 100
    plans = {"smoke": {t: 1 for t in smoke_t}, "pilot": {t: 5 for t in TEMPLATES},
             "main": {t: 12 + (t in main_plus) for t in TEMPLATES}}
    ids, pos = {}, {t: 0 for t in TEMPLATES}
    for name in ("smoke", "pilot", "main"):
        out = []
        for t in TEMPLATES:
            n = plans[name].get(t, 0)
            out += [f"{t}_{i}" for i in pools[t][pos[t]:pos[t] + n]]
            pos[t] += n
        ids[name] = out
    return {"plans": plans, "ids": ids}


def make_subset(name: str) -> list[str]:
    """Frozen, seeded subsets (seed 20261006): smoke = 4 tasks of 4 templates, pilot = 40 (5 per template),
    main = 100 (12-13 per template), mutually disjoint, instances 200-399."""
    p = TB / "subsets" / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))["ids"]
    d = _draw()
    if name not in d["ids"]:
        raise SystemExit(f"unknown subset {name}; known: {sorted(d['ids'])}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"name": name, "seed": SEED, "plan": d["plans"][name], "instances": "200-399",
                             "ids": d["ids"][name]}, indent=1), encoding="utf-8")
    return d["ids"][name]


def focal_step(task) -> str:
    """Where the lost_ack fault is injected: the first write step that is not a batch insert (the source never
    injects lost_ack into a batch; for T5 this is the event append)."""
    return next(s.name for s in task.steps if s.tool != "batch_insert")


# ------------------------------------------------------------------ environment
def _docker_engine() -> bool:
    for i in range(60):  # wait up to ~15 min for Docker Desktop's engine (as run_clinical.check_env)
        try:
            r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True,
                               timeout=60)
            if r.returncode == 0 and r.stdout.strip():
                print("docker engine up:", r.stdout.strip())
                return True
        except (OSError, subprocess.TimeoutExpired) as e:
            print("docker info failed:", type(e).__name__)
        if i == 0:
            print("Docker engine not running; launching Docker Desktop...", flush=True)
            for exe in (r"C:\Program Files\Docker\Docker\Docker Desktop.exe",
                        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Docker\Docker\Docker Desktop.exe")):
                if os.path.exists(exe):
                    subprocess.Popen([exe], close_fds=True, creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
                    print("started", exe, flush=True)
                    break
            else:
                print("Docker Desktop executable not found", flush=True)
        time.sleep(15)
    print("Docker engine not available after 15 min")
    return False


def _stop_source_stack() -> None:
    """The action-recovery stack (compose project `actrec`) binds the same host ports. If it is running, stop
    its containers (docker stop only: its volumes and files are untouched) so this project's stack can start."""
    r = subprocess.run(["docker", "ps", "--filter", "label=com.docker.compose.project=actrec", "--format", "{{.Names}}"],
                       capture_output=True, text=True, timeout=60)
    names = [n for n in r.stdout.split() if n]
    if names:
        print("stopping the action-recovery stack's containers (ports are shared):", " ".join(names), flush=True)
        subprocess.run(["docker", "stop", *names], capture_output=True, text=True, timeout=300)


def check_env(task_ids: list[str]) -> bool:
    ensure_stack_env(create=True)
    from actrec import ops_stack
    tasks = [task_of(t) for t in task_ids]
    svcs = ops_stack.services_for(tasks)
    print("services needed:", ", ".join(svcs), flush=True)
    down = [s for s in svcs if not ops_stack.probe(s)]
    print("reachable:", ", ".join(s for s in svcs if s not in down) or "-", "| not reachable:", ", ".join(down) or "-")
    if down:
        if not _docker_engine():
            return False
        try:
            _stop_source_stack()
            cmd = ops_stack.compose_cmd("up", "-d", *svcs)
            print("running:", " ".join(cmd), flush=True)
            r = ops_stack.compose("up", "-d", *svcs, timeout=1800)
            print((r.stdout or "")[-1500:], (r.stderr or "")[-1500:])
            if r.returncode != 0:
                print("compose up failed (exit", r.returncode, ")")
        except Exception as e:  # noqa: BLE001
            print("docker error:", type(e).__name__, str(e)[:300])
        h = ops_stack.wait_healthy(svcs, timeout=300)
        print("health (s to healthy, None = not healthy):", h)
        down = [s for s, v in h.items() if v is None]
        if down:
            print("NOT HEALTHY:", ", ".join(down))
            return False
    try:
        info = ops_stack.bootstrap(svcs)
        print("bootstrap:", info)
    except Exception as e:  # noqa: BLE001
        from actrec.config import redact
        print("bootstrap failed:", type(e).__name__, redact(str(e))[:300])
        return False
    ok = all(ops_stack.probe(s) for s in svcs)
    if ok and "gitea" in svcs:
        from actrec.config import stack_secrets
        ok = ops_stack.gitea_token_ok(stack_secrets()["gitea_token"])
        print("gitea token valid:", ok)
    return ok


# ------------------------------------------------------------------ one episode
def run_episode(spec, task_id, temperature, rep, world, subset, out_file, slots, chat=None):
    with _LOCK:
        slot = slots.pop()
    try:
        return _episode(spec, task_id, temperature, rep, world, subset, out_file, slot, chat or mmllm.chat)
    finally:
        with _LOCK:
            slots.append(slot)


def _episode(spec, task_id, temperature, rep, world, subset, out_file, slot, chat):
    from actrec.config import redact, stack_secrets
    from actrec.contracts import Toolbox
    from actrec.driver_chat import run_chat
    from actrec.durable import Log
    from actrec.episode import Dispatcher, EpisodeConfig
    from actrec.grader import grade
    from actrec.ops_grade import action_verdicts, classify, scan_extras
    from actrec.services.ledger_tb import Ledger

    task = task_of(task_id)
    label = spec["label"]
    eid = hashlib.sha1(f"{TESTBED}|{label}|{temperature}|{rep}|{task_id}|{world}".encode()).hexdigest()[:12]
    run = f"{eid[:8]}-{uuid.uuid4().hex[:6]}"      # a fresh namespace for every attempt (as actrec.runner)
    state = Path(tempfile.mkdtemp(prefix=f"ops_{eid}_"))
    t0 = time.time()
    try:
        token = stack_secrets().get("gitea_token", "")
        cfg = EpisodeConfig(episode_id=eid, run=run, task=task.to_dict(), focal_step=focal_step(task), mode=world,
                            variant="exact", condition=CONDITION, driver=f"chat:{label}", worker=slot, repeat=rep,
                            seed=int(eid, 16) % (2 ** 31), state_dir=str(state),
                            extra={"grid": f"{TESTBED}/{subset}"})
        try:
            tb0 = Toolbox(run, token)
            if any(s.tool in ("issue_create", "issue_comment") for s in task.steps):
                code = tb0.gitea.ensure_repo()
                if code not in (201, 409):
                    raise HarnessError(f"gitea repo create HTTP {code}")
            if any(s.tool == "transfer_create" for s in task.steps):
                led = Ledger()
                led.setup_accounts()
                cfg.tb_accounts = [str(led.debit), str(led.credit)]
            tb = Toolbox(run, token, variant="exact", tb_accounts=cfg.tb_accounts)
            disp = Dispatcher(cfg, tb)
        except HarnessError:
            raise
        except Exception as e:  # noqa: BLE001
            raise HarnessError(f"setup: {type(e).__name__}: {str(e)[:200]}") from e
        salt = f"T{temperature}|r{rep}|{task_id}|{world}"
        try:
            report = run_chat(cfg, tb, disp, spec, chat, temperature=temperature, salt=salt,
                              fatal=(mmllm.BudgetExceeded, mmllm.QuotaExhausted), stage=f"{TESTBED}:{subset}")
        except (mmllm.BudgetExceeded, mmllm.QuotaExhausted):
            raise
        except Exception as e:  # noqa: BLE001 - a bug or service failure inside a tool call
            raise HarnessError(f"agent loop: {type(e).__name__}: {str(e)[:200]}") from e
        finally:
            disp.drain(120)
        if str(report["stop"]).startswith("model error"):
            raise HarnessError(report["stop"])
        time.sleep(QUIESCE_S)
        tbg = Toolbox(run, token, variant="exact", tb_accounts=cfg.tb_accounts)
        try:
            g = grade(state, task, tbg)
            x = scan_extras(task, tbg, report["actions"])
        except Exception as e:  # noqa: BLE001
            raise HarnessError(f"grading: {type(e).__name__}: {str(e)[:200]}") from e
        if x["scan_errors"]:
            raise HarnessError("grading scan: " + "; ".join(x["scan_errors"])[:300])
        status = "completed" if report["stop"] == "finished" else report["stop"]
        category, success, detail = classify(status, report["claimed"], g, x)
        action_verdicts(report["actions"], g)
        writes = [a for a in report["actions"] if a.get("kind") == "write"]
        rec = {"testbed": TESTBED, "subset": subset, "task_id": task_id, "label": label, "model": spec["model"],
               "provider": spec["provider"], "temperature": temperature, "repeat": rep, "success": success,
               "status": status, "error": None, "category": category, "category_detail": detail,
               "template": task.template, "instance": task.instance, "world": world, "condition": CONDITION,
               "focal_step": cfg.focal_step, "goal": task.goal, "claimed": report["claimed"],
               "model_summary": report["model_summary"], "uncertain": report["uncertain"],
               "layer_uncertain": report["layer_uncertain"], "n_turns": report["turns"],
               "n_tool_calls": report["tool_calls"], "n_writes": len(writes), "n_nudges": report["nudges"],
               "n_malformed": report["malformed_calls"], "stop": report["stop"], "agent_cost": report["agent_cost"],
               "paid_calls": report["paid_calls"], "cached_calls": report["cached_calls"],
               "tokens_in": report["tokens_in"], "tokens_out": report["tokens_out"],
               "tokens_reasoning": report["tokens_reasoning"], "length_stops": report["length_stops"],
               "served": report["served"], "model_latency_s": report["model_latency_s"],
               "step_map": report["step_map"], "actions": report["actions"], "grade": g,
               "extras": x["extras"], "mismatches": x["mismatches"],
               "dispatch_log": Log(state / "dispatch.jsonl").records(), "messages": report["messages"],
               "episode_id": eid, "run_namespace": run, "wall_s": round(time.time() - t0, 2)}
        rec = redact(rec)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(tmp, out_file)
        return {"task_id": task_id, "label": label, "T": temperature, "r": rep, "success": success, "status": status,
                "category": category, "claimed": report["claimed"], "n_turns": report["turns"],
                "n_tool_calls": report["tool_calls"], "n_writes": len(writes), "cost": report["agent_cost"],
                "error": None, "template": task.template, "world": world}
    finally:
        shutil.rmtree(state, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--subset", default="smoke")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--rep-start", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--world", default="none", choices=WORLDS)
    ap.add_argument("--check-env", action="store_true")
    a = ap.parse_args(argv)
    ids = make_subset(a.subset)
    env_ok = check_env(ids)
    if a.check_env or not env_ok:
        print("environment OK" if env_ok else "ENVIRONMENT NOT READY")
        sys.exit(0 if env_ok else 2)
    workers = max(1, min(a.workers, MAX_WORKERS))
    models = mmllm.load_models()
    sub_dir = a.subset if a.world == "none" else f"{a.subset}-{a.world}"
    base = ROOT / "runs" / TESTBED / sub_dir
    summary, errlog = base / "episodes.jsonl", base / "errors.jsonl"
    base.mkdir(parents=True, exist_ok=True)
    print(f"subset {a.subset}: {len(ids)} tasks; world {a.world}; condition {CONDITION}; models {a.models}; "
          f"T={a.temperature}; repeats {a.rep_start}..{a.rep_start + a.repeats - 1}; workers {workers}; "
          f"job cap ${mmllm.job_cap():.2f}")
    n_errors_total = 0
    from actrec.config import redact
    for label in [m for m in a.models.split(",") if m]:
        spec = models[label]
        for rep in range(a.rep_start, a.rep_start + a.repeats):
            d = base / label / f"T{a.temperature:g}" / f"r{rep}"
            todo = [i for i in ids if not (d / f"{i}.json").exists()]
            if not todo:
                print(f"[{label} r{rep}] all {len(ids)} done")
                continue
            t0, n_ok, n_err, stop = time.time(), 0, 0, None
            slots = list(range(workers))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(run_episode, spec, i, a.temperature, rep, a.world, a.subset, d / f"{i}.json",
                                  slots): i for i in todo}
                for f in as_completed(futs):
                    try:
                        row = f.result()
                    except (mmllm.BudgetExceeded, mmllm.QuotaExhausted) as e:
                        stop = e
                        continue
                    except Exception as e:  # noqa: BLE001 - not measured; re-run next time
                        n_err += 1
                        with _LOCK:
                            base.mkdir(parents=True, exist_ok=True)
                            with errlog.open("a", encoding="utf-8") as fh:
                                fh.write(json.dumps(redact({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "task_id": futs[f],
                                                            "label": label, "T": a.temperature, "r": rep,
                                                            "world": a.world,
                                                            "error": f"{type(e).__name__}: {str(e)[:400]}"})) + "\n")
                        continue
                    n_ok += row["success"]
                    with _LOCK:
                        with summary.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(row) + "\n")
            done = sum(1 for i in ids if (d / f"{i}.json").exists())
            n_errors_total += n_err
            print(f"[{label} r{rep}] {done}/{len(ids)} done, success {n_ok}/{len(todo)} this pass, "
                  f"unmeasured (errors, will re-run) {n_err}, {time.time() - t0:.0f}s, "
                  f"job spend ${mmllm.job_spent():.4f}", flush=True)
            if stop:
                print(f"STOPPED: {type(stop).__name__}: {stop}")
                sys.exit(3)
    if n_errors_total:
        print(f"{n_errors_total} episodes errored (see {errlog.relative_to(ROOT)}); re-queue to retry them")
        sys.exit(4)


if __name__ == "__main__":
    main()
