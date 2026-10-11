"""Parent side of an episode and of a grid: namespace setup, child launch (and relaunch after a crash),
quiescence, grading, one JSON record per episode. Resumable: finished episode ids are skipped."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import ROOT, RUNS, SECRETS, append_jsonl, redact, stack_secrets
from .contracts import SPECS, Toolbox
from .episode import CRASH_EXIT, CRASH_MODES, EpisodeConfig, tail_delay
from .faults.toxi import Toxi
from .grader import grade
from .services.ledger_tb import Ledger
from .tasks import DEV, TEST_ONLY, TaskInstance, make

QUIESCE_S = 3.0
CONTENT_TOOLS = {"issue_create", "issue_comment", "mail_send", "mail_send_plain", "row_insert", "event_append"}
AMBIG = ["lost_ack", "not_exec_err", "not_exec_to", "late"]
NET_MODES = ["none", "lost_ack", "not_exec_err", "held", "not_exec_to", "late", "late_tail", "redelivery"]
ALL_CONDITIONS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "Gw", "C8", "S", "S-j", "S-t", "S-h", "S-c", "O"]


def episode_id(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def worlds(task: TaskInstance, modes: list[str], variants: list[str], focal: list[str] | None = None) -> list[tuple]:
    """(focal step, mode, variant) combinations that are meaningful for this task."""
    out = []
    for st in task.steps:
        if focal and st.name not in focal:
            continue
        tool = st.tool
        if tool == "batch_insert":
            ms = [m for m in ["none", "partial"] + CRASH_MODES if m in modes]
        else:
            ms = [m for m in modes if m != "partial"]
        for v in variants:
            if v == "exact":
                vm = ms
            elif v == "lookalike" and tool in CONTENT_TOOLS:
                vm = [m for m in ms if m in AMBIG]
            elif v == "lookalike_other" and tool in ("issue_create", "issue_comment"):
                vm = [m for m in ms if m in AMBIG]
            elif v == "lagged" and tool == "issue_create":
                vm = [m for m in ms if m in AMBIG]
            elif v == "agg_shared" and SPECS[tool].cls == "U":
                vm = [m for m in ms if m in AMBIG]
            else:
                vm = []
            out += [(st.name, m, v) for m in vm]
    return out


def build(templates, instances, modes, variants, conditions, driver="scripted", repeats=1, deadline=4.0,
          delay=12.0, delta=20.0, focal=None) -> list[EpisodeConfig]:
    eps = []
    for t in templates:
        for i in instances:
            task = make(t, i)
            for (fs, m, v) in worlds(task, modes, variants, focal):
                seed = int(episode_id(t, i, fs, m, v), 16) % (2 ** 31)
                d = tail_delay(seed) if m == "late_tail" else delay
                for c in conditions:
                    for r in range(repeats):
                        eid = episode_id(t, i, fs, m, v, c, driver, r)
                        eps.append(EpisodeConfig(episode_id=eid, run=eid, task=task.to_dict(), focal_step=fs, mode=m,
                                                 variant=v, condition=c, driver=driver, deadline_s=deadline,
                                                 delay_s=d, delta_assumed_s=delta, repeat=r, seed=seed))
    return eps


def _token_file() -> Path:
    tok = stack_secrets()["gitea_token"]
    p = SECRETS / "gitea_token.txt"
    p.write_text(tok, encoding="utf-8")
    return p


def run_episode(cfg: EpisodeConfig, out_root: Path, worker: int, token_file: Path) -> dict:
    task = TaskInstance.from_dict(cfg.task)
    state = out_root / "episodes" / cfg.episode_id
    if state.exists():                                   # a half-finished earlier attempt: start clean
        import shutil
        shutil.rmtree(state, ignore_errors=True)
    state.mkdir(parents=True, exist_ok=True)
    # A fresh service namespace for EVERY attempt: a re-run must never see the effects of an earlier attempt
    # (same repo, mail address, run id, stream keys, object prefix would otherwise be shared).
    import uuid as _uuid
    cfg.run = f"{cfg.episode_id[:8]}-{_uuid.uuid4().hex[:6]}"
    cfg.state_dir, cfg.worker, cfg.gitea_token_file = str(state), worker, str(token_file)
    token = token_file.read_text(encoding="utf-8").strip()
    tb = Toolbox(cfg.run, token, variant=cfg.variant)
    tb.gitea.ensure_repo()
    if any(s.tool == "transfer_create" for s in task.steps):
        led = Ledger()
        led.setup_accounts()
        cfg.tb_accounts = [str(led.debit), str(led.credit)]
    cfg.save(state / "config.json")
    t0 = time.time()
    rcs = []
    for attempt in range(2):
        p = subprocess.run([sys.executable, str(ROOT / "code" / "run.py"), "child", str(state / "config.json")],
                           capture_output=True, text=True, timeout=1800, encoding="utf-8", errors="replace")
        (state / f"child{attempt}.log").write_text(redact((p.stdout or "") + (p.stderr or ""))[-20000:], encoding="utf-8")
        rcs.append(p.returncode)
        if p.returncode == CRASH_EXIT and cfg.mode in CRASH_MODES and attempt == 0:
            continue
        break
    time.sleep(QUIESCE_S)
    rep_path = state / "report_resumed.json" if (state / "report_resumed.json").exists() else state / "report.json"
    report = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {"claimed": "no_report"}
    tbg = Toolbox(cfg.run, token, variant=cfg.variant, tb_accounts=cfg.tb_accounts)
    try:
        g = grade(state, task, tbg)
    except Exception as e:  # noqa: BLE001
        g = {"grade_error": f"{type(e).__name__}: {str(e)[:200]}"}
    rec = {k: getattr(cfg, k) for k in ("episode_id", "run", "focal_step", "mode", "variant", "condition", "driver",
                                         "delay_s", "delta_assumed_s", "deadline_s", "repeat", "seed")}
    rec.update(template=task.template, instance=task.instance,
               focal_tool=next(s.tool for s in task.steps if s.name == cfg.focal_step),
               child_rcs=rcs, wall_s=round(time.time() - t0, 2),
               agent_elapsed_s=report.get("agent_elapsed_s"), claimed=report.get("claimed"),
               uncertain=report.get("uncertain"), step_status=report.get("step_status"),
               layer_compensations=report.get("compensations"), duplicates_reported=report.get("duplicates_reported"),
               restarts=report.get("restarts"), grade=g)
    if cfg.driver != "scripted":
        rec.update({k: report.get(k) for k in ("usd", "paid_calls", "cached_calls", "tokens_in", "tokens_cached",
                                               "tokens_out", "tokens_reasoning", "incomplete_turns", "model_latency_s",
                                               "nudges", "turns", "tool_calls", "stop", "layer_uncertain",
                                               "resolved_models", "step_map", "model_summary")})
    if "residual_inconsistent" in g:
        rec["overclaim"] = rec["claimed"] == "completed" and g["residual_inconsistent"]
    rec["focal_cls"] = SPECS[rec["focal_tool"]].cls
    rec["focal_reversible"] = SPECS[rec["focal_tool"]].reversible
    return rec


def unfinished(r: dict) -> bool:
    """A record that does not measure the condition: a runner or grading error, an LLM episode stopped by the
    provider (model error) or the spending cap, or (amendment 2) an agent process that crashed with an exception
    and left no report. Such records are archived and the episode is run again."""
    crashed = r.get("claimed") == "no_report" and bool(r.get("child_rcs")) and r["child_rcs"][-1] not in (0,)
    return ("error" in r or "grade_error" in (r.get("grade") or {}) or crashed
            or str(r.get("stop") or "").startswith(("model error", "budget")))


def invalidate(name: str, template: str | None = None, condition: str | None = None, reason: str = "",
               where: dict | None = None) -> int:
    """Move finished records matching the filter to episodes_invalidated.jsonl so they are run again."""
    out = RUNS / name
    f = out / "episodes.jsonl"
    if not f.exists():
        return 0
    keep, bad = [], []
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        hit = (template is None or r.get("template") == template) and (condition is None or r.get("condition") == condition)
        for k, v in (where or {}).items():
            vals = v.split("|")
            hit = hit and str(r.get(k)) in vals
        (bad if hit else keep).append(line)
    if bad:
        with (out / "episodes_invalidated.jsonl").open("a", encoding="utf-8") as g:
            for line in bad:
                r = json.loads(line)
                r["invalidated_reason"] = reason
                g.write(json.dumps(r) + "\n")
        f.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
    return len(bad)


def run_grid(name: str, episodes: list[EpisodeConfig], workers: int = 3, meta: dict | None = None) -> Path:
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    done_file = out / "episodes.jsonl"
    done = set()
    if done_file.exists():
        # Unfinished episodes (runner/grading error, model error, budget stop) are not finished: archive their records and re-run them,
        # so episodes.jsonl keeps exactly one record per episode.
        keep, errored = [], []
        for line in done_file.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if unfinished(r):
                errored.append(line)
            else:
                keep.append(line)
                done.add(r["episode_id"])
        if errored:
            with (out / "episodes_errored.jsonl").open("a", encoding="utf-8") as f:
                f.write("\n".join(errored) + "\n")
            done_file.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
    todo = [e for e in episodes if e.episode_id not in done]
    for e in todo:
        e.extra["grid"] = name
    (out / "grid.json").write_text(json.dumps({"n_total": len(episodes), "n_todo": len(todo),
                                               "started": time.strftime("%Y-%m-%d %H:%M:%S"), **(meta or {})},
                                              indent=1))
    if meta and meta.get("freeze_digest"):                 # one line per (re)start: which freeze each part ran under
        append_jsonl(out / "freeze_log.jsonl", {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **meta})
    Toxi().ensure_proxies()
    from .services.objstore import Objects
    Objects("actrec").ensure_bucket()
    token_file = _token_file()
    lock = threading.Lock()
    slots = list(range(workers))
    counter = {"n": 0}

    def job(cfg):
        with lock:
            w = slots.pop()
        try:
            rec = run_episode(cfg, out, w, token_file)
        except Exception as e:  # noqa: BLE001
            rec = {"episode_id": cfg.episode_id, "condition": cfg.condition, "mode": cfg.mode,
                   "error": f"{type(e).__name__}: {str(e)[:300]}"}
        finally:
            with lock:
                slots.append(w)
        with lock:
            append_jsonl(done_file, rec)
            counter["n"] += 1
            g = rec.get("grade", {})
            print(f"[{counter['n']}/{len(todo)}] {rec.get('template')} {rec.get('focal_step')} {rec.get('mode')} "
                  f"{rec.get('variant')} {rec.get('condition'):4s} inconsistent={g.get('residual_inconsistent')} "
                  f"dups={g.get('dup_execs')} claimed={rec.get('claimed')} {rec.get('wall_s')}s"
                  + (f" ERROR {rec['error']}" if "error" in rec else ""), flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(job, todo))
    return out


LLM_MODELS = ["gpt-5.6-luna", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-4.1-mini"]
TEST_INSTANCES = [100, 101, 102]
ALL_TEMPLATES = DEV + TEST_ONLY

PRESETS = {
    "smoke": dict(templates=DEV, instances=[0], modes=["none", "lost_ack", "late", "crash_after", "partial"],
                  variants=["exact"], conditions=["C0", "C3", "C6", "S"], focal="first_two"),
    "dev": dict(templates=DEV, instances=[0], modes=NET_MODES + ["partial"] + CRASH_MODES,
                variants=["exact", "lookalike", "lagged", "agg_shared"], conditions=ALL_CONDITIONS, focal=None),
    # paid pilot (dev split only): first write step of T1, T4, T7; four worlds; five conditions incl. reflection
    "llm_pilot": dict(templates=["T1", "T4", "T7"], instances=[0], modes=["none", "lost_ack", "not_exec_err", "late"],
                      variants=["exact"], conditions=["C0", "C5", "C6", "C7", "S"], focal="first_one"),
    # cheap per-model check: tool-calling reliability and cost of the other models
    "llm_models": dict(templates=["T1", "T7"], instances=[0], modes=["lost_ack", "late"], variants=["exact"],
                       conditions=["C0", "S"], focal="first_one"),
    # dev-split LLM check of the hard worlds added to test_llm after the pilot (long-tail delay, crash + replay,
    # look-alikes, lagged search, shared aggregates); harness check only
    "llm_hardcheck": dict(templates=["T1", "T4", "T7"], instances=[0], modes=["late_tail", "crash_after", "lost_ack"],
                          variants=["exact", "lookalike", "lagged", "agg_shared"], conditions=["C0", "S"],
                          focal="first_one"),
    # harness check of the test-only templates on a NON-test instance (900); only errors and the fault-free
    # sanity table are inspected (docs/PROTOCOL.md §4)
    "harness_check": dict(templates=TEST_ONLY, instances=[900], modes=["none", "lost_ack", "late", "crash_after"],
                          variants=["exact", "lookalike"], conditions=["C0", "C3", "C4", "C6", "S", "O"], focal=None),
    # ---- frozen test split (docs/PROTOCOL.md) ----
    "test_scripted": dict(templates=ALL_TEMPLATES, instances=TEST_INSTANCES, modes=NET_MODES + ["partial"] + CRASH_MODES,
                          variants=["exact", "lookalike", "lagged", "agg_shared"], conditions=ALL_CONDITIONS, focal=None),
    # first two write steps of every template; easy worlds (exact reads, short delay) and hard ones (long-tail delay,
    # crash + replay, look-alikes, lagged search, shared aggregates); 670 episodes per repeat
    "test_llm": dict(templates=ALL_TEMPLATES, instances=[TEST_INSTANCES[0]],
                     modes=["none", "lost_ack", "not_exec_err", "late", "late_tail", "crash_after"],
                     variants=["exact", "lookalike", "lagged", "agg_shared"],
                     conditions=["C0", "C5", "C6", "C7", "S"], focal="first_two"),
    # ---- follow-up study (docs/07_followup_plan.md; freeze v3) ----
    # (a) the tag wait-then-count baseline and Settle-as-designed on every world of the frozen scripted grid
    "followup_a": dict(templates=ALL_TEMPLATES, instances=TEST_INSTANCES, modes=NET_MODES + ["partial"] + CRASH_MODES,
                       variants=["exact", "lookalike", "lagged", "agg_shared"], conditions=["Sp", "S2"], focal=None),
    # (b) delayed transport redelivery (the duplicate arrives 8 s after the acknowledged request)
    "followup_b": dict(templates=ALL_TEMPLATES, instances=TEST_INSTANCES, modes=["redelivery_late"], variants=["exact"],
                       conditions=["C0", "Gw", "S", "Sp", "S2", "O"], focal=None),
    # (c) look-alikes from another account, against author-filtered content checks
    "followup_c": dict(templates=ALL_TEMPLATES, instances=TEST_INSTANCES, modes=AMBIG, variants=["lookalike_other"],
                       conditions=["C5", "C5a", "Gw", "Gwa", "S", "Sp", "S2"], focal=None),
    # (d) Settle with plan-free intent identity under the three original models (test_llm worlds)
    "followup_llm_np": dict(templates=ALL_TEMPLATES, instances=[TEST_INSTANCES[0]],
                            modes=["none", "lost_ack", "not_exec_err", "late", "late_tail", "crash_after"],
                            variants=["exact", "lookalike", "lagged", "agg_shared"], conditions=["S-np"],
                            focal="first_two"),
    # (e) a frontier model, plain agent vs plan-free Settle (test_llm worlds)
    "followup_llm_frontier": dict(templates=ALL_TEMPLATES, instances=[TEST_INSTANCES[0]],
                                  modes=["none", "lost_ack", "not_exec_err", "late", "late_tail", "crash_after"],
                                  variants=["exact", "lookalike", "lagged", "agg_shared"], conditions=["C0", "S-np"],
                                  focal="first_two"),
}


def main(argv: list[str]) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="run.py grid")
    ap.add_argument("--name", required=True)
    ap.add_argument("--preset", choices=list(PRESETS))
    ap.add_argument("--templates")
    ap.add_argument("--instances")
    ap.add_argument("--modes")
    ap.add_argument("--variants")
    ap.add_argument("--conditions")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--driver", default="scripted", help="scripted | llm:<model>")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="only print the episode count")
    ap.add_argument("--invalidate-template", help="re-run all finished episodes of this template")
    ap.add_argument("--reason", default="")
    ap.add_argument("--invalidate-where", help="key=v1|v2;key2=v (e.g. 'condition=O' or 'template=T7;variant=lookalike')")
    a = ap.parse_args(argv)
    if a.invalidate_template:
        n = invalidate(a.name, template=a.invalidate_template, reason=a.reason)
        print(f"invalidated {n} finished episodes of {a.invalidate_template}; they will be re-run")
    if a.invalidate_where:
        w = dict(kv.split("=", 1) for kv in a.invalidate_where.split(";"))
        n = invalidate(a.name, reason=a.reason, where=w)
        print(f"invalidated {n} finished episodes matching {w}; they will be re-run")
    freeze_digest = None
    if a.preset and a.preset.startswith(("test_", "followup_")):
        from .freeze import check
        ok, diff, freeze_digest = check()
        if not ok:
            raise SystemExit("test split refused: the pre-registration freeze does not match the working tree: "
                             + ", ".join(diff[:10]))
        if any([a.templates, a.instances, a.modes, a.variants, a.conditions]) or (
                a.limit and a.preset not in ("test_llm", "followup_llm_frontier")):
            raise SystemExit("test split refused: frozen presets cannot be narrowed or changed from the command line")
    p = dict(PRESETS[a.preset]) if a.preset else dict(templates=DEV, instances=[0], modes=NET_MODES,
                                                       variants=["exact"], conditions=ALL_CONDITIONS, focal=None)
    if a.templates:
        p["templates"] = a.templates.split(",")
    if a.instances:
        p["instances"] = [int(x) for x in a.instances.split(",")]
    if a.modes:
        p["modes"] = a.modes.split(",")
    if a.variants:
        p["variants"] = a.variants.split(",")
    if a.conditions:
        p["conditions"] = a.conditions.split(",")
    focal = p.pop("focal")
    eps = []
    for t in p["templates"]:
        f = None
        if focal in ("first_two", "first_one"):
            f = [s.name for s in make(t, p["instances"][0]).steps][:2 if focal == "first_two" else 1]
        eps += build([t], p["instances"], p["modes"], p["variants"], p["conditions"], driver=a.driver,
                     repeats=a.repeats, focal=f)
    if a.limit:
        eps = eps[: a.limit]
    print(f"{len(eps)} episodes")
    if a.dry:
        from collections import Counter
        print(Counter(e.condition for e in eps))
        print(Counter(e.mode for e in eps))
        return
    if a.driver.startswith("llm:"):
        from .llm_client import PRICES, spent
        from .config import ENV
        m = a.driver.split(":", 1)[1]
        if m not in PRICES:
            raise SystemExit(f"no price on file for {m}")
        print(f"model {m}; spent so far ${spent():.4f} (project cap ${ENV.get('MAX_SPEND_USD')}), "
              f"this grid ${spent(a.name):.4f} (run cap ${ENV.get('MAX_SPEND_PER_RUN_USD')})")
    if "C4" in p["conditions"]:
        import socket
        try:
            socket.create_connection(("127.0.0.1", 7233), timeout=2).close()
        except OSError:
            raise SystemExit("C4 needs the workflow engine: run `python code/run.py stack up --engine` first")
    run_grid(a.name, eps, a.workers, meta={"preset": a.preset, "driver": a.driver, "repeats": a.repeats,
                                          "freeze_digest": freeze_digest})
