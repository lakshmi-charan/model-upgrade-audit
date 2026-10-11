"""Run the clinical (MedAgentBench FHIR) testbed for one model, one temperature and k repeats.

Usage:
  run_clinical.py --models gpt-4.1-nano,gpt-5.6-luna --subset pilot --repeats 3 --temperature 0 --workers 4
  run_clinical.py --check-env              # verify the FHIR server and the stress setup only (free)

Output: runs/clinical/<subset>/<label>/T<temp>/r<k>/<task_id>.json (one file per episode) and
        runs/clinical/<subset>/episodes.jsonl (one summary row per episode, appended).
Episodes already on disk are skipped, so a job can be re-queued after a quota stop or crash.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(os.environ.get("MM_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
TB = ROOT / "testbeds" / "clinical"
os.environ["CAVE_CONFIG"] = str(TB / "config.yaml")
sys.path.insert(0, str(TB))
sys.path.insert(0, str(ROOT / "code"))

from cave import agent, fhir, llm as shim  # noqa: E402
from cave.reference import label_units  # noqa: E402
from mm import llm as mmllm  # noqa: E402

import re  # noqa: E402

# Environment failures seen by the agent (FHIR server down or refusing connections). An episode that contains one
# was not a fair test of the model: per the protocol's exclusion rule it is not recorded and is re-run.
INFRA_RE = re.compile(r"WinError 1006|Failed to establish a new connection|Max retries exceeded|Read timed out|"
                      r"Connection aborted|RemoteDisconnected|Connection refused", re.I)


class InfraDown(RuntimeError):
    pass


_PING = {"t": 0.0, "ok": False}


def fhir_ok(max_age_s: float = 15.0) -> bool:
    with _LOCK:
        if time.time() - _PING["t"] < max_age_s and _PING["ok"]:
            return True
    ok = False
    for _ in range(3):
        try:
            ok = fhir.ping()
        except Exception:
            ok = False
        if ok:
            break
        time.sleep(2)
    with _LOCK:
        _PING.update(t=time.time(), ok=ok)
    return ok


agent.model_of = lambda name: name
agent.salt_of = lambda name: None
_LOCK = threading.Lock()


# ------------------------------------------------------------------ tasks and subsets
def all_tasks() -> dict:
    tasks = json.loads((TB / "data" / "base_tasks.json").read_text(encoding="utf-8"))
    tasks += json.loads((TB / "data" / "stress_tasks.json").read_text(encoding="utf-8"))
    live = TB / "data" / "refs_live.json"     # references recomputed on the live server (code/clinical_refs.py)
    if live.exists():
        refs = json.loads(live.read_text(encoding="utf-8"))
        for t in tasks:
            t["ref"] = refs.get(t["id"], t["ref"])
    return {t["id"]: t for t in tasks}


def make_subset(name: str, tasks: dict) -> list[str]:
    """Frozen, seeded subsets. Action-bearing templates are over-weighted (they can produce side-effect flips)."""
    p = TB / "subsets" / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))["ids"]
    rng = random.Random(20261006)
    by_group: dict[str, list[str]] = {}
    for tid, t in tasks.items():
        g = t["id"].split("_")[0] if t.get("stressor", "base") == "base" else t["id"].rsplit("_", 1)[0]
        by_group.setdefault(g, []).append(tid)
    plan = {
        "smoke": {"task3": 1, "task5": 1, "task9": 1, "S6_duplicate": 1},
        "pilot": {"task1": 3, "task2": 3, "task4": 3, "task6": 3, "task7": 3,
                  "task3": 6, "task5": 6, "task8": 6, "task9": 6, "task10": 6,
                  "S4_allergy": 4, "S5_renal": 4, "S6_duplicate": 4, "S7_lookalike": 3},
        "main": {"task1": 4, "task2": 4, "task4": 4, "task6": 4, "task7": 4,
                 "task3": 8, "task5": 12, "task8": 8, "task9": 12, "task10": 12,
                 "S4_allergy": 7, "S5_renal": 7, "S6_duplicate": 7, "S7_lookalike": 7},
    }[name]
    exclude = set()
    if name == "main":                       # confirmatory set: disjoint from the pilot (and smoke) tasks
        for other in ("pilot", "smoke"):
            q = TB / "subsets" / f"{other}.json"
            if q.exists():
                exclude |= set(json.loads(q.read_text(encoding="utf-8"))["ids"])
    ids = []
    for g, n in plan.items():
        pool = sorted([x for x in by_group.get(g, []) if x not in exclude], key=lambda s: (len(s), s))
        ids += rng.sample(pool, min(n, len(pool)))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"name": name, "seed": 20261006, "plan": plan, "ids": ids}, indent=1), encoding="utf-8")
    return ids


# ------------------------------------------------------------------ environment
def check_env() -> bool:
    """Bring up / verify the FHIR server (code/clinical_env.py) and point the testbed at it."""
    import clinical_env
    ok = clinical_env.main()
    rt = TB / "runtime.json"
    if ok and rt.exists():
        from cave import common
        common.cfg()["fhir_api_base"] = json.loads(rt.read_text(encoding="utf-8"))["fhir_api_base"]
        ok = fhir.ping()
    print("FHIR base:", fhir.base(), "reachable:", ok, flush=True)
    return ok


# ------------------------------------------------------------------ one episode
def run_episode(spec, task, temperature, rep, out_file):
    shim.set_active(spec, salt=f"T{temperature}|r{rep}", temperature=temperature)
    t0 = time.time()
    err = None
    if not fhir_ok(0.0):
        raise InfraDown("FHIR server not reachable before the episode")
    try:
        rec = agent.run_task(spec["label"], task)
        bad = [s for s in rec.get("steps", []) if INFRA_RE.search(str(s.get("response", "")))]
        if bad:
            with _LOCK:
                _PING.update(t=0.0, ok=False)
            raise InfraDown(f"{len(bad)} step(s) hit a FHIR connection error")
        units, success = label_units(task["ref"], rec["units"])
        rec["units"] = units
    except (mmllm.BudgetExceeded, mmllm.QuotaExhausted, mmllm.TransientAPIError, InfraDown):
        raise                       # never write an episode the model did not really get to play
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:300]}"
        rec, success = {"steps": [], "units": [], "status": "harness_error", "agent_cost": 0.0, "n_rounds": 0}, 0
    rec.update({"task_id": task["id"], "label": spec["label"], "model": spec["model"], "provider": spec["provider"],
                "temperature": temperature, "repeat": rep, "success": int(bool(success)), "error": err,
                "stressor": task.get("stressor", "base"), "template": task["template"],
                "wall_s": round(time.time() - t0, 2)})
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    posts = [u for u in rec["units"] if u["type"] == "POST"]
    row = {"task_id": task["id"], "label": spec["label"], "T": temperature, "r": rep, "success": rec["success"],
           "status": rec["status"], "n_rounds": rec["n_rounds"], "n_post": len(posts),
           "post_resources": [u.get("resource") for u in posts], "cost": rec["agent_cost"], "error": err,
           "template": task["template"], "stressor": rec["stressor"]}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--subset", default="smoke")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--rep-start", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check-env", action="store_true")
    a = ap.parse_args()
    env_ok = check_env()
    if a.check_env or not env_ok:
        print("environment OK" if env_ok else "ENVIRONMENT NOT READY")
        sys.exit(0 if env_ok else 2)
    models = mmllm.load_models()
    tasks = all_tasks()
    ids = make_subset(a.subset, tasks)
    base = ROOT / "runs" / "clinical" / a.subset
    summary = base / "episodes.jsonl"
    print(f"subset {a.subset}: {len(ids)} tasks; models {a.models}; T={a.temperature}; "
          f"repeats {a.rep_start}..{a.rep_start + a.repeats - 1}; job cap ${mmllm.job_cap():.2f}")
    for label in [m for m in a.models.split(",") if m]:
        spec = models[label]
        quota_hit = False
        for rep in range(a.rep_start, a.rep_start + a.repeats):
            d = base / label / f"T{a.temperature:g}" / f"r{rep}"
            for attempt in range(6):          # repeat the pass while episodes are rejected for infrastructure errors
                todo = [tasks[i] for i in ids if not (d / f"{i}.json").exists()]
                if not todo:
                    print(f"[{label} r{rep}] all {len(ids)} done")
                    break
                if not fhir_ok(0.0):
                    print("FHIR down; re-checking the environment", flush=True)
                    if not check_env():
                        print("STOPPED: FHIR server not reachable")
                        sys.exit(4)
                t0, n_ok, n_err, n_infra, stop = time.time(), 0, 0, 0, None
                with ThreadPoolExecutor(max_workers=a.workers) as ex:
                    futs = {ex.submit(run_episode, spec, t, a.temperature, rep, d / f"{t['id']}.json"): t
                            for t in todo}
                    for f in as_completed(futs):
                        try:
                            row = f.result()
                        except (mmllm.BudgetExceeded, mmllm.QuotaExhausted) as e:
                            stop = e
                            continue
                        except InfraDown as e:
                            n_infra += 1
                            with _LOCK:
                                with (base / "infra_errors.jsonl").open("a", encoding="utf-8") as fh:
                                    fh.write(json.dumps({"t": time.time(), "label": label, "r": rep,
                                                         "task": futs[f]["id"], "error": str(e)[:300]}) + "\n")
                            continue
                        except mmllm.TransientAPIError as e:
                            n_err += 1
                            with _LOCK:
                                with (base / "api_errors.jsonl").open("a", encoding="utf-8") as fh:
                                    fh.write(json.dumps({"label": label, "r": rep, "task": futs[f]["id"],
                                                         "error": str(e)[:300]}) + "\n")
                            continue
                        n_ok += row["success"]
                        n_err += row["error"] is not None
                        with _LOCK:
                            with summary.open("a", encoding="utf-8") as fh:
                                fh.write(json.dumps(row) + "\n")
                done = sum(1 for i in ids if (d / f"{i}.json").exists())
                print(f"[{label} r{rep}] {done}/{len(ids)} done, success {n_ok}/{len(todo)} this pass, "
                      f"harness errors {n_err}, infra-rejected {n_infra}, {time.time() - t0:.0f}s, "
                      f"job spend ${mmllm.job_spent():.4f}", flush=True)
                if isinstance(stop, mmllm.BudgetExceeded):
                    print(f"STOPPED: {type(stop).__name__}: {stop}")
                    sys.exit(3)
                if isinstance(stop, mmllm.QuotaExhausted):
                    quota_hit = True
                    break
                if n_infra == 0:
                    break
                time.sleep(20)
            if quota_hit:
                print(f"[{label}] free-tier quota exhausted; skipping to the next model (re-queue later)", flush=True)
                break


if __name__ == "__main__":
    main()
