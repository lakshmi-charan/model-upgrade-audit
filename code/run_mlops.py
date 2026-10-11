"""Run the mlops (drift-or-defect diagnosis) testbed for one or more models, one temperature and k repeats.

The agent is the source project's tools-mode LLM diagnostic agent (drift-or-defect, v2/code/dod/llm_agent.py,
"llm_tools"): it reads a monitoring report of a degraded production model and its data pipeline, may run up to
8 diagnostic actions (rollback / replay / effect / boundary, executed for real on the pipeline under a compute
budget of three full pipeline runs), and ends with {"action": "final", "fix": [...], "retrain": bool}.

Usage:
  run_mlops.py --models gpt-4.1-nano,gpt-5.6-luna --subset pilot --repeats 3 --temperature 0 --workers 4
  run_mlops.py --check-env               # data, envs, code copy, and one offline tool action (no LLM; free)

Output: runs/mlops/<subset>/<label>/T<temp>/r<k>/<task_id>.json (one file per episode) and
        runs/mlops/<subset>/episodes.jsonl (one summary row per episode, appended).
Episodes already on disk are skipped, so a job can be re-queued after a quota stop or crash.

Pipelines run in a tool-server subprocess (testbeds/mlops/mlops_worker.py) under the SOURCE project's venv
(<model-migration>/../drift-or-defect/.venv), reading the source's data, models and envs in place; every cache
or temporary file it writes goes to runs/mlops_cache/ in this project. See testbeds/mlops/README.md.
Env overrides: MM_MLOPS_SOURCE (source project root), MM_MLOPS_PYTHON (python for the tool server),
MM_MLOPS_CACHE (pipeline cache dir; must be outside the source project).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(os.environ.get("MM_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
TB = ROOT / "testbeds" / "mlops"
SOURCE = Path(os.environ.get("MM_MLOPS_SOURCE") or (ROOT.parent / "drift-or-defect")).resolve()
CACHE_DIR = Path(os.environ.get("MM_MLOPS_CACHE") or (ROOT / "runs" / "mlops_cache")).resolve()
DOMAINS = ("tlc", "news", "bts", "hmda", "divvy")
TESTBED = "mlops"
sys.path.insert(0, str(ROOT / "code"))

from mm import llm as mmllm  # noqa: E402

_LOCK = threading.Lock()
WORKER_TIMEOUT_S = float(os.environ.get("MM_MLOPS_WORKER_TIMEOUT_S", "900"))


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def source_python() -> Path:
    p = os.environ.get("MM_MLOPS_PYTHON")
    return Path(p) if p else _venv_python(SOURCE / ".venv")


# ------------------------------------------------------------------ tool server (subprocess)
class Worker:
    """One tool-server process (stateful: one open episode at a time)."""

    def __init__(self):
        env = {k: v for k, v in os.environ.items()
               if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
        env.update({"DOD_PROJECT_ROOT": str(SOURCE), "DOD_CACHE": str(CACHE_DIR), "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        env.pop("PYTHONPATH", None)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen([str(source_python()), "-u", str(TB / "mlops_worker.py")], cwd=str(TB), env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        self.info = self.call({"cmd": "hello"}, timeout=300)

    def _reader(self):
        for line in self.proc.stdout:
            self.q.put(line)
        self.q.put(None)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def call(self, req: dict, timeout: float = WORKER_TIMEOUT_S) -> dict:
        if not self.alive():
            raise RuntimeError("tool server is not running")
        self.proc.stdin.write((json.dumps(req, default=str) + "\n").encode("utf-8"))
        self.proc.stdin.flush()
        try:
            line = self.q.get(timeout=timeout)
        except queue.Empty:
            self.kill()
            raise RuntimeError(f"tool server timeout after {timeout:.0f}s on {req.get('cmd')}")
        if line is None:
            raise RuntimeError(f"tool server exited (code {self.proc.poll()})")
        rep = json.loads(line.decode("utf-8"))
        if not rep.get("ok"):
            raise RuntimeError(f"tool server error on {req.get('cmd')}: {rep.get('error')}")
        return rep

    def kill(self):
        try:
            self.proc.kill()
        except Exception:
            pass

    def quit(self):
        try:
            self.proc.stdin.write(b'{"cmd": "quit"}\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        except Exception:
            self.kill()


class Pool:
    def __init__(self, n: int):
        self.free: queue.Queue = queue.Queue()
        self.all: list[Worker] = []
        self.n, self.made = n, 0
        self.lock = threading.Lock()

    def get(self) -> Worker:
        with self.lock:
            if self.free.empty() and self.made < self.n:
                self.made += 1
                w = Worker()
                self.all.append(w)
                return w
        return self.free.get()

    def put(self, w: Worker):
        if not w.alive():                       # replace a dead server so the pool keeps its size
            w = Worker()
            with self.lock:
                self.all.append(w)
        self.free.put(w)

    def close(self):
        for w in self.all:
            if w.alive():
                w.quit()


# ------------------------------------------------------------------ cases and subsets
def all_cases() -> dict:
    p = TB / "data" / "cases.jsonl"
    return {r["task_id"]: r for r in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())}


def make_subset(name: str) -> list[str]:
    """Frozen, seeded subsets (seed 20261006), built once by testbeds/mlops/build_cases.py from the source's
    test cases: smoke 4, pilot 40, main 100 (disjoint from pilot), stratified by domain x oracle type."""
    p = TB / "subsets" / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"unknown subset {name}: {p} not found (subsets are frozen by testbeds/mlops/build_cases.py)")
    return json.loads(p.read_text(encoding="utf-8"))["ids"]


# ------------------------------------------------------------------ agent (port of dod/llm_agent.run, tools mode)
def parse(text: str) -> dict:
    """dod/llm_agent._parse (verbatim logic); non-dict JSON counts as unparseable."""
    try:
        a = json.loads(text)
    except Exception:
        a = {}
        s, e = text.find("{"), text.rfind("}")
        if s >= 0 and e > s:
            try:
                a = json.loads(text[s:e + 1])
            except Exception:
                pass
    return a if isinstance(a, dict) else {}


def valid_action(a: dict, comps: list[str]) -> bool:
    act = a.get("action")
    if act in ("rollback", "replay"):
        return a.get("component") in comps
    if act == "effect":
        cs = a.get("components")
        return isinstance(cs, list) and all(c in comps for c in cs)
    if act == "boundary":
        return a.get("component") in comps and a.get("period") is not None
    return False


def validate_final(a: dict, comps: list[str]):
    """Returns (decision, problem). A valid final decision is a JSON object whose action is "final" (or absent),
    with "fix" a list of pipeline component names (matched case-insensitively) and "retrain" a boolean."""
    if not a:
        return None, "unparseable"
    if a.get("action", "final") != "final":
        return None, "not_final"
    fix, rt = a.get("fix"), a.get("retrain")
    if not isinstance(fix, list):
        return None, "fix_not_list"
    if isinstance(rt, str) and rt.strip().lower() in ("true", "false"):
        rt = rt.strip().lower() == "true"
    if not isinstance(rt, bool):
        return None, "retrain_not_bool"
    canon = {c.lower(): c for c in comps}
    out, bad = [], []
    for c in fix:
        k = str(c).strip().lower()
        (out.append(canon[k]) if k in canon else bad.append(str(c)))
    if bad:
        return None, f"unknown_component:{','.join(bad)[:100]}"
    return {"fix": sorted(set(out)), "retrain": rt, "reason": str(a.get("reason", ""))[:300]}, None


def grade(dec: dict | None, status: str, forced: bool, oracle: dict) -> dict:
    """success = the source's 'correct' endpoint (consequential ⊆ fix ⊆ effective, and retrain == oracle);
    category per testbeds/mlops/README.md."""
    C, E, Ro = set(oracle["fix"]), set(oracle["effective"]), bool(oracle["retrain"])
    if status != "completed" or dec is None:
        return {"success": 0, "exact_match": 0, "category": "interface", "flags": {}}
    F, R = set(dec["fix"]), bool(dec["retrain"])
    extra, missing = sorted(F - E), sorted(C - F)
    xr, mr = R and not Ro, Ro and not R
    success = int(C <= F <= E and R == Ro)
    flags = {"extra_fix": extra, "missing_fix": missing, "extra_retrain": xr, "missing_retrain": mr,
             "fix_inconsequential": sorted((F & E) - C), "forced_final": forced}
    if success:
        cat = "ok"
    elif forced:
        cat = "interface"
    elif extra and missing:
        cat = "wrong_write"
        flags["overlap"] = "partial" if F & C else "disjoint"
    elif (missing and xr) or (extra and mr):
        cat = "wrong_answer"
    elif extra or xr:
        cat = "unrequested_write"
    elif missing or mr:
        cat = "missing_write"
    else:
        cat = "other"
    return {"success": success, "exact_match": int(F == C and R == Ro), "category": cat, "flags": flags}


def run_agent(spec: dict, rec: dict, temperature: float, rep: int, w: Worker) -> dict:
    o = w.call({"cmd": "open", "record": rec})
    msgs, comps, limit_msg = o["messages"], o["components"], o["limit_msg"]
    max_steps = int(w.info["max_steps"])
    salt = f"{TESTBED}|T{temperature:g}|r{rep}|{rec['task_id']}"
    extra_tok = 2000 if spec.get("reasoning") not in (None, "off") else 0     # as the source: room for reasoning
    turns, trace = [], []
    cost, tin, tout = 0.0, 0, 0
    steps, n_invalid, forced, budget_hit = 0, 0, False, False

    def ask(max_tokens: int, stage: str) -> dict:
        nonlocal cost, tin, tout
        r = mmllm.chat(spec, msgs, temperature=temperature, max_tokens=max_tokens + extra_tok, salt=salt,
                       stage=stage)
        cost += float(r.get("cost") or 0.0)
        u = r.get("usage") or {}
        tin += int(u.get("prompt_tokens") or 0)
        tout += int(u.get("completion_tokens") or 0)
        text = r.get("text") or ""
        turns.append({"text": text[:4000], "usage": u, "cost": r.get("cost"), "cached": r.get("cached"),
                      "finish_reason": r.get("finish_reason")})
        msgs.append({"role": "assistant", "content": text})
        return parse(text)

    while True:
        a = ask(600, f"{TESTBED}_agent")
        act = a.get("action", "final")
        if act == "final" or steps >= max_steps or budget_hit:
            if act != "final":
                forced = True
                msgs.append({"role": "user", "content": limit_msg})
                a = ask(400, f"{TESTBED}_final")
            final_raw = a
            break
        steps += 1
        ok = valid_action(a, comps)
        n_invalid += not ok
        res = w.call({"cmd": "act", "action": a})
        trace.append({"step": steps, "action": act, "args": {k: v for k, v in a.items() if k != "action"},
                      "valid": ok, "tool_error": res["tool_error"], "result": res["result"],
                      "cost_s": res["cost_s"], "spent_seconds": res["spent_seconds"], "wall_s": res["wall_s"]})
        msgs.append({"role": "user", "content": res["content"]})
        budget_hit = res["budget_exhausted"]
    sp = w.call({"cmd": "close"})
    dec, problem = validate_final(final_raw, comps)
    status = "completed" if dec else ("no_final" if problem in ("unparseable", "not_final") else "invalid_final")
    return {"status": status, "final_problem": problem, "final": dec, "final_raw": final_raw,
            "forced_final": forced, "stop_reason": ("agent" if not forced else
                                                   ("budget" if budget_hit else "step_limit")),
            "n_steps": steps, "n_invalid_actions": n_invalid, "n_llm_calls": len(turns), "trace": trace,
            "turns": turns, "agent_cost": round(cost, 6), "tokens_in": tin, "tokens_out": tout,
            "compute_spent_s": sp["spent_seconds"], "compute_budget_s": o["budget_s"],
            "source_cache_hits": sp.get("source_cache_hits"), "components": comps}


# ------------------------------------------------------------------ one episode
def run_episode(spec, rec, temperature, rep, out_file, pool: Pool):
    t0 = time.time()
    err = None
    w = pool.get()
    try:
        out = run_agent(spec, rec, temperature, rep, w)
    except (mmllm.BudgetExceeded, mmllm.QuotaExhausted, mmllm.TransientAPIError):
        try:
            w.call({"cmd": "close"}, timeout=60)
        except Exception:
            w.kill()
        pool.put(w)
        raise                       # an API failure is not a model decision: never record the episode
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:300]}"
        out = {"status": "harness_error", "final": None, "trace": [], "turns": [], "agent_cost": 0.0, "n_steps": 0,
               "n_llm_calls": 0, "forced_final": False, "tokens_in": 0, "tokens_out": 0}
        try:
            w.call({"cmd": "close"}, timeout=60)
        except Exception:
            w.kill()
    pool.put(w)
    g = grade(out.get("final"), out["status"], out.get("forced_final", False), rec["oracle"])
    ep = {"task_id": rec["task_id"], "label": spec["label"], "model": spec["model"], "provider": spec["provider"],
          "temperature": temperature, "repeat": rep, "success": g["success"], "status": out["status"], "error": err,
          "category": g["category"], "exact_match": g["exact_match"], "grade_flags": g["flags"],
          "agent_cost": out["agent_cost"], "n_steps": out["n_steps"], "actions": out["trace"],
          "final_decision": out.get("final"), "oracle": rec["oracle"],
          "domain": rec["domain"], "otype": rec["otype"], "scenario": rec["scenario"], "tier": rec["tier"],
          "wall_s": round(time.time() - t0, 2),
          **{k: v for k, v in out.items() if k not in ("trace", "final", "status", "agent_cost", "n_steps")}}
    if out_file is None:
        return ep
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(ep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, out_file)
    fd = ep.get("final_decision") or {}
    return {"task_id": rec["task_id"], "label": spec["label"], "T": temperature, "r": rep, "success": ep["success"],
            "status": ep["status"], "category": ep["category"], "exact_match": ep["exact_match"],
            "n_steps": ep["n_steps"], "n_llm_calls": ep["n_llm_calls"],
            "actions": [s["action"] for s in ep["actions"]], "fix": fd.get("fix"), "retrain": fd.get("retrain"),
            "oracle_fix": rec["oracle"]["fix"], "oracle_retrain": rec["oracle"]["retrain"],
            "cost": ep["agent_cost"], "tokens_in": ep["tokens_in"], "tokens_out": ep["tokens_out"],
            "error": err, "domain": rec["domain"], "otype": rec["otype"], "scenario": rec["scenario"]}


# ------------------------------------------------------------------ environment
def _sha_tree(d: Path) -> dict:
    return {str(p.relative_to(d)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*.py")) if "__pycache__" not in p.parts}


def check_env(subset: str = "smoke", run_tool: bool = True) -> bool:
    ok = True

    def need(cond, msg, fatal=True):
        nonlocal ok
        print(("OK   " if cond else ("FAIL " if fatal else "WARN ")) + msg, flush=True)
        if not cond and fatal:
            ok = False
    print("source project:", SOURCE)
    print("target cache  :", CACHE_DIR)
    need(SOURCE.is_dir(), "source project folder exists")
    need(SOURCE not in CACHE_DIR.resolve().parents, "cache is outside the source project")
    py = source_python()
    need(py.exists(), f"tool-server python exists: {py}")
    for env in ("p15", "p30"):
        need(_venv_python(SOURCE / "envs" / env).exists(), f"library env {env}: {_venv_python(SOURCE / 'envs' / env)}",
             fatal=(os.name == "nt"))
    for d in DOMAINS:
        need((SOURCE / "data" / "bronze" / d).is_dir(), f"bronze data {d}")
        need((SOURCE / "data" / "artifacts" / d).is_dir(), f"artifacts {d}")
        need(len(list((SOURCE / "data" / "models" / "v3" / d).glob("p0_*.pkl"))) == 2, f"reference models v3/{d}")
        need((SOURCE / "data" / "cache" / d).is_dir(), f"source pipeline cache {d} (read-through, optional)", False)
    src_code = SOURCE / "v2" / "code" / "dod"
    if src_code.is_dir():
        need(_sha_tree(src_code) == _sha_tree(TB / "dod"), "testbeds/mlops/dod is byte-identical to source v2/code/dod")
    try:
        cases = all_cases()
        ids = make_subset(subset)
        need(all(i in cases for i in ids), f"subset {subset}: {len(ids)} cases present in data/cases.jsonl")
    except SystemExit as e:
        need(False, str(e))
        return False
    try:
        mmllm.load_models()
        need(True, "config/models.yaml loads")
    except Exception as e:
        need(False, f"config/models.yaml: {type(e).__name__}: {e}")
    if not ok or not run_tool:
        return ok
    w = None
    try:
        t0 = time.time()
        w = Worker()
        i = w.info
        print(f"tool server: python {i['python']} pandas {i['pandas']} numpy {i['numpy']} sklearn {i['sklearn']} "
              f"({time.time() - t0:.1f}s)")
        need(i["pandas"].startswith("2.2"), "tool server runs pandas 2.2 (the source main environment)",
             fatal=(os.name == "nt"))
        rec = cases[ids[0]]
        o = w.call({"cmd": "open", "record": rec})
        chg = [c["component"] for c in rec["obs"].get("change_log") or []]
        comp = chg[0] if chg else o["components"][0]
        print(f"case {rec['task_id']} ({rec['domain']}, oracle {rec['oracle']['fix']}/{rec['oracle']['retrain']}): "
              f"prompt {sum(len(m['content']) for m in o['messages'])} chars, budget {o['budget_s']:.1f}s")
        for a in ({"action": "rollback", "component": comp}, {"action": "effect", "components": [comp]}):
            t1 = time.time()
            r = w.call({"cmd": "act", "action": a})
            print(f"  {json.dumps(a)} -> {r['content'][:400]} ({time.time() - t1:.1f}s wall)")
            need(not r["tool_error"], f"tool action {a['action']} executed")
        sp = w.call({"cmd": "close"})
        print(f"  compute charged {sp['spent_seconds']}s, source-cache hits {sp['source_cache_hits']}")
    except Exception as e:
        need(False, f"tool server: {type(e).__name__}: {e}")
    finally:
        if w:
            w.quit()
    return ok


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
    env_ok = check_env(a.subset, run_tool=a.check_env)
    if a.check_env or not env_ok:
        print("environment OK" if env_ok else "ENVIRONMENT NOT READY")
        sys.exit(0 if env_ok else 2)
    models = mmllm.load_models()
    cases = all_cases()
    ids = make_subset(a.subset)
    base = ROOT / "runs" / TESTBED / a.subset
    summary = base / "episodes.jsonl"
    print(f"subset {a.subset}: {len(ids)} tasks; models {a.models}; T={a.temperature}; "
          f"repeats {a.rep_start}..{a.rep_start + a.repeats - 1}; job cap ${mmllm.job_cap():.2f}")
    pool = Pool(max(1, a.workers))
    try:
        for label in [m for m in a.models.split(",") if m]:
            spec = models[label]
            for rep in range(a.rep_start, a.rep_start + a.repeats):
                d = base / label / f"T{a.temperature:g}" / f"r{rep}"
                todo = [cases[i] for i in ids if not (d / f"{i}.json").exists()]
                if not todo:
                    print(f"[{label} r{rep}] all {len(ids)} done")
                    continue
                t0, n_ok, n_err, stop = time.time(), 0, 0, None
                with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
                    futs = {ex.submit(run_episode, spec, t, a.temperature, rep, d / f"{t['task_id']}.json", pool): t
                            for t in todo}
                    for f in as_completed(futs):
                        try:
                            row = f.result()
                        except (mmllm.BudgetExceeded, mmllm.QuotaExhausted) as e:
                            stop = e
                            continue
                        except mmllm.TransientAPIError as e:
                            n_err += 1
                            with _LOCK:
                                with (summary.parent / "api_errors.jsonl").open("a", encoding="utf-8") as fh:
                                    fh.write(json.dumps({"label": label, "r": rep, "task": futs[f]["task_id"],
                                                         "error": str(e)[:300]}) + "\n")
                            continue
                        n_ok += row["success"]
                        n_err += row["error"] is not None
                        with _LOCK:
                            summary.parent.mkdir(parents=True, exist_ok=True)
                            with summary.open("a", encoding="utf-8") as fh:
                                fh.write(json.dumps(row) + "\n")
                done = sum(1 for i in ids if (d / f"{i}.json").exists())
                print(f"[{label} r{rep}] {done}/{len(ids)} done, success {n_ok}/{len(todo)} this pass, "
                      f"harness errors {n_err}, {time.time() - t0:.0f}s, job spend ${mmllm.job_spent():.4f}",
                      flush=True)
                if isinstance(stop, mmllm.QuotaExhausted):
                    print(f"[{label}] free-tier quota exhausted; skipping to the next model", flush=True)
                    break
                if stop:
                    print(f"STOPPED: {type(stop).__name__}: {stop}")
                    sys.exit(3)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
