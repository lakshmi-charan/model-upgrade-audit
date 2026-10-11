"""Tool server for the mlops testbed. Runs under the SOURCE project's main Python environment (pandas 2.2,
scikit-learn 1.9, pyarrow), because the pipelines, the reference models (pickles) and the ground truth were
produced there; the model-migration venv has neither scikit-learn nor pyarrow and runs pandas 3.

Started by code/run_mlops.py as a subprocess; speaks JSON lines on stdin/stdout (one request, one reply):
  {"cmd": "hello"}                       -> versions, the source prompts (SYSTEM, TOOLS_DOC, FINAL_FMT), MAX_STEPS
  {"cmd": "open", "record": {...}}       -> initial messages exactly as dod/llm_agent.py builds them (tools mode),
                                            component names, compute budget (3 full pipeline runs)
  {"cmd": "act", "action": {...}}        -> executes one rollback/replay/effect/boundary action on the open case
                                            and returns the message content the source agent would append
  {"cmd": "spent"} / {"cmd": "close"} / {"cmd": "quit"}

Where things are read and written (the source folder is never written):
  DOD_PROJECT_ROOT = <source>           bronze data, reference artifacts, reference models, envs/p15, envs/p30
  DOD_CACHE        = <target>/runs/mlops_cache   every pipeline output this worker computes (+ _xfer temp files)
  Read-through: a component output already in <source>/data/cache (content-addressed, same keys) is read from
  there; misses are computed and written only under DOD_CACHE. Reference models are only loaded, never built
  (a missing model is an error), and dod.domains.*.save_artifact is disabled.
The code under testbeds/mlops/dod is a byte-identical copy of <source>/v2/code/dod.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
import traceback
import warnings
from pathlib import Path

TB = Path(__file__).resolve().parent
if "DOD_PROJECT_ROOT" not in os.environ or "DOD_CACHE" not in os.environ:
    raise SystemExit("mlops_worker: DOD_PROJECT_ROOT and DOD_CACHE must be set by the caller")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["DOD_REPLAY_ONLY"] = "1"          # dod.llm must never reach an API from here (it is not used anyway)
sys.path.insert(0, str(TB))
warnings.simplefilter("ignore")

_PROTO = sys.stdout                            # protocol channel; everything else printed goes to stderr
sys.stdout = sys.stderr

from dod import bench, diag, llm_agent  # noqa: E402
from dod.core import Executor, state_from_json  # noqa: E402
from dod.domains import DOMAINS, get_domain  # noqa: E402

SOURCE = Path(os.environ["DOD_PROJECT_ROOT"]).resolve()
SRC_CACHE = SOURCE / "data" / "cache"
TGT_CACHE = Path(os.environ["DOD_CACHE"]).resolve()
if TGT_CACHE == SRC_CACHE or SOURCE in TGT_CACHE.parents:
    raise SystemExit("mlops_worker: DOD_CACHE must not be inside the source project")


def _no_save(*a, **k):
    raise RuntimeError("mlops testbed: writing reference artifacts into the source project is disabled")


for _d in DOMAINS:
    try:
        get_domain(_d).save_artifact = _no_save
    except Exception:
        pass


class ReadThroughExecutor(Executor):
    """Executor whose cache lives under the target project, reading hits from the source cache first."""

    def __init__(self, pipe):
        super().__init__(pipe, cache=TGT_CACHE)
        self.src_cache = SRC_CACHE / pipe.domain
        self.src_hits = 0

    def _paths(self, node, key):
        out_p, meta_p = super()._paths(node, key)
        if not (out_p.exists() and meta_p.exists()):
            s_out, s_meta = self.src_cache / node / f"{key}.parquet", self.src_cache / node / f"{key}.json"
            if s_out.exists() and s_meta.exists():
                self.src_hits += 1
                return s_out, s_meta          # Executor.node only reads when both files exist
        return out_p, meta_p


def load_reference(domain: str, p0: str) -> bench.Reference:
    path = bench.MODELS / domain / f"p0_{p0}.pkl"
    if not path.exists():
        raise RuntimeError(f"reference model missing: {path} (build it in the source project)")
    return pickle.loads(path.read_bytes())


_EX: dict = {}
_REF: dict = {}
S: dict = {}                                   # the open episode


def _ex(domain):
    if domain not in _EX:
        _EX[domain] = ReadThroughExecutor(get_domain(domain).PIPELINE)
    return _EX[domain]


def _clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def hello(_):
    import numpy, pandas, sklearn
    return {"ok": True, "python": sys.version.split()[0], "pandas": pandas.__version__, "numpy": numpy.__version__,
            "sklearn": sklearn.__version__, "source": str(SOURCE), "cache": str(TGT_CACHE),
            "src_cache_exists": SRC_CACHE.exists(), "max_steps": llm_agent.MAX_STEPS,
            "system": llm_agent.SYSTEM, "tools_doc": llm_agent.TOOLS_DOC, "final_fmt": llm_agent.FINAL_FMT}


def open_case(req):
    r = req["record"]
    cj = r["case"]
    domain = cj["domain"]
    dom = get_domain(domain)
    ex = _ex(domain)
    case = bench.Case(cj["case_id"], domain, cj["p0"], cj["p1"], state_from_json(cj["state"]),
                      state_from_json(cj["spec"]), cj["injected"], cj["tier"], cj["meta"])
    key = (domain, case.p0)
    if key not in _REF:
        _REF[key] = load_reference(domain, case.p0)
    ref = _REF[key]
    o = dict(r["obs"])
    o["lineage"] = dom.LINEAGE
    o["components"] = [{"name": c, "inputs": dom.PIPELINE.components[c].inputs} for c in dom.PIPELINE.order()]
    obs = diag.Observation(**o)
    tb = diag.Toolbox(case, ref, ex)
    budget_s = 3.0 * max(obs.pipeline_cost_s, 0.5)       # as dod/methods.py _llm("tools")
    comps = [c["name"] for c in obs.components]
    msgs = [{"role": "system", "content": llm_agent.SYSTEM},
            {"role": "user", "content": "Monitoring report (JSON):\n" + json.dumps(llm_agent.report(obs), default=str)
             + "\n\n" + llm_agent.TOOLS_DOC + f"\nBudget: {budget_s:.1f} seconds."}]
    S.clear()
    S.update(case=case, tb=tb, comps=comps, budget_s=budget_s, hits0=ex.src_hits, domain=domain)
    return {"ok": True, "messages": msgs, "components": comps, "budget_s": budget_s,
            "limit_msg": "Budget or step limit reached. " + llm_agent.FINAL_FMT}


def act(req):
    """One tool action, exactly the body of the loop in dod/llm_agent.run (tools mode)."""
    tb, comps = S["tb"], S["comps"]
    a = req["action"]
    act_ = a.get("action")
    t0, c0 = time.perf_counter(), tb.cost_s
    try:
        if act_ == "rollback":
            res = tb.rollback(a["component"])
        elif act_ == "replay":
            res = tb.replay(a["component"])
        elif act_ == "effect":
            res = tb.effect([c for c in a.get("components", []) if c in comps])
        elif act_ == "boundary":
            res = tb.boundary(a["component"], str(a["period"]))
        else:
            res = {"error": f"unknown action {act_}"}
    except Exception as e:
        res = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    summ = llm_agent._summ(res)
    content = json.dumps({"result": summ, "spent_seconds": round(tb.cost_s, 3)}, default=str)
    return {"ok": True, "content": content, "result": _clean(json.loads(json.dumps(summ, default=str))),
            "tool_error": ("error" in res) or str(res.get("result", "")).endswith("error"),
            "spent_seconds": round(tb.cost_s, 4), "cost_s": round(tb.cost_s - c0, 4),
            "wall_s": round(time.perf_counter() - t0, 3), "budget_exhausted": tb.cost_s >= S["budget_s"]}


def spent(_):
    tb = S.get("tb")
    return {"ok": True, "spent_seconds": round(tb.cost_s, 4) if tb else 0.0,
            "budget_exhausted": bool(tb and tb.cost_s >= S["budget_s"]),
            "source_cache_hits": _EX[S["domain"]].src_hits - S["hits0"] if tb else 0}


def close(_):
    out = spent(None)
    S.clear()
    return out


CMDS = {"hello": hello, "open": open_case, "act": act, "spent": spent, "close": close}


def main():
    stdin = sys.stdin.buffer
    out = _PROTO.buffer
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line.decode("utf-8"))
            if req.get("cmd") == "quit":
                break
            rep = CMDS[req["cmd"]](req)
        except Exception as e:
            rep = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:500]}", "trace": traceback.format_exc()[-1500:]}
        out.write((json.dumps(rep, default=str) + "\n").encode("utf-8"))
        out.flush()


if __name__ == "__main__":
    main()
