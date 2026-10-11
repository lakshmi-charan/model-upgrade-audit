"""Generic LLM diagnostic agent (baseline). Two variants:
  report : sees the same monitoring report, lineage, change log and schema events as every other method; answers once.
  tools  : additionally may call the same re-execution tools as the method (rollback, replay, effect, boundary)
           under the same compute budget, for at most MAX_STEPS turns.
Model and provider come from DOD_LLM_PROVIDER / DOD_LLM_MODEL (default: local Ollama qwen2.5:3b, free)."""
from __future__ import annotations

import json
import math
import os

from dod import llm
from dod.methods import DELTA_RETRAIN, Decision

MAX_STEPS = 8

SYSTEM = """You are an experienced MLOps engineer on call for a production ML model whose quality may have degraded.
The model is fed by a data pipeline (a DAG of components). Between the reference period (when the model was trained)
and now, the world may have changed (covariate, label or concept drift), pipeline components may have changed (code,
config, reference data, library environment; changes can be harmless or harmful), and upstream data sources may have
changed their published schema. Some components may be stale (should have been updated for a source change but were not).
Your job: decide which components must be fixed (possibly none) and whether the model should be retrained.
Retraining helps for genuine drift but does not repair a broken pipeline. Fixing a component means an engineer restores
or repairs it. Be precise: blame only components you believe are actually defective.
Always answer with a single JSON object."""

FINAL_FMT = ('{"action": "final", "fix": [<component names>], "retrain": true|false, '
             '"reason": "<one sentence>"}')
TOOLS_DOC = """You may run experiments before answering. Reply with ONE JSON object per turn:
{"action": "rollback", "component": C}  re-run component C at its reference (known-good) version on its current inputs;
                                        tells you whether C's current output differs from the reference version's output.
{"action": "replay", "component": C}    run C's current version on the reference period's inputs; tells you whether C's
                                        behaviour changed on the old data.
{"action": "effect", "components": [..]} model accuracy on current data if these components were rolled back.
{"action": "boundary", "component": C, "period": P}  compare C's output just before and at source-schema event P with
                                        ordinary period-to-period change.
""" + FINAL_FMT + """
Each experiment costs compute; budget is given in seconds. Stop when you are confident."""


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(x, nd) if isinstance(x, float) else x


def report(obs) -> dict:
    feats = {}
    for f, r in obs.features.items():
        if f.startswith("_"):
            continue
        feats[f] = {k: _fmt(v) for k, v in r.items()}
    nd = {c: {k: v for k, v in d.items() if k in ("max_psi", "max_missing_jump", "rows_ratio", "missing_cols")}
          for c, d in obs.extra.get("node_drift", {}).items()}
    return {
        "reference_period": obs.p0, "current_period": obs.p1,
        "accuracy_reference": _fmt(obs.ref_perf), "accuracy_current": _fmt(obs.cur_perf),
        "pipeline_crash": obs.cur_crash,
        "rows": obs.features.get("_rows"),
        "feature_monitoring": feats,
        "component_output_drift": nd,
        "pipeline": obs.components, "lineage_feature_to_components": obs.lineage,
        "change_log": obs.change_log, "source_schema_events": obs.source_events,
        "component_cost_seconds": obs.component_cost_s,
        "alarm_threshold_accuracy_drop": DELTA_RETRAIN,
    }


def _parse(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except Exception:
                pass
    return {}


def _summ(res: dict) -> dict:
    out = {k: v for k, v in res.items() if k != "diff"}
    if "diff" in res:
        out["differing_columns"] = dict(list(res["diff"].get("col_diff_frac", {}).items())[:12])
        out["rows"] = res["diff"].get("rows")
    if "columns" in out:
        cols = out.pop("columns")
        top = sorted(cols.items(), key=lambda kv: -kv[1]["boundary_psi"])[:8]
        out["columns_top"] = dict(top)
    return out


def run(obs, tb, dom, mode: str = "report", budget_s: float | None = None) -> Decision:
    provider = os.environ.get("DOD_LLM_PROVIDER", "ollama")
    model = os.environ.get("DOD_LLM_MODEL", "qwen2.5:3b")
    name = f"llm_{mode}"
    comps = [c["name"] for c in obs.components]
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Monitoring report (JSON):\n" + json.dumps(report(obs), default=str)
             + "\n\n" + (TOOLS_DOC + f"\nBudget: {budget_s:.1f} seconds." if mode == "tools" else
                         "Answer with: " + FINAL_FMT)}]
    usd, calls_llm, steps, final = 0.0, 0, 0, None
    while True:
        r = llm.chat(provider, model, msgs, temperature=0.0, max_tokens=600, seed=0)
        usd += r["usd"]
        calls_llm += 1
        msgs.append({"role": "assistant", "content": r["text"]})
        a = _parse(r["text"])
        act = a.get("action", "final") if mode == "tools" else "final"
        if act == "final" or mode != "tools" or steps >= MAX_STEPS or (budget_s is not None and tb.cost_s >= budget_s):
            if act != "final" or not a:
                if mode == "tools" and act != "final":
                    msgs.append({"role": "user", "content": "Budget or step limit reached. " + FINAL_FMT})
                    r = llm.chat(provider, model, msgs, temperature=0.0, max_tokens=400, seed=0)
                    usd += r["usd"]
                    calls_llm += 1
                    a = _parse(r["text"])
            final = a
            break
        steps += 1
        try:
            if act == "rollback":
                res = tb.rollback(a["component"])
            elif act == "replay":
                res = tb.replay(a["component"])
            elif act == "effect":
                res = tb.effect([c for c in a.get("components", []) if c in comps])
            elif act == "boundary":
                res = tb.boundary(a["component"], str(a["period"]))
            else:
                res = {"error": f"unknown action {act}"}
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        msgs.append({"role": "user", "content": json.dumps({"result": _summ(res), "spent_seconds": round(tb.cost_s, 3)},
                                                           default=str)})
    fix = [c for c in (final.get("fix") or []) if c in comps] if isinstance(final.get("fix"), list) else []
    retrain = bool(final.get("retrain", False))
    return Decision(name, fix, retrain, tb.cost_s, list(tb.calls),
                    {"llm_usd": round(usd, 6), "llm_calls": calls_llm, "steps": steps, "model": model,
                     "provider": provider, "reason": str(final.get("reason", ""))[:300],
                     "parse_ok": bool(final)})
