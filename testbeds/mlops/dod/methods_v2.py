"""Version-2 diagnosis method (designed after the v1 test results, on already-seen data only; evaluated on the v2
held-out domains). v1 methods (methods.py) are unchanged and are run alongside.

DoD-2 differs from DoD only in how a source-coupled (forced or latent) component is judged at a raw-schema event:
  * history rule: the introduced change at the event must exceed every one of the M_HIST preceding ordinary
    transitions of the deployment history (and BOUNDARY_K x their median, and BOUNDARY_MIN); Proposition 7;
  * invariant evidence (Theorem 5): invariants mined from the deployment history of the component's output with
    fixed templates; a violation increase at the event flags the component; a column jump covered by an intact
    invariant that also involves an unflagged anchor column is attributed to the world (excused).
Everything else (candidates, ordering, rollback and effect tests, residual retrain decision, thresholds) is DoD.
"""
from __future__ import annotations

import numpy as np

from dod import diag_v2
from dod.methods import (BOUNDARY_K, BOUNDARY_MIN, DELTA_RETRAIN, TAU_EFFECT, Decision, _changed, _drop,
                         _introduced_drift, _source_readers)

M_HIST = 4


def dod2(obs, tb, dom, budget_s: float | None = None, use_invariants: bool = True, use_history: bool = True,
         name: str = "dod2") -> Decision:
    pipe = dom.PIPELINE
    changed = _changed(obs)
    order_all = [c["name"] for c in obs.components]

    def raw_tables_upstream(c):
        return {i[4:] for u in pipe.upstream(c) for i in pipe.components[u].inputs if i.startswith("raw:")}
    raw_cols = getattr(dom, "RAW_COLUMNS", {})

    def relevant(c, e):
        if e["table"] not in raw_tables_upstream(c):
            return False
        if c not in raw_cols:
            return True
        return bool(set(raw_cols[c].get(e["table"], [])) & set(e.get("columns", [])))
    downstream_ev = {c for c in order_all if any(relevant(c, e) for e in obs.source_events)}
    readers = set(_source_readers(obs, dom))
    intro = _introduced_drift(obs)
    latent = [c for c in order_all if c not in changed and c in downstream_ev
              and (c in readers or intro.get(c, 0.0) >= 0.05)]
    cand = list(dict.fromkeys(changed + latent))
    drifted, failing = set(obs.drifted()), set(obs.failing())
    prior = {}
    for c in cand:
        feats = [f for f, lin in obs.lineage.items() if c in lin]
        ev = sum((f in drifted) + 2 * (f in failing) for f in feats) / max(1, len(feats))
        ev += max(0.0, intro.get(c, 0.0))
        prior[c] = 0.2 + min(1.0, ev) + (0.3 if c in downstream_ev else 0.0)
    cost = {c: max(1e-3, obs.component_cost_s.get(c) or 1.0) for c in cand}
    order = sorted(cand, key=lambda c: -prior[c] / cost[c])
    harmful, beneficial, benign, source_suspect = [], [], [], []
    notes = {"order": order, "prior": {k: round(v, 3) for k, v in prior.items()}, "latent_candidates": latent}

    def over() -> bool:
        return budget_s is not None and tb.cost_s >= budget_s

    def boundary_flag(c):
        flags, excused = [], []
        for e in obs.source_events:
            if not relevant(c, e) or over():
                continue
            ep = str(e.get("period", e.get("year")))
            m = M_HIST if use_history else 1
            r = diag_v2.boundary_hist(tb, c, ep, e.get("columns"), m=m)
            if r.get("result") == "error":
                flags.append({"event": ep, "why": "current version fails around the event"})
                continue
            if r.get("result") != "ok":
                continue
            col_flags = {}
            for col, v in r["columns"].items():
                if v.get("inherited"):
                    continue
                if v.get("event_psi") == float("inf"):
                    col_flags[col] = {"event": ep, "column": col, "why": "column missing"}
                    continue
                h = v["hist_introduced"] or [0.0]
                thr = max(BOUNDARY_K * float(np.median(h)), max(h) if use_history else 0.0, BOUNDARY_MIN)
                hm = max(v["hist_missing"] or [0.0])
                if v["introduced"] > thr or v["missing_jump"] > max(0.01, hm + 0.01):
                    col_flags[col] = {"event": ep, "column": col, "introduced": v["introduced"], "threshold": thr,
                                      "missing_jump": v["missing_jump"]}
            if use_invariants:
                inv = diag_v2.invariants(tb, c, ep, m=M_HIST, event_cols=e.get("columns"))
                if inv.get("result") == "ok":
                    for row in inv["invariants"]:
                        if row["flag"]:
                            flags.append({"event": ep, "invariant": row["type"], "cols": row["cols"],
                                          "increase": row["increase"]})
                    # only invariants that pin down the column (its zero pattern or its value) can excuse its jump;
                    # an order constraint is preserved by most jumps and proves nothing
                    intact = [row for row in inv["invariants"] if not row["flag"]
                              and row["type"] in ("support", "sum", "diff") and row["event_rate"] <= 1 - diag_v2.INV_HOLD]
                    for col in list(col_flags):
                        for row in intact:
                            anchors = [k for k in row["cols"] if k != col and k not in col_flags]
                            if col in row["cols"] and anchors:
                                excused.append({"event": ep, "column": col, "by": row["type"], "cols": row["cols"]})
                                col_flags.pop(col)
                                break
            flags.extend(col_flags.values())
            drop_hist = max(r["hist_retention_drop"] or [0.0])
            if r["retention_drop"] > max(0.0, drop_hist) + 0.02:
                flags.append({"event": ep, "retention": r["retention"]})
        notes.setdefault("boundary", {})[c] = flags
        if excused:
            notes.setdefault("excused", {})[c] = excused
        return bool(flags)

    for c in order:
        if over():
            notes["budget_exhausted_before"] = c
            break
        if c not in changed:
            if boundary_flag(c):
                source_suspect.append(c)
            continue
        r = tb.rollback(c)
        if over():
            notes["budget_exhausted_at"] = c
            break
        if r["result"] == "equal":
            benign.append(c)
            continue
        if r["result"] == "differ":
            diff_cols = set(r.get("diff", {}).get("col_diff_frac", {}))
            rows = r.get("diff", {}).get("rows", [0, 0])
            if dom.LABEL in diff_cols or rows[0] != rows[1]:
                if c in downstream_ev:
                    (source_suspect if boundary_flag(c) else benign).append(c)
                else:
                    harmful.append(c)
                continue
            if c not in downstream_ev:
                harmful.append(c)
                continue
            e = tb.effect([c])
            if e["result"] == "ok":
                gain = e["perf"] - e["cur_perf"]
                if gain > TAU_EFFECT:
                    harmful.append(c)
                elif gain < -TAU_EFFECT:
                    (source_suspect if boundary_flag(c) else beneficial).append(c)
                elif boundary_flag(c):
                    source_suspect.append(c)
                else:
                    benign.append(c)
                continue
        if c in downstream_ev:
            (source_suspect if boundary_flag(c) else benign).append(c)
        else:
            harmful.append(c)
    fix = sorted(set(harmful + source_suspect))
    masked = []
    if source_suspect and dom.TASK == "binary":
        susp_feats = {f for f, lin in obs.lineage.items() if any(c in lin for c in source_suspect)}
        masked = sorted(susp_feats & (set(obs.drifted()) | set(obs.failing())) - {dom.LABEL})
    if harmful or masked:
        e = tb.effect(harmful, mask=masked)
        resid = (e["ref_perf"] - e["perf"]) if e["result"] == "ok" else float("inf")
        notes["residual"] = {"rolled_back": harmful, "masked": masked, "residual_drop": round(resid, 4)}
    else:
        resid = _drop(obs)
    retrain = resid >= DELTA_RETRAIN
    notes.update({"harmful": harmful, "beneficial": beneficial, "benign": benign, "source_suspect": source_suspect})
    return Decision(name, fix, retrain, tb.cost_s, list(tb.calls), notes)


METHODS_V2 = {
    "dod2": dod2,
    "dod2_no_inv": lambda o, t, d: dod2(o, t, d, use_invariants=False, name="dod2_no_inv"),
    "dod2_no_hist": lambda o, t, d: dod2(o, t, d, use_history=False, name="dod2_no_hist"),
}
