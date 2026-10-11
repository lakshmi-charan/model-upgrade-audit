"""Diagnosis methods. Each takes the observation bundle and a fresh Toolbox and returns a Decision.

Thresholds are set on the dev split only (docs/04_dev_tuning.md) and frozen before the test split is run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from dod.diag import Observation, Toolbox

DELTA_RETRAIN = 0.01     # performance drop on labelled serve data that raises the alarm / justifies retraining
TAU_EFFECT = 0.003       # paired counterfactual improvement that counts as a harmful change
BOUNDARY_K = 3.0         # boundary PSI must exceed K x ordinary period-to-period PSI ...
BOUNDARY_MIN = 0.05      # ... and this absolute floor (or a missing-rate jump of 1%)


@dataclass
class Decision:
    method: str
    fix: list = field(default_factory=list)
    retrain: bool = False
    cost_s: float = 0.0
    calls: list = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"method": self.method, "fix": sorted(set(self.fix)), "retrain": self.retrain,
                "cost_s": round(self.cost_s, 4), "n_calls": len(self.calls), "calls": self.calls, "notes": self.notes}


def _drop(obs: Observation) -> float:
    if obs.cur_perf is None or obs.cur_perf != obs.cur_perf:
        return float("inf")
    return obs.ref_perf - obs.cur_perf


def _changed(obs: Observation) -> list[str]:
    order = [c["name"] for c in obs.components]
    return sorted({c["component"] for c in obs.change_log}, key=order.index)


def _source_readers(obs: Observation, dom) -> list[str]:
    tables = {e["table"] for e in obs.source_events}
    return [c for t in tables for c in dom.SOURCE_READERS.get(t, [])]


# ----------------------------------------------------------------------------------------------- baselines
def retrain_on_alarm(obs, tb, dom) -> Decision:
    return Decision("retrain_on_alarm", [], _drop(obs) >= DELTA_RETRAIN)


def validation_first(obs, tb, dom) -> Decision:
    """Schema/missing/range checks first (TFDV / Great Expectations style); a failure is routed through lineage to
    the changed component closest to the failing feature (else the source reader); otherwise a performance alarm
    means retrain."""
    if obs.cur_crash:
        cands = _changed(obs) or _source_readers(obs, dom) or [obs.components[0]["name"]]
        return Decision("validation_first", cands[-1:], False, notes={"crash": True})
    failing = obs.failing()
    rows = obs.features.get("_rows", {})
    if rows and rows.get("ref") and rows.get("cur", 0) / rows["ref"] < 0.5:
        failing = failing + ["_rows"]
    if dom.TASK != "binary" and obs.features.get("y", {}).get("unmapped_cur", 0) > 0.01:
        failing = failing + ["y"]
    if not failing:
        return Decision("validation_first", [], _drop(obs) >= DELTA_RETRAIN, notes={"failing": []})
    changed, readers = set(_changed(obs)), set(_source_readers(obs, dom))
    fix = set()
    for f in failing:
        lin = obs.lineage.get(f, [c["name"] for c in obs.components])
        hit = [c for c in lin if c in changed] or [c for c in lin if c in readers] or lin[:1]
        fix.add(hit[-1])
    return Decision("validation_first", sorted(fix), False, notes={"failing": failing})


def _introduced_drift(obs) -> dict:
    """Drift introduced at each node of the dataflow graph: for every output column, its drift (reference run vs
    current run) beyond the same column's drift at the node's inputs; a column whose input is already drifting is
    inherited, not introduced. Missing-value increases count 10x."""
    nd = obs.extra.get("node_drift", {})
    out = {}
    for comp in obs.components:
        c = comp["name"]
        if c not in nd or nd[c].get("error"):
            continue
        best = 0.0
        for k, v in nd[c].get("columns", {}).items():
            ps = [nd[i]["columns"][k] for i in comp["inputs"] if i in nd and k in nd[i].get("columns", {})]
            in_psi = max([p["psi"] for p in ps] + [0.0])
            in_miss = max([p["missing_jump"] for p in ps] + [0.0])
            if in_psi >= 0.1:
                continue
            best = max(best, max(0.0, v["psi"] - in_psi) + 10 * max(0.0, v["missing_jump"] - in_miss))
        out[c] = round(best, 4)
    return out


def observational_attribution(obs, tb, dom) -> Decision:
    """Observational root-cause attribution on the dataflow graph (in the spirit of Paleyes & Lawrence 2023):
    compare every node's archived output in the reference and current runs and blame the changed or
    source-coupled node that *introduces* the most drift; otherwise attribute the drop to the world."""
    if obs.cur_crash:
        return validation_first(obs, tb, dom)
    cand = set(_changed(obs)) | set(_source_readers(obs, dom))
    intro = _introduced_drift(obs)
    share = {c: round(v, 4) for c, v in intro.items() if c in cand}
    top = max(share, key=share.get) if share else None
    if top is not None and share[top] >= 0.1:
        return Decision("observational_attribution", [top], False, notes={"introduced": intro})
    return Decision("observational_attribution", [], _drop(obs) >= DELTA_RETRAIN, notes={"introduced": intro})


def bugdoc_shortcut(obs, tb, dom) -> Decision:
    """Pipeline-instance debugging (BugDoc 'Shortcut'): for each changed component, re-run the whole pipeline with
    that component set back to its known-good version; blame components whose swap restores performance."""
    found = []
    base = tb.full_run(tb.case.state)
    base_perf = base.get("perf", float("nan"))
    for c in _changed(obs):
        r = tb.full_run(tb._rolled([c]))
        if r["result"] == "ok" and (base["result"] != "ok" or r["perf"] - base_perf > TAU_EFFECT):
            found.append(c)
    retrain = False
    if found:
        r = tb.full_run(tb._rolled(found))
        retrain = r["result"] == "ok" and obs.ref_perf - r["perf"] >= DELTA_RETRAIN
    else:
        retrain = _drop(obs) >= DELTA_RETRAIN
    return Decision("bugdoc_shortcut", found, retrain, tb.cost_s, list(tb.calls))


def bisect(obs, tb, dom) -> Decision:
    """git-bisect over the ordered change list with full re-runs; finds up to two culprits."""
    changes = _changed(obs)
    found = []
    base = tb.full_run(tb.case.state)
    base_perf = base.get("perf", float("nan"))

    def improves(comps):
        r = tb.full_run(tb._rolled(comps + found))
        ref = base_perf if not found else cur_perf[0]
        return r["result"] == "ok" and (base["result"] != "ok" or r["perf"] - ref > TAU_EFFECT), r
    cur_perf = [base_perf]
    for _ in range(2):
        remaining = [c for c in changes if c not in found]
        if not remaining:
            break
        ok, r = improves(remaining)
        if not ok:
            break
        lo = remaining
        while len(lo) > 1:
            half = lo[: len(lo) // 2]
            ok, _ = improves(half)
            lo = half if ok else lo[len(lo) // 2:]
        found.append(lo[0])
        rr = tb.full_run(tb._rolled(found))
        cur_perf[0] = rr.get("perf", cur_perf[0])
    retrain = (obs.ref_perf - cur_perf[0] >= DELTA_RETRAIN) if found else (_drop(obs) >= DELTA_RETRAIN)
    return Decision("bisect", found, retrain, tb.cost_s, list(tb.calls))


def full_rollback(obs, tb, dom) -> Decision:
    """One full re-run with every changed component at its known-good version: detects that *some* change is
    harmful, without localising it."""
    changes = _changed(obs)
    if not changes:
        return Decision("full_rollback", [], _drop(obs) >= DELTA_RETRAIN, tb.cost_s, list(tb.calls))
    r = tb.full_run(tb._rolled(changes))
    if r["result"] == "ok" and (obs.cur_perf is None or r["perf"] - (obs.cur_perf or 0) > TAU_EFFECT):
        return Decision("full_rollback", changes, obs.ref_perf - r["perf"] >= DELTA_RETRAIN, tb.cost_s,
                        list(tb.calls))
    return Decision("full_rollback", [], _drop(obs) >= DELTA_RETRAIN, tb.cost_s, list(tb.calls))


# ----------------------------------------------------------------------------------------------- the method
def dod(obs, tb, dom, budget_s: float | None = None, use_boundary: bool = True, use_prior: bool = True,
        uniform_cost: bool = False, name: str = "dod") -> Decision:
    """Drift-or-Defect diagnosis.

    1. Candidates: changed components (change log), plus unchanged components downstream of a relevant raw-schema
       event that read the source or introduce drift at their node (possible latent/stale defects).
    2. Order by prior (drift introduced at the node, drift/validation evidence on the inputs it feeds, source
       coupling) per unit of measured execution cost (ratio rule, Proposition 4).
    3. Changed component: rollback test. equal -> benign. differ:
       - unforced change (not downstream of a relevant schema event): the reference version is a valid known-good
         version, so the difference is an effective defect (Theorem 2);
       - forced change, if the target or the set of records changed: boundary test;
       - forced change otherwise: effect test; harmful if rolling back improves the deployed model by > TAU_EFFECT,
         else boundary test (the reference version is not a valid known-good version).
       error (the reference version cannot run on the current inputs) -> boundary test.
    4. Boundary test (assumption: world smoothness at schema boundaries): flag if a column the component creates
       or modifies jumps at the event by more than BOUNDARY_K x the ordinary period-to-period change (and an
       absolute floor), if missing values appear, or if records are lost; jumps already present at the
       component's source columns are inherited.
    5. Action: fix every harmful or flagged component; retrain if the residual drop (counterfactual with harmful
       components rolled back, inputs produced by flagged source-coupled components masked) is >= DELTA_RETRAIN.
    """
    pipe = dom.PIPELINE
    changed = _changed(obs)
    order_all = [c["name"] for c in obs.components]
    ev_tables = {e["table"] for e in obs.source_events}

    def raw_tables_upstream(c):
        return {i[4:] for u in pipe.upstream(c) for i in pipe.components[u].inputs if i.startswith("raw:")}
    raw_cols = getattr(dom, "RAW_COLUMNS", {})

    def relevant(c, e):
        if e["table"] not in raw_tables_upstream(c):
            return False
        if c not in raw_cols:          # no column-level lineage for c: any event on an upstream table counts
            return True
        return bool(set(raw_cols[c].get(e["table"], [])) & set(e.get("columns", [])))
    downstream_ev = {c for c in order_all if any(relevant(c, e) for e in obs.source_events)}
    readers = set(_source_readers(obs, dom))
    intro = _introduced_drift(obs)
    latent = [c for c in order_all if c not in changed and c in downstream_ev
              and (c in readers or intro.get(c, 0.0) >= 0.05)] if use_boundary else []
    cand = list(dict.fromkeys(changed + latent))
    drifted, failing = set(obs.drifted()), set(obs.failing())
    prior = {}
    for c in cand:
        feats = [f for f, lin in obs.lineage.items() if c in lin]
        ev = sum((f in drifted) + 2 * (f in failing) for f in feats) / max(1, len(feats))
        ev += max(0.0, intro.get(c, 0.0))
        prior[c] = 0.2 + (min(1.0, ev) if use_prior else 0.0) + (0.3 if c in downstream_ev else 0.0)
    cost = {c: (1.0 if uniform_cost else max(1e-3, obs.component_cost_s.get(c) or 1.0)) for c in cand}
    order = sorted(cand, key=lambda c: -prior[c] / cost[c])
    harmful, beneficial, benign, source_suspect = [], [], [], []
    notes = {"order": order, "prior": {k: round(v, 3) for k, v in prior.items()}, "latent_candidates": latent}

    def over() -> bool:
        return budget_s is not None and tb.cost_s >= budget_s

    def boundary_flag(c):
        flags = []
        for e in obs.source_events:
            if not relevant(c, e) or over():
                continue
            ep = str(e.get("period", e.get("year")))
            r = tb.boundary(c, ep, e.get("columns"))
            if r.get("result") == "error":
                flags.append({"event": ep, "why": "current version fails around the event"})
                continue
            if r.get("result") != "ok":
                continue
            for col, v in r["columns"].items():
                inherited = (v.get("input_boundary_psi", 0.0) >= BOUNDARY_MIN
                             or abs(v.get("input_missing_jump", 0.0)) > 0.01)
                introduced = v["boundary_psi"] - v.get("input_boundary_psi", 0.0)
                if not inherited and (introduced > max(BOUNDARY_K * v["baseline_psi"], BOUNDARY_MIN)
                                      or v["missing_jump"] > 0.01):
                    flags.append({"event": ep, "column": col, **v})
            if r["retention_jump"] > 0.02:
                flags.append({"event": ep, "retention": r["retention"]})
        notes.setdefault("boundary", {})[c] = flags
        return bool(flags)

    for c in order:
        if budget_s is not None and tb.cost_s >= budget_s:
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
                # the change redefines the target or the set of records: accuracy comparisons across them are
                # meaningless, so such a change is a defect unless it was forced by a schema event
                if use_boundary and c in downstream_ev:
                    (source_suspect if boundary_flag(c) else benign).append(c)
                else:
                    harmful.append(c)
                continue
            if c not in downstream_ev:
                # unforced change (Theorem 2): its reference version is valid on the current inputs, so an
                # output difference is an effective defect; the effect test only grades its consequence
                harmful.append(c)
                continue
            e = tb.effect([c])
            if e["result"] == "ok":
                gain = e["perf"] - e["cur_perf"]
                if gain > TAU_EFFECT:
                    harmful.append(c)
                elif gain < -TAU_EFFECT:
                    if use_boundary and c in downstream_ev and boundary_flag(c):
                        source_suspect.append(c)
                    else:
                        beneficial.append(c)
                elif use_boundary and c in downstream_ev and boundary_flag(c):
                    source_suspect.append(c)
                else:
                    benign.append(c)
                continue
        # rollback error (old version cannot read the current inputs) or the rolled-back pipeline fails downstream
        if use_boundary and c in downstream_ev:
            if boundary_flag(c):
                source_suspect.append(c)
            else:
                benign.append(c)
        else:
            harmful.append(c)
    fix = sorted(set(harmful + source_suspect))
    # residual drop once the identified defects are fixed: counterfactual for rolled-back defects; features
    # produced by source-coupled suspects (which cannot be rolled back) are masked in both periods
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


METHODS = {
    "retrain_on_alarm": retrain_on_alarm,
    "validation_first": validation_first,
    "observational_attribution": observational_attribution,
    "full_rollback": full_rollback,
    "bisect": bisect,
    "bugdoc_shortcut": bugdoc_shortcut,
    "dod": dod,
    "dod_no_boundary": lambda o, t, d: dod(o, t, d, use_boundary=False, name="dod_no_boundary"),
    # selection under a compute budget of one full pipeline run: ratio order vs ablated orderings
    "dod_b1": lambda o, t, d: dod(o, t, d, budget_s=o.pipeline_cost_s, name="dod_b1"),
    "dod_b1_no_prior": lambda o, t, d: dod(o, t, d, budget_s=o.pipeline_cost_s, use_prior=False,
                                           name="dod_b1_no_prior"),
    "dod_b1_uniform": lambda o, t, d: dod(o, t, d, budget_s=o.pipeline_cost_s, use_prior=False,
                                          uniform_cost=True, name="dod_b1_uniform"),
}


def _llm(mode: str):
    def f(obs, tb, dom):
        from dod import llm_agent
        budget = 3.0 * max(obs.pipeline_cost_s, 0.5)       # three full pipeline runs of compute
        return llm_agent.run(obs, tb, dom, mode, budget if mode == "tools" else None)
    return f


LLM_METHODS = {"llm_report": _llm("report"), "llm_tools": _llm("tools")}
ALL_METHODS = {**METHODS, **LLM_METHODS}
