"""What a diagnoser can see and do.

Observations (free): monitoring statistics of the reference and current serve splits, measured performance on
labelled serve data (labels as the *current* pipeline produces them), data-validation results, the lineage map,
the anonymised change log and raw-schema events.

Tools (charged in measured compute seconds; archived production outputs are free to read):
  rollback(c)        run the reference version of c on c's current inputs; compare with c's current output
  replay(c)          run the current version of c on c's reference-period inputs; compare with the archived output
  effect(S)          counterfactual pipeline with components S rolled back; performance of the deployed model
  boundary(c, event) run the current version of c on the periods around a raw-schema event and compare the
                     discontinuity with ordinary period-to-period change
  full_run(state)    re-run the whole pipeline for a state (full charge, as a pipeline-instance debugger would)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from dod import bench
from dod.core import Executor, ExecError, State, _eq_cols, diff_summary, frames_equal
from dod.domains import get_domain


# ----------------------------------------------------------------------------------------------- statistics
def psi(a: np.ndarray, b: np.ndarray, categorical: bool) -> float:
    a, b = a[~pd.isna(a)], b[~pd.isna(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    if categorical:
        cats = np.union1d(np.unique(a), np.unique(b))
        pa = np.array([(a == c).mean() for c in cats])
        pb = np.array([(b == c).mean() for c in cats])
    else:
        edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
        if len(edges) < 2:
            edges = np.array([a.min() - 1e-9, a.max() + 1e-9])
        edges[0], edges[-1] = -np.inf, np.inf
        pa = np.histogram(a, edges)[0] / len(a)
        pb = np.histogram(b, edges)[0] / len(b)
    pa, pb = np.clip(pa, 1e-4, None), np.clip(pb, 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def feature_report(dom, ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    """Per-feature drift tests (KS / chi-square as in common monitoring tools), PSI, and validation checks
    learned from the reference profile (missing rate, range, unseen categories)."""
    rep = {}
    cols = list(dom.FEATURES) if dom.TASK == "binary" else []
    for c in cols:
        a = ref[c].to_numpy(dtype="float64")
        b = cur[c].to_numpy(dtype="float64") if c in cur else np.full(len(cur), np.nan)
        cat = c in dom.CAT_FEATURES
        a_ok, b_ok = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a_ok) and len(b_ok):
            if cat:
                cats = np.union1d(np.unique(a_ok), np.unique(b_ok))
                table = np.array([[(a_ok == k).sum() for k in cats], [(b_ok == k).sum() for k in cats]]) + 0.5
                pval = float(stats.chi2_contingency(table)[1])
            else:
                pval = float(stats.ks_2samp(a_ok, b_ok).pvalue)
        else:
            pval = 0.0
        miss_a, miss_b = float(np.isnan(a).mean()), float(np.isnan(b).mean())
        lo, hi = (np.nanmin(a), np.nanmax(a)) if len(a_ok) else (np.nan, np.nan)
        out_of_range = float(((b_ok < lo) | (b_ok > hi)).mean()) if len(b_ok) and not cat else 0.0
        rep[c] = {"pvalue": pval, "psi": psi(a, b, cat), "miss_ref": miss_a, "miss_cur": miss_b,
                  "out_of_range": out_of_range,
                  "validation_fail": bool((miss_b - miss_a) > 0.01 or out_of_range > 0.01 or
                                          (miss_b > 0.99 and miss_a < 0.99))}
    if dom.TASK != "binary":
        ta = ref["tok"].fillna("").map(lambda s: len(s.split()) if s else 0).to_numpy(dtype="float64")
        tb = cur["tok"].fillna("").map(lambda s: len(s.split()) if s else 0).to_numpy(dtype="float64")
        rep["tok"] = {"pvalue": float(stats.ks_2samp(ta, tb).pvalue) if len(ta) and len(tb) else 0.0,
                      "psi": psi(ta, tb, False), "miss_ref": float((ta == 0).mean()), "miss_cur": float((tb == 0).mean()),
                      "out_of_range": 0.0,
                      "validation_fail": bool(((tb == 0).mean() - (ta == 0).mean()) > 0.01)}
    la, lb = ref[dom.LABEL].to_numpy(), cur[dom.LABEL].to_numpy()
    if dom.TASK == "binary":
        rep["y"] = {"rate_ref": float(la.mean()), "rate_cur": float(lb.mean()) if len(lb) else float("nan"),
                    "psi": psi(la.astype(float), lb.astype(float), True)}
    else:
        rep["y"] = {"psi": psi(la.astype(float), lb.astype(float), True),
                    "unmapped_cur": float((lb < 0).mean()) if len(lb) else float("nan")}
    rep["_rows"] = {"ref": int(len(ref)), "cur": int(len(cur))}
    return rep


def column_drift(a: pd.Series, b: pd.Series) -> dict:
    """PSI and missing-rate change of one column between two samples (high-cardinality strings -> length)."""
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        av, bv = a.to_numpy(dtype="float64", na_value=np.nan), b.to_numpy(dtype="float64", na_value=np.nan)
        cat = max(pd.Series(av).nunique(), pd.Series(bv).nunique()) <= 30
    else:
        na, nb = a.isna().to_numpy(), b.isna().to_numpy()
        sa = pd.Series(np.where(na, "<NA>", a.astype(str).to_numpy(dtype=object)), dtype=object)
        sb = pd.Series(np.where(nb, "<NA>", b.astype(str).to_numpy(dtype=object)), dtype=object)
        if max(sa.nunique(), sb.nunique()) > 200:
            av = np.where(na, np.nan, sa.map(len).to_numpy(dtype="float64"))
            bv = np.where(nb, np.nan, sb.map(len).to_numpy(dtype="float64"))
            cat = False
        else:
            av, bv = sa.to_numpy(dtype=object), sb.to_numpy(dtype=object)
            cat = True
    miss = float(pd.isna(b).mean() - pd.isna(a).mean())
    return {"psi": psi(av, bv, cat), "missing_jump": miss}


def node_drift(ex: Executor, case: bench.Case, ref: bench.Reference, sample: int = 20000) -> dict:
    """Drift of every component's output between the archived reference run (p0) and the current production
    run (p1): free observational evidence at every node of the dataflow graph."""
    out = {}
    pipe = get_domain(case.domain).PIPELINE
    for c in pipe.order():
        try:
            a = ex.node(case.p0, ref.state, c, charge=False)
            b = ex.node(case.p1, case.state, c, charge=False)
        except ExecError:
            out[c] = {"max_psi": float("inf"), "error": True}
            continue
        a = a.sample(min(sample, len(a)), random_state=0) if len(a) else a
        b = b.sample(min(sample, len(b)), random_state=0) if len(b) else b
        cols = {}
        for k in a.columns:
            if k in b.columns and not k.startswith("_") and k not in ("trip_id", "link", "SERIALNO", "SPORDER") \
                    and not pd.api.types.is_datetime64_any_dtype(a[k]):     # timestamps move with time by design
                d = column_drift(a[k], b[k])
                if d["psi"] == d["psi"]:
                    cols[k] = d
        mx = max([v["psi"] for v in cols.values()] + [0.0])
        mj = max([v["missing_jump"] for v in cols.values()] + [0.0])
        out[c] = {"max_psi": round(mx, 4), "max_missing_jump": round(mj, 4),
                  "rows_ratio": round(len(b) / max(1, len(a)), 4),
                  "missing_cols": sorted(set(a.columns) - set(b.columns)),
                  "columns": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in cols.items()}}
    return out


# ----------------------------------------------------------------------------------------------- observation bundle
@dataclass
class Observation:
    case_id: str
    domain: str
    p0: str
    p1: str
    ref_perf: float
    cur_perf: float | None
    cur_crash: str | None
    features: dict
    change_log: list
    source_events: list
    lineage: dict
    components: list
    component_cost_s: dict             # measured seconds of each component in the current production run
    pipeline_cost_s: float
    extra: dict = field(default_factory=dict)

    def drifted(self, alpha: float = 0.01, psi_min: float = 0.1) -> list[str]:
        return [f for f, r in self.features.items() if not f.startswith("_") and f != "y"
                and r.get("pvalue", 1) < alpha and r.get("psi", 0) >= psi_min]

    def failing(self) -> list[str]:
        return [f for f, r in self.features.items() if not f.startswith("_") and r.get("validation_fail")]


def observe(case: bench.Case, ref: bench.Reference, ex: Executor) -> Observation:
    dom = get_domain(case.domain)
    ref_out = bench.pipeline_output(ex, case.p0, ref.state, dom)
    ref_serve = ref_out[ref_out["_split"] == "serve"]
    ref_perf = bench.score(dom, ref.model, ref_serve)
    crash = None
    try:
        cur_out = bench.with_split(dom, ex.run(case.p1, case.state))
        cur_serve = cur_out[cur_out["_split"] == "serve"]
        cur_perf = bench.score(dom, ref.model, cur_serve) if len(cur_serve) else float("nan")
        feats = feature_report(dom, ref_serve, cur_serve)
    except ExecError as e:
        crash, cur_perf, feats = str(e)[:400], None, {}
    costs = {}
    for c in dom.PIPELINE.order():
        costs[c] = ex.node_cost(case.p1, case.state, c)
    extra = {"node_drift": node_drift(ex, case, ref) if not crash else {}}
    return Observation(case.case_id, case.domain, case.p0, case.p1, ref_perf, cur_perf, crash, feats,
                       bench.change_log(ref, case), dom.source_events(case.p0, case.p1), dom.LINEAGE,
                       [{"name": c, "inputs": dom.PIPELINE.components[c].inputs} for c in dom.PIPELINE.order()],
                       {k: (round(v, 4) if v is not None else None) for k, v in costs.items()},
                       float(sum(v for v in costs.values() if v)), extra)


# ----------------------------------------------------------------------------------------------- tools
class Toolbox:
    def __init__(self, case: bench.Case, ref: bench.Reference, ex: Executor):
        self.case, self.ref, self.ex = case, ref, ex
        self.dom = get_domain(case.domain)
        self.pipe = self.dom.PIPELINE
        ex.free_keys = ex.all_keys(case.p1, case.state) | ex.all_keys(case.p0, ref.state)
        ex.charge_all = False
        ex.reset_charge()
        self.calls: list[dict] = []
        self._cur_perf = None

    @property
    def cost_s(self) -> float:
        return self.ex.charged_s

    def _log(self, tool: str, arg, result, t0: float, c0: float) -> None:
        self.calls.append({"tool": tool, "arg": arg, "result": result, "cost_s": round(self.ex.charged_s - c0, 4),
                           "wall_s": round(time.perf_counter() - t0, 3)})

    def _rolled(self, comps) -> State:
        st = dict(self.case.state)
        for c in comps:
            st[c] = self.ref.state[c]
        return st

    def rollback(self, c: str) -> dict:
        t0, c0 = time.perf_counter(), self.ex.charged_s
        try:
            cur = self.ex.node(self.case.p1, self.case.state, c)
        except ExecError as e:
            res = {"result": "current_error", "error": str(e)[:200]}
            self._log("rollback", c, res, t0, c0)
            return res
        try:
            alt = self.ex.node(self.case.p1, self._rolled([c]), c)
        except ExecError as e:
            res = {"result": "error", "error": str(e)[:200]}
            self._log("rollback", c, res, t0, c0)
            return res
        eq = frames_equal(cur, alt)
        res = {"result": "equal" if eq else "differ"}
        if not eq:
            res["diff"] = diff_summary(cur, alt)
        self._log("rollback", c, res, t0, c0)
        return res

    def replay(self, c: str) -> dict:
        t0, c0 = time.perf_counter(), self.ex.charged_s
        ref_state = self.ref.state
        try:
            arch = self.ex.node(self.case.p0, ref_state, c)
        except ExecError as e:
            res = {"result": "ref_error", "error": str(e)[:200]}
            self._log("replay", c, res, t0, c0)
            return res
        st = dict(ref_state)
        st[c] = self.case.state[c]
        try:
            new = self.ex.node(self.case.p0, st, c)
        except ExecError as e:
            res = {"result": "error", "error": str(e)[:200]}
            self._log("replay", c, res, t0, c0)
            return res
        eq = frames_equal(arch, new)
        res = {"result": "equal" if eq else "differ"}
        if not eq:
            res["diff"] = diff_summary(arch, new)
        self._log("replay", c, res, t0, c0)
        return res

    def current_perf(self) -> float | None:
        if self._cur_perf is None:
            try:
                out = bench.with_split(self.dom, self.ex.run(self.case.p1, self.case.state))
                self._cur_perf = bench.score(self.dom, self.ref.model, out[out["_split"] == "serve"])
            except ExecError:
                self._cur_perf = float("nan")
        return self._cur_perf

    def _cur_serve(self):
        out = bench.with_split(self.dom, self.ex.run(self.case.p1, self.case.state))
        return out[out["_split"] == "serve"]

    def effect(self, comps: list[str], mask: list[str] | None = None) -> dict:
        """Deployed-model accuracy on the serve split of the counterfactual pipeline with comps rolled back.
        With mask, those model inputs are set to missing in both the counterfactual and the reference serve data
        (to estimate the drop that remains once features from unrecoverable components are discounted)."""
        t0, c0 = time.perf_counter(), self.ex.charged_s
        try:
            out = bench.with_split(self.dom, self.ex.run(self.case.p1, self._rolled(comps)))
            serve = out[out["_split"] == "serve"].copy()
            ts = time.perf_counter()
            ref_perf = None
            if mask:
                ref_out = bench.with_split(self.dom, self.ex.run(self.case.p0, self.ref.state))
                ref_serve = ref_out[ref_out["_split"] == "serve"].copy()
                for f in mask:
                    serve[f] = float("nan")
                    ref_serve[f] = float("nan")
                ref_perf = bench.score(self.dom, self.ref.model, ref_serve)
            perf = bench.score(self.dom, self.ref.model, serve)
            cur_perf = self.current_perf()
            n_common = None
            if comps:
                try:   # if the rollback changes which records exist, compare on the records both versions keep
                    cur_serve = self._cur_serve()
                    if set(cur_serve["_id"]) != set(serve["_id"]):
                        common = set(cur_serve["_id"]) & set(serve["_id"])
                        n_common = len(common)
                        sv = serve[serve["_id"].isin(common)]
                        cs = cur_serve[cur_serve["_id"].isin(common)]
                        if mask:
                            cs = cs.copy()
                            for f in mask:
                                cs[f] = float("nan")
                        perf = bench.score(self.dom, self.ref.model, sv)
                        cur_perf = bench.score(self.dom, self.ref.model, cs)
                except ExecError:
                    pass
            self.ex.charged_s += time.perf_counter() - ts
            if ref_perf is None:
                ref_out = bench.with_split(self.dom, self.ex.run(self.case.p0, self.ref.state))
                ref_perf = bench.score(self.dom, self.ref.model, ref_out[ref_out["_split"] == "serve"])
            res = {"result": "ok", "perf": perf, "cur_perf": cur_perf, "ref_perf": ref_perf,
                   "rows": int(len(serve)), "common_rows": n_common, "masked": mask or []}
        except ExecError as e:
            res = {"result": "error", "error": str(e)[:200]}
        self._log("effect", sorted(comps), res, t0, c0)
        return res

    def full_run(self, state: State) -> dict:
        """Re-execute an entire pipeline instance (every component charged) and score it on the serve split."""
        t0, c0 = time.perf_counter(), self.ex.charged_s
        self.ex.charge_all = True
        try:
            out = bench.with_split(self.dom, self.ex.run(self.case.p1, state))
            serve = out[out["_split"] == "serve"]
            ts = time.perf_counter()
            perf = bench.score(self.dom, self.ref.model, serve)
            self.ex.charged_s += time.perf_counter() - ts
            res = {"result": "ok", "perf": perf, "rows": int(len(serve))}
        except ExecError as e:
            res = {"result": "error", "error": str(e)[:200]}
        finally:
            self.ex.charge_all = False
        self._log("full_run", {k: v.key()[:8] for k, v in state.items() if v != self.case.state[k]}, res, t0, c0)
        return res

    def boundary(self, c: str, event_period: str, event_cols: list[str] | None = None) -> dict:
        """Discontinuity *introduced by* c (current version) at a raw-schema event, relative to ordinary change
        between the two preceding periods. Only columns that c creates or modifies are compared, plus c's row
        retention. Needs archived raw data of those periods (re-executed, so charged)."""
        t0, c0 = time.perf_counter(), self.ex.charged_s
        periods = [p for p in self.dom.PERIODS if p <= event_period]
        if len(periods) < 3:
            res = {"result": "insufficient_history"}
            self._log("boundary", [c, event_period], res, t0, c0)
            return res
        ps = periods[-3:]
        comp = self.pipe.components[c]
        outs, ins, used = {}, {}, {}
        # history: before the event the reference (then-deployed) versions ran; at the event, the current ones
        for p in ps[:-1]:     # past production runs of the reference pipeline are archived (free to read)
            self.ex.free_keys |= self.ex.all_keys(p, self.ref.state)
        for p in ps:
            states = [self.case.state] if p == ps[-1] else [self.ref.state, self.case.state]
            outs[p] = None
            for st in states:
                try:
                    outs[p] = self.ex.node(p, st, c)
                    ins[p] = [self.ex.node(p, st, i) if not i.startswith("raw:") else self.pipe.load_raw(p, i[4:])
                              for i in comp.inputs]
                    used[p] = "reference" if st is self.ref.state else "current"
                    break
                except ExecError:
                    continue
        if any(v is None for v in outs.values()):
            res = {"result": "error", "periods": ps, "failed": [p for p, v in outs.items() if v is None]}
            self._log("boundary", [c, event_period], res, t0, c0)
            return res
        p2, p1, pe = ps
        full_rows = {p: len(outs[p]) for p in ps}
        full_in = {p: max(len(f) for f in ins[p]) for p in ps}
        for p in ps:                    # statistics on a fixed subsample (same positions for aligned frames)
            n = len(outs[p])
            if n > 20000:
                idx = np.random.default_rng(0).choice(n, 20000, replace=False)
                idx.sort()
                outs[p] = outs[p].iloc[idx].reset_index(drop=True)
                ins[p] = [f.iloc[idx].reset_index(drop=True) if len(f) == n else
                          f.sample(min(len(f), 20000), random_state=0).reset_index(drop=True) for f in ins[p]]
            else:
                ins[p] = [f if len(f) == n else f.sample(min(len(f), 20000), random_state=0).reset_index(drop=True)
                          for f in ins[p]]

        def introduced_cols(p):
            o, inp = outs[p], ins[p]
            cols = []
            for k in o.columns:
                if k.startswith("_") or k in self.dom.ID_COLS:
                    continue
                same = False
                for f in inp:
                    if k in f.columns and len(f) == len(o):
                        try:
                            same = bool(_eq_cols(f[k], o[k]).all())
                        except Exception:
                            same = False
                        if same:
                            break
                if not same:
                    cols.append(k)
            return cols
        cols = sorted(set(introduced_cols(pe)) | set(introduced_cols(p1)))
        srcmap = getattr(self.dom, "COLUMN_SOURCES", {}).get(c, {})
        if event_cols:      # only columns computed from the event's columns (within-component column lineage)
            cols = [k for k in cols if set(srcmap.get(k, [k])) & set(event_cols)]
        # a column whose sources were all absent before the event (a newly published quantity) cannot be judged
        # by smoothness: there is no "before" to compare with
        raw_in = [j for j, i in enumerate(comp.inputs) if i.startswith("raw:")]
        if raw_in:
            cols = [k for k in cols if any(sk in ins[p1][j].columns and ins[p1][j][sk].notna().any()
                                           for sk in srcmap.get(k, [k]) for j in raw_in)]
        per_col = {}
        cols = [k for k in cols if not (k in outs[pe].columns and pd.api.types.is_datetime64_any_dtype(outs[pe][k]))]
        for k in cols:
            if not all(k in outs[p].columns for p in ps):
                per_col[k] = {"baseline_psi": 0.0, "boundary_psi": float("inf"), "missing_jump": 1.0}
                continue
            base = column_drift(outs[p2][k], outs[p1][k])
            jump = column_drift(outs[p1][k], outs[pe][k])
            in_jump, in_miss = 0.0, 0.0
            srcs = getattr(self.dom, "COLUMN_SOURCES", {}).get(c, {}).get(k, [k])
            for j in range(len(comp.inputs)):          # the source columns' jump on the way in (not introduced by c)
                for sk in srcs:
                    if all(sk in ins[p][j].columns for p in (p1, pe)) and \
                            not pd.api.types.is_datetime64_any_dtype(ins[pe][j][sk]):
                        dj = column_drift(ins[p1][j][sk], ins[pe][j][sk])
                        in_jump = max(in_jump, dj["psi"] if dj["psi"] == dj["psi"] else 0.0)
                        in_miss = max(in_miss, dj["missing_jump"])
            per_col[k] = {"baseline_psi": round(base["psi"], 4), "boundary_psi": round(jump["psi"], 4),
                          "input_boundary_psi": round(in_jump, 4), "input_missing_jump": round(in_miss, 4),
                          "missing_jump": round(jump["missing_jump"] - max(0.0, base["missing_jump"])
                                                - max(0.0, in_miss), 4)}
        ret = {p: full_rows[p] / max(1, full_in[p]) for p in ps}
        res = {"result": "ok", "periods": ps, "versions": used, "columns": per_col,
               "retention": {p: round(v, 4) for p, v in ret.items()},
               # a defect loses records; regaining them (e.g. a refreshed mapping) is not a defect
               "retention_jump": round((ret[p1] - ret[pe]) - abs(ret[p1] - ret[p2]), 4)}
        self._log("boundary", [c, event_period], res, t0, c0)
        return res
