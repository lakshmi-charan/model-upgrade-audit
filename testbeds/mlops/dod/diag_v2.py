"""Version-2 diagnosis tools (added after the v1 test results; v1 tools in diag.py are unchanged).

boundary_hist(tb, c, event, cols, m)
    Like Toolbox.boundary, but the ordinary period-to-period change is measured over the m transitions before the
    event (the deployment history: the reference version's archived outputs), not over a single transition. A column
    is flagged only if the change introduced at the event exceeds every historical transition (rank rule,
    Proposition 7: false-flag probability <= 1/(m+1) per column under exchangeable world changes) as well as
    BOUNDARY_K x the historical median and an absolute floor.

invariants(tb, c, event, m)
    Source-internal invariants of c's output, mined from the deployment history with fixed templates (no domain
    knowledge), and their violation rates before and at the event:
      support   numeric k is non-zero exactly for the records whose context value z is in a set Z
                (e.g. a fee is charged exactly on trips from certain zones)
      order     a <= b (e.g. start <= end)
      sum       t = a + b (+ c) (accounting identities)
      diff      d = (b - a) / u for datetimes a, b and a unit u (durations)
    An invariant is accepted if it holds on >= INV_HOLD of the records in every history period and is not trivial.
    Theorem 5: a correct adapter keeps an exact invariant whatever the world does inside the invariant's region, so
    a violation increase at the event is evidence of an encoding/adapter defect that does not rely on world
    smoothness; conversely a distribution jump in a column covered by an intact invariant is attributable to the
    world.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from dod.core import ExecError, _eq_cols
from dod.diag import column_drift

SAMPLE = 20000
INV_HOLD = 0.98          # an invariant must hold on this share of records in every history period
INV_MIN = 0.01           # minimum increase of the violation rate at the event that counts
MAX_NUM = 24             # numeric columns considered by the sum template
_CACHE: dict = {}


def _history(tb, event_period: str, m: int) -> list[str]:
    return [p for p in tb.dom.PERIODS if p < event_period][-(m + 1):]


def _outputs(tb, c: str, event_period: str, m: int):
    """Outputs (and inputs) of c: reference version over the history periods (archived: free), current version at the
    event period (charged). Falls back to the current version where the reference version cannot run."""
    if event_period not in tb.dom.PERIODS:      # no archived data for the event period
        return [], {}, {}, {}
    hist = _history(tb, event_period, m)
    comp = tb.pipe.components[c]
    for p in hist:
        tb.ex.free_keys |= tb.ex.all_keys(p, tb.ref.state)
    outs, ins, used = {}, {}, {}
    for p in hist + [event_period]:
        # history: the deployment history (archived outputs of the reference version); at the event: the current one
        states = [tb.case.state] if p == event_period else [tb.ref.state, tb.case.state]
        for st in states:
            try:
                outs[p] = tb.ex.node(p, st, c)
                ins[p] = [tb.ex.node(p, st, i) if not i.startswith("raw:") else tb.pipe.load_raw(p, i[4:])
                          for i in comp.inputs]
                used[p] = "reference" if st is tb.ref.state else "current"
                break
            except ExecError:
                continue
    return hist, outs, ins, used


def _sub(df: pd.DataFrame, n: int = SAMPLE) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    idx = np.sort(np.random.default_rng(0).choice(len(df), n, replace=False))
    return df.iloc[idx].reset_index(drop=True)


def _introduced(out: pd.DataFrame, inp: list[pd.DataFrame], id_cols) -> list[str]:
    cols = []
    for k in out.columns:
        if k.startswith("_") or k in id_cols:
            continue
        same = False
        for f in inp:
            if k in f.columns and len(f) == len(out):
                try:
                    same = bool(_eq_cols(f[k], out[k]).all())
                except Exception:
                    same = False
                if same:
                    break
        if not same:
            cols.append(k)
    return cols


def _psi(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    d = column_drift(a, b)
    v = d["psi"] if d["psi"] == d["psi"] else 0.0
    return float(v), float(d["missing_jump"])


# ----------------------------------------------------------------------------------------------- boundary, history
def boundary_hist(tb, c: str, event_period: str, event_cols: list[str] | None = None, m: int = 4) -> dict:
    t0, c0 = time.perf_counter(), tb.ex.charged_s
    hist, outs, ins, used = _outputs(tb, c, event_period, m)
    if len(hist) < 2:
        res = {"result": "insufficient_history"}
        tb._log("boundary_hist", [c, event_period], res, t0, c0)
        return res
    ps = hist + [event_period]
    if any(p not in outs for p in ps):
        res = {"result": "error", "periods": ps, "failed": [p for p in ps if p not in outs]}
        tb._log("boundary_hist", [c, event_period], res, t0, c0)
        return res
    comp = tb.pipe.components[c]
    ts = time.perf_counter()
    full_rows = {p: len(outs[p]) for p in ps}
    full_in = {p: max(len(f) for f in ins[p]) for p in ps}
    o = {p: _sub(outs[p]) for p in ps}
    i_ = {p: [_sub(f) for f in ins[p]] for p in ps}     # same positions as o[p] for frames aligned with it
    pe, p1 = ps[-1], ps[-2]
    cols = sorted(set(_introduced(o[pe], i_[pe], tb.dom.ID_COLS)) | set(_introduced(o[p1], i_[p1], tb.dom.ID_COLS)))
    srcmap = getattr(tb.dom, "COLUMN_SOURCES", {}).get(c, {})
    if event_cols:
        cols = [k for k in cols if set(srcmap.get(k, [k])) & set(event_cols)]
    raw_in = [j for j, inp in enumerate(comp.inputs) if inp.startswith("raw:")]
    if raw_in:     # newly published quantities have no "before"
        cols = [k for k in cols if any(sk in i_[p1][j].columns and i_[p1][j][sk].notna().any()
                                       for sk in srcmap.get(k, [k]) for j in raw_in)]
    cols = [k for k in cols if not (k in o[pe].columns and pd.api.types.is_datetime64_any_dtype(o[pe][k]))]

    def in_jump(k, pa, pb):
        srcs = srcmap.get(k, [k])
        jmax, mmax = 0.0, 0.0
        for j in range(len(comp.inputs)):
            for sk in srcs:
                if all(sk in i_[p][j].columns for p in (pa, pb)) and \
                        not pd.api.types.is_datetime64_any_dtype(i_[pb][j][sk]):
                    v, mj = _psi(i_[pa][j][sk], i_[pb][j][sk])
                    jmax, mmax = max(jmax, v), max(mmax, mj)
        return jmax, mmax
    per = {}
    for k in cols:
        if not all(k in o[p].columns for p in ps):
            per[k] = {"event_psi": float("inf"), "hist_psi": [], "missing_jump": 1.0, "inherited": False}
            continue
        hj, hm = [], []
        for a, b in zip(ps[:-2], ps[1:-1]):
            v, mj = _psi(o[a][k], o[b][k])
            iv, im = in_jump(k, a, b)
            hj.append(max(0.0, v - iv))
            hm.append(mj - im)
        v, mj = _psi(o[p1][k], o[pe][k])
        iv, im = in_jump(k, p1, pe)
        per[k] = {"event_psi": round(v, 4), "input_psi": round(iv, 4), "introduced": round(max(0.0, v - iv), 4),
                  "hist_introduced": [round(x, 4) for x in hj],
                  "missing_jump": round(mj - max(0.0, im), 4), "hist_missing": [round(x, 4) for x in hm],
                  "inherited": bool(iv >= 0.05 or abs(im) > 0.01)}
    ret = {p: full_rows[p] / max(1, full_in[p]) for p in ps}
    hist_drop = [ret[a] - ret[b] for a, b in zip(ps[:-2], ps[1:-1])]
    tb.ex.charged_s += time.perf_counter() - ts
    res = {"result": "ok", "periods": ps, "versions": used, "columns": per,
           "retention": {p: round(v, 4) for p, v in ret.items()},
           "retention_drop": round(ret[p1] - ret[pe], 4), "hist_retention_drop": [round(x, 4) for x in hist_drop]}
    tb._log("boundary_hist", [c, event_period], res, t0, c0)
    return res


# ----------------------------------------------------------------------------------------------- invariants
def _num(df: pd.DataFrame) -> list[str]:
    return [k for k in df.columns if not k.startswith("_") and pd.api.types.is_numeric_dtype(df[k])
            and not pd.api.types.is_bool_dtype(df[k])]


def _ctx(df: pd.DataFrame) -> list[str]:
    out = []
    for k in df.columns:
        if k.startswith("_") or pd.api.types.is_float_dtype(df[k]) or pd.api.types.is_datetime64_any_dtype(df[k]):
            continue
        try:
            if 2 <= df[k].nunique(dropna=False) <= 300:
                out.append(k)
        except TypeError:
            continue
    return out


def _f(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)


def _viol(inv: dict, df: pd.DataFrame) -> np.ndarray | None:
    """Per-record violation indicator (True = violated), or None if a column is missing."""
    if any(k not in df.columns for k in inv["cols"]):
        return None
    t = inv["type"]
    if t == "support":
        k, z = inv["cols"]
        nz = np.nan_to_num(_f(df[k]), nan=0.0) != 0
        inz = df[z].astype(str).isin(inv["Z"]).to_numpy()
        return nz != inz
    if t == "order":
        a, b = inv["cols"]
        if inv.get("dt"):
            va, vb = pd.to_datetime(df[a], errors="coerce"), pd.to_datetime(df[b], errors="coerce")
            ok = (va <= vb).to_numpy() | va.isna().to_numpy() | vb.isna().to_numpy()
            return ~ok
        va, vb = _f(df[a]), _f(df[b])
        return ~((va <= vb) | np.isnan(va) | np.isnan(vb))
    if t == "sum":
        tgt, parts = inv["cols"][0], inv["cols"][1:]
        s = sum(np.nan_to_num(_f(df[p]), nan=0.0) for p in parts)
        v = np.nan_to_num(_f(df[tgt]), nan=0.0)
        return np.abs(v - s) > 0.011 + 1e-6 * np.abs(v)
    if t == "diff":
        d, a, b = inv["cols"]
        va, vb = pd.to_datetime(df[a], errors="coerce"), pd.to_datetime(df[b], errors="coerce")
        sec = (vb - va).dt.total_seconds().to_numpy() / inv["unit"]
        vd = _f(df[d])
        ok = (np.abs(vd - sec) <= np.maximum(1.0, 0.01 * np.abs(sec))) | np.isnan(vd) | np.isnan(sec)
        return ~ok
    raise ValueError(t)


def mine(frames: list[pd.DataFrame], cols_of_interest: list[str] | None = None, id_cols=()) -> list[dict]:
    """Invariants holding on >= INV_HOLD of the records of every frame (fixed templates, no domain knowledge).
    Record identifiers are excluded."""
    base = frames[-1]
    common = [k for k in base.columns if all(k in f.columns for f in frames) and k not in id_cols
              and not k.startswith("_")]
    num = [k for k in _num(base) if k in common]
    ctx = [k for k in _ctx(base) if k in common]
    dts = [k for k in common if pd.api.types.is_datetime64_any_dtype(base[k])]
    cand: list[dict] = []
    # support: k != 0 exactly for z in Z
    for k in num:
        nzr = float(np.mean(np.nan_to_num(_f(base[k]), nan=0.0) != 0))
        if not 0.005 <= nzr <= 0.995:
            continue
        best = None
        for z in ctx:
            if z == k:
                continue
            nz = pd.Series(np.nan_to_num(_f(base[k]), nan=0.0) != 0)
            zs = base[z].astype(str).reset_index(drop=True)
            g = nz.groupby(zs).agg(["mean", "size"])
            Z = sorted(g[(g["mean"] >= 0.5) & (g["size"] >= 20)].index)
            if not Z:
                continue
            inv = {"type": "support", "cols": [k, z], "Z": Z}
            acc = 1 - float(np.mean(_viol(inv, base)))
            if acc - max(nzr, 1 - nzr) >= 0.01 and (best is None or acc > best[0]):
                best = (acc, inv)
        if best:
            cand.append(best[1])
    # order: a <= b
    for a, b in itertools.permutations(num[:MAX_NUM], 2):
        va, vb = _f(base[a]), _f(base[b])
        ok = ~(np.isnan(va) | np.isnan(vb))
        if ok.mean() < 0.5 or np.mean(va[ok] < vb[ok]) < 0.5:
            continue
        cand.append({"type": "order", "cols": [a, b]})
    for a, b in itertools.permutations(dts, 2):
        cand.append({"type": "order", "cols": [a, b], "dt": True})
    # sum: t = a + b (+ c)
    nn = [k for k in num[:MAX_NUM] if np.mean(np.nan_to_num(_f(base[k]), nan=0.0) != 0) >= 0.05]
    vals = {k: np.nan_to_num(_f(base[k]), nan=0.0) for k in nn}
    sizes = (2, 3) if len(nn) <= 16 else (2,)
    for r in sizes:
        for parts in itertools.combinations(nn, r):
            s = sum(vals[p] for p in parts)
            for t in nn:
                if t in parts:
                    continue
                if np.mean(np.abs(vals[t] - s) <= 0.011 + 1e-6 * np.abs(vals[t])) >= INV_HOLD:
                    if all(np.mean(np.abs(vals[t] - vals[p]) <= 0.011) < 0.9 for p in parts):
                        cand.append({"type": "sum", "cols": [t, *parts]})
    # diff: d = (b - a) / unit
    for d in num:
        for a, b in itertools.permutations(dts, 2):
            for u in (1.0, 60.0, 3600.0):
                cand.append({"type": "diff", "cols": [d, a, b], "unit": u})
    out = []
    perm = np.random.default_rng(1).permutation(len(base))
    for inv in cand:
        rates = []
        for f in frames:
            v = _viol(inv, f)
            rates.append(1.0 if v is None else float(np.mean(v)))
        if max(rates) > 1 - INV_HOLD:
            continue
        # informative: breaking the link between the first column and the others (a permutation of that column)
        # must violate it far more often; trivial invariants (e.g. "a fee of 0 is within 1 unit of the duration in
        # hours") hold under the permutation too
        pb = base.copy()
        k0 = inv["cols"][0]
        pb[k0] = base[k0].to_numpy()[perm]
        pr = float(np.mean(_viol(inv, pb)))
        if pr >= max(5 * max(rates), max(rates) + 0.05):
            inv["hist_rates"] = [round(x, 5) for x in rates]
            inv["perm_rate"] = round(pr, 4)
            out.append(inv)
    if cols_of_interest is not None:
        out = [i for i in out if set(i["cols"]) & set(cols_of_interest)]
    return out


def invariants(tb, c: str, event_period: str, m: int = 4, event_cols: list[str] | None = None) -> dict:
    t0, c0 = time.perf_counter(), tb.ex.charged_s
    hist, outs, ins, used = _outputs(tb, c, event_period, m)
    ps = hist + [event_period]
    if len(hist) < 2 or any(p not in outs for p in ps):
        res = {"result": "unavailable"}
        tb._log("invariants", [c, event_period], res, t0, c0)
        return res
    ts = time.perf_counter()
    o = {p: _sub(outs[p]) for p in ps}
    intro = set(_introduced(o[event_period], [_sub(f) for f in ins[event_period]], tb.dom.ID_COLS)) | \
        set(_introduced(o[hist[-1]], [_sub(f) for f in ins[hist[-1]]], tb.dom.ID_COLS))
    key = (tb.case.domain, c, tuple((p, (tb.case.state if used[p] == "current" else tb.ref.state)[c].key())
                                    for p in hist))
    if key not in _CACHE:
        _CACHE[key] = mine([o[p] for p in hist], id_cols=tuple(tb.dom.ID_COLS) + (tb.dom.LABEL,))
    invs = [dict(i) for i in _CACHE[key] if set(i["cols"]) & intro]
    comp = tb.pipe.components[c]
    srcmap = getattr(tb.dom, "COLUMN_SOURCES", {}).get(c, {})
    pl, pe = hist[-1], event_period
    il, ie = [_sub(f) for f in ins[pl]], [_sub(f) for f in ins[pe]]
    inherited = {}
    for k in intro:
        inh = False
        for j in range(len(comp.inputs)):
            for sk in srcmap.get(k, [k]):
                if sk in il[j].columns and sk in ie[j].columns and \
                        not pd.api.types.is_datetime64_any_dtype(ie[j][sk]):
                    v, mj = _psi(il[j][sk], ie[j][sk])
                    inh = inh or v >= 0.05 or abs(mj) > 0.01
        inherited[k] = inh
    rows = []
    for inv in invs:
        v = _viol(inv, o[event_period])
        rate_e = 1.0 if v is None else float(np.mean(v))
        # the same invariant on c's inputs at the event (if computable): a violation already present is inherited
        inh = None
        for f in ins[event_period]:
            vi = _viol(inv, _sub(f))
            if vi is not None:
                inh = float(np.mean(vi))
                break
        hr = inv["hist_rates"]
        hist_inc = max([b - a for a, b in zip(hr[:-1], hr[1:])] + [0.0])
        inc = rate_e - hr[-1]
        own = [k for k in inv["cols"] if k in intro and not inherited.get(k, False)
               and (not event_cols or set(srcmap.get(k, [k])) & set(event_cols))]
        flag = inc > max(hist_inc, INV_MIN) and bool(own) and \
            not (inh is not None and inh - hr[-1] > max(hist_inc, INV_MIN))
        rows.append({**{k: v for k, v in inv.items() if k != "Z"}, "n_Z": len(inv.get("Z", [])),
                     "event_rate": round(rate_e, 5), "increase": round(inc, 5), "hist_max_increase": round(hist_inc, 5),
                     "input_rate": None if inh is None else round(inh, 5), "own_cols": own, "flag": bool(flag)})
    tb.ex.charged_s += time.perf_counter() - ts
    res = {"result": "ok", "periods": ps, "versions": used, "invariants": rows}
    tb._log("invariants", [c, event_period], {**res, "invariants": [r for r in rows if r["flag"]][:10],
                                              "n_invariants": len(rows)}, t0, c0)
    return res
