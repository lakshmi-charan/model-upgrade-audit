"""Benchmark machinery shared by all domains: splits, reference models, cases, ground truth and action outcomes.

A *case* is (domain, reference period p0, current period p1, current pipeline state θ_t). Diagnosers see:
reference and current monitoring data (serve split, with labels as the current pipeline produces them), the
lineage graph, an anonymised change log (which components changed, and whether code/config/artifact/env changed),
raw-schema events, and execution tools. They never see SPEC states or ground truth.

Ground truth (grader only) comes from executing the spec state: a component is *defective* if replacing its
current version by the spec version changes the pipeline output. Action outcomes are measured by executing each
action and scoring on the holdout split against labels produced by the spec pipeline.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from dod.core import ROOT, Executor, ExecError, State, Version, frames_equal, stable_hash, state_to_json
from dod.domains import get_domain

BENCH_VERSION = "v3"      # bump when model training, splits or outcome definitions change
MODELS = ROOT / "data" / "models" / BENCH_VERSION
SPLIT_FRACS = {"train": 0.4, "serve": 0.3, "hold": 0.3}
SEED = 20261005


# ----------------------------------------------------------------------------------------------- splits
_SPLIT_MEMO: dict = {}


def split_of(ids: pd.Series) -> pd.Series:
    """Deterministic split by hashed record id (same record -> same split in every pipeline state)."""
    def f(s: str) -> str:
        r = _SPLIT_MEMO.get(s)
        if r is None:
            u = int.from_bytes(hashlib.blake2b(f"{SEED}:{s}".encode(), digest_size=8).digest(), "little") / 2 ** 64
            r = "train" if u < SPLIT_FRACS["train"] else ("serve" if u < 1 - SPLIT_FRACS["hold"] else "hold")
            if len(_SPLIT_MEMO) < 5_000_000:
                _SPLIT_MEMO[s] = r
        return r
    return ids.map(f)


def record_ids(dom, df: pd.DataFrame) -> pd.Series:
    ids = df[dom.ID_COLS[0]].astype(str)
    for c in dom.ID_COLS[1:]:
        ids = ids + "|" + df[c].astype(str)
    return ids


def with_split(dom, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_id"] = record_ids(dom, df)
    df["_split"] = split_of(df["_id"])
    return df


# ----------------------------------------------------------------------------------------------- models
def make_model(dom):
    if dom.TASK == "binary":
        cat_idx = [dom.FEATURES.index(c) for c in dom.CAT_FEATURES]
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, categorical_features=cat_idx,
                                              random_state=SEED)
    return LogisticRegression(max_iter=300, C=4.0)


def xy(dom, df: pd.DataFrame):
    if dom.TASK == "binary":
        return df[dom.FEATURES].to_numpy(dtype="float64"), df[dom.LABEL].to_numpy()
    return dom.design_matrix(df), df[dom.LABEL].to_numpy()


def fit(dom, df: pd.DataFrame):
    X, y = xy(dom, df)
    m = make_model(dom)
    m.fit(X, y)
    return m


def score_detail(dom, model, feats: pd.DataFrame, true_y: np.ndarray | None = None) -> dict:
    """Accuracy (primary, decision quality at the deployed threshold) and, for binary tasks, AUROC.
    true_y defaults to the frame's own labels."""
    if len(feats) == 0:
        return {"acc": float("nan"), "auroc": float("nan")}
    X, y = xy(dom, feats)
    y = y if true_y is None else true_y
    if dom.TASK == "binary":
        p = model.predict_proba(X)[:, 1]
        au = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
        return {"acc": float(accuracy_score(y, (p >= 0.5).astype(int))), "auroc": au}
    return {"acc": float(accuracy_score(y, model.predict(X))), "auroc": float("nan")}


def score(dom, model, feats: pd.DataFrame, true_y: np.ndarray | None = None) -> float:
    return score_detail(dom, model, feats, true_y)["acc"]


# ----------------------------------------------------------------------------------------------- reference models
@dataclass
class Reference:
    domain: str
    p0: str
    encoder_id: str
    state: State
    model: object
    perf_hold: float


def build_reference(domain: str, p0: str, ex: Executor | None = None, force: bool = False) -> Reference:
    """Fit the encoder on p0 train rows (spec pipeline), train the reference model, record holdout performance."""
    dom = get_domain(domain)
    ex = ex or Executor(dom.PIPELINE)
    enc_id = f"p0_{p0}"
    path = MODELS / domain / f"{enc_id}.pkl"
    if path.exists() and not force:
        return pickle.loads(path.read_bytes())
    # the pipeline deployed when the model was trained: the correct pipeline for the reference period
    state = dom.spec_state(p0, enc_id)
    pre = dom.PIPELINE.components[dom.PIPELINE.output].inputs[0]
    feats_pre = with_split(dom, ex.node(p0, state, pre, charge=False))
    dom.save_artifact("encoder", enc_id, dom.fit_encoder(feats_pre[feats_pre["_split"] == "train"]))
    if hasattr(dom, "fit_text_artifacts"):
        dom.fit_text_artifacts(enc_id, feats_pre[feats_pre["_split"] == "train"])
    feats = with_split(dom, ex.run(p0, state))
    model = fit(dom, feats[feats["_split"] == "train"])
    perf = score(dom, model, feats[feats["_split"] == "hold"])
    ref = Reference(domain, p0, enc_id, state, model, perf)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(ref))
    return ref


# ----------------------------------------------------------------------------------------------- cases
@dataclass
class Case:
    case_id: str
    domain: str
    p0: str
    p1: str
    state: State                         # θ_t, the deployed (current) pipeline
    spec: State                          # correct pipeline for p1 (grader only)
    injected: list = field(default_factory=list)   # what the generator changed (grader only)
    tier: str = ""                       # T1 / T2 / T3 / none
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"case_id": self.case_id, "domain": self.domain, "p0": self.p0, "p1": self.p1,
                "state": state_to_json(self.state), "spec": state_to_json(self.spec), "injected": self.injected,
                "tier": self.tier, "meta": self.meta}


def change_log(ref: Reference, case: Case) -> list[dict]:
    """Anonymised change log a diagnoser may read: component, and which aspects of its version changed."""
    log = []
    for c, v in case.state.items():
        v0 = ref.state[c]
        if v != v0:
            kinds = []
            if v.variant != v0.variant:
                kinds.append("code")
            if v.params != v0.params:
                kinds.append("config")
            if v.artifacts != v0.artifacts:
                kinds.append("artifact")
            if v.env != v0.env:
                kinds.append("environment")
            log.append({"component": c, "changed": kinds, "version_id": v.key()[:8], "ref_version_id": v0.key()[:8]})
    return log


# ----------------------------------------------------------------------------------------------- ground truth
def pipeline_output(ex: Executor, period: str, state: State, dom) -> pd.DataFrame | None:
    try:
        return with_split(dom, ex.run(period, state))
    except ExecError:
        return None


EPS_CONSEQUENTIAL = 0.002   # a defect is consequential if fixing it moves some action's holdout accuracy this much


def ground_truth(case: Case, ref: Reference, ex: Executor, delta: float) -> dict:
    """Effective defects (fixing changes the pipeline output), consequential defects (fixing changes the outcome
    of 'nothing' or of 'retrain' by >= EPS_CONSEQUENTIAL), outcomes of the canonical actions, and the correct
    action: fix the consequential defects, and retrain iff retraining (after the fix) gains more than delta."""
    dom = get_domain(case.domain)
    spec_out = pipeline_output(ex, case.p1, case.spec, dom)
    assert spec_out is not None, f"spec pipeline fails for {case.case_id}"
    cur_out = pipeline_output(ex, case.p1, case.state, dom)
    defective = []
    for c in case.state:
        if case.state[c] == case.spec[c]:
            continue
        alt = dict(case.state)
        alt[c] = case.spec[c]
        alt_out = pipeline_output(ex, case.p1, alt, dom)
        if cur_out is None or alt_out is None or not frames_equal(cur_out, alt_out):
            defective.append(c)
    if not defective and cur_out is not None and not frames_equal(cur_out, spec_out):
        defective = [c for c in case.state if case.state[c] != case.spec[c]]
    defective = sorted(defective)
    fix_sets = [[c] for c in defective] + ([defective] if len(defective) > 1 else [])
    acts = action_outcomes(case, ref, ex, fix_sets=fix_sets, spec_out=spec_out)
    nothing, retrain = acts["nothing"]["score"], acts["retrain"]["score"]

    def gap(a, b):
        return abs(a - b) if (a == a and b == b) else float("inf")
    consequential = [c for c in defective
                     if gap(acts[f"fix:{c}"]["score"], nothing) >= EPS_CONSEQUENTIAL
                     or gap(acts[f"fix:{c}+retrain"]["score"], retrain) >= EPS_CONSEQUENTIAL
                     or cur_out is None]
    if defective:
        k = "fix:" + ",".join(defective)
        gain = acts[k + "+retrain"]["score"] - acts[k]["score"]
    else:
        gain = retrain - nothing
    scores = [v["score"] for v in acts.values() if v["score"] == v["score"]]
    return {"defective": defective, "consequential": sorted(consequential), "crashes": cur_out is None,
            "outcomes": acts, "correct_action": {"fix": sorted(consequential), "retrain": bool(gain > delta)},
            "retrain_gain_after_fix": round(float(gain), 5), "best_score": max(scores) if scores else None,
            "ref_perf": ref.perf_hold,
            "drop_vs_ref": None if nothing != nothing else round(ref.perf_hold - nothing, 5)}


def action_outcomes(case: Case, ref: Reference, ex: Executor, fix_sets: list[list[str]],
                    spec_out: pd.DataFrame | None = None) -> dict:
    """Score 'nothing', 'retrain', and fix(S) / fix(S)+retrain for each S on the p1 holdout split, against
    labels from the spec pipeline. Results are cached on disk per (case, action)."""
    dom = get_domain(case.domain)
    cache_dir = ROOT / "runs" / "outcomes" / BENCH_VERSION / case.domain
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = cache_dir / f"{case.case_id}.json"
    cache = json.loads(cpath.read_text()) if cpath.exists() else {}
    if spec_out is None:
        spec_out = pipeline_output(ex, case.p1, case.spec, dom)
    truth = spec_out[spec_out["_split"] == "hold"][["_id", dom.LABEL]].drop_duplicates("_id")
    truth = truth.rename(columns={dom.LABEL: "_true"})

    def evaluate(state: State, retrain: bool) -> dict:
        out = pipeline_output(ex, case.p1, state, dom)
        if out is None:
            return {"score": float("nan"), "crash": True}
        hold = out[out["_split"] == "hold"].merge(truth, on="_id", how="inner")
        model = ref.model
        if retrain:
            tr = out[out["_split"] == "train"]
            tr = dom.refit_for_retrain(tr, ex, case.p1, state) if hasattr(dom, "refit_for_retrain") else tr
            if len(tr) == 0 or tr[dom.LABEL].nunique() < 2:
                return {"score": float("nan"), "crash": True}
            try:
                model = fit(dom, tr)
            except Exception as e:      # e.g. a feature that is entirely missing in the retraining data
                return {"score": float("nan"), "crash": True, "error": f"retrain failed: {type(e).__name__}"}
            if hasattr(dom, "retrain_transform"):
                hold = dom.retrain_transform(hold, tr)
        d = score_detail(dom, model, hold, hold["_true"].to_numpy())
        res = {"score": d["acc"], "auroc": d["auroc"], "n_hold": int(len(hold)),
               "coverage": round(len(hold) / max(1, len(truth)), 4)}
        if retrain:
            # fragility: the same model once the pipeline is repaired (features from the spec pipeline)
            spec_hold = spec_out[spec_out["_split"] == "hold"]
            if hasattr(dom, "retrain_transform"):
                spec_hold = dom.retrain_transform(spec_hold, tr)
            res["score_after_repair"] = score(dom, model, spec_hold)
        return res

    keys = {"nothing": (case.state, False), "retrain": (case.state, True)}
    for S in fix_sets:
        if not S:
            continue
        st = dict(case.state)
        for c in S:
            st[c] = case.spec[c]
        keys["fix:" + ",".join(sorted(S))] = (st, False)
        keys["fix:" + ",".join(sorted(S)) + "+retrain"] = (st, True)
    changed = False
    for k, (st, rt) in keys.items():
        if k not in cache:
            cache[k] = evaluate(st, rt)
            changed = True
    if changed:
        cpath.write_text(json.dumps(cache, indent=1))
    return cache


def action_key(fix: list[str], retrain: bool) -> str:
    if fix:
        return "fix:" + ",".join(sorted(fix)) + ("+retrain" if retrain else "")
    return "retrain" if retrain else "nothing"
