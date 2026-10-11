"""ACS income pipeline (tabular): US Census ACS 1-year PUMS person + housing records, CA/TX/NY, 2014–2024.

Task: does a working adult earn more than $50,000 (survey-year dollars, ADJINC applied)? (cf. folktables ACSIncome,
extended with a household join and an occupation crosswalk so the pipeline has real joins and reference data.)

Real, dated source changes this pipeline must absorb (verified in the published files, see runs/data_probe):
  2017  SERIALNO becomes a 13-digit id with a year prefix (still numeric)
  2018  SERIALNO becomes alphanumeric ("2018GQ0000004", "2018HU..."); OCCP moves to 2018 SOC-based codes
  2019  RELP -> RELSHIPP (new code list), WKW (6 bins) -> WKWN (weeks 1-52), JWTR -> JWTRNS (recoded modes)
  2023  ST -> STATE
Each component version below is either the original (2014) code, the correct update for a source change, or a
realistic faulty update/staleness. Which versions are "correct" is defined in SPEC; diagnosers never see it.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from dod.core import Component, Pipeline, Version

ROOT = Path(os.environ.get("DOD_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
BRONZE = ROOT / "data" / "bronze" / "acs"
ART = ROOT / "data" / "artifacts" / "acs"
CODEBOOK = ROOT / "data" / "raw" / "acs_codebook"

PERIODS = [str(y) for y in range(2014, 2025)]
CAT_FEATURES = ["COW", "MAR", "OCCG", "RELP", "SEX", "RAC1P", "JWTR", "ST", "TEN"]
NUM_FEATURES = ["AGEP", "SCHL", "WKHP", "WKW", "POBP", "NP", "VEH"]
FEATURES = NUM_FEATURES + CAT_FEATURES
LABEL = "y"
ID_COLS = ["SERIALNO", "SPORDER"]
TASK = "binary"

# Official harmonisation of 2019+ codes to the pre-2019 code lists (Census API value labels, 2018 vs 2019).
RELSHIPP_TO_RELP = {20: 0, 21: 1, 22: 13, 23: 1, 24: 13, 25: 2, 26: 3, 27: 4, 28: 5, 29: 6, 30: 7, 31: 8, 32: 9,
                    33: 10, 34: 12, 35: 14, 36: 15, 37: 16, 38: 17}
JWTRNS_TO_JWTR = {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 3, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12}


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("", np.nan), errors="coerce")


def wkwn_to_wkw(w: pd.Series, mode: str) -> pd.Series:
    w = num(w)
    bins = [0, 13.5, 26.5, 39.5, 47.5, 49.5, 52.5]
    if mode == "bins":          # official WKW categories: 1=50-52 ... 6=<14
        lab = [6, 5, 4, 3, 2, 1]
    elif mode == "reversed":    # faulty: categories numbered from the low end
        lab = [1, 2, 3, 4, 5, 6]
    elif mode == "passthrough":  # faulty: weeks passed through as if they were WKW categories
        return w
    else:
        raise ValueError(mode)
    return pd.cut(w, bins=bins, labels=lab).astype("float64")


# ----------------------------------------------------------------------------------------------- components
C = {}


def comp(name, inputs, doc=""):
    C[name] = Component(name, inputs, doc=doc)
    return C[name]


person_adapter = comp("person_adapter", ["raw:person"], "map published person columns to the canonical schema")


@person_adapter.variant("v2014")
def _pa_v2014(inp, p, a):
    r = inp["raw:person"]
    out = pd.DataFrame({
        "SERIALNO": r["SERIALNO"].astype(str), "SPORDER": num(r["SPORDER"]), "ST": r["ST"].astype(str),
        "AGEP": num(r["AGEP"]), "COW": num(r["COW"]), "SCHL": num(r["SCHL"]), "MAR": num(r["MAR"]),
        "OCCP": r["OCCP"].astype(str).str.strip(), "POBP": num(r["POBP"]), "RELP": num(r["RELP"]),
        "WKHP": num(r["WKHP"]), "SEX": num(r["SEX"]), "RAC1P": num(r["RAC1P"]), "WKW": num(r["WKW"]),
        "JWTR": num(r["JWTR"]), "PINCP": num(r["PINCP"]), "ADJINC": num(r["ADJINC"]), "PWGTP": num(r["PWGTP"]),
    })
    return out


@person_adapter.variant("v2019")
def _pa_v2019(inp, p, a):
    """Vintage-aware adapter written for the 2019 schema change (and later the 2023 ST->STATE rename)."""
    r = inp["raw:person"]
    cols = set(r.columns)
    if "RELP" in cols:
        relp = num(r["RELP"])
    elif p.get("relp", "official") == "official":
        relp = num(r["RELSHIPP"]).map(RELSHIPP_TO_RELP)
    else:                                   # faulty: shift the new codes down to the old range
        relp = num(r["RELSHIPP"]) - 20
    wkw = num(r["WKW"]) if "WKW" in cols else wkwn_to_wkw(r["WKWN"], p.get("wkw", "bins"))
    if "JWTR" in cols:
        jwtr = num(r["JWTR"])
    elif p.get("jwtr", "official") == "official":
        jwtr = num(r["JWTRNS"]).map(JWTRNS_TO_JWTR)
    else:                                   # faulty: treat the renamed column as the old one
        jwtr = num(r["JWTRNS"])
    if "ST" in cols:
        st = r["ST"].astype(str)
    elif p.get("state", "fallback") == "fallback":
        st = r["STATE"].astype(str)
    else:                                   # faulty: column missing, filled with the most common training state
        st = pd.Series("06", index=r.index)
    return pd.DataFrame({
        "SERIALNO": r["SERIALNO"].astype(str), "SPORDER": num(r["SPORDER"]), "ST": st,
        "AGEP": num(r["AGEP"]), "COW": num(r["COW"]), "SCHL": num(r["SCHL"]), "MAR": num(r["MAR"]),
        "OCCP": r["OCCP"].astype(str).str.strip(), "POBP": num(r["POBP"]), "RELP": relp,
        "WKHP": num(r["WKHP"]), "SEX": num(r["SEX"]), "RAC1P": num(r["RAC1P"]), "WKW": wkw,
        "JWTR": jwtr, "PINCP": num(r["PINCP"]), "ADJINC": num(r["ADJINC"]), "PWGTP": num(r["PWGTP"]),
    })


@person_adapter.variant("v2019_refactor")
def _pa_v2019_refactor(inp, p, a):
    """Benign refactor: same outputs as v2019 with the same params (column-wise loop instead of literals)."""
    out = _pa_v2019(inp, p, a)
    return out[list(out.columns)].copy()


housing_adapter = comp("housing_adapter", ["raw:housing"], "household attributes")


@housing_adapter.variant("v2014")
def _ha_v2014(inp, p, a):
    r = inp["raw:housing"]
    return pd.DataFrame({"SERIALNO": r["SERIALNO"].astype(str), "NP": num(r["NP"]), "TEN": num(r["TEN"]),
                         "VEH": num(r["VEH"])})


person_clean = comp("person_clean", ["person_adapter"], "folktables-style universe filter and missing codes")


@person_clean.variant("v2014")
def _pc_v2014(inp, p, a):
    df = inp["person_adapter"]
    df = df[(df["AGEP"] > 16) & (df["PINCP"] > 100) & (df["WKHP"] > 0) & (df["PWGTP"] >= 1)].copy()
    # Written in the pandas-1.x era: chained assignment to fill "not in universe" blanks with 0.
    # (Under pandas 3 copy-on-write this assignment silently does nothing.)
    for c in ("COW", "JWTR", "WKW", "MAR"):
        df[c][df[c].isna()] = 0
    return df.reset_index(drop=True)


@person_clean.variant("v2024_loc")
def _pc_loc(inp, p, a):
    """Correct modernisation of v2014 (uses .loc, behaves the same on every pandas version)."""
    df = inp["person_adapter"]
    df = df[(df["AGEP"] > 16) & (df["PINCP"] > 100) & (df["WKHP"] > 0) & (df["PWGTP"] >= 1)].copy()
    for c in ("COW", "JWTR", "WKW", "MAR"):
        df.loc[df[c].isna(), c] = 0
    return df.reset_index(drop=True)


join_household = comp("join_household", ["person_clean", "housing_adapter"], "left join persons to households")


@join_household.variant("v2014")
def _jh(inp, p, a):
    per, hh = inp["person_clean"], inp["housing_adapter"]
    key = p.get("key", "numeric")
    if key == "numeric":            # original: ids were numeric in 2014
        lk, rk = pd.to_numeric(per["SERIALNO"], errors="coerce"), pd.to_numeric(hh["SERIALNO"], errors="coerce")
    elif key == "string":           # correct for 2018+ alphanumeric ids
        lk, rk = per["SERIALNO"].str.strip(), hh["SERIALNO"].str.strip()
    elif key == "digits":           # faulty "fix": strip letters so ids parse as numbers (GQ/HU ids collide)
        lk = pd.to_numeric(per["SERIALNO"].str.replace(r"\D", "", regex=True), errors="coerce")
        rk = pd.to_numeric(hh["SERIALNO"].str.replace(r"\D", "", regex=True), errors="coerce")
    else:
        raise ValueError(key)
    left = per.assign(_k=lk.values)
    right = hh.drop(columns=["SERIALNO"]).assign(_k=rk.values)
    if p.get("dedupe_right", True):
        right = right.drop_duplicates("_k")
    out = left.merge(right, on="_k", how="left").drop(columns="_k")
    return out.reset_index(drop=True)


occ_group = comp("occ_group", ["join_household"], "occupation code -> SOC major group via crosswalk artifact")


@occ_group.variant("v2014")
def _og(inp, p, a):
    df = inp["join_household"].copy()
    xw = a["occ_xwalk"]
    df["OCCG"] = df["OCCP"].str.zfill(4).map(xw).fillna(-1).astype("float64")
    return df


features = comp("features", ["occ_group"], "model features and label")


@features.variant("v2014")
def _ft(inp, p, a):
    df = inp["occ_group"]
    adj = df["ADJINC"] / 1e6 if p.get("adjinc", "apply") == "apply" else 1.0
    out = df[["SERIALNO", "SPORDER"] + [c for c in FEATURES if c != "OCCG"] + ["OCCG"]].copy()
    out = out[["SERIALNO", "SPORDER"] + FEATURES]
    unit = p.get("wkhp_unit", "week")
    if unit == "month":            # faulty unit change: hours per month instead of per week
        out["WKHP"] = out["WKHP"] * 4.33
    out[LABEL] = ((df["PINCP"] * adj) > 50_000).astype("int8")
    stuck = p.get("stuck", "")
    if stuck:                      # incident class: a feature stuck at a default value for one state
        col, st, val = stuck.split(":")
        out.loc[out["ST"] == st, col] = float(val)
    return out.reset_index(drop=True)


encode = comp("encode", ["features"], "categorical codes from the fitted encoder artifact")


@encode.variant("v2014")
def _enc(inp, p, a):
    df = inp["features"].copy()
    maps = a["encoder"]
    if p.get("refit", False):      # skew defect: mapping re-fitted on serving data
        maps = fit_encoder(df)
    for c in CAT_FEATURES:
        df[c] = pd.Series(_keys(df[c]), index=df.index).map(maps[c]).astype("float64")
    return df


def _k(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)

def _keys(s: pd.Series) -> np.ndarray:
    """Vectorised _k: 'NA' for missing, integer-valued numbers without a decimal point, otherwise str(v)."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = s.to_numpy(dtype="float64", na_value=np.nan)
        out = np.empty(len(v), dtype=object)
        nan = np.isnan(v)
        isint = ~nan & np.isfinite(v) & (v == np.floor(v))
        out[isint] = v[isint].astype(np.int64).astype(str)
        other = ~nan & ~isint
        out[other] = [str(x) for x in v[other]]
        out[nan] = "NA"
        return out
    return np.array([_k(x) for x in s.tolist()], dtype=object)



def fit_encoder(df: pd.DataFrame) -> dict:
    # codes in order of first appearance (as pandas.factorize assigns them)
    return {c: {k: i for i, k in enumerate(dict.fromkeys(_keys(df[c]).tolist()))} for c in CAT_FEATURES}


# ----------------------------------------------------------------------------------------------- data and artifacts
def load_raw(period: str, table: str) -> pd.DataFrame:
    return pd.read_parquet(BRONZE / f"{period}_{table}.parquet")


@lru_cache(maxsize=64)
def _load_art(name: str, version: str) -> str:
    return (ART / name / f"{version}.json").read_text(encoding="utf-8")


def load_artifact(name: str, version: str):
    return json.loads(_load_art(name, version))


def save_artifact(name: str, version: str, obj) -> None:
    (ART / name).mkdir(parents=True, exist_ok=True)
    (ART / name / f"{version}.json").write_text(json.dumps(obj, sort_keys=True), encoding="utf-8")
    _load_art.cache_clear()


def build_occ_xwalks() -> dict:
    """Crosswalks OCCP -> major group from the official value labels (prefix before '-'), per code vintage:
    '2012' built from the 2017 codebook (codes used 2014-2017), '2018' from the 2018 codebook (2018+)."""
    groups: dict[str, int] = {}
    res = {}
    for vint, year in (("2012", 2017), ("2018", 2018)):
        items = json.loads((CODEBOOK / str(year) / "OCCP.json").read_text())["values"]["item"]
        xw = {}
        for code, label in items.items():
            pref = label.split("-")[0].strip() if "-" in label[:5] else "NOT"
            g = groups.setdefault(pref, len(groups))
            xw[str(code).zfill(4)] = g
        save_artifact("occ_xwalk", vint, xw)
        res[vint] = len(xw)
    save_artifact("occ_groups", "all", groups)
    return res


PIPELINE = Pipeline(domain="acs", components=C, output="encode", raw_tables=["person", "housing"],
                    load_raw=load_raw, load_artifact=load_artifact)

# ----------------------------------------------------------------------------------------------- reference and spec
V = Version.make


def reference_state(encoder_id: str) -> dict:
    """The pipeline as deployed in 2014 (pandas 2.2 main environment)."""
    return {
        "person_adapter": V("v2014"),
        "housing_adapter": V("v2014"),
        "person_clean": V("v2014"),
        "join_household": V("v2014", {"key": "numeric"}),
        "occ_group": V("v2014", artifacts={"occ_xwalk": "2012"}),
        "features": V("v2014"),
        "encode": V("v2014", artifacts={"encoder": encoder_id}),
    }


def spec_state(period: str, encoder_id: str) -> dict:
    """The correct pipeline for a period: every forced source adaptation done right, nothing else changed."""
    y = int(period)
    s = reference_state(encoder_id)
    if y >= 2019:
        s["person_adapter"] = V("v2019", {"relp": "official", "wkw": "bins", "jwtr": "official",
                                          "state": "fallback"})
    if y >= 2018:
        s["join_household"] = V("v2014", {"key": "string"})
        s["occ_group"] = V("v2014", artifacts={"occ_xwalk": "2018"})
    return s


def source_events(period_from: str, period_to: str) -> list[dict]:
    """Published-schema changes between two periods (what a schema diff of the raw tables shows)."""
    ev = []
    for y, table, what, cols in (
            (2017, "person", "SERIALNO format: 13-digit year-prefixed id", ["SERIALNO"]),
            (2017, "housing", "SERIALNO format: 13-digit year-prefixed id", ["SERIALNO"]),
            (2018, "person", "SERIALNO alphanumeric (GQ/HU); OCCP 2018 code list", ["SERIALNO", "OCCP"]),
            (2018, "housing", "SERIALNO alphanumeric (GQ/HU)", ["SERIALNO"]),
            (2019, "person", "RELP->RELSHIPP, WKW->WKWN, JWTR->JWTRNS",
             ["RELP", "RELSHIPP", "WKW", "WKWN", "JWTR", "JWTRNS"]),
            (2023, "person", "ST->STATE", ["ST", "STATE"])):
        if int(period_from) < y <= int(period_to):
            ev.append({"period": str(y), "table": table, "change": what, "columns": cols})
    return ev


# ----------------------------------------------------------------------------------------------- lineage and catalog
# Which components compute or transform each model input (what a lineage tool such as mlinspect extracts).
LINEAGE = {
    "AGEP": ["person_adapter"], "SCHL": ["person_adapter"], "WKHP": ["person_adapter", "features"],
    "POBP": ["person_adapter"], "SEX": ["person_adapter"], "RAC1P": ["person_adapter"],
    "COW": ["person_adapter", "person_clean"], "MAR": ["person_adapter", "person_clean"],
    "WKW": ["person_adapter", "person_clean"], "JWTR": ["person_adapter", "person_clean"],
    "RELP": ["person_adapter"], "ST": ["person_adapter", "features"],
    "NP": ["housing_adapter", "join_household"], "TEN": ["housing_adapter", "join_household"],
    "VEH": ["housing_adapter", "join_household", "features"],
    "OCCG": ["person_adapter", "occ_group"],
    "y": ["person_adapter", "person_clean", "features"],
}
for _f in LINEAGE:
    LINEAGE[_f] = LINEAGE[_f] + ["encode"] if _f in CAT_FEATURES else LINEAGE[_f]
SOURCE_READERS = {"person": ["person_adapter"], "housing": ["housing_adapter"]}
# Column-level lineage: raw columns each component depends on (transitively), as a column-lineage tool reports.
_P_ALL = ["SERIALNO", "SPORDER", "ST", "STATE", "AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "RELSHIPP",
          "WKHP", "SEX", "RAC1P", "WKW", "WKWN", "JWTR", "JWTRNS", "PINCP", "ADJINC", "PWGTP"]
RAW_COLUMNS = {
    "person_adapter": {"person": _P_ALL},
    "housing_adapter": {"housing": ["SERIALNO", "NP", "TEN", "VEH"]},
    "person_clean": {"person": ["AGEP", "PINCP", "WKHP", "PWGTP", "COW", "JWTR", "JWTRNS", "WKW", "WKWN", "MAR"]},
    "join_household": {"person": ["SERIALNO"], "housing": ["SERIALNO"]},
    "occ_group": {"person": ["OCCP"]},
    "features": {"person": ["PINCP", "ADJINC", "WKHP", "ST", "STATE"], "housing": ["VEH"]},
    "encode": {"person": _P_ALL, "housing": ["NP", "TEN", "VEH"]},
}


def catalog(period: str, encoder_id: str) -> list[dict]:
    """Changes that can be applied to the spec state of `period`.
    kind: 'defect' (deviates from spec) or 'benign' (deviates from the 2014 reference but not harmful by design;
    whether it actually matters is measured, never assumed). tier: T1 source-coupled, T2 library, T3 incident."""
    y = int(period)
    sp = spec_state(period, encoder_id)
    out = []

    def add(cid, comp_, version, kind, tier, silent_hint, note):
        out.append({"id": cid, "component": comp_, "version": version, "kind": kind, "tier": tier,
                    "silent_hint": silent_hint, "note": note})
    pa = sp["person_adapter"]
    if y >= 2019:
        base = pa.p()
        add("acs.relp_minus20", "person_adapter", V("v2019", {**base, "relp": "minus20"}), "defect", "T1", True,
            "RELSHIPP codes shifted by -20 instead of the official crosswalk")
        add("acs.wkw_reversed", "person_adapter", V("v2019", {**base, "wkw": "reversed"}), "defect", "T1", True,
            "WKWN binned into WKW categories in reverse order")
        add("acs.wkw_passthrough", "person_adapter", V("v2019", {**base, "wkw": "passthrough"}), "defect", "T1",
            False, "WKWN weeks passed through as WKW categories")
        add("acs.jwtr_rename", "person_adapter", V("v2019", {**base, "jwtr": "rename"}), "defect", "T1", True,
            "JWTRNS used as JWTR without recoding (subway/rail/streetcar codes permuted)")
        add("acs.adapter_refactor", "person_adapter", V("v2019_refactor", base), "benign", "B", True,
            "refactor of the 2019 adapter")
    else:
        add("acs.adapter_early_upgrade", "person_adapter",
            V("v2019", {"relp": "official", "wkw": "bins", "jwtr": "official", "state": "fallback"}),
            "benign", "B", True, "2019-ready adapter deployed early")
    if y >= 2023:
        add("acs.state_fill", "person_adapter", V("v2019", {**pa.p(), "state": "fill"}), "defect", "T1", True,
            "missing ST column filled with the most common training state")
    if y >= 2018:
        add("acs.key_digits", "join_household", V("v2014", {"key": "digits"}), "defect", "T1", True,
            "alphanumeric SERIALNO 'fixed' by stripping letters (GQ/HU ids collide)")
        add("acs.key_numeric_stale", "join_household", V("v2014", {"key": "numeric"}), "defect", "T1", False,
            "join keys still parsed as numbers after the id format change")
        add("acs.occ_xwalk_stale", "occ_group", V("v2014", artifacts={"occ_xwalk": "2012"}), "defect", "T1", False,
            "occupation crosswalk not refreshed for 2018 codes")
    jk = sp["join_household"].p()
    add("acs.join_nodedupe", "join_household", V("v2014", {**jk, "dedupe_right": False}), "benign", "B", True,
        "dedupe of the household side removed (keys are unique)")
    add("acs.clean_env_p30", "person_clean", V("v2014", env="p30"), "defect", "T2", True,
        "pandas 3 upgrade: chained-assignment fill becomes a no-op")
    add("acs.clean_modernised", "person_clean", V("v2024_loc"), "benign", "B", True, "chained assignment -> .loc")
    add("acs.clean_modernised_p30", "person_clean", V("v2024_loc", env="p30"), "benign", "T2b", True,
        "pandas 3 upgrade of the modernised cleaning step")
    add("acs.occ_env_p30", "occ_group", V("v2014", artifacts=sp["occ_group"].a(), env="p30"), "benign", "T2b",
        True, "pandas 3 upgrade of the crosswalk step")
    add("acs.housing_env_p30", "housing_adapter", V("v2014", env="p30"), "benign", "T2b", True,
        "pandas 3 upgrade of the housing adapter")
    ft = sp["features"].p()
    add("acs.adjinc_ignored", "features", V("v2014", {**ft, "adjinc": "ignore"}), "defect", "T3", True,
        "income adjustment factor dropped (stale reference)")
    add("acs.wkhp_month", "features", V("v2014", {**ft, "wkhp_unit": "month"}), "defect", "T3", False,
        "hours per week reported per month (unit change)")
    add("acs.veh_stuck_tx", "features", V("v2014", {**ft, "stuck": "VEH:48:0"}), "defect", "T3", True,
        "vehicles feature stuck at 0 for Texas (upstream default)")
    add("acs.encoder_refit", "encode", V("v2014", {"refit": True}, sp["encode"].a()), "defect", "T3", True,
        "categorical codes re-fitted at serving time (training/serving skew)")
    return out

# Column derivations inside components (output column -> input columns it is computed from), for the boundary
# test's inherited-jump check; columns not listed are compared with the same-named input column.
COLUMN_SOURCES = {
    "person_adapter": {"ST": ["ST", "STATE"], "RELP": ["RELP", "RELSHIPP"], "WKW": ["WKW", "WKWN"],
                       "JWTR": ["JWTR", "JWTRNS"]},
    "join_household": {"NP": ["NP", "SERIALNO"], "TEN": ["TEN", "SERIALNO"], "VEH": ["VEH", "SERIALNO"]},
    "occ_group": {"OCCG": ["OCCP"]},
    "features": {"y": ["PINCP", "ADJINC"]},
}
