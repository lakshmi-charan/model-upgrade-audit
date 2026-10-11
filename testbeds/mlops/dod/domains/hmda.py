"""HMDA mortgage-application pipeline (v2 held-out domain): Home Mortgage Disclosure Act loan/application
records for Colorado, Oregon and Tennessee, 2012-2025.

Task: for an application that was either originated or denied, was it denied?

Real, dated source changes in the published records (CFPB historic data -> FFIEC Data Browser; found from
data/bronze/hmda/_profile.json): value formats in 2014 (blank missing values), 2015 ('NA' again), 2017 (no zero
padding), 2020 and 2024 (tract income percentage with / without decimals), and the major one:
  2018  the record format was rebuilt (Regulation C amendments): loan amount published in dollars (was thousands);
        loan purpose refinancing split into 31 / 32 (cash-out) and new codes 4 / 5; owner occupancy replaced by
        occupancy type (1 principal, 2 second residence, 3 investment); property type replaced by construction
        method and total units; state published as a postal code and county as a 5-digit FIPS code; tract fields
        renamed; open-end lines of credit, reverse mortgages and business-purpose loans flagged in new fields.
The adapter written for the 2012 format cannot read the 2018 records at all (a loud failure); the correct
pipeline needs a new adapter (a forced change). Realistic mistakes in that new adapter are the source-coupled
defects of this domain.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from dod.core import Component, Pipeline, Version

ROOT = Path(os.environ.get("DOD_PROJECT_ROOT", Path(__file__).resolve().parents[4]))
BRONZE = ROOT / "data" / "bronze" / "hmda"
ART = ROOT / "data" / "artifacts" / "hmda"

PERIODS = sorted(p.stem for p in BRONZE.glob("[0-9]*.parquet")) if BRONZE.exists() else []
CAT_FEATURES = ["state", "county", "loan_type", "purpose", "occupancy", "property", "lien", "preapproval"]
NUM_FEATURES = ["amount_k", "income_k", "lti", "tract_minority_pct", "tract_income_pct", "msa_income_k"]
FEATURES = NUM_FEATURES + CAT_FEATURES
LABEL = "y"
ID_COLS = ["app_id"]
TASK = "binary"
POSTAL_TO_FIPS = {"CO": "08", "OR": "41", "TN": "47"}

C = {}


def comp(name, inputs, doc=""):
    C[name] = Component(name, inputs, doc=doc)
    return C[name]


def _num(s: pd.Series) -> pd.Series:
    """Published numeric text -> float ('NA', 'Exempt', blanks -> NaN)."""
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce")


def _code(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


lar_adapter = comp("lar_adapter", ["raw:lar"], "published application records -> canonical schema")


@lar_adapter.variant("v2012")
def _la_2012(inp, p, a):
    """Adapter for the 2012-2017 record format."""
    r = inp["raw:lar"]
    st = _code(r["state_code"]).str.zfill(2)
    return pd.DataFrame({
        "app_id": r["_row"].astype("int64"),
        "state": st, "county": st + _code(r["county_code"]).str.zfill(3),
        "loan_type": _code(r["loan_type"]), "purpose": _code(r["loan_purpose"]),
        "occupancy": _code(r["owner_occupancy"]), "property": _code(r["property_type"]),
        "lien": _code(r["lien_status"]), "preapproval": _code(r["preapproval"]),
        "amount_k": _num(r["loan_amount_000s"]), "income_k": _num(r["applicant_income_000s"]),
        "tract_minority_pct": _num(r["minority_population"]), "tract_income_pct": _num(r["tract_to_msamd_income"]),
        "msa_income_k": _num(r["hud_median_family_income"]) / 1000.0,
        "action": _code(r["action_taken"]),
        "open_end": "2", "reverse": "2", "business": "2",
    })


@lar_adapter.variant("v2018")
def _la_2018(inp, p, a):
    """Adapter for the 2018 record format, mapping the new fields onto the 2012 semantics. Each parameter switches
    on one realistic mistake (default: correct)."""
    r = inp["raw:lar"]
    st = _code(r["state_code"]).map(POSTAL_TO_FIPS).fillna("NA")
    cty = _code(r["county_code"])
    purpose = _code(r["loan_purpose"])
    if not p.get("purpose_alias", False):          # 31/32 refinance (cash-out) -> 3; 4 other, 5 n/a -> 4
        purpose = purpose.replace({"31": "3", "32": "3", "5": "4"})
    occ = _code(r["occupancy_type"])
    if not p.get("occupancy_alias", False):        # 1 principal residence; 2 second, 3 investment -> not owner-occ.
        occ = occ.replace({"3": "2"})
    units = _code(r["total_units"])
    prop = np.where(_code(r["construction_method"]) == "2", "2",
                    np.where(units.isin(["1", "2", "3", "4"]), "1", "3"))
    amount = _num(r["loan_amount"])
    if not p.get("amount_dollars", False):
        amount = amount / 1000.0
    out = pd.DataFrame({
        "app_id": r["_row"].astype("int64"),
        "state": st,
        "county": (st + cty) if p.get("county_concat", False) else cty.str.zfill(5),
        "loan_type": _code(r["loan_type"]), "purpose": purpose, "occupancy": occ, "property": prop,
        "lien": _code(r["lien_status"]), "preapproval": _code(r["preapproval"]),
        "amount_k": amount, "income_k": _num(r["income"]),
        "tract_minority_pct": _num(r["tract_minority_population_percent"]),
        "tract_income_pct": _num(r["tract_to_msa_income_percentage"]),
        "msa_income_k": _num(r["ffiec_msa_md_median_family_income"]) / 1000.0,
        "action": _code(r["action_taken"]),
        "open_end": _code(r["open-end_line_of_credit"]), "reverse": _code(r["reverse_mortgage"]),
        "business": _code(r["business_or_commercial_purpose"]),
    })
    if p.get("ignore_coverage", False):            # new product types counted as ordinary applications
        out[["open_end", "reverse", "business"]] = "2"
    return out


clean = comp("clean", ["lar_adapter"], "applications in scope, label, missing income")


@clean.variant("v2012")
def _cl(inp, p, a):
    df = inp["lar_adapter"]
    df = df[df["action"].isin(["1", "3"]) & df["lien"].isin(["1", "2"]) & df["property"].isin(["1", "2"])
            & (df["open_end"] != "1") & (df["reverse"] != "1") & (df["business"] != "1")].copy()
    # pandas-1.x idiom: missing income -> sentinel -1 (chained assignment; a no-op under copy-on-write)
    df["income_k"][df["income_k"].isna()] = -1.0
    df["y"] = (df["action"] == "3").astype("int8")
    return df.reset_index(drop=True)


@clean.variant("v2024_loc")
def _cl_loc(inp, p, a):
    df = inp["lar_adapter"]
    df = df[df["action"].isin(["1", "3"]) & df["lien"].isin(["1", "2"]) & df["property"].isin(["1", "2"])
            & (df["open_end"] != "1") & (df["reverse"] != "1") & (df["business"] != "1")].copy()
    df.loc[df["income_k"].isna(), "income_k"] = -1.0
    df["y"] = (df["action"] == "3").astype("int8")
    return df.reset_index(drop=True)


@clean.variant("v2024_np")
def _cl_np(inp, p, a):
    """numpy rewrite of the sentinel step. Correct form: np.nan_to_num(x, nan=-1). The incident replay (folktables
    issue #39) writes np.nan_to_num(x, -1): -1 is taken as `copy`, so missing income becomes 0."""
    df = inp["lar_adapter"]
    df = df[df["action"].isin(["1", "3"]) & df["lien"].isin(["1", "2"]) & df["property"].isin(["1", "2"])
            & (df["open_end"] != "1") & (df["reverse"] != "1") & (df["business"] != "1")].copy()
    x = df["income_k"].to_numpy(dtype="float64")
    df["income_k"] = np.nan_to_num(x, -1) if p.get("positional", False) else np.nan_to_num(x, nan=-1.0)
    df["y"] = (df["action"] == "3").astype("int8")
    return df.reset_index(drop=True)


features = comp("features", ["clean"], "loan-to-income ratio")


@features.variant("v2012")
def _ft(inp, p, a):
    df = inp["clean"].copy()
    inc = df["income_k"].where(df["income_k"] > 0)
    df["lti"] = (inc / df["amount_k"]) if p.get("lti_inverted", False) else (df["amount_k"] / inc)
    return df


encode = comp("encode", ["features"], "model matrix with categorical codes from the encoder artifact")


def fit_encoder(df: pd.DataFrame) -> dict:
    return {c: {k: i for i, k in enumerate(dict.fromkeys(df[c].astype(str).tolist()))} for c in CAT_FEATURES}


@encode.variant("v2012")
def _enc(inp, p, a):
    df = inp["features"]
    out = df[ID_COLS + NUM_FEATURES].copy()
    maps = fit_encoder(df) if p.get("refit", False) else a["encoder"]
    for c in CAT_FEATURES:
        codes = df[c].astype(str).map(maps[c])
        if p.get("unknown_as_reference", False):
            # incident replay (scikit-learn issue #34549): with drop='first' and handle_unknown='ignore', unseen
            # categories are encoded exactly like the dropped reference category
            codes = codes.fillna(0)
        out[c] = codes.astype("float64")
    out[LABEL] = df[LABEL].values
    return out[ID_COLS + FEATURES + [LABEL]]


# ----------------------------------------------------------------------------------------------- data and artifacts
def load_raw(period: str, table: str) -> pd.DataFrame:
    df = pd.read_parquet(BRONZE / f"{period}.parquet")
    df.insert(0, "_row", np.arange(len(df), dtype="int64"))
    return df


@lru_cache(maxsize=64)
def _load_art(name: str, version: str) -> str:
    return (ART / name / f"{version}.json").read_text(encoding="utf-8")


def load_artifact(name: str, version: str):
    return json.loads(_load_art(name, version))


def save_artifact(name: str, version: str, obj) -> None:
    (ART / name).mkdir(parents=True, exist_ok=True)
    (ART / name / f"{version}.json").write_text(json.dumps(obj, sort_keys=True), encoding="utf-8")
    _load_art.cache_clear()


PIPELINE = Pipeline(domain="hmda", components=C, output="encode", raw_tables=["lar"],
                    load_raw=load_raw, load_artifact=load_artifact)
V = Version.make


def reference_state(encoder_id: str) -> dict:
    return {
        "lar_adapter": V("v2012"),
        "clean": V("v2012", env="p15"),          # deployed in a pandas-1.5 environment
        "features": V("v2012"),
        "encode": V("v2012", artifacts={"encoder": encoder_id}),
    }


def spec_state(period: str, encoder_id: str) -> dict:
    s = reference_state(encoder_id)
    if period >= "2018":
        s["lar_adapter"] = V("v2018")
    return s


_FMT_EVENTS = [
    ("2014", "missing values published as blanks instead of 'NA'", ["applicant_income_000s", "county_code",
                                                                   "tract_to_msamd_income", "minority_population",
                                                                   "hud_median_family_income"]),
    ("2015", "missing values published as 'NA' again", ["applicant_income_000s", "county_code",
                                                         "tract_to_msamd_income", "minority_population",
                                                         "hud_median_family_income"]),
    ("2017", "numbers published without zero padding", ["loan_amount_000s", "applicant_income_000s",
                                                        "hud_median_family_income", "minority_population"]),
    ("2020", "tract income percentage published with decimals", ["tract_to_msa_income_percentage"]),
    ("2024", "tract income percentage published as integers again", ["tract_to_msa_income_percentage"]),
]


def source_events(period_from: str, period_to: str) -> list[dict]:
    ev = [{"period": w, "table": "lar", "change": what, "columns": cols}
          for w, what, cols in _FMT_EVENTS if period_from < w <= period_to]
    if period_from < "2018" <= period_to:
        ev.append({"period": "2018", "table": "lar",
                   "change": "record format rebuilt (Regulation C): loan amount in dollars; loan purpose, occupancy "
                             "and property recoded; state as postal code, county as 5-digit FIPS; tract fields "
                             "renamed; open-end, reverse-mortgage and business-purpose flags added",
                   "columns": _ALL_2012 + _ALL_2018})
    return sorted(ev, key=lambda e: e["period"])


# ----------------------------------------------------------------------------------------------- lineage and catalog
LINEAGE = {
    "amount_k": ["lar_adapter"], "income_k": ["lar_adapter", "clean"], "lti": ["lar_adapter", "clean", "features"],
    "tract_minority_pct": ["lar_adapter"], "tract_income_pct": ["lar_adapter"], "msa_income_k": ["lar_adapter"],
    **{c: ["lar_adapter", "encode"] for c in CAT_FEATURES},
    "y": ["lar_adapter", "clean"],
}
SOURCE_READERS = {"lar": ["lar_adapter"]}
_ALL_2012 = ["state_code", "county_code", "loan_type", "loan_purpose", "owner_occupancy", "property_type",
             "lien_status", "preapproval", "loan_amount_000s", "applicant_income_000s", "minority_population",
             "tract_to_msamd_income", "hud_median_family_income", "action_taken"]
_ALL_2018 = ["state_code", "county_code", "loan_type", "loan_purpose", "occupancy_type", "total_units",
             "construction_method", "lien_status", "preapproval", "loan_amount", "income",
             "tract_minority_population_percent", "tract_to_msa_income_percentage",
             "ffiec_msa_md_median_family_income", "action_taken", "open-end_line_of_credit", "reverse_mortgage",
             "business_or_commercial_purpose"]
RAW_COLUMNS = {
    "lar_adapter": {"lar": _ALL_2012 + _ALL_2018},
    "clean": {"lar": ["action_taken", "lien_status", "property_type", "construction_method", "total_units",
                      "applicant_income_000s", "income", "open-end_line_of_credit", "reverse_mortgage",
                      "business_or_commercial_purpose"]},
    "features": {"lar": ["loan_amount_000s", "loan_amount", "applicant_income_000s", "income"]},
    "encode": {"lar": _ALL_2012 + _ALL_2018},
}
COLUMN_SOURCES = {
    "lar_adapter": {"state": ["state_code"], "county": ["state_code", "county_code"],
                    "purpose": ["loan_purpose"], "occupancy": ["owner_occupancy", "occupancy_type"],
                    "property": ["property_type", "construction_method", "total_units"],
                    "amount_k": ["loan_amount_000s", "loan_amount"], "income_k": ["applicant_income_000s", "income"],
                    "tract_minority_pct": ["minority_population", "tract_minority_population_percent"],
                    "tract_income_pct": ["tract_to_msamd_income", "tract_to_msa_income_percentage"],
                    "msa_income_k": ["hud_median_family_income", "ffiec_msa_md_median_family_income"],
                    "action": ["action_taken"], "app_id": ["_row"],
                    "open_end": ["open-end_line_of_credit"], "reverse": ["reverse_mortgage"],
                    "business": ["business_or_commercial_purpose"]},
    "clean": {"y": ["action"]},
    "features": {"lti": ["amount_k", "income_k"]},
}


def catalog(period: str, encoder_id: str) -> list[dict]:
    sp = spec_state(period, encoder_id)
    out = []

    def add(cid, comp_, version, kind, tier, silent_hint, note, source=""):
        out.append({"id": cid, "component": comp_, "version": version, "kind": kind, "tier": tier,
                    "silent_hint": silent_hint, "note": note, "source": source})
    if period >= "2018":
        for k, note in (("amount_dollars", "new adapter keeps the loan amount in dollars (was thousands)"),
                        ("purpose_alias", "new loan-purpose codes 31/32/5 passed through instead of recoded "
                                          "(rename without recode)"),
                        ("occupancy_alias", "occupancy type read as the old owner-occupancy code (investment "
                                            "property becomes 'not applicable')"),
                        ("county_concat", "county key built the old way (state + county code) from the new "
                                          "5-digit county code"),
                        ("ignore_coverage", "new open-end / reverse / business-purpose flags ignored: products "
                                            "outside the old scope enter the data")):
            add(f"hmda.{k}", "lar_adapter", V("v2018", {k: True}), "defect", "T1", True, note,
                "folktables issue #22 (rename without recode)" if k == "purpose_alias" else "")
        add("hmda.adapter_2018_env_p30", "lar_adapter", V("v2018", env="p30"), "benign", "T2b", True,
            "pandas 3 upgrade of the new adapter")
    else:
        add("hmda.adapter_env_p30", "lar_adapter", V("v2012", env="p30"), "benign", "T2b", True,
            "pandas 3 upgrade of the adapter")
    add("hmda.clean_env_main", "clean", V("v2012"), "benign", "T2b", True,
        "pandas 2.2 upgrade of the cleaning step (chained assignment still effective)")
    add("hmda.clean_env_p30", "clean", V("v2012", env="p30"), "defect", "T2", True,
        "pandas 3 upgrade: chained-assignment missing-income sentinel becomes a no-op",
        "pandas 3.0 release notes (copy-on-write)")
    add("hmda.clean_modernised", "clean", V("v2024_loc"), "benign", "B", True, "chained assignment -> .loc")
    add("hmda.clean_numpy", "clean", V("v2024_np"), "benign", "B", True, "sentinel via np.nan_to_num(x, nan=-1)")
    add("hmda.income_nan_zero", "clean", V("v2024_np", {"positional": True}), "defect", "T3r", True,
        "np.nan_to_num(x, -1): missing income becomes 0 instead of -1", "folktables issue #39")
    add("hmda.lti_inverted", "features", V("v2012", {"lti_inverted": True}), "defect", "T3", True,
        "loan-to-income ratio computed upside down")
    add("hmda.features_env_p30", "features", V("v2012", env="p30"), "benign", "T2b", True,
        "pandas 3 upgrade of the feature step")
    add("hmda.encoder_refit", "encode", V("v2012", {"refit": True}, sp["encode"].a()), "defect", "T3", True,
        "categorical codes re-fitted at serving time (training/serving skew)")
    add("hmda.unknown_as_reference", "encode", V("v2012", {"unknown_as_reference": True}, sp["encode"].a()),
        "defect", "T3r", True, "unseen categories encoded as the reference category", "scikit-learn issue #34549")
    return out
