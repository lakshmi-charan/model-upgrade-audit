"""US airline on-time pipeline (v2 held-out domain): BTS Reporting Carrier On-Time Performance, months 1/4/7/10
of 2015-2026.

Task: for a scheduled domestic flight that operated (not cancelled or diverted), did it arrive 15 or more minutes
late? Features are known at scheduling time.

The BTS distributes every year in the current record layout (it republishes history), so this domain has no
dated published-schema change: its degradations are real-world changes (carrier mergers, the 2020 pandemic, new
routes) and internal pipeline defects (library upgrades and incident replays). It tests whether a diagnoser
blames the pipeline when the world moved, and vice versa, without the help (or the trap) of a schema event.
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
BRONZE = ROOT / "data" / "bronze" / "bts"
ART = ROOT / "data" / "artifacts" / "bts"

PERIODS = sorted(p.stem for p in BRONZE.glob("[0-9]*.parquet")) if BRONZE.exists() else []
CAT_FEATURES = ["carrier", "origin_state", "dest_state", "origin_size", "dest_size"]
NUM_FEATURES = ["month", "dow", "day_of_year", "dep_min", "arr_min", "sched_min", "distance"]
FEATURES = NUM_FEATURES + CAT_FEATURES
LABEL = "y"
ID_COLS = ["flight_id"]
TASK = "binary"

C = {}


def comp(name, inputs, doc=""):
    C[name] = Component(name, inputs, doc=doc)
    return C[name]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce")


flight_adapter = comp("flight_adapter", ["raw:flights"], "published flight records -> canonical schema")


@flight_adapter.variant("v2015")
def _fa(inp, p, a):
    r = inp["raw:flights"]
    return pd.DataFrame({
        "flight_id": r["_row"].astype("int64"),
        "date": r["FlightDate"].astype(str),
        "dow_raw": _num(r["DayOfWeek"]),
        "carrier": r["Reporting_Airline"].astype(str),
        "origin_id": r["OriginAirportID"].astype(str), "dest_id": r["DestAirportID"].astype(str),
        "origin_state": r["OriginState"].astype(str), "dest_state": r["DestState"].astype(str),
        "crs_dep": r["CRSDepTime"].astype(str), "crs_arr": r["CRSArrTime"].astype(str),
        "sched_min": _num(r["CRSElapsedTime"]), "distance": _num(r["Distance"]),
        "cancelled": _num(r["Cancelled"]), "diverted": _num(r["Diverted"]),
        "arr_del15": _num(r["ArrDel15"]), "arr_delay": _num(r["ArrDelay"]),
    })


time_features = comp("time_features", ["flight_adapter"], "calendar and scheduled clock times")


def _hhmm(s: pd.Series, as_minutes_bug: bool) -> pd.Series:
    x = pd.to_numeric(s.str.strip(), errors="coerce")
    if as_minutes_bug:            # incident class: "hhmm" read as a number of minutes
        return x
    return (x // 100) * 60 + (x % 100)


@time_features.variant("v2015")
def _tf(inp, p, a):
    df = inp["flight_adapter"].copy()
    d = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["month"] = d.dt.month.astype("float64")
    df["dow"] = (d.dt.dayofweek.astype("float64") if p.get("dow_from_date", False)
                 else df["dow_raw"] - 1.0)     # BTS: 1 = Monday; pandas: 0 = Monday
    # original idiom: day of year from epoch nanoseconds (resolution-dependent across pandas versions)
    jan1 = pd.to_datetime(d.dt.year.astype(str) + "-01-01", format="%Y-%m-%d")
    df["day_of_year"] = ((d.astype("int64") - jan1.astype("int64")) // 86_400_000_000_000).astype("float64")
    df["dep_min"] = _hhmm(df["crs_dep"], p.get("hhmm_as_minutes", False)).astype("float64")
    df["arr_min"] = _hhmm(df["crs_arr"], p.get("hhmm_as_minutes", False)).astype("float64")
    return df.drop(columns=["date", "dow_raw", "crs_dep", "crs_arr"])


@time_features.variant("v2024_dt")
def _tf_dt(inp, p, a):
    """Correct modernisation: day of year from the datetime accessor."""
    df = inp["flight_adapter"].copy()
    d = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["month"] = d.dt.month.astype("float64")
    df["dow"] = df["dow_raw"] - 1.0
    df["day_of_year"] = (d.dt.dayofyear - 1).astype("float64")
    df["dep_min"] = _hhmm(df["crs_dep"], False).astype("float64")
    df["arr_min"] = _hhmm(df["crs_arr"], False).astype("float64")
    return df.drop(columns=["date", "dow_raw", "crs_dep", "crs_arr"])


clean = comp("clean", ["time_features"], "operated flights, label")


@clean.variant("v2015")
def _cl(inp, p, a):
    df = inp["time_features"]
    keep = (df["cancelled"] == 0) & (df["diverted"] == 0) if not p.get("keep_diverted", False) \
        else (df["cancelled"] == 0)
    df = df[keep & df["sched_min"].notna()].copy()
    # pandas-1.x idiom: overnight flights arrive "before" they depart on the clock; continue the arrival clock past
    # midnight (chained assignment; a no-op under copy-on-write)
    df["arr_min"][df["arr_min"] < df["dep_min"]] = df["arr_min"] + 1440
    if p.get("label_from_delay", False):    # incident class: label recomputed with the wrong boundary
        df["y"] = (df["arr_delay"] > 15).astype("int8")
    else:
        df["y"] = (df["arr_del15"] == 1).astype("int8")
    return df.drop(columns=["cancelled", "diverted", "arr_del15", "arr_delay"]).reset_index(drop=True)


@clean.variant("v2024_loc")
def _cl_loc(inp, p, a):
    df = inp["time_features"]
    df = df[(df["cancelled"] == 0) & (df["diverted"] == 0) & df["sched_min"].notna()].copy()
    df.loc[df["arr_min"] < df["dep_min"], "arr_min"] = df["arr_min"] + 1440
    df["y"] = (df["arr_del15"] == 1).astype("int8")
    return df.drop(columns=["cancelled", "diverted", "arr_del15", "arr_delay"]).reset_index(drop=True)


airport_join = comp("airport_join", ["clean"], "origin/destination airport size class from the airport artifact")


@airport_join.variant("v2015")
def _aj(inp, p, a):
    df = inp["clean"].copy()
    size = a["airports"]
    dest_key = "origin_id" if p.get("dest_key", "dest_id") == "origin_id" else "dest_id"
    df["origin_size"] = df["origin_id"].map(size).fillna("unknown")
    df["dest_size"] = df[dest_key].map(size).fillna("unknown")
    if p.get("distance_km", False):
        df["distance"] = df["distance"] * 1.609
    return df.drop(columns=["origin_id", "dest_id"])


encode = comp("encode", ["airport_join"], "model matrix with categorical codes from the encoder artifact")


def fit_encoder(df: pd.DataFrame) -> dict:
    return {c: {k: i for i, k in enumerate(dict.fromkeys(df[c].astype(str).tolist()))} for c in CAT_FEATURES}


@encode.variant("v2015")
def _enc(inp, p, a):
    df = inp["airport_join"]
    out = df[ID_COLS + NUM_FEATURES].copy()
    maps = fit_encoder(df) if p.get("refit", False) else a["encoder"]
    for c in CAT_FEATURES:
        codes = df[c].astype(str).map(maps[c])
        if p.get("unknown_as_reference", False):    # incident replay (scikit-learn issue #34549)
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


def build_airport_artifact(period: str) -> None:
    """Airport size class (large / medium / small by departures in the period's records), as an operations team
    would maintain it; built from the 2015-01 records (version 'v2015')."""
    r = pd.read_parquet(BRONZE / f"{period}.parquet", columns=["OriginAirportID"])
    n = r["OriginAirportID"].astype(str).value_counts()
    share = n.cumsum() / n.sum()
    cls = {k: ("large" if share[k] <= 0.5 else "medium" if share[k] <= 0.85 else "small") for k in n.index}
    save_artifact("airports", "v2015", cls)


PIPELINE = Pipeline(domain="bts", components=C, output="encode", raw_tables=["flights"],
                    load_raw=load_raw, load_artifact=load_artifact)
V = Version.make


def reference_state(encoder_id: str) -> dict:
    return {
        "flight_adapter": V("v2015"),
        "time_features": V("v2015", env="p15"),    # deployed in a pandas-1.5 environment
        "clean": V("v2015", env="p15"),
        "airport_join": V("v2015", artifacts={"airports": "v2015"}),
        "encode": V("v2015", artifacts={"encoder": encoder_id}),
    }


def spec_state(period: str, encoder_id: str) -> dict:
    return reference_state(encoder_id)


def source_events(period_from: str, period_to: str) -> list[dict]:
    return []


LINEAGE = {
    "month": ["flight_adapter", "time_features"], "dow": ["flight_adapter", "time_features"],
    "day_of_year": ["flight_adapter", "time_features"], "dep_min": ["flight_adapter", "time_features"],
    "arr_min": ["flight_adapter", "time_features", "clean"], "sched_min": ["flight_adapter", "clean"],
    "distance": ["flight_adapter", "airport_join"],
    "carrier": ["flight_adapter", "encode"], "origin_state": ["flight_adapter", "encode"],
    "dest_state": ["flight_adapter", "encode"],
    "origin_size": ["flight_adapter", "airport_join", "encode"], "dest_size": ["flight_adapter", "airport_join", "encode"],
    "y": ["flight_adapter", "clean"],
}
SOURCE_READERS = {"flights": ["flight_adapter"]}
COLUMN_SOURCES = {
    "time_features": {"month": ["date"], "dow": ["date", "dow_raw"], "day_of_year": ["date"],
                      "dep_min": ["crs_dep"], "arr_min": ["crs_arr"]},
    "airport_join": {"origin_size": ["origin_id"], "dest_size": ["dest_id"], "distance": ["distance"]},
    "clean": {"y": ["arr_del15", "arr_delay"], "arr_min": ["arr_min", "dep_min"]},
}


def catalog(period: str, encoder_id: str) -> list[dict]:
    sp = spec_state(period, encoder_id)
    out = []

    def add(cid, comp_, version, kind, tier, silent_hint, note, source=""):
        out.append({"id": cid, "component": comp_, "version": version, "kind": kind, "tier": tier,
                    "silent_hint": silent_hint, "note": note, "source": source})
    add("bts.adapter_env_p30", "flight_adapter", V("v2015", env="p30"), "benign", "T2b", True,
        "pandas 3 upgrade of the adapter")
    add("bts.time_env_main", "time_features", V("v2015"), "benign", "T2b", True,
        "pandas 2.2 upgrade of the time step (string dates still parse to nanoseconds)")
    add("bts.time_env_p30", "time_features", V("v2015", env="p30"), "defect", "T2", True,
        "pandas 3 upgrade: day of year via astype(int64) changes resolution",
        "pandas issue #65308; pandas 3.0 release notes (datetime resolution inference)")
    add("bts.time_modernised", "time_features", V("v2024_dt"), "benign", "B", True, "day of year via .dt accessor")
    add("bts.hhmm_as_minutes", "time_features", V("v2015", {"hhmm_as_minutes": True}, env="p15"), "defect", "T3",
        True, "scheduled hhmm clock times read as minutes")
    add("bts.dow_from_date", "time_features", V("v2015", {"dow_from_date": True}, env="p15"), "benign", "B", True,
        "day of week from the date instead of the published field (same values)")
    add("bts.clean_env_main", "clean", V("v2015"), "benign", "T2b", True,
        "pandas 2.2 upgrade of the cleaning step")
    add("bts.clean_env_p30", "clean", V("v2015", env="p30"), "defect", "T2", True,
        "pandas 3 upgrade: chained-assignment overnight arrival fix becomes a no-op",
        "pandas 3.0 release notes (copy-on-write)")
    add("bts.clean_modernised", "clean", V("v2024_loc"), "benign", "B", True, "chained assignment -> .loc")
    add("bts.label_from_delay", "clean", V("v2015", {"label_from_delay": True}, env="p15"), "defect", "T3", True,
        "label recomputed as delay > 15 instead of the published >= 15 flag")
    add("bts.keep_diverted", "clean", V("v2015", {"keep_diverted": True}, env="p15"), "defect", "T3", True,
        "diverted flights kept (scope filter bug)")
    add("bts.dest_key_bug", "airport_join", V("v2015", {"dest_key": "origin_id"}, {"airports": "v2015"}), "defect",
        "T3", True, "destination size joined on the origin airport id (copy-paste key bug)")
    add("bts.distance_km", "airport_join", V("v2015", {"distance_km": True}, {"airports": "v2015"}), "defect", "T3",
        True, "distance converted to km (unit change)")
    add("bts.join_env_p30", "airport_join", V("v2015", artifacts={"airports": "v2015"}, env="p30"), "benign",
        "T2b", True, "pandas 3 upgrade of the join")
    add("bts.encoder_refit", "encode", V("v2015", {"refit": True}, sp["encode"].a()), "defect", "T3", True,
        "categorical codes re-fitted at serving time (training/serving skew)")
    add("bts.unknown_as_reference", "encode", V("v2015", {"unknown_as_reference": True}, sp["encode"].a()),
        "defect", "T3r", True, "unseen categories encoded as the reference category", "scikit-learn issue #34549")
    return out
