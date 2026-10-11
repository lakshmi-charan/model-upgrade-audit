"""NYC yellow-taxi tipping pipeline (spatio-temporal tabular): TLC trip records, quarterly months 2015-2026.

Task: for a credit-card trip, is the tip at least 20% of the fare?

Real, dated source changes in the published files (verified in runs/data_probe):
  2018-07  passenger_count and RatecodeID published as double instead of int64
  2019-01  congestion_surcharge populated
  2021-01  airport_fee populated
  2023-04  airport_fee renamed Airport_fee; VendorID/LocationIDs int32
  2025-01  cbd_congestion_fee added (Manhattan congestion pricing, a real-world change at the same time)
  2026-07  request_source added
Unlike ACS, several of these break the original pipeline *without any change to its code* (latent defects
exposed by a source change): a case-sensitive column lookup, and a dtype-dependent string key.
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
BRONZE = ROOT / "data" / "bronze" / "tlc"
RAW = ROOT / "data" / "raw" / "tlc"
ART = ROOT / "data" / "artifacts" / "tlc"

PERIODS = sorted(p.stem for p in BRONZE.glob("*.parquet")) if BRONZE.exists() else []
CAT_FEATURES = ["vendor", "ratecode", "pu_borough", "do_borough", "pu_service", "do_service"]
NUM_FEATURES = ["hour", "dow", "duration_min", "trip_distance", "speed_mph", "passenger_count", "fare_amount",
                "extra", "tolls_amount", "congestion_surcharge", "airport_fee", "other_charges"]
FEATURES = NUM_FEATURES + CAT_FEATURES
LABEL = "y"
ID_COLS = ["trip_id"]
TASK = "binary"

C = {}


def comp(name, inputs, doc=""):
    C[name] = Component(name, inputs, doc=doc)
    return C[name]


def col(r: pd.DataFrame, name: str, case_insensitive: bool) -> pd.Series:
    if name in r.columns:
        return r[name]
    if case_insensitive:
        for c in r.columns:
            if c.lower() == name.lower():
                return r[c]
    return pd.Series(np.nan, index=r.index)


trip_adapter = comp("trip_adapter", ["raw:trips"], "published trip columns -> canonical schema")


@trip_adapter.variant("v2015")
def _ta_v2015(inp, p, a):
    r = inp["raw:trips"]
    ci = p.get("case_insensitive", False)
    out = pd.DataFrame({
        "trip_id": r["_row"].astype("int64"),
        "vendor": r["VendorID"].astype(str),
        "pickup": r["tpep_pickup_datetime"], "dropoff": r["tpep_dropoff_datetime"],
        "passenger_count": r["passenger_count"].astype("float64"),
        "trip_distance": r["trip_distance"].astype("float64"),
        # original code: string category straight from the published dtype ("1" for int64; "1.0" for double)
        "ratecode": (r["RatecodeID"].astype("Int64").astype(str) if p.get("ratecode_int", False)
                     else r["RatecodeID"].astype(str)),
        "PULocationID": r["PULocationID"].astype("int64"), "DOLocationID": r["DOLocationID"].astype("int64"),
        "payment_type": r["payment_type"].astype("float64"),
    })
    for c in ("fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge",
              "total_amount", "congestion_surcharge"):
        out[c] = col(r, c, ci).astype("float64").fillna(0.0)
    out["airport_fee"] = col(r, "airport_fee", ci).astype("float64").fillna(0.0)
    out["cbd_congestion_fee"] = (col(r, "cbd_congestion_fee", ci).astype("float64").fillna(0.0)
                                 if p.get("cbd", False) else 0.0)
    return out


time_features = comp("time_features", ["trip_adapter"], "hour, day of week, duration")


@time_features.variant("v2015")
def _tf(inp, p, a):
    df = inp["trip_adapter"].copy()
    pu, do = pd.to_datetime(df["pickup"]), pd.to_datetime(df["dropoff"])
    if p.get("tz", "local") == "utc":       # incident class: timestamps converted as if they were UTC
        pu, do = pu + pd.Timedelta(hours=-5), do + pd.Timedelta(hours=-5)
    df["hour"] = pu.dt.hour.astype("float64")
    df["dow"] = pu.dt.dayofweek.astype("float64")
    # original idiom: epoch nanoseconds via astype(int64) (resolution-dependent across pandas versions)
    df["duration_min"] = (do.astype("int64") - pu.astype("int64")) / 6e10
    return df.drop(columns=["pickup", "dropoff"])


@time_features.variant("v2024_td")
def _tf_td(inp, p, a):
    """Correct modernisation: timedelta arithmetic, independent of the datetime resolution."""
    df = inp["trip_adapter"].copy()
    pu, do = pd.to_datetime(df["pickup"]), pd.to_datetime(df["dropoff"])
    if p.get("tz", "local") == "utc":
        pu, do = pu + pd.Timedelta(hours=-5), do + pd.Timedelta(hours=-5)
    df["hour"] = pu.dt.hour.astype("float64")
    df["dow"] = pu.dt.dayofweek.astype("float64")
    df["duration_min"] = (do - pu).dt.total_seconds() / 60.0
    return df.drop(columns=["pickup", "dropoff"])


clean = comp("clean", ["time_features"], "credit-card trips with plausible fare, distance and duration")


@clean.variant("v2015")
def _cl(inp, p, a):
    df = inp["time_features"]
    df = df[(df["payment_type"] == 1) & (df["fare_amount"] > 0) & (df["trip_distance"] > 0)
            & (df["duration_min"] > 0) & (df["duration_min"] < 180)].copy()
    # pandas-1.x idiom: zero passengers means "not recorded"
    df["passenger_count"][df["passenger_count"] == 0] = np.nan
    return df.reset_index(drop=True)


@clean.variant("v2024_loc")
def _cl_loc(inp, p, a):
    df = inp["time_features"]
    df = df[(df["payment_type"] == 1) & (df["fare_amount"] > 0) & (df["trip_distance"] > 0)
            & (df["duration_min"] > 0) & (df["duration_min"] < 180)].copy()
    df.loc[df["passenger_count"] == 0, "passenger_count"] = np.nan
    return df.reset_index(drop=True)


zone_join = comp("zone_join", ["clean"], "pickup/dropoff borough and service zone from the zone lookup")


@zone_join.variant("v2015")
def _zj(inp, p, a):
    df = inp["clean"].copy()
    z = pd.DataFrame(a["zones"]).astype({"LocationID": "int64"})
    zp = z.rename(columns={"LocationID": "PULocationID", "Borough": "pu_borough", "service_zone": "pu_service"})
    do_key = "PULocationID" if p.get("do_key", "DOLocationID") == "PULocationID" else "DOLocationID"
    zd = z.rename(columns={"LocationID": do_key, "Borough": "do_borough", "service_zone": "do_service"})
    out = df.merge(zp[["PULocationID", "pu_borough", "pu_service"]], on="PULocationID", how="left")
    out = out.merge(zd[[do_key, "do_borough", "do_service"]], on=do_key, how="left")
    return out


fare_features = comp("fare_features", ["zone_join"], "fare components, speed, and label")


@fare_features.variant("v2015")
def _ff(inp, p, a):
    df = inp["zone_join"].copy()
    known = (df["fare_amount"] + df["extra"] + df["mta_tax"] + df["tip_amount"] + df["tolls_amount"]
             + df["improvement_surcharge"] + df["congestion_surcharge"] + df["airport_fee"]
             + df["cbd_congestion_fee"])
    df["other_charges"] = (df["total_amount"] - known).round(2)
    dist = df["trip_distance"] * (1.609 if p.get("distance_unit", "mi") == "km" else 1.0)
    df["trip_distance"] = dist
    df["speed_mph"] = dist / (df["duration_min"] / 60.0)
    df[LABEL] = (df["tip_amount"] / df["fare_amount"] >= 0.2).astype("int8")
    return df


encode = comp("encode", ["fare_features"], "model matrix with categorical codes from the encoder artifact")


def _k(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    return str(v)

def _keys(s: pd.Series) -> np.ndarray:
    """_k applied to a whole column (all categorical inputs of this pipeline are strings)."""
    return np.array([_k(x) for x in s.tolist()], dtype=object)



def fit_encoder(df: pd.DataFrame) -> dict:
    # codes in order of first appearance (as pandas.factorize assigns them)
    return {c: {k: i for i, k in enumerate(dict.fromkeys(_keys(df[c]).tolist()))} for c in CAT_FEATURES}


@encode.variant("v2015")
def _enc(inp, p, a):
    df = inp["fare_features"]
    out = df[ID_COLS + NUM_FEATURES].copy()
    maps = fit_encoder(df) if p.get("refit", False) else a["encoder"]
    for c in CAT_FEATURES:
        out[c] = pd.Series(_keys(df[c]), index=df.index).map(maps[c]).astype("float64")
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


def build_zone_artifact() -> None:
    z = pd.read_csv(RAW / "taxi_zone_lookup.csv")
    save_artifact("zones", "official", z[["LocationID", "Borough", "service_zone"]].to_dict(orient="list"))


PIPELINE = Pipeline(domain="tlc", components=C, output="encode", raw_tables=["trips"],
                    load_raw=load_raw, load_artifact=load_artifact)
V = Version.make


def reference_state(encoder_id: str) -> dict:
    return {
        "trip_adapter": V("v2015"),
        "time_features": V("v2015", env="p15"),    # 2015-era environment: pandas 1.5, nanosecond timestamps
        "clean": V("v2015"),
        "zone_join": V("v2015", artifacts={"zones": "official"}),
        "fare_features": V("v2015"),
        "encode": V("v2015", artifacts={"encoder": encoder_id}),
    }


def spec_state(period: str, encoder_id: str) -> dict:
    """Correct pipeline for a period: adapters keep feature semantics across the published-schema changes."""
    s = reference_state(encoder_id)
    params = {}
    if period >= "2018-07":
        params["ratecode_int"] = True
    if period >= "2023-04":
        params["case_insensitive"] = True
    if period >= "2025-01":
        params["cbd"] = True
    if params:
        s["trip_adapter"] = V("v2015", params)
    return s


def source_events(period_from: str, period_to: str) -> list[dict]:
    ev = []
    for when, what, cols in (
            ("2018-07", "passenger_count, RatecodeID published as double", ["passenger_count", "RatecodeID"]),
            ("2019-01", "congestion_surcharge populated", ["congestion_surcharge"]),
            ("2021-01", "airport_fee populated", ["airport_fee"]),
            ("2023-04", "airport_fee renamed Airport_fee; VendorID/LocationIDs int32",
             ["airport_fee", "Airport_fee", "VendorID", "PULocationID", "DOLocationID"]),
            ("2025-01", "cbd_congestion_fee column added", ["cbd_congestion_fee"]),
            ("2026-07", "request_source column added", ["request_source"])):
        if period_from < when <= period_to:
            ev.append({"period": when, "table": "trips", "change": what, "columns": cols})
    return ev


# ----------------------------------------------------------------------------------------------- lineage and catalog
LINEAGE = {
    "hour": ["trip_adapter", "time_features"], "dow": ["trip_adapter", "time_features"],
    "duration_min": ["trip_adapter", "time_features"], "trip_distance": ["trip_adapter", "fare_features"],
    "speed_mph": ["trip_adapter", "time_features", "fare_features"],
    "passenger_count": ["trip_adapter", "clean"], "fare_amount": ["trip_adapter"], "extra": ["trip_adapter"],
    "tolls_amount": ["trip_adapter"], "congestion_surcharge": ["trip_adapter"], "airport_fee": ["trip_adapter"],
    "other_charges": ["trip_adapter", "fare_features"],
    "vendor": ["trip_adapter", "encode"], "ratecode": ["trip_adapter", "encode"],
    "pu_borough": ["trip_adapter", "zone_join", "encode"], "do_borough": ["trip_adapter", "zone_join", "encode"],
    "pu_service": ["trip_adapter", "zone_join", "encode"], "do_service": ["trip_adapter", "zone_join", "encode"],
    "y": ["trip_adapter", "fare_features"],
}
SOURCE_READERS = {"trips": ["trip_adapter"]}
_FARE = ["fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge", "total_amount",
         "congestion_surcharge", "airport_fee", "Airport_fee", "cbd_congestion_fee"]
_ALL = ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count", "trip_distance",
        "RatecodeID", "PULocationID", "DOLocationID", "payment_type"] + _FARE
RAW_COLUMNS = {
    "trip_adapter": {"trips": _ALL},
    "time_features": {"trips": ["tpep_pickup_datetime", "tpep_dropoff_datetime"]},
    "clean": {"trips": ["payment_type", "fare_amount", "trip_distance", "passenger_count",
                        "tpep_pickup_datetime", "tpep_dropoff_datetime"]},
    "zone_join": {"trips": ["PULocationID", "DOLocationID"]},
    "fare_features": {"trips": _FARE + ["trip_distance"]},
    "encode": {"trips": _ALL},
}


def catalog(period: str, encoder_id: str) -> list[dict]:
    sp = spec_state(period, encoder_id)
    out = []

    def add(cid, comp_, version, kind, tier, silent_hint, note):
        out.append({"id": cid, "component": comp_, "version": version, "kind": kind, "tier": tier,
                    "silent_hint": silent_hint, "note": note})
    ta = sp["trip_adapter"].p()
    if period >= "2018-07":
        add("tlc.ratecode_stale", "trip_adapter", V("v2015", {k: v for k, v in ta.items() if k != "ratecode_int"}),
            "defect", "T1", True, "rate code key built from the published dtype ('1.0' after the double change)")
    if period >= "2023-04":
        add("tlc.airport_case_stale", "trip_adapter",
            V("v2015", {k: v for k, v in ta.items() if k != "case_insensitive"}), "defect", "T1", True,
            "case-sensitive lookup misses the renamed Airport_fee column (filled with 0)")
    if period >= "2025-01":
        add("tlc.cbd_stale", "trip_adapter", V("v2015", {k: v for k, v in ta.items() if k != "cbd"}), "defect",
            "T1", True, "new congestion fee not separated; lands in other_charges")
    if period < "2023-04":
        add("tlc.adapter_ci_early", "trip_adapter", V("v2015", {**ta, "case_insensitive": True}), "benign", "B",
            True, "case-insensitive column lookup deployed early")
    add("tlc.time_env_main", "time_features", V("v2015"), "defect", "T2", True,
        "pandas 2.2 / pyarrow 25 upgrade: parquet timestamps keep microsecond unit; epoch math via astype(int64)")
    add("tlc.time_env_p30", "time_features", V("v2015", env="p30"), "defect", "T2", True,
        "pandas 3 upgrade: epoch via astype(int64) changes resolution")
    add("tlc.time_modernised", "time_features", V("v2024_td"), "benign", "B", True,
        "timedelta arithmetic in the upgraded environment")
    add("tlc.time_modernised_p30", "time_features", V("v2024_td", env="p30"), "benign", "T2b", True,
        "pandas 3 upgrade of the modernised time step")
    add("tlc.clean_env_p30", "clean", V("v2015", env="p30"), "defect", "T2", True,
        "pandas 3 upgrade: chained-assignment cleaning becomes a no-op")
    add("tlc.clean_modernised", "clean", V("v2024_loc"), "benign", "B", True, "chained assignment -> .loc")
    add("tlc.zone_env_p30", "zone_join", V("v2015", artifacts={"zones": "official"}, env="p30"), "benign", "T2b",
        True, "pandas 3 upgrade of the zone join")
    add("tlc.tz_utc", "time_features", V("v2015", {"tz": "utc"}, env="p15"), "defect", "T3", True,
        "timestamps shifted as if converted from UTC")
    add("tlc.do_key_bug", "zone_join", V("v2015", {"do_key": "PULocationID"}, {"zones": "official"}), "defect",
        "T3", True, "dropoff zone joined on the pickup id (copy-paste key bug)")
    add("tlc.distance_km", "fare_features", V("v2015", {"distance_unit": "km"}), "defect", "T3", True,
        "trip distance converted to km (unit change)")
    add("tlc.encoder_refit", "encode", V("v2015", {"refit": True}, sp["encode"].a()), "defect", "T3", True,
        "categorical codes re-fitted at serving time (training/serving skew)")
    return out

COLUMN_SOURCES = {
    "trip_adapter": {"vendor": ["VendorID"], "pickup": ["tpep_pickup_datetime"],
                     "dropoff": ["tpep_dropoff_datetime"], "ratecode": ["RatecodeID"],
                     "airport_fee": ["airport_fee", "Airport_fee"], "trip_id": ["_row"]},
    "time_features": {"hour": ["pickup"], "dow": ["pickup"], "duration_min": ["pickup", "dropoff"]},
    "zone_join": {"pu_borough": ["PULocationID"], "pu_service": ["PULocationID"],
                  "do_borough": ["DOLocationID"], "do_service": ["DOLocationID"]},
    "fare_features": {"other_charges": ["total_amount", "fare_amount", "extra", "mta_tax", "tip_amount",
                                         "tolls_amount", "improvement_surcharge", "congestion_surcharge",
                                         "airport_fee", "cbd_congestion_fee"],
                      "speed_mph": ["trip_distance", "duration_min"], "y": ["tip_amount", "fare_amount"]},
}
