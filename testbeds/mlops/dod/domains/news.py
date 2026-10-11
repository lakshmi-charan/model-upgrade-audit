"""News-category pipeline (text): HuffPost News Category Dataset (headline + short description), 2012-2022.

Task: classify an article into one of the harmonised sections below from its headline and description.

Real, dated source change: the publisher's section taxonomy changed (verified in runs/data_probe and the
category-by-year table): e.g. WELLNESS -> HEALTHY LIVING, PARENTING -> PARENTS, STYLE & BEAUTY -> STYLE,
FOOD & DRINK -> TASTE, ENVIRONMENT -> GREEN, CULTURE & ARTS -> ARTS -> ARTS & CULTURE, WORLDPOST ->
THE WORLDPOST -> WORLD NEWS, new U.S. NEWS. The label pipeline keeps a harmonisation table (reference
artifact) that must be refreshed when new section names appear.
"""
from __future__ import annotations

import html
import json
import os
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from dod.core import Component, Pipeline, Version

ROOT = Path(os.environ.get("DOD_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
BRONZE = ROOT / "data" / "bronze" / "news"
ART = ROOT / "data" / "artifacts" / "news"
PERIODS = [str(y) for y in range(2012, 2023)]
LABEL = "y"
ID_COLS = ["link"]
TASK = "multiclass"
FEATURES = ["tok"]
CAT_FEATURES: list[str] = []

# Harmonised sections (the label space). Raw names map to these through the harmonisation artifact.
SECTIONS = ["POLITICS", "WELLNESS", "ENTERTAINMENT", "TRAVEL", "STYLE", "PARENTING", "FOOD", "WORLD", "BUSINESS",
            "COMEDY", "SPORTS", "HOME", "VOICES", "ARTS", "ENVIRONMENT", "CRIME", "MEDIA", "TECH", "SCIENCE",
            "RELIGION", "WEDDINGS", "DIVORCE", "IMPACT", "WOMEN", "EDUCATION", "WEIRD NEWS", "US NEWS"]
# Full harmonisation (the correct table once all names have appeared).
HARMONISE_FULL = {
    "POLITICS": "POLITICS", "WELLNESS": "WELLNESS", "HEALTHY LIVING": "WELLNESS", "ENTERTAINMENT": "ENTERTAINMENT",
    "TRAVEL": "TRAVEL", "STYLE & BEAUTY": "STYLE", "STYLE": "STYLE", "PARENTING": "PARENTING", "PARENTS": "PARENTING",
    "FOOD & DRINK": "FOOD", "TASTE": "FOOD", "WORLDPOST": "WORLD", "THE WORLDPOST": "WORLD", "WORLD NEWS": "WORLD",
    "BUSINESS": "BUSINESS", "MONEY": "BUSINESS", "COMEDY": "COMEDY", "SPORTS": "SPORTS", "HOME & LIVING": "HOME",
    "BLACK VOICES": "VOICES", "QUEER VOICES": "VOICES", "LATINO VOICES": "VOICES", "ARTS": "ARTS",
    "ARTS & CULTURE": "ARTS", "CULTURE & ARTS": "ARTS", "ENVIRONMENT": "ENVIRONMENT", "GREEN": "ENVIRONMENT",
    "CRIME": "CRIME", "MEDIA": "MEDIA", "TECH": "TECH", "SCIENCE": "SCIENCE", "RELIGION": "RELIGION",
    "WEDDINGS": "WEDDINGS", "DIVORCE": "DIVORCE", "IMPACT": "IMPACT", "GOOD NEWS": "IMPACT", "WOMEN": "WOMEN",
    "EDUCATION": "EDUCATION", "COLLEGE": "EDUCATION", "WEIRD NEWS": "WEIRD NEWS", "U.S. NEWS": "US NEWS",
    "FIFTY": "WELLNESS",
}

C = {}


def comp(name, inputs, doc=""):
    C[name] = Component(name, inputs, doc=doc)
    return C[name]


ingest = comp("ingest", ["raw:articles"], "published article records")


@ingest.variant("v2012")
def _ing(inp, p, a):
    r = inp["raw:articles"]
    return pd.DataFrame({"link": r["link"].astype(str), "headline": r["headline"],
                         "short_description": r["short_description"], "category": r["category"],
                         "date": r["date"].astype(str)})


label_map = comp("label_map", ["ingest"], "raw section name -> harmonised section via the harmonisation table")


@label_map.variant("v2012")
def _lm(inp, p, a):
    df = inp["ingest"].copy()
    table = a["harmonise"]
    df["section"] = df["category"].map(table)
    return df[df["section"].notna()].reset_index(drop=True)


text_norm = comp("text_norm", ["label_map"], "text = headline + description, normalised")


@text_norm.variant("v2012")
def _tn(inp, p, a):
    df = inp["label_map"].copy()
    # original idiom: missing descriptions become strings before concatenation
    text = df["headline"].astype(str) + " " + df["short_description"].astype(str)
    if p.get("field", "both") == "headline":     # incident class: description dropped by a field-mapping bug
        text = df["headline"].astype(str)
    if p.get("mojibake", False):                 # incident class: UTF-8 bytes decoded as Latin-1 upstream
        text = text.map(lambda s: s.encode("utf-8", "replace").decode("latin-1", "replace"))
    text = text.map(html.unescape).map(lambda s: unicodedata.normalize("NFKC", s))
    if p.get("lower", True):
        text = text.str.lower()
    df["text"] = text
    return df[["link", "text", "section"]]


vectorize = comp("vectorize", ["text_norm"], "token ids from the fitted vocabulary artifact")
TOKEN = re.compile(r"(?u)\b\w\w+\b")


def tokens(s: str) -> list[str]:
    return TOKEN.findall(s if isinstance(s, str) else "")


def fit_vocab(texts: pd.Series, max_features: int = 20000) -> dict:
    df_count = Counter()
    for t in texts:
        df_count.update(set(tokens(t)))
    vocab = [w for w, c in df_count.most_common() if c >= 3][:max_features]
    n = len(texts)
    idf = [float(np.log((1 + n) / (1 + df_count[w])) + 1) for w in vocab]
    return {"vocab": vocab, "idf": idf}


@vectorize.variant("v2012")
def _vec(inp, p, a):
    df = inp["text_norm"]
    art = fit_vocab(df["text"]) if p.get("refit", False) else a["vocab"]
    index = {w: i for i, w in enumerate(art["vocab"])}
    toks = []
    for t in df["text"]:
        c = Counter(index[w] for w in tokens(t) if w in index)
        toks.append(" ".join(f"{k}:{v}" for k, v in sorted(c.items())))
    out = pd.DataFrame({"link": df["link"].values, "tok": toks})
    labels = a["labels"]
    out[LABEL] = df["section"].map(labels).fillna(-1).astype("int64").values
    return out


# ----------------------------------------------------------------------------------------------- design matrix
_IDF_CACHE: dict = {}


def design_matrix(df: pd.DataFrame, n_features: int = 20000) -> sp.csr_matrix:
    rows, cols, vals = [], [], []
    for i, t in enumerate(df["tok"].tolist()):
        if not t:
            continue
        for pair in t.split(" "):
            k, v = pair.split(":")
            rows.append(i)
            cols.append(int(k))
            vals.append(1.0 + np.log(float(v)))
    m = sp.csr_matrix((vals, (rows, cols)), shape=(len(df), n_features))
    norms = np.sqrt(m.multiply(m).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms) @ m


# ----------------------------------------------------------------------------------------------- data and artifacts
def load_raw(period: str, table: str) -> pd.DataFrame:
    return pd.read_parquet(BRONZE / f"{period}.parquet")


@lru_cache(maxsize=64)
def _load_art(name: str, version: str) -> str:
    return (ART / name / f"{version}.json").read_text(encoding="utf-8")


def load_artifact(name: str, version: str):
    return json.loads(_load_art(name, version))


def save_artifact(name: str, version: str, obj) -> None:
    (ART / name).mkdir(parents=True, exist_ok=True)
    (ART / name / f"{version}.json").write_text(json.dumps(obj, sort_keys=True), encoding="utf-8")
    _load_art.cache_clear()


def build_harmonise_artifacts() -> dict:
    """Harmonisation tables as they would have been maintained: 'as_of_<year>' contains every raw name that had
    appeared by the end of that year (the correct table for that year); 'full' contains all names."""
    seen: set[str] = set()
    out = {}
    for y in PERIODS:
        seen |= set(pd.read_parquet(BRONZE / f"{y}.parquet", columns=["category"])["category"].unique())
        table = {k: v for k, v in HARMONISE_FULL.items() if k in seen}
        save_artifact("harmonise", f"as_of_{y}", table)
        out[y] = len(table)
    save_artifact("harmonise", "full", HARMONISE_FULL)
    save_artifact("labels", "v1", {s: i for i, s in enumerate(SECTIONS)})
    return out


def fit_encoder(df: pd.DataFrame) -> dict:      # interface parity with the tabular domains
    return {}


def fit_text_artifacts(enc_id: str, train_pre: pd.DataFrame) -> None:
    """Called by build_reference with the pre-vectorisation train rows: fits the vocabulary artifact."""
    save_artifact("vocab", enc_id, fit_vocab(train_pre["text"]))


PIPELINE = Pipeline(domain="news", components=C, output="vectorize", raw_tables=["articles"],
                    load_raw=load_raw, load_artifact=load_artifact)
V = Version.make


def reference_state(encoder_id: str) -> dict:
    y0 = encoder_id.split("_")[-1]
    return {
        "ingest": V("v2012"),
        "label_map": V("v2012", artifacts={"harmonise": f"as_of_{y0}"}),
        "text_norm": V("v2012"),
        "vectorize": V("v2012", artifacts={"vocab": encoder_id, "labels": "v1"}),
    }


def spec_state(period: str, encoder_id: str) -> dict:
    s = reference_state(encoder_id)
    s["label_map"] = V("v2012", artifacts={"harmonise": f"as_of_{period}"})
    return s


def source_events(period_from: str, period_to: str) -> list[dict]:
    """New raw section names appearing after period_from (a profile diff of the raw category column)."""
    ev = []
    seen = set(pd.read_parquet(BRONZE / f"{period_from}.parquet", columns=["category"])["category"].unique())
    for y in PERIODS:
        if period_from < y <= period_to:
            cur = set(pd.read_parquet(BRONZE / f"{y}.parquet", columns=["category"])["category"].unique())
            new = sorted(cur - seen)
            if new:
                ev.append({"period": y, "table": "articles", "change": f"new category values: {new}",
                           "columns": ["category"]})
            seen |= cur
    return ev


# ----------------------------------------------------------------------------------------------- lineage and catalog
LINEAGE = {"tok": ["ingest", "text_norm", "vectorize"], "y": ["ingest", "label_map", "vectorize"]}
SOURCE_READERS = {"articles": ["ingest"]}
RAW_COLUMNS = {
    "ingest": {"articles": ["link", "headline", "short_description", "category", "date"]},
    "label_map": {"articles": ["category"]},
    "text_norm": {"articles": ["headline", "short_description"]},
    "vectorize": {"articles": ["headline", "short_description", "category"]},
}


def catalog(period: str, encoder_id: str) -> list[dict]:
    sp = spec_state(period, encoder_id)
    ref = reference_state(encoder_id)
    out = []

    def add(cid, comp_, version, kind, tier, silent_hint, note):
        out.append({"id": cid, "component": comp_, "version": version, "kind": kind, "tier": tier,
                    "silent_hint": silent_hint, "note": note})
    if sp["label_map"] != ref["label_map"]:
        add("news.harmonise_stale", "label_map", ref["label_map"], "defect", "T1", True,
            "section harmonisation table not refreshed for new section names")
    tn = sp["text_norm"]
    add("news.text_env_p30", "text_norm", V("v2012", env="p30"), "benign", "T2b", True,
        "pandas 3 upgrade of text normalisation (astype(str) keeps missing values; none occur in this source)")
    add("news.ingest_env_p30", "ingest", V("v2012", env="p30"), "benign", "T2b", True,
        "pandas 3 upgrade of ingestion")
    add("news.no_lower", "text_norm", V("v2012", {"lower": False}), "defect", "T3", True,
        "lower-casing step dropped")
    add("news.headline_only", "text_norm", V("v2012", {"field": "headline"}), "defect", "T3", True,
        "description dropped by a field-mapping bug")
    add("news.mojibake", "text_norm", V("v2012", {"mojibake": True}), "defect", "T3", True,
        "UTF-8 text decoded as Latin-1 upstream")
    add("news.vocab_refit", "vectorize", V("v2012", {"refit": True}, sp["vectorize"].a()), "defect", "T3", True,
        "vocabulary re-fitted at serving time (training/serving skew)")
    add("news.label_map_full_early", "label_map", V("v2012", artifacts={"harmonise": "full"}), "benign", "B",
        True, "complete harmonisation table deployed early")
    return out

COLUMN_SOURCES = {"label_map": {"section": ["category"]},
                  "text_norm": {"text": ["headline", "short_description"]},
                  "vectorize": {"tok": ["text"], "y": ["section"]}}
