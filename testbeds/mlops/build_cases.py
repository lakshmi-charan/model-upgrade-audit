"""Freeze the mlops testbed's case subsets and extract their records from the source project (read-only).

Source records (produced by the source project's run_cases.py; never modified here):
  <source>/runs/final/{tlc,news}.jsonl   v1 domains, split == "test" only (ACS is dev-only and excluded)
  <source>/runs/v2main/{bts,hmda,divvy}.jsonl   v2 held-out domains (every case is a test case)
and the source's gpt-5.6-luna llm_tools results (<domain>.llm_luna.jsonl) as a reference column.

Writes (target only):
  testbeds/mlops/subsets/{smoke,pilot,main}.json   frozen ids (seed 20261006); refuses to overwrite
  testbeds/mlops/data/cases.jsonl                  compact case records for the union of the subsets

Strata: domain x oracle type, where oracle type is
  none        oracle fix = [] and retrain = false
  drift_only  oracle fix = [] and retrain = true
  defect_only oracle fix != [] and retrain = false
  both        oracle fix != [] and retrain = true
pilot: 8 per domain (2 per oracle type), main: 20 per domain (5 per oracle type), disjoint from pilot; a stratum
with too few cases passes its shortfall to the domain's other strata (both, defect_only, drift_only, none order).
smoke: 4 pilot cases, one per oracle type, from four different domains.

usage: python testbeds/mlops/build_cases.py [--source PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

TB = Path(__file__).resolve().parent
SEED = 20261006
OTYPES = ("none", "drift_only", "defect_only", "both")
SPILL = ("both", "defect_only", "drift_only", "none")
SOURCES = [("final", "tlc"), ("final", "news"), ("v2main", "bts"), ("v2main", "hmda"), ("v2main", "divvy")]
PLAN = {"pilot": 2, "main": 5}           # cases per (domain, oracle type)


def source_root(arg: str | None = None) -> Path:
    if arg:
        return Path(arg).resolve()
    env = os.environ.get("MM_MLOPS_SOURCE")
    return Path(env).resolve() if env else (TB.parents[1].parent / "drift-or-defect").resolve()


def otype(o: dict) -> str:
    return {(False, False): "none", (False, True): "drift_only", (True, False): "defect_only",
            (True, True): "both"}[(bool(o["fix"]), bool(o["retrain"]))]


def compact(r: dict, run: str) -> dict:
    g = r["gt"]
    return {"task_id": r["case"]["case_id"], "domain": r["case"]["domain"], "source_run": run, "split": r["split"],
            "scenario": r["case"]["meta"].get("scenario"), "tier": r["case"]["tier"], "otype": otype(r["oracle"]),
            "oracle": {"fix": sorted(r["oracle"]["fix"]), "retrain": bool(r["oracle"]["retrain"]),
                       "effective": sorted(g["defective"]), "consequential": sorted(g["consequential"])},
            "gt": {k: g.get(k) for k in ("defective", "consequential", "crashes", "retrain_gain_after_fix",
                                         "ref_perf", "drop_vs_ref", "best_score")},
            "case": r["case"], "obs": r["obs"]}


def take(rng, pools: dict, dom: str, per: int, exclude: set) -> list[str]:
    avail = {t: [c for c in pools.get((dom, t), []) if c not in exclude] for t in OTYPES}
    quota = {t: min(per, len(avail[t])) for t in OTYPES}
    short = per * len(OTYPES) - sum(quota.values())
    while short > 0:
        moved = False
        for t in SPILL:
            if short and quota[t] < len(avail[t]):
                quota[t] += 1
                short -= 1
                moved = True
        if not moved:
            break
    out = []
    for t in OTYPES:
        out += rng.sample(avail[t], quota[t])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    a = ap.parse_args()
    src = source_root(a.source)
    sub_dir = TB / "subsets"
    if any((sub_dir / f"{n}.json").exists() for n in ("smoke", "pilot", "main")):
        raise SystemExit("subsets already frozen; refusing to overwrite")
    recs, pools, luna = {}, {}, {}
    for run, dom in SOURCES:
        with (src / "runs" / run / f"{dom}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["split"] != "test":
                    continue
                c = compact(r, run)
                recs[c["task_id"]] = c
                pools.setdefault((dom, c["otype"]), []).append(c["task_id"])
        lp = src / "runs" / run / f"{dom}.llm_luna.jsonl"
        if lp.exists():
            for line in lp.open(encoding="utf-8"):
                x = json.loads(line)
                m = x["methods"].get("llm_tools")
                if m:
                    n = m.get("notes") or {}
                    luna[x["case_id"]] = {"fix": m.get("fix"), "retrain": m.get("retrain"), "steps": n.get("steps"),
                                          "llm_usd": n.get("llm_usd"), "error": m.get("error")}
    for k in pools:
        pools[k].sort()
    rng = random.Random(SEED)
    doms = [d for _, d in SOURCES]
    pilot = [cid for d in doms for cid in take(rng, pools, d, PLAN["pilot"], set())]
    main_ids = [cid for d in doms for cid in take(rng, pools, d, PLAN["main"], set(pilot))]
    used, smoke = set(), []
    for t in OTYPES:
        cands = sorted(c for c in pilot if recs[c]["otype"] == t and recs[c]["domain"] not in used)
        pick = rng.choice(cands)
        smoke.append(pick)
        used.add(recs[pick]["domain"])
    assert not set(pilot) & set(main_ids) and set(smoke) <= set(pilot)
    sub_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in (("smoke", smoke), ("pilot", pilot), ("main", main_ids)):
        mix = {}
        for c in ids:
            r = recs[c]
            for key in ("domain", "otype", "scenario"):
                mix.setdefault(key, {}).setdefault(r[key], 0)
                mix[key][r[key]] += 1
        (sub_dir / f"{name}.json").write_text(json.dumps(
            {"name": name, "seed": SEED, "per_domain_otype": PLAN.get(name, "1 per oracle type"),
             "source_runs": [f"{r}/{d}" for r, d in SOURCES], "mix": mix, "ids": ids}, indent=1), encoding="utf-8")
        print(name, len(ids), json.dumps(mix))
    out = TB / "data" / "cases.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in sorted(set(pilot) | set(main_ids)):
            r = dict(recs[c])
            r["source_luna_tools"] = luna.get(c)
            f.write(json.dumps(r, default=str) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
