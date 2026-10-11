"""Offline evaluation of migration-testing procedures (docs/03_study_design.md), from logged k-run data.

Ground truth per pair: G = reliable negative flips (old k/k pass, new 0/k pass) among the pool P of tasks the old
model passes k/k (in practice: known from the old model's history). Each procedure spends NEW-model episodes
(and, for FlipScout, cheap first-action probes) and outputs a set of flagged tasks.

Procedures
  oneshot        run every P task once (repeat r1); flag every failure.            [the usual regression suite]
  rerun_fail     oneshot, then re-run each failure up to k-1 more times; flag tasks failing all k runs.
  oracle_k       run every P task with early stopping (stop at the first pass, max k runs); flag 0/k tasks.
  flipscout      probe every P task with ONE model call (the new model's first action, = turn 1 of r1); tasks whose
                 first action differs from the old model's modal first action are confirmed like oracle_k; a uniform
                 random audit of m non-divergent tasks is confirmed too; flag 0/k tasks. Escape bound: one-sided
                 Clopper-Pearson upper limit (95%) on the flip rate among non-divergent, non-audited tasks.
  random_budget  confirm tasks in random order (oracle_k rule) until the same episode budget as flipscout is spent.
Costs are in episode units (1 = a full new-model episode); a probe costs 1/rounds of its episode (measured).
Usage: flipscout.py --testbed clinical --subset pilot --pairs a>b,... [--k 3] [--audit 10] [--seeds 200]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse, parse_qsl

import numpy as np
from scipy.stats import beta

ROOT = Path(os.environ.get("MM_PROJECT_ROOT", Path(__file__).resolve().parent.parent))


def first_action(rec, testbed):
    steps = rec.get("steps") or []
    if testbed == "clinical":
        if not steps:
            return "none"
        s = steps[0]
        k = s.get("kind")
        if k == "GET":
            u = urlparse(s.get("url", ""))
            path = u.path.rstrip("/").split("/")[-1]
            q = {a: b for a, b in parse_qsl(u.query) if a not in ("_format", "_count", "_sort")}
            keep = {a: q[a] for a in sorted(q) if a in ("code", "patient", "category", "name", "birthdate", "family",
                                                      "given", "identifier")}
            return f"GET {path} {json.dumps(keep, sort_keys=True)}"
        if k == "POST":
            p = s.get("payload") or {}
            return f"POST {s.get('resource')} " + json.dumps({x: str(p.get(x))[:40] for x in ("resourceType", "status", "intent")}, sort_keys=True)
        return str(k)
    # generic fallback for other testbeds: first recorded action/tool call
    acts = rec.get("actions") or rec.get("trace") or steps
    if not acts:
        return "none"
    a = acts[0]
    if isinstance(a, dict):
        name = a.get("tool") or a.get("name") or a.get("action") or a.get("kind")
        args = a.get("args") or a.get("arguments") or {k: v for k, v in a.items() if k in ("component", "components")}
        return f"{name} {json.dumps(args, sort_keys=True, default=str)[:120]}"
    return str(a)[:120]


def load(testbed, subset, label, T, k):
    d = ROOT / "runs" / testbed / subset / label / f"T{T:g}"
    out = {}
    for r in range(1, k + 1):
        for f in (d / f"r{r}").glob("*.json"):
            rec = json.loads(f.read_text(encoding="utf-8"))
            out.setdefault(rec["task_id"], {})[r] = rec
    return {t: v for t, v in out.items() if len(v) == k}


def cp_upper(x, n, conf=0.95):
    return 1.0 if n == 0 else (1.0 if x == n else float(beta.ppf(conf, x + 1, n - x)))


def evaluate(old, new, testbed, k, audit, seeds):
    tasks = sorted(set(old) & set(new))
    P = [t for t in tasks if all(old[t][r]["success"] for r in range(1, k + 1))]
    G = {t for t in P if not any(new[t][r]["success"] for r in range(1, k + 1))}
    seq = {t: [new[t][r]["success"] for r in range(1, k + 1)] for t in P}

    def confirm_cost(t):                      # early stop at first pass
        c = 0
        for s in seq[t]:
            c += 1
            if s:
                return c, False
        return c, True

    res = {"n_pool": len(P), "n_flips": len(G)}
    # oneshot
    fails = [t for t in P if not seq[t][0]]
    res["oneshot"] = {"cost": len(P), "flagged": len(fails), "true": len(set(fails) & G), "false_alarms": len(set(fails) - G)}
    # rerun failures
    c = len(P) + sum(min(k - 1, next((i for i, s in enumerate(seq[t][1:], 1) if s), k - 1)) for t in fails)
    res["rerun_fail"] = {"cost": c, "flagged": len(G), "true": len(G), "false_alarms": 0}
    # oracle with early stop
    res["oracle_k"] = {"cost": sum(confirm_cost(t)[0] for t in P), "flagged": len(G), "true": len(G), "false_alarms": 0}
    # flipscout
    modal_old = {t: Counter(first_action(old[t][r], testbed) for r in range(1, k + 1)).most_common(1)[0][0] for t in P}
    probe = {t: first_action(new[t][1], testbed) for t in P}
    probe_cost = sum(1.0 / max(1, new[t][1].get("n_rounds") or 1) for t in P)
    div = [t for t in P if probe[t] != modal_old[t]]
    rest = [t for t in P if t not in div]
    rng = np.random.default_rng(0)
    fs = []
    rb = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        aud = list(rng.choice(rest, size=min(audit, len(rest)), replace=False)) if rest else []
        checked = div + aud
        cost = probe_cost + sum(confirm_cost(t)[0] for t in checked)
        found = {t for t in checked if t in G}
        unaud = [t for t in rest if t not in aud]
        x = sum(1 for t in aud if t in G)
        ub = cp_upper(x, len(aud))
        escapes = len(G - found)
        fs.append((cost, len(found), escapes, ub, ub * len(unaud), escapes <= ub * len(unaud) + 1e-9))
        # random order with the same budget
        order = list(rng.permutation(P))
        spent, found_r = 0.0, set()
        for t in order:
            cc, flip = confirm_cost(t)
            if spent + cc > cost + 1e-9:
                break
            spent += cc
            if flip:
                found_r.add(t)
        rb.append((spent, len(found_r)))
    fs = np.array(fs, dtype=float)
    res["flipscout"] = {"cost": round(fs[:, 0].mean(), 2), "probe_cost": round(probe_cost, 2), "n_divergent": len(div),
                        "flips_in_divergent": len(set(div) & G), "true": round(fs[:, 1].mean(), 2),
                        "escapes": round(fs[:, 2].mean(), 2), "false_alarms": 0,
                        "escape_rate_ub95": round(fs[:, 3].mean(), 4), "escape_count_ub95": round(fs[:, 4].mean(), 2),
                        "bound_holds_frac": round(fs[:, 5].mean(), 3)}
    rb = np.array(rb, dtype=float)
    res["random_budget"] = {"cost": round(rb[:, 0].mean(), 2), "true": round(rb[:, 1].mean(), 2), "false_alarms": 0}
    res["recall_divergence_signal"] = round(len(set(div) & G) / len(G), 3) if G else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", default="clinical")
    ap.add_argument("--subset", default="pilot")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--T", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--audit", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=200)
    a = ap.parse_args()
    out = {}
    for p in a.pairs.split(","):
        o, n = p.split(">")
        old, new = load(a.testbed, a.subset, o, a.T, a.k), load(a.testbed, a.subset, n, a.T, a.k)
        if not old or not new:
            print("skip", p, "(missing runs)")
            continue
        out[p] = evaluate(old, new, a.testbed, a.k, a.audit, a.seeds)
        print(p, json.dumps(out[p]))
    f = ROOT / "analysis" / "out" / f"flipscout_{a.testbed}_{a.subset}_T{a.T:g}_k{a.k}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
