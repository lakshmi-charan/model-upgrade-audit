"""Flip analysis for one testbed (pre-registered in docs/02_pilot_plan.md).

For each upgrade pair (old -> new, consecutive members of a family in config/models.yaml) on the tasks both
models completed k times at temperature T:
  * mean success old / new (paired by task) and single-run flip counts;
  * reliable negative flips  RNF = #tasks with old k/k pass and new 0/k pass;
  * reliable positive flips  RPF = #tasks with old 0/k and new k/k;
  * a within-task label-permutation test of RNF (H0: the 2k runs of a task are exchangeable between models,
    i.e. any difference is run-to-run noise). One-sided p = P_perm(RNF >= observed). 10,000 permutations, seed 0.
    Excess RNF = observed - mean RNF under the permutation null;
  * failure taxonomy of the NEW model on RNF tasks (modal category over its k runs):
      interface  - no FINISH (invalid action format, round limit, harness error)
      unrequested_write - a POST of a resource type the reference does not order at all
      wrong_write - a POST that matches no required order but the reference orders that resource type
                    (wrong dose/code/patient, duplicate)
      missing_write - FINISH with a required order missing
      wrong_answer - all POSTs correct but the final answer is wrong
Usage: flips.py --testbed clinical --subset pilot --T 0 --k 3 [--labels a,b,...]
"""
from __future__ import annotations

import argparse
import json
import re
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(os.environ.get("MM_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
try:                                   # clinical answer checker, used only to detect answer-format flips
    import sys as _sys
    os.environ.setdefault("CAVE_CONFIG", str(ROOT / "testbeds" / "clinical" / "config.yaml"))
    _sys.path.insert(0, str(ROOT / "testbeds" / "clinical"))
    from cave.reference import answer_ok as _ANSWER_OK
except Exception:
    _ANSWER_OK = None


def load_runs(testbed, subset, T, k, labels):
    base = ROOT / "runs" / testbed / subset
    out = {}
    for lab in labels:
        d = base / lab / f"T{T:g}"
        per = defaultdict(dict)
        for r in range(1, k + 1):
            for f in (d / f"r{r}").glob("*.json"):
                rec = json.loads(f.read_text(encoding="utf-8"))
                per[rec["task_id"]][r] = rec
        out[lab] = {t: v for t, v in per.items() if len(v) == k}
    return out


def category(rec, ref_resources):
    if "category" in rec:          # testbeds that grade categories themselves (ops, mlops)
        return rec["category"]
    if rec.get("status") != "completed" or rec.get("error"):
        return "interface"
    posts = [u for u in rec["units"] if u["type"] == "POST"]
    bad = [u for u in posts if not u.get("correct")]
    if bad:
        if any(u.get("resource") not in ref_resources for u in bad):
            return "unrequested_write"
        return "wrong_write"
    fin = [u for u in rec["units"] if u["type"] == "FINISH"]
    why = fin[-1].get("why") if fin else ""
    if why == "required order missing":
        return "missing_write"
    if why == "wrong answer":
        # answer-format flip: the value is right but wrapped/shaped differently (e.g. [[-1]] instead of [-1])
        try:
            raw = fin[-1].get("answer_raw") or ""
            v = json.loads(raw if raw.strip().startswith("[") else "[" + raw + "]")
            flat = []
            def _fl(x):
                if isinstance(x, list):
                    for y in x:
                        _fl(y)
                else:
                    flat.append(x)
            _fl(v)
            cands = [json.dumps(flat)]
            if flat and isinstance(flat[0], str):    # value embedded in text, e.g. "[-1]" or "Last value: 7.2 ..."
                nums = re.findall(r"-?\d+(?:\.\d+)?", flat[0])
                cands += [json.dumps([float(n)] + flat[1:]) for n in nums[:3]]
            if _ANSWER_OK and any(_ANSWER_OK(rec.get("_ref_answer"), c) for c in cands):
                return "answer_format"
        except Exception:
            pass
        return "wrong_answer"
    return "ok" if rec.get("success") else "other"


def perm_test(old: np.ndarray, new: np.ndarray, n_perm=10000, seed=0):
    """old,new: (n_tasks, k) 0/1 arrays. Returns observed RNF, null mean, one-sided p."""
    k = old.shape[1]
    obs = int(np.sum((old.sum(1) == k) & (new.sum(1) == 0)))
    both = np.concatenate([old, new], axis=1)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        idx = np.argsort(rng.random(both.shape), axis=1)
        p = np.take_along_axis(both, idx, axis=1)
        null[b] = np.sum((p[:, :k].sum(1) == k) & (p[:, k:].sum(1) == 0))
    pval = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return obs, float(null.mean()), float(pval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", default="clinical")
    ap.add_argument("--subset", default="pilot")
    ap.add_argument("--T", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--labels", default="")
    ap.add_argument("--pairs", default="", help="explicit old>new pairs, comma-separated (overrides families)")
    a = ap.parse_args()
    models = {m["label"]: m for m in yaml.safe_load((ROOT / "config" / "models.yaml").read_text())["models"]}
    labels = [x for x in a.labels.split(",") if x] or list(models)
    runs = load_runs(a.testbed, a.subset, a.T, a.k, labels)
    fams = defaultdict(list)
    for lab in labels:
        if runs.get(lab):
            fams[models[lab]["family"]].append(lab)
    tasks_meta = {}
    ref_answers = {}
    if a.testbed == "clinical":
        tb = ROOT / "testbeds" / "clinical" / "data"
        for f in ("base_tasks.json", "stress_tasks.json"):
            for t in json.loads((tb / f).read_text(encoding="utf-8")):
                tasks_meta[t["id"]] = {r["resource"] for r in t["ref"]["orders"] + t["ref"]["optional_orders"]}
        live = ROOT / "testbeds" / "clinical" / "data" / "refs_live.json"
        if live.exists():
            for tid, ref in json.loads(live.read_text(encoding="utf-8")).items():
                tasks_meta[tid] = {r["resource"] for r in ref["orders"] + ref["optional_orders"]}
                ref_answers[tid] = ref["answer"]
    results = []
    if a.pairs:
        pair_list = [tuple(p.split(">")) for p in a.pairs.split(",") if p]
    else:
        pair_list = []
        for fam, labs in fams.items():
            labs.sort(key=lambda l: models[l]["order"])
            pair_list += list(zip(labs, labs[1:]))
    for old, new in pair_list:
            fam = models[old]["family"]
            if not runs.get(old) or not runs.get(new):
                continue
            common = sorted(set(runs[old]) & set(runs[new]))
            if not common:
                continue
            O = np.array([[runs[old][t][r]["success"] for r in range(1, a.k + 1)] for t in common])
            N = np.array([[runs[new][t][r]["success"] for r in range(1, a.k + 1)] for t in common])
            obs, null_mean, p = perm_test(O, N)
            rpf = int(np.sum((O.sum(1) == 0) & (N.sum(1) == a.k)))
            rel_old = int(np.sum(O.sum(1) == a.k))
            rnf_tasks = [t for t, o, n in zip(common, O, N) if o.sum() == a.k and n.sum() == 0]
            tax = Counter()
            for t in rnf_tasks:
                for r in range(1, a.k + 1):
                    runs[new][t][r]["_ref_answer"] = ref_answers.get(t)
                cats = Counter(category(runs[new][t][r], tasks_meta.get(t, set())) for r in range(1, a.k + 1))
                tax[cats.most_common(1)[0][0]] += 1
            single_neg = int(np.sum((O[:, 0] == 1) & (N[:, 0] == 0)))
            noise_old = int(np.sum((O[:, 0] == 1) & (O[:, 1] == 0))) if a.k > 1 else None
            noise_new = int(np.sum((N[:, 0] == 1) & (N[:, 1] == 0))) if a.k > 1 else None
            results.append({
                "family": fam, "old": old, "new": new, "n_tasks": len(common),
                "mean_old": round(float(O.mean()), 4), "mean_new": round(float(N.mean()), 4),
                "reliable_old_pass": rel_old, "RNF": obs, "RNF_null_mean": round(null_mean, 3),
                "excess_RNF": round(obs - null_mean, 3), "p_perm": round(p, 5), "RPF": rpf,
                "RNF_rate_of_reliable": round(obs / rel_old, 4) if rel_old else None,
                "single_run_neg_flips": single_neg, "same_model_neg_r1r2_old": noise_old,
                "same_model_neg_r1r2_new": noise_new, "taxonomy": dict(tax), "RNF_tasks": rnf_tasks})
    out = ROOT / "analysis" / "out" / f"flips_{a.testbed}_{a.subset}_T{a.T:g}_k{a.k}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    for r in results:
        print(f"{r['old']:>22} -> {r['new']:<22} n={r['n_tasks']:3d} mean {r['mean_old']:.3f}->{r['mean_new']:.3f} "
              f"RNF {r['RNF']} (null {r['RNF_null_mean']}, p={r['p_perm']}) RPF {r['RPF']} tax {r['taxonomy']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
