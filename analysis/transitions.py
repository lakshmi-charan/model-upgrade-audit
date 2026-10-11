"""EXPLORATORY (post hoc, not in the frozen protocol): how does per-task reliability move between versions?
For each pair, a 4x4 matrix of old pass count (0..3) -> new pass count, and a decomposition of the mean change into
reliable breaks (3->0), partial losses (3->1/2: the task became flaky), partial gains and reliable fixes (0->3)."""
import json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get("MM_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "analysis"))
import flips as F
from confirmatory import PAIRS
smap = dict(x.split("=") for x in (sys.argv[1] if len(sys.argv) > 1 else "clinical=main").split(","))
out = []
_RUNS = {}
for tb, sub in smap.items():
    runs = _RUNS[tb] = F.load_runs(tb, sub, 0.0, 3, sorted({l for p in PAIRS for l in p}))
    for o, n in PAIRS:
        O, N = runs.get(o) or {}, runs.get(n) or {}
        common = sorted(set(O) & set(N))
        if not common: continue
        a = np.array([sum(O[t][r]["success"] for r in (1, 2, 3)) for t in common])
        b = np.array([sum(N[t][r]["success"] for r in (1, 2, 3)) for t in common])
        M = np.zeros((4, 4), int)
        for x, y in zip(a, b): M[x, y] += 1
        d = (b - a) / 3.0
        rel_break = d[(a == 3) & (b == 0)].sum(); flaky_loss = d[(a == 3) & (b > 0) & (b < 3)].sum()
        other_loss = d[(d < 0) & ~(a == 3)].sum(); gains = d[d > 0].sum()
        out.append({"testbed": tb, "old": o, "new": n, "n": len(common), "matrix": M.tolist(),
                    "mean_change": float(d.mean()), "loss_reliable_break": float(rel_break / len(common)),
                    "loss_reliable_to_flaky": float(flaky_loss / len(common)), "loss_other": float(other_loss / len(common)),
                    "gains": float(gains / len(common)),
                    "tasks_reliable_old": int((a == 3).sum()), "reliable_to_flaky": int(((a == 3) & (b > 0) & (b < 3)).sum()),
                    "reliable_to_broken": int(((a == 3) & (b == 0)).sum())})
f = ROOT / "analysis" / "out" / ("transitions_" + "_".join(f"{k}-{v}" for k, v in smap.items()) + ".json")
f.write_text(json.dumps(out, indent=1), encoding="utf-8")
for r in out:
    print(f"{r['testbed']:8s} {r['old']:>22s}->{r['new']:<22s} Δ={r['mean_change']:+.3f}  loss: broken {r['loss_reliable_break']:+.3f} "
          f"flaky {r['loss_reliable_to_flaky']:+.3f} other {r['loss_other']:+.3f} | gains {r['gains']:+.3f} | reliable_old {r['tasks_reliable_old']} -> flaky {r['reliable_to_flaky']}, broken {r['reliable_to_broken']}")


# ---- EXPLORATORY test: is the loss of "always passes" (pass^3) beyond run-to-run noise?
def erosion_test(a_runs, b_runs, n_perm=4000, seed=0):
    """a_runs,b_runs: (n,3) 0/1. Statistic: #tasks reliable under old but not new minus the reverse.
    Null: the 6 runs of a task are exchangeable between versions (permute within task)."""
    def stat(A, B):
        ra, rb = A.sum(1) == 3, B.sum(1) == 3
        return int((ra & ~rb).sum() - (rb & ~ra).sum())
    obs = stat(a_runs, b_runs)
    both = np.concatenate([a_runs, b_runs], 1); rng = np.random.default_rng(seed); ge = 0
    for _ in range(n_perm):
        p = np.take_along_axis(both, np.argsort(rng.random(both.shape), 1), 1)
        ge += stat(p[:, :3], p[:, 3:]) >= obs
    return obs, (1 + ge) / (n_perm + 1)


if __name__ == "__main__":
    print("\nEXPLORATORY erosion test (net loss of always-passing tasks; one-sided permutation p):")
    for tb, sub in smap.items():
        runs = _RUNS[tb]
        for o, n in PAIRS:
            O, N = runs.get(o) or {}, runs.get(n) or {}
            common = sorted(set(O) & set(N))
            if not common: continue
            A = np.array([[O[t][r]["success"] for r in (1, 2, 3)] for t in common])
            B = np.array([[N[t][r]["success"] for r in (1, 2, 3)] for t in common])
            obs, p = erosion_test(A, B)
            print(f"  {tb:8s} {o:>22s}->{n:<22s} pass^3 old {np.mean(A.sum(1)==3):.2f} new {np.mean(B.sum(1)==3):.2f}  net loss {obs:+d}  p={p:.4f}")
