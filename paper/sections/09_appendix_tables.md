# Per-cell results {#sec:tables}

| Testbed | Pair | n | Δ [95% CI] | Old 3/3 | 1-run regr. (vanish) | RNF | p (Holm) | pass^3 old → new | Erosion p (Holm) |
|---|---|--:|---|--:|---|--:|---|---|---|
| Clin. | P1 | 100 | +0.18 [+0.10, +0.25] | 38 | 6 (6) | 0 | 1 (1) | 0.38 → 0.48 | .994 (1) |
| Clin. | P2 | 100 | +0.03 [-0.03, +0.08] | 49 | 7 (6) | 1 | .307 (1) | 0.49 → 0.48 | .450 (1) |
| Clin. | P3 | 100 | +0.15 [+0.06, +0.23] | 38 | 5 (3) | 2 | .100 (1) | 0.38 → 0.49 | .999 (1) |
| Clin. | P4 | 100 | +0.11 [+0.03, +0.19] | 56 | 5 (1) | 4 | .008 (.144) | 0.56 → 0.71 | 1 (1) |
| Clin. | P5 | 96 | -0.00 [-0.07, +0.06] | 66 | 5 (3) | 2 | .012 (.197) | 0.69 → 0.64 | .126 (.757) |
| Clin. | P6 | 100 | +0.13 [+0.04, +0.22] | 51 | 6 (2) | 4 | .005 (.088) | 0.51 → 0.72 | 1 (1) |
| Ops | P1 | 50 | +0.31 [+0.23, +0.41] | 18 | 0 (0) | 0 | 1 (1) | 0.36 → 0.92 | 1 (1) |
| Ops | P2 | 50 | +0.13 [+0.05, +0.21] | 34 | 2 (2) | 0 | 1 (1) | 0.68 → 0.92 | .999 (1) |
| Ops | P3 | 50 | +0.19 [+0.07, +0.31] | 18 | 5 (4) | 1 | .434 (1) | 0.36 → 0.68 | 1 (1) |
| Ops | P4 | 50 | +0.67 [+0.57, +0.77] | 5 | 0 (0) | 0 | 1 (1) | 0.10 → 0.88 | 1 (1) |
| Ops | P5 | 50 | +0.15 [+0.09, +0.22] | 25 | 0 (0) | 0 | 1 (1) | 0.50 → 0.84 | 1 (1) |
| Ops | P6 | 50 | +0.80 [+0.69, +0.89] | 6 | 0 (0) | 0 | 1 (1) | 0.12 → 0.82 | 1 (1) |
| MLOps | P1 | 40 | +0.21 [+0.08, +0.35] | 6 | 1 (1) | 0 | 1 (1) | 0.15 → 0.20 | .933 (1) |
| MLOps | P2 | 40 | +0.23 [+0.12, +0.33] | 3 | 1 (1) | 0 | 1 (1) | 0.07 → 0.20 | .989 (1) |
| MLOps | P3 | 40 | -0.03 [-0.11, +0.06] | 6 | 1 (1) | 0 | 1 (1) | 0.15 → 0.07 | .052 (.315) |
| MLOps | P4 | 40 | -0.08 [-0.20, +0.03] | 16 | 5 (4) | 1 | .091 (1) | 0.40 → 0.30 | .066 (.328) |
| MLOps | P5 | 40 | +0.12 [-0.02, +0.28] | 3 | 2 (0) | 2 | .032 (.454) | 0.07 → 0.12 | .950 (1) |
| MLOps | P6 | 40 | +0.08 [-0.07, +0.24] | 11 | 3 (0) | 3 | .013 (.197) | 0.28 → 0.35 | .997 (1) |

: Every pair–testbed cell on the repaired data (pairs as in Table [-@tbl:pairs]). Δ is the change in mean success (new − old) with a task-level bootstrap 95% CI; "Old 3/3" counts tasks the old model passed in all runs; single-run regressions are old-3/3 tasks the new model failed in its first run, with the number that passed in run 2 or 3 in parentheses; RNF p-values are from the within-task permutation test, Holm-corrected across the 18 cells; erosion p-values are one-sided, Holm-corrected within each testbed (exploratory on the clinical testbed). {#tbl:cells}
