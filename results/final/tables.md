## Table: hypothesis outcomes

| Hypothesis | Statistic | Outcome |
|---|---|---|
| H1 one-shot regressions mostly noise | 34/54 = 0.63 vanish on re-run, 95% CI [0.50, 0.76] | not supported |
| H2 reliable regressions exist but are rare | 0 significant cells; pooled RNF 20/449 = 0.045 (UB95 0.068) | not supported (rare: yes; exist: no significant cell) |
| H3 unlicensed writes change, in both directions | 5 pairs significant, directions: down | not supported (all significant changes are decreases) |
| H4 re-running failures is cheap | +88/449 episodes = 19.6% | supported |
| H5 better on average, broken in places | improving pairs with a significant RNF cell: 0 | not supported |
| H6 reliability erodes more than tasks break (confirmatory on Ops, MLOps) | significant erosion cells: Ops 0, MLOps 0 | not supported |

## Table: per pair and testbed (clean data)

| Testbed | Pair | n | success old → new | Δ [95% CI] | old 3/3 | single-run regr. (vanish) | RNF | p (Holm) | pass^3 old → new | erosion p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| Clinical | GPT-4.1 nano → GPT-5.6 Luna | 100 | 0.45 → 0.63 | +0.18 [+0.10, +0.25] | 38 | 6 (6) | 0 | 1.000 (1.000) | 0.38 → 0.48 | 0.994 (1.000) |
| Clinical | GPT-5 nano → GPT-5.6 Luna | 100 | 0.60 → 0.63 | +0.03 [-0.03, +0.08] | 49 | 7 (6) | 1 | 0.307 (1.000) | 0.49 → 0.48 | 0.450 (1.000) |
| Clinical | GPT-4.1 nano → GPT-5 nano | 100 | 0.45 → 0.60 | +0.15 [+0.06, +0.23] | 38 | 5 (3) | 2 | 0.100 (1.000) | 0.38 → 0.49 | 0.999 (1.000) |
| Clinical | Qwen3-30B-A3B → Qwen3.5-35B-A3B | 100 | 0.65 → 0.76 | +0.11 [+0.03, +0.19] | 56 | 5 (1) | 4 | 0.008 (0.144) | 0.56 → 0.71 | 1.000 (1.000) |
| Clinical | Mistral Small 3.2 → Mistral Small 2603 | 96 | 0.76 → 0.76 | -0.00 [-0.07, +0.06] | 66 | 5 (3) | 2 | 0.012 (0.197) | 0.69 → 0.64 | 0.126 (0.757) |
| Clinical | Gemma 3 27B → Gemma 4 31B | 100 | 0.62 → 0.75 | +0.13 [+0.04, +0.22] | 51 | 6 (2) | 4 | 0.005 (0.088) | 0.51 → 0.72 | 1.000 (1.000) |
| Ops | GPT-4.1 nano → GPT-5.6 Luna | 50 | 0.66 → 0.97 | +0.31 [+0.23, +0.41] | 18 | 0 (0) | 0 | 1.000 (1.000) | 0.36 → 0.92 | 1.000 (1.000) |
| Ops | GPT-5 nano → GPT-5.6 Luna | 50 | 0.85 → 0.97 | +0.13 [+0.05, +0.21] | 34 | 2 (2) | 0 | 1.000 (1.000) | 0.68 → 0.92 | 0.999 (1.000) |
| Ops | GPT-4.1 nano → GPT-5 nano | 50 | 0.66 → 0.85 | +0.19 [+0.07, +0.31] | 18 | 5 (4) | 1 | 0.434 (1.000) | 0.36 → 0.68 | 1.000 (1.000) |
| Ops | Qwen3-30B-A3B → Qwen3.5-35B-A3B | 50 | 0.29 → 0.96 | +0.67 [+0.57, +0.77] | 5 | 0 (0) | 0 | 1.000 (1.000) | 0.10 → 0.88 | 1.000 (1.000) |
| Ops | Mistral Small 3.2 → Mistral Small 2603 | 50 | 0.79 → 0.95 | +0.15 [+0.09, +0.22] | 25 | 0 (0) | 0 | 1.000 (1.000) | 0.50 → 0.84 | 1.000 (1.000) |
| Ops | Gemma 3 27B → Gemma 4 31B | 50 | 0.13 → 0.93 | +0.80 [+0.69, +0.89] | 6 | 0 (0) | 0 | 1.000 (1.000) | 0.12 → 0.82 | 1.000 (1.000) |
| MLOps | GPT-4.1 nano → GPT-5.6 Luna | 40 | 0.20 → 0.41 | +0.21 [+0.08, +0.35] | 6 | 1 (1) | 0 | 1.000 (1.000) | 0.15 → 0.20 | 0.933 (1.000) |
| MLOps | GPT-5 nano → GPT-5.6 Luna | 40 | 0.17 → 0.41 | +0.23 [+0.12, +0.33] | 3 | 1 (1) | 0 | 1.000 (1.000) | 0.07 → 0.20 | 0.989 (1.000) |
| MLOps | GPT-4.1 nano → GPT-5 nano | 40 | 0.20 → 0.17 | -0.03 [-0.11, +0.06] | 6 | 1 (1) | 0 | 1.000 (1.000) | 0.15 → 0.07 | 0.052 (0.315) |
| MLOps | Qwen3-30B-A3B → Qwen3.5-35B-A3B | 40 | 0.47 → 0.38 | -0.08 [-0.20, +0.03] | 16 | 5 (4) | 1 | 0.091 (1.000) | 0.40 → 0.30 | 0.066 (0.328) |
| MLOps | Mistral Small 3.2 → Mistral Small 2603 | 40 | 0.12 → 0.25 | +0.12 [-0.02, +0.28] | 3 | 2 (0) | 2 | 0.032 (0.454) | 0.07 → 0.12 | 0.950 (1.000) |
| MLOps | Gemma 3 27B → Gemma 4 31B | 40 | 0.29 → 0.38 | +0.08 [-0.07, +0.24] | 11 | 3 (0) | 3 | 0.013 (0.197) | 0.28 → 0.35 | 0.997 (1.000) |

## Table: unlicensed writes (H3; clinical tasks with empty licensed set + all Ops tasks)

| Pair | tasks | rate old | rate new | p | p (Holm) |
|---|---|---|---|---|---|
| GPT-4.1 nano → GPT-5.6 Luna | 119 | 0.165 | 0.003 | <0.001 | <0.001 |
| GPT-5 nano → GPT-5.6 Luna | 119 | 0.081 | 0.003 | <0.001 | <0.001 |
| GPT-4.1 nano → GPT-5 nano | 119 | 0.165 | 0.081 | 0.012 | 0.024 |
| Qwen3-30B-A3B → Qwen3.5-35B-A3B | 119 | 0.106 | 0.028 | 0.007 | 0.022 |
| Mistral Small 3.2 → Mistral Small 2603 | 117 | 0.077 | 0.088 | 0.903 | 0.903 |
| Gemma 3 27B → Gemma 4 31B | 119 | 0.151 | 0.022 | <0.001 | 0.002 |

## Table: clinical testbed, contaminated vs clean data

| Pair | Δ contaminated | Δ clean | RNF cont. (p Holm) | RNF clean (p Holm) | pass^3 cont. old → new | pass^3 clean old → new |
|---|---|---|---|---|---|---|
| GPT-4.1 nano → GPT-5.6 Luna | +0.040 | +0.177 | 5 (0.006) | 0 (1.000) | 0.38 → 0.44 | 0.38 → 0.48 |
| GPT-5 nano → GPT-5.6 Luna | -0.057 | +0.027 | 0 (1.000) | 1 (1.000) | 0.38 → 0.44 | 0.49 → 0.48 |
| GPT-4.1 nano → GPT-5 nano | +0.097 | +0.150 | 2 (0.496) | 2 (1.000) | 0.38 → 0.38 | 0.38 → 0.49 |
| Qwen3-30B-A3B → Qwen3.5-35B-A3B | -0.030 | +0.110 | 0 (1.000) | 4 (0.144) | 0.55 → 0.47 | 0.56 → 0.71 |
| Mistral Small 3.2 → Mistral Small 2603 | +0.060 | -0.003 | 0 (1.000) | 2 (0.197) | 0.41 → 0.44 | 0.69 → 0.64 |
| Gemma 3 27B → Gemma 4 31B | -0.110 | +0.130 | 1 (0.775) | 4 (0.088) | 0.58 → 0.35 | 0.51 → 0.72 |

Contaminated pooled: H1 0.72 [0.60, 0.83] (n=58); H2 significant cells 1, pooled RNF 16/468; H4 extra 17.9%.