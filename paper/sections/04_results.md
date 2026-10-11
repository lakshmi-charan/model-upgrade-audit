# Results {#sec:results}

All numbers below are computed on the repaired data (Section [-@sec:integrity]) by the frozen analysis scripts. Six pairs × three testbeds give 18 cells. One pair lost 4 of its 100 clinical tasks to provider rate limits that persisted through three re-run passes; those tasks are excluded from that pair's clinical cell, and every other cell is complete. Table [-@tbl:hyp] summarises the pre-registered tests; Table [-@tbl:cells] in Appendix [-@sec:tables] gives every cell.

| Hypothesis (pre-registered criterion) | Result | Verdict |
|---|---|---|
| H1: of single-run regressions, > 50% pass on re-run (CI lower bound > 0.5) | 34 of 54 (63%), 95% CI [0.50, 0.76] | not supported (bound = 0.50) |
| H2: ≥ 1 cell with significant RNF, and pooled RNF rate < 10% (upper bound) | 0 cells significant; 20 of 449 (4.5%), upper bound 6.8% | not supported (rare, but none significant) |
| H3: ≥ 2 pairs change unlicensed writes significantly, not all in one direction | 5 of 6 pairs significant, all decreases | not supported (one direction only) |
| H4: re-running failures costs ≤ 25% more episodes | +88 of 449 (19.6%) | supported |
| H5: ≥ 2 improving pairs with significant RNF | 0 | not supported |
| H6: ≥ 2 cells with significant erosion in Ops or MLOps | 0 in Ops, 0 in MLOps | not supported |

: Pre-registered hypotheses and outcomes on the repaired data. {#tbl:hyp}

## Upgrades are better on average {#sec:avg}

Every new model is at least as good as its predecessor on average in almost every cell (Figure [-@fig:decomp]). Mean success rises in 15 of 18 cells, and its 95% bootstrap interval lies above zero in 12; no interval lies below zero. The gains are largest on the operations testbed, where both open-weight successors go from failing most workflows to passing nearly all of them (Gemma 0.13 → 0.93, Qwen 0.29 → 0.96). Almost all of the older Qwen model's failures there are interface errors (malformed or unusable tool calls), while the older Gemma model's are a mix of wrong outcome reports, wrong writes and interface errors. On the clinical testbed the provider's replacement for its oldest small model gains 18 points (0.45 → 0.63). The only point decreases are small and uncertain: one OpenAI step (−0.03) and the Qwen step (−0.08) on MLOps, and the Mistral step on the clinical testbed (−0.003).

![Where each upgrade's change in mean success comes from. Bars to the right are gains; bars to the left split the losses into tasks the old model always passed that became flaky (1–2 of 3), tasks it always passed that the new model always fails (reliable negative flips), and losses on other tasks. Diamonds mark the net change.](figs/fig_decomposition.png){#fig:decomp width=100%}

## One-shot regression reports are mostly false alarms {#sec:noise}

A one-shot regression suite runs each task once with the new model and flags every task the old model handled that now fails. In our data that procedure flags 54 tasks across the 18 cells (counting only tasks the old model passed in all three runs, so that the old model's own flakiness cannot cause the flag). Of these, 34 (63%) pass in at least one of the new model's next two runs (Figure [-@fig:noise]); only 20 fail all three times. The pre-registered test required the lower end of the 95% interval to exceed one half; it equals 0.50, so H1 is formally not supported, but the point estimate and the interval both say that at least half of what a one-shot suite reports is run-to-run noise.

The rule of re-running each failure up to twice before flagging it finds all 20 reliable flips with no false alarms, at a cost of 88 extra episodes (19.6% more than one shot; H4). As a secondary analysis we pre-registered a first-action probing heuristic: it runs only the new model's first step on every task, fully checks the tasks whose first action differs from the old model's usual one plus a small random audit of the rest, and bounds the flips it may have missed. Over 200 random seeds it cost more (670 episodes on average against 537) and missed on average 4.8 of the 20 flips, because many reliable flips begin with the same first action as the old model. Confirming tasks in random order under the probe's budget (capped at what checking every task costs) found 17 of the 20 on average. The simple rule is the best of the procedures we tried.

![Single-run regressions per testbed and whether they persist when the new model is run twice more.](figs/fig_noise.png){#fig:noise width=60%}

## Reliable regressions are rare and not separable from churn {#sec:rnf}

Pooled over cells, 20 of the 449 tasks that an old model always solved are always failed by its successor (4.5%; 95% upper bound 6.8%). Reliable positive flips outnumber them six to one (129). No cell's RNF count is significant after Holm correction across the 18 cells; the smallest corrected p-value is 0.088 (Gemma on the clinical testbed, 4 RNF where 0.75 are expected under exchangeability). Five cells are below 0.05 before correction: Gemma and Qwen on the clinical testbed (4 RNF each), Gemma and Mistral on MLOps (3 and 2), and Mistral on the clinical testbed (2). Several of these concentrate on one task template. Three of Qwen3.5's four clinical RNFs are instances of one template (report the last HbA1c value and its date, and order a new test if it is older than a year; 12 instances in the set), and two of Gemma 4's four are instances of another (report the most recent capillary blood glucose; 4 instances). A team whose workflow is one of these templates would experience a real regression. But with three runs and 18 cells, our data cannot distinguish them from the churn that repeated runs of a single model produce, and H5, which required such breaks in improving pairs, is not supported.

Of the 20 RNFs, 9 are unrequested writes, 6 wrong answers, 2 wrong writes, 2 answer-format failures and 1 missing write. None is an interface failure.

## Newer generations take fewer unlicensed actions {#sec:overreach}

Unlicensed writes fall in every pair but one (Figure [-@fig:overreach]). Measured on the 69 clinical tasks whose correct action is to write nothing and on all 50 operations tasks, the rate per task falls from 0.165 to 0.003 for GPT-4.1 nano to GPT-5.6 Luna, from 0.081 to 0.003 for GPT-5 nano to GPT-5.6 Luna, from 0.165 to 0.081 between the two older OpenAI models, from 0.151 to 0.022 for Gemma and from 0.106 to 0.028 for Qwen, all significant after Holm correction. Mistral is unchanged (0.077 → 0.088, p = 0.90). Because no pair increased significantly, H3, which predicted family-specific directions, is not supported; the exploratory pilot's suggestion that some families became more reckless did not replicate.

![Unlicensed-write rate per task, old and new model; * marks pairs significant after Holm correction.](figs/fig_overreach.png){#fig:overreach width=70%}

## Reliability does not erode {#sec:erosion}

H6 predicted that upgrades would turn tasks the old model always passed into flaky ones more often than the reverse. On the two testbeds where it was confirmatory, no cell shows significant erosion; the two smallest one-sided p-values are 0.052 and 0.066, both on MLOps (net losses of 3 and 4 tasks), and neither is close to significant after Holm correction (0.32, 0.33). On the repaired clinical data, where the hypothesis was formed and is exploratory, pass^3 rises in four of six pairs and erosion is never significant. As Section [-@sec:integrity] shows, the pattern that motivated H6 was produced by the outage.
