# Study Design {#sec:setup}

## Upgrade pairs {#sec:pairs}

An *upgrade pair* is an older and a newer model from the same family that a team would plausibly migrate between. We study six pairs from four families (Table [-@tbl:pairs]): two official retirement replacements published by a provider (each older small model is retired in favour of the same newer one), the step between those two older models, and the successive releases of three open-weight families. A seventh, pre-registered pair of two Gemini Flash-Lite versions ran on a free tier as a best-effort addition. Its daily quota of 50 requests per model made it impossible to replace its episodes lost to the outage described in Section [-@sec:integrity], so it is excluded from all analyses.

| Pair | Old model | New model | Relation |
|------|--------------------------|--------------------------|----------------------------------------|
| P1 | GPT-4.1 nano | GPT-5.6 Luna | provider's official replacement |
| P2 | GPT-5 nano | GPT-5.6 Luna | provider's official replacement |
| P3 | GPT-4.1 nano | GPT-5 nano | successive small models |
| P4 | Qwen3-30B-A3B (2507) | Qwen3.5-35B-A3B | successive open-weight MoE releases |
| P5 | Mistral Small 3.2 | Mistral Small (2603) | successive releases |
| P6 | Gemma 3 27B | Gemma 4 31B | successive open-weight releases |

: Upgrade pairs. {#tbl:pairs}

All models are reached through one routing service. To keep a model's behaviour from depending on which host answered, every model is pinned to a single serving provider with fallbacks disabled, and the provider that served each call is logged. All models run at temperature 0 with the same prompts, tools, round limits and response truncation, and with the cheapest reasoning setting where a model has one.

## Testbeds {#sec:testbeds}

All three testbeds are executable and graded by code that reads the resulting state; no model judges any outcome.

**Clinical (100 tasks).** The agent answers questions about patients and places orders through the FHIR API of an electronic health record server, using MedAgentBench task instances [@jiang2025medagentbench] and hazard variants (unfamiliar API aliases and value drift, an allergy, renal impairment, a duplicate order already on file, and a look-alike patient). Sixty-four of the 100 tasks are conditional: an order is licensed only if a lab value meets a threshold, or only if no recent result exists. A reference solver computes from the live server state the answer and the set of licensed orders for every task. An episode succeeds when the final answer is correct, every required order is placed, and no unlicensed order is placed. Write requests are recorded and acknowledged but not applied, so every episode sees the same record.

**Operations (50 tasks).** The agent completes short workflows (open an issue and comment on it, send a notification, record inventory and ledger movements, store an object) on real services running in containers: PostgreSQL, Gitea, Mailpit, Redis, an S3-compatible store and TigerBeetle. Every episode gets its own namespace. Grading reads only service-side records: the required effects must be present exactly once with the right content, and a scan of the namespace finds any extra or collateral effect. The agent must also report its outcome truthfully.

**MLOps (40 cases).** The agent is on call for a production model whose performance has dropped. It may roll back or replay pipeline components, measure the effect of a change and test the time boundary of a shift, within a budget of pipeline executions, and must then decide which components to fix and whether to retrain. The oracle comes from re-executing the real pipelines on public datasets from five domains with injected defects, genuine drift, both or neither.

The confirmatory task sets are disjoint from the pilot and smoke sets; the operations and MLOps sets are seeded stratified subsamples of larger pools.

## Measurements {#sec:measures}

For each pair and testbed, every task is run $k=3$ times by each model, each run an independent model call. Episodes in which the model API failed, or in which the environment returned a connection error to the agent, are not graded but re-run. A task enters a pair's analysis only when both models have all three runs. Let $s^{o}_{t,r}, s^{n}_{t,r}\in\{0,1\}$ be the success of the old and new model on task $t$ in run $r$.

* **Single-run regression:** $s^{o}_{t,1}=1$ and $s^{n}_{t,1}=0$ among tasks the old model passes in all runs. This is what a one-shot regression suite reports.
* **Reliable negative flip (RNF):** $\sum_r s^{o}_{t,r}=k$ and $\sum_r s^{n}_{t,r}=0$. Reliable positive flips are defined symmetrically.
* **Noise test.** Under the null hypothesis that a task's $2k$ runs are exchangeable between the two models, we permute the run labels within each task (10,000 permutations) and compare the observed RNF count with its permutation distribution [@good2005permutation]. Holm's correction [@holm1979] is applied across all pair–testbed cells.
* **pass^3^ and erosion.** pass^3^ is the share of tasks passed in all three runs [@yao2024taubench]. The erosion statistic of a cell is the number of tasks always passed by the old model but not the new one, minus the reverse, tested one-sided with the same within-task permutation (4,000 permutations) and Holm's correction within each testbed.
* **Failure categories** of the new model on RNF tasks: interface (malformed actions, round limit), unrequested write, wrong write (a requested write with wrong content or duplicated), missing write, wrong answer, and answer format (the right value in a shape the task's output contract rejects).
* **Unlicensed writes (overreach):** a write outside the licensed set. In the clinical testbed it is measured on the tasks whose licensed set is empty; in the operations testbed it is any extra or collateral effect found by the scan. Per pair, per-task rates (mean over three runs) are compared with a two-sided Wilcoxon signed-rank test, Holm-corrected across pairs.
* **Triage procedures** for finding RNFs: *one-shot* (run once, flag every failure), *re-run failures* (re-run each failure up to twice, flag only 0/3), and *first-action probing* (run only the new model's first step on every task; fully check tasks whose first action differs from the old model's usual one, plus a random audit of ten others), compared by episodes spent (a probe costs the measured fraction of an episode), flips found and false alarms, averaged over 200 random seeds.
