# Limitations {#sec:limits}

**Scale.** We study six pairs from four families on 100 clinical tasks, 50 operations tasks and 40 MLOps cases, with three runs per model. This is enough to separate reliable flips from run-to-run churn per cell, but not to estimate small RNF rates precisely; the pooled upper bounds we report are the honest summary. All models are small or "lite" tiers chosen to fit a budget of under ten US dollars of paid inference; larger models may churn less, and our conclusions about noise in one-shot suites apply most directly to the cost-sensitive tiers that teams run at volume.

**Three runs.** With $k=3$ a task the new model passes with probability one half is classed as an RNF one time in eight. The permutation test accounts for this, but the definition of "reliable" is still coarse. More runs would sharpen it at linear cost; our released episodes allow any reader to recompute every statistic for $k \in \{1,2,3\}$.

**Serving.** Each model is pinned to one serving provider with fallbacks disabled, and the provider that answered every call is logged. A different provider, quantisation or decoding stack could change behaviour; for open-weight models our results describe *that model as served by that provider*. Temperature-zero decoding is not deterministic on these services, which is precisely the churn we measure, so exact episodes are not reproducible, but every episode we graded is released.

**Testbeds.** All three testbeds are code-graded with state-based oracles, which rules out judge error but limits the task types to those whose outcome can be read from a system's state. The clinical tasks run against a public synthetic record, not real patients; nothing here supports clinical use of any model. The operations and MLOps testbeds are our own constructions on real services and real pipelines, and their task mix shapes absolute success rates.

**Free-tier models.** A pre-registered seventh pair (Gemini Flash-Lite) ran on a free tier. Its quota of 50 requests per model per day made it impossible to replace the episodes lost to the outage, and it is excluded; our conclusions cover four families.

**Rate limits.** One provider's rate limits left 4 of 100 clinical tasks incomplete for one model after three re-run passes; they are excluded from that pair's clinical cell. Excluding tasks the provider would not serve could bias that cell if the excluded tasks differ systematically, for example by length.

**Same-family scope.** We study upgrades within a family because that is the migration providers recommend when a model is retired. Cross-family switches are likely to change more, and our pairs do not speak to them.

**Hypothesis H6 was added after the clinical data.** We state this in the protocol, treat H6 as exploratory on the clinical testbed, and test it as confirmatory only on the operations and MLOps sets, which had not been analysed when it was written.
