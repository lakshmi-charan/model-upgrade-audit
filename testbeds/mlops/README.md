# Testbed `mlops`: drift-or-defect diagnosis agent (ported from drift-or-defect v2, tools mode)

## What the agent does
An on-call MLOps agent sees the monitoring report of a production model whose quality may have degraded:
reference vs current accuracy, per-feature drift and validation checks, per-component output drift, the pipeline
DAG and lineage, an anonymised change log (which components changed, and whether code/config/artifact/environment
changed) and source-schema events. It must decide **which pipeline components to fix (possibly none) and whether to
retrain**. Before answering it may run diagnostic actions, one JSON object per turn, which really execute the
pipeline (pandas, scikit-learn, two older library environments) on the case's data:

| action | what it does |
|---|---|
| `{"action":"rollback","component":C}` | re-run C at its reference version on current inputs: does C's current output differ? |
| `{"action":"replay","component":C}` | run C's current version on the reference period's inputs: did C's behaviour change? |
| `{"action":"effect","components":[...]}` | deployed-model accuracy on current serve data if these components were rolled back |
| `{"action":"boundary","component":C,"period":P}` | C's output jump at schema event P vs ordinary period-to-period change |
| `{"action":"final","fix":[...],"retrain":bool,"reason":"..."}` | the decision (ends the episode) |

Limits as in the source: at most **8** tool actions and a compute budget of 3 full pipeline runs (measured seconds);
when either is reached the agent is told so and asked once more for a final answer ("forced final").
System prompt, report, tool documentation and result summaries are produced by the source code itself
(`dod/llm_agent.py`: `SYSTEM`, `report`, `TOOLS_DOC`, `_summ`), so prompts are byte-identical to the source study.

Differences from the source runner (documented deviations): the LLM goes through `code/mm/llm.py` (OpenRouter /
Gemini / Groq) instead of the source's OpenAI client; no `response_format=json_object` is sent (not all providers
support it; the source's tolerant JSON extraction is kept); temperature is the CLI value (source: 0 / provider
default for reasoning models); reasoning models get +2000 completion tokens of headroom like the source; an
unparseable or ill-formed final is a failure here (the source silently treated it as "fix nothing, no retrain").

## Grading
Oracle per case (from the source's ground truth by execution, `bench.ground_truth`):
`fix` = consequential defects C (fixing them changes holdout accuracy of an action by ≥ 0.002), `retrain` = Ro
(retraining after the fix gains > 0.01), plus `effective` defects E ⊇ C (fixing changes the pipeline output at all).

**success = 1** iff the episode produced a valid final decision (F, R) with **C ⊆ F ⊆ E and R = Ro** — the source's
"correct" endpoint (`analysis/summarize.py`). Fixing an effective-but-inconsequential defect is therefore not an
error. `exact_match` (F = C and R = Ro) is stored as well; C ≠ E in about 40% of cases, so the two differ.

A valid final decision: a JSON object with `action` "final" (or no action), `fix` a list of pipeline component
names (case-insensitive match; any unknown name makes it invalid) and `retrain` a boolean ("true"/"false" accepted).
`status`: `completed` (valid final), `no_final` (unparseable reply or a tool action where the final was due),
`invalid_final` (wrong types / unknown component), `harness_error`.

## Category mapping (first matching rule)
With extra = F − E (blamed components that are not defective), missing = C − F, xr = R ∧ ¬Ro, mr = Ro ∧ ¬R:

| category | rule |
|---|---|
| `ok` | success |
| `interface` | status ≠ completed (no valid final, malformed final, harness error), or the final was forced by the step/budget limit and is wrong |
| `wrong_write` | extra ≠ ∅ and missing ≠ ∅: fixes, but the wrong component set (`grade_flags.overlap` = partial / disjoint) |
| `wrong_answer` | wrong cause type: (missing ≠ ∅ and xr) retrains instead of fixing, or (extra ≠ ∅ and mr) fixes a healthy component instead of a needed retrain |
| `unrequested_write` | extra ≠ ∅ (fix of a non-defective component), or fix set right but xr (unneeded retrain) |
| `missing_write` | missing ≠ ∅ (needed fix omitted), or fix set right but mr (needed retrain omitted) |
| `other` | none of the above (cannot occur with a valid final; kept for safety) |

`grade_flags` keeps the raw ingredients (extra_fix, missing_fix, extra_retrain, missing_retrain,
fix_inconsequential, forced_final) so categories can be recomputed. Malformed intermediate actions are counted in
`n_invalid_actions` (they still cost a step, as in the source) and do not by themselves change the category.

## Cases and subsets (frozen, seed 20261006; `build_cases.py`, refuses to overwrite)
Source TEST cases only: v1 test split of `tlc` (NYC taxi) and `news` (HuffPost) from `runs/final`, and all cases of
the v2 held-out domains `bts` (airline on-time), `hmda` (mortgages), `divvy` (bike share) from `runs/v2main`;
`acs` is dev-only and excluded. Strata = domain × oracle type (none / drift_only / defect_only / both).
- `pilot`: 40 = 8 per domain, 2 per oracle type (news has no "none" test case; shortfall moved to other types)
- `main`: 100 = 20 per domain, 5 per oracle type where available, disjoint from pilot
- `smoke`: 4 pilot cases, one per oracle type, from four domains
`data/cases.jsonl` holds the compact records (case, observation, oracle, ground-truth summary, and the source's
gpt-5.6-luna tools-mode answer for reference) of the 140 pilot+main cases; the mix is in each subset file.

## Data, environments and where files are written
The source project is used **in place, read-only**: `<model-migration>/../drift-or-defect` (override
`MM_MLOPS_SOURCE`). Needed there: `.venv` (pandas 2.2.3, scikit-learn 1.9.1, pyarrow; the main pipeline env and the
env that pickled the reference models), `envs/p15` (pandas 1.5) and `envs/p30` (pandas 3.0) for components deployed
in other library versions, `data/bronze/<domain>`, `data/artifacts/<domain>`, `data/raw/tlc`, `data/models/v3/<domain>`.
`dod/` here is a byte-identical copy of `<source>/v2/code/dod` (checked by `--check-env`); new files only:
`mlops_worker.py` (tool server), `build_cases.py`, this README, and `code/run_mlops.py`.

`run_mlops.py` (model-migration venv) starts one tool server per worker thread with the source venv's python
(override `MM_MLOPS_PYTHON`), passing `DOD_PROJECT_ROOT=<source>` and `DOD_CACHE=<model-migration>/runs/mlops_cache`
(override `MM_MLOPS_CACHE`; refused if inside the source). All pipeline outputs computed by the tools, and the
temporary transfer files of the p15/p30 subprocesses, go to `runs/mlops_cache/`. The source's content-addressed
cache `data/cache` is **read through** (a hit is read there; a miss is computed and written only to the target).
Reference models are only loaded (never built), `save_artifact` is disabled, ground-truth/outcome code that writes
`runs/outcomes` and the source LLM client are never called, API keys are stripped from the tool server's environment,
and `PYTHONDONTWRITEBYTECODE=1` is set.

## Commands
```
python code/run_mlops.py --check-env                       # free: paths, envs, code copy, one rollback + effect
python code/run_mlops.py --models gpt-4.1-nano,gpt-5.6-luna --subset smoke --repeats 1 --temperature 0 --workers 4
python code/run_mlops.py --models ... --subset pilot --repeats 3 --rep-start 1 --temperature 0.7 --workers 4
```
Output: `runs/mlops/<subset>/<label>/T<T>/r<r>/<task_id>.json` (task_id, label, model, provider, temperature, repeat,
success, status, error, category, exact_match, grade_flags, agent_cost, tokens_in/out, n_steps, n_llm_calls,
actions [step, action, args, valid, tool_error, result summary, cost_s], turns [model text, usage], final_decision,
final_raw, oracle, domain, otype, scenario, compute spent/budget) and one row per episode in
`runs/mlops/<subset>/episodes.jsonl`. Existing episode files are skipped. BudgetExceeded / QuotaExhausted stop the
job with exit code 3; `--check-env` failure exits 2. LLM cache salt: `mlops|T<T>|r<r>|<task_id>`.

## Cost and size
Source ledger (gpt-5.6-luna, OpenAI direct, reasoning low, tools mode): 2.7–3.7 tool steps (3.7–4.7 LLM calls) and
$0.0011–0.0018 per episode depending on domain. Initial prompt 3.4k–10.2k characters (median 6.4k, ≈1.8k tokens);
an episode is ≈ 8–12k input tokens and 0.3–0.8k output tokens (incl. reasoning), i.e. ≈ $0.002–0.003 for luna at
OpenRouter list prices without prompt caching. Tool compute is seconds per action when the source cache hits.
