# Re-run Before You Roll Back

**A pre-registered audit of same-family model upgrades for tool-using agents.**

When a provider retires the model behind an agent, teams switch to the next model in the same family and need to know which workflows will break. This repository contains the paper, results and analysis scripts for a repeat-aware audit of six same-family upgrade pairs (GPT-4.1 nano → GPT-5.6 Luna, GPT-5 nano → GPT-5.6 Luna, GPT-4.1 nano → GPT-5 nano, Qwen3-30B-A3B → Qwen3.5-35B-A3B, Mistral Small 3.2 → Mistral Small 2603, Gemma 3 27B → Gemma 4 31B). Every model ran every task three times on three executable, code-graded testbeds:

| Testbed | Tasks | What the agent does |
|---|---|---|
| Clinical | 100 | Reads and writes a FHIR health record (MedAgentBench tasks plus hazard variants) |
| Operations | 50 | Completes workflows on PostgreSQL, Gitea, Mailpit, Redis, an S3 store and TigerBeetle |
| MLOps | 40 | Diagnoses production model degradation by re-executing pipeline components |

## Main findings

- Mean success rose in 15 of 18 pair–testbed cells and fell significantly in none.
- 63% (95% CI 50–76%) of the regressions a one-shot comparison reports pass again on re-run. Re-running each failure up to twice removes these false alarms for about 20% more episodes.
- Reliable regressions (always passed before, always failed after) affect 4.5% of previously reliable tasks; none survives correction for multiple testing.
- Unlicensed writes fell significantly in five of six pairs and rose significantly in none.
- A ten-hour outage of the clinical server during data collection silently corrupted 63% of clinical episodes and made three beneficial upgrades look harmful; a per-episode environment check removes the artifact.

## Repository layout

```
paper/                    paper sources (Markdown), figures, bibliography and the built PDF
results/
  final/                  confirmatory analysis on the repaired data (JSON outputs + full log)
  audit_contaminated_run/ superseded analysis on data collected during the outage, with the list of affected episodes
runs/                     every graded episode (one JSON file per episode) and per-call serving-provider logs
  clinical/main, ops/main50, mlops/main40   confirmatory sets (k = 3 runs per model)
  clinical/pilot          exploratory pilot
  quarantine/             episodes excluded because the environment failed during the episode
  calls.jsonl             one row per model call: model, serving provider, tokens, cost, latency
code/                     runners (run_clinical.py, run_ops.py, run_mlops.py), the unified LLM client (mm/llm.py),
                          environment checks and the infrastructure scan (infra_scan.py)
analysis/                 confirmatory tests (H1–H5), transitions and erosion test (H6), flip and triage analyses
testbeds/                 clinical (FHIR / MedAgentBench + hazard variants), ops (real services), mlops (pipeline diagnosis)
config/models.yaml        models, routing and pinned serving providers
docs/PROTOCOL.md          pre-registered protocol with dated amendments; docs/freeze/ holds the SHA-256 freeze records
scripts/                  regenerate every table and figure of the paper from results/
```

## Reproducing

Tables and figures from the released results (no API keys needed):

```bash
pip install -r requirements.txt
python scripts/make_tables.py results/final results/audit_contaminated_run results/final/tables.md
python scripts/figs.py results/final paper/figs
python scripts/figs_integrity.py results/final results/audit_contaminated_run paper/figs
```

Re-running the statistics from the episode files:

```bash
python code/run_analysis.py          # writes analysis/out/ and prints the full log
```

Collecting new episodes requires API keys (copy `.env.example` to `.env`) and the testbed services:

- **Clinical:** the MedAgentBench FHIR server (`docker run -p 8080:8080 jyxsu6/medagentbench`), then
  `python code/run_clinical.py --check-env` and `python code/run_clinical.py --models <labels> --subset main --repeats 3`.
- **Operations:** `docker compose -f testbeds/ops/docker/compose.yml up -d`, then `python code/run_ops.py ...`.
- **MLOps:** requires the drift-or-defect source project (set `MM_MLOPS_SOURCE`), then `python code/run_mlops.py ...`.

`python code/infra_scan.py` scans graded episodes for environment connection errors.

## Citation

See `CITATION.cff`.

## License

MIT (see `LICENSE`).
