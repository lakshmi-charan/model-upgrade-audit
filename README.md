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
paper/                   paper sources (Markdown), figures, bibliography and the built PDF
  sections/              one file per section
  build.sh               builds paper.pdf (pandoc + pandoc-crossref + xelatex)
results/
  final/                 confirmatory analysis on the repaired data (JSON outputs + full log)
  audit_contaminated_run/ the superseded analysis on data collected during the outage (kept as an audit record)
scripts/
  make_tables.py         regenerates every results table from results/
  figs.py, figs_integrity.py   regenerate the figures from results/
  infra_scan.py          scans graded episodes for environment connection errors
```

## Reproducing the tables and figures

```bash
pip install -r requirements.txt
python scripts/make_tables.py results/final results/audit_contaminated_run results/final/tables.md
python scripts/figs.py results/final paper/figs
python scripts/figs_integrity.py results/final results/audit_contaminated_run paper/figs
bash paper/build.sh /path/to/pandoc-crossref     # optional: rebuild the PDF
```

Testbed code, runners and the graded episode files will be added in a follow-up release.

## Citation

See `CITATION.cff`.

## License

MIT (see `LICENSE`).
