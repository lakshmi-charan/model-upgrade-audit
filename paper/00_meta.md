---
title: "Re-run Before You Roll Back: A Pre-registered Audit of Same-Family Model Upgrades for Tool-Using Agents"
author: "[Author names and affiliations withheld for review]"
date: ""
abstract: |
  When a provider retires the model behind an agent, teams switch to the next model in the same family and ask which of their workflows will break. We answer this question with a pre-registered, repeat-aware audit of six same-family upgrade pairs from four families, including a provider's two official retirement replacements. Each model ran every task three times on three executable, code-graded testbeds with side effects: a clinical agent on a FHIR health record, an operations agent on six real services, and an MLOps agent that diagnoses production degradation by re-executing pipelines. Mean success rose in 15 of 18 pair–testbed cells and fell significantly in none. Of the tasks a one-shot regression suite flags as broken, 63% (95% CI 50–76%) pass again on re-run. Re-running each failure up to twice removes these false alarms for 20% more episodes, and a first-action probing heuristic did not do better. Reliable regressions (always passed before, always failed after) affect 4.5% of previously reliable tasks, and none survives correction for multiple testing. Unlicensed writes fell significantly in five of six pairs and rose significantly in none. Five of our six hypotheses (five pre-registered, one added by a documented amendment) were not supported, and we report all of them. Finally, a ten-hour outage of the clinical server during our own collection silently corrupted 63% of clinical episodes. On that data, three upgrades appeared to lower success and an official replacement model appeared to reliably break a clinical workflow; checking the environment on every episode removed both artifacts. We release all episodes, serving-provider logs, the per-episode environment check and the analysis code.
link-citations: true
numbersections: true
geometry: margin=1in
fontsize: 10pt
mainfont: "DejaVu Serif"
sansfont: "DejaVu Sans"
monofont: "DejaVu Sans Mono"
figPrefix: ["Figure", "Figures"]
tblPrefix: ["Table", "Tables"]
secPrefix: ["Section", "Sections"]
---
