"""Generate the paper's result tables (markdown) from analysis outputs only.
Usage: python paper/make_tables.py <clean analysis dir> <contaminated analysis dir> <out .md>"""
import json
import re
import sys
from pathlib import Path

NICE = {"gpt-4.1-nano": "GPT-4.1 nano", "gpt-5-nano": "GPT-5 nano", "gpt-5.6-luna": "GPT-5.6 Luna",
        "qwen3-30b-a3b-2507": "Qwen3-30B-A3B", "qwen3.5-35b-a3b": "Qwen3.5-35B-A3B", "mistral-small-3.2": "Mistral Small 3.2",
        "mistral-small-2603": "Mistral Small 2603", "gemma-3-27b": "Gemma 3 27B", "gemma-4-31b": "Gemma 4 31B",
        "gemini-3.1-flash-lite": "Gemini 3.1 FL", "gemini-3.5-flash-lite": "Gemini 3.5 FL"}
TBN = {"clinical": "Clinical", "ops": "Ops", "mlops": "MLOps"}


def pl(o, n):
    return f"{NICE.get(o, o)} → {NICE.get(n, n)}"


def fp(p):
    if p is None:
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def erosion(log):
    pat = re.compile(r"^\s+(\w+)\s+(\S+)->(\S+)\s+pass\^3 old ([\d.]+) new ([\d.]+)\s+net loss ([+-]\d+)\s+p=([\d.]+)")
    out = {}
    for line in log.splitlines():
        m = pat.match(line)
        if m:
            out[(m.group(1), m.group(2), m.group(3))] = dict(p3o=float(m.group(4)), p3n=float(m.group(5)),
                                                           net=int(m.group(6)), p=float(m.group(7)))
    by = {}
    for k, v in out.items():
        by.setdefault(k[0], []).append(k)
    for tb, ks in by.items():                      # Holm within testbed
        ks.sort(key=lambda k: out[k]["p"])
        run, m = 0.0, len(ks)
        for i, k in enumerate(ks):
            run = max(run, (m - i) * out[k]["p"])
            out[k]["p_holm"] = min(1.0, run)
    return out


def main():
    clean, cont, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    C = json.loads((clean / "confirmatory.json").read_text())
    K = json.loads((cont / "confirmatory.json").read_text())
    E = erosion((clean / "analysis_log.txt").read_text())
    EK = erosion((cont / "analysis_log.txt").read_text())
    L = []
    L.append("## Table: hypothesis outcomes\n")
    L.append("| Hypothesis | Statistic | Outcome |\n|---|---|---|")
    h1, h2, h4 = C["H1"], C["H2"], C["H4"]
    L.append(f"| H1 one-shot regressions mostly noise | {round(h1['share_noise'] * h1['n_single_run_regressions'])}/"
             f"{h1['n_single_run_regressions']} = {h1['share_noise']:.2f} vanish on re-run, 95% CI "
             f"[{h1['ci95'][0]:.2f}, {h1['ci95'][1]:.2f}] | {'supported' if h1['holds'] else 'not supported'} |")
    L.append(f"| H2 reliable regressions exist but are rare | {h2['cells_significant']} significant cells; pooled RNF "
             f"{h2['pooled_RNF']}/{h2['pooled_reliable']} = {h2['pooled_rate']:.3f} (UB95 {h2['rate_ub95']:.3f}) | "
             f"{'supported' if h2['holds'] else 'not supported (rare: yes; exist: no significant cell)'} |")
    sig3 = [p for p in C["H3"]["pairs"] if p["p_holm"] < 0.05]
    dirs = {("down" if p["overreach_new"] < p["overreach_old"] else "up") for p in sig3}
    L.append(f"| H3 unlicensed writes change, in both directions | {len(sig3)} pairs significant, directions: "
             f"{', '.join(sorted(dirs)) or '–'} | {'supported' if C['H3']['holds'] else 'not supported (all significant changes are decreases)'} |")
    L.append(f"| H4 re-running failures is cheap | +{h4['rerun_extra_episodes']}/{h4['oneshot_episodes']} episodes "
             f"= {h4['extra_share']:.1%} | {'supported' if h4['holds'] else 'not supported'} |")
    L.append(f"| H5 better on average, broken in places | improving pairs with a significant RNF cell: "
             f"{len(C['H5'].get('cells', []))} | {'supported' if C['H5']['holds'] else 'not supported'} |")
    h6 = {tb: sum(1 for k, v in E.items() if k[0] == tb and v["p_holm"] < 0.05) for tb in ("ops", "mlops")}
    L.append(f"| H6 reliability erodes more than tasks break (confirmatory on Ops, MLOps) | significant erosion cells: "
             f"Ops {h6['ops']}, MLOps {h6['mlops']} | {'supported' if max(h6.values()) >= 2 else 'not supported'} |")
    L.append("")
    L.append("## Table: per pair and testbed (clean data)\n")
    L.append("| Testbed | Pair | n | success old → new | Δ [95% CI] | old 3/3 | single-run regr. (vanish) | RNF | p (Holm) | pass^3 old → new | erosion p (Holm) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in C["cells"]:
        e = E.get((c["testbed"], c["old"], c["new"]), {})
        L.append(f"| {TBN[c['testbed']]} | {pl(c['old'], c['new'])} | {c['n']} | {c['mean_old']:.2f} → {c['mean_new']:.2f} | "
                 f"{c['delta']:+.2f} [{c['delta_ci'][0]:+.2f}, {c['delta_ci'][1]:+.2f}] | {c['reliable_old']} | "
                 f"{c['single_run_regressions']} ({c['recovered_on_rerun']}) | {c['RNF']} | {fp(c['p'])} ({fp(c['p_holm'])}) | "
                 f"{e.get('p3o', float('nan')):.2f} → {e.get('p3n', float('nan')):.2f} | {fp(e.get('p'))} ({fp(e.get('p_holm'))}) |")
    L.append("")
    L.append("## Table: unlicensed writes (H3; clinical tasks with empty licensed set + all Ops tasks)\n")
    L.append("| Pair | tasks | rate old | rate new | p | p (Holm) |\n|---|---|---|---|---|---|")
    for p in C["H3"]["pairs"]:
        L.append(f"| {pl(p['old'], p['new'])} | {p['n_tasks']} | {p['overreach_old']:.3f} | {p['overreach_new']:.3f} | "
                 f"{fp(p['p'])} | {fp(p['p_holm'])} |")
    L.append("")
    L.append("## Table: clinical testbed, contaminated vs clean data\n")
    L.append("| Pair | Δ contaminated | Δ clean | RNF cont. (p Holm) | RNF clean (p Holm) | pass^3 cont. old → new | pass^3 clean old → new |")
    L.append("|---|---|---|---|---|---|---|")
    kc = {(c["old"], c["new"]): c for c in K["cells"] if c["testbed"] == "clinical"}
    for c in [c for c in C["cells"] if c["testbed"] == "clinical"]:
        k = kc.get((c["old"], c["new"]))
        ek, ec = EK.get(("clinical", c["old"], c["new"]), {}), E.get(("clinical", c["old"], c["new"]), {})
        if not k:
            continue
        L.append(f"| {pl(c['old'], c['new'])} | {k['delta']:+.3f} | {c['delta']:+.3f} | {k['RNF']} ({fp(k['p_holm'])}) | "
                 f"{c['RNF']} ({fp(c['p_holm'])}) | {ek.get('p3o', 0):.2f} → {ek.get('p3n', 0):.2f} | "
                 f"{ec.get('p3o', 0):.2f} → {ec.get('p3n', 0):.2f} |")
    L.append("")
    L.append("Contaminated pooled: H1 " + f"{K['H1']['share_noise']:.2f} [{K['H1']['ci95'][0]:.2f}, {K['H1']['ci95'][1]:.2f}] "
             f"(n={K['H1']['n_single_run_regressions']}); H2 significant cells {K['H2']['cells_significant']}, pooled RNF "
             f"{K['H2']['pooled_RNF']}/{K['H2']['pooled_reliable']}; H4 extra {K['H4']['extra_share']:.1%}.")
    dst.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
