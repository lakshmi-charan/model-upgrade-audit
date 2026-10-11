"""Paper figures, generated only from analysis outputs (analysis/out/*.json and the analysis job log).
Usage: python paper/figs.py <dir with confirmatory.json, transitions_*.json, flips_*.json, analysis_log.txt> <out dir>
Palette: validated categorical slots (blue, orange, violet) + a labelled neutral gray; light surface, print-safe."""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, VIOLET, GRAY = "#2a78d6", "#eb6834", "#4a3aa7", "#7a7974"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
NICE = {"gpt-4.1-nano": "GPT-4.1 nano", "gpt-5-nano": "GPT-5 nano", "gpt-5.6-luna": "GPT-5.6 Luna",
        "qwen3-30b-a3b-2507": "Qwen3-30B", "qwen3.5-35b-a3b": "Qwen3.5-35B", "mistral-small-3.2": "Mistral S 3.2",
        "mistral-small-2603": "Mistral S 2603", "gemma-3-27b": "Gemma 3 27B", "gemma-4-31b": "Gemma 4 31B",
        "gemini-3.1-flash-lite": "Gemini 3.1 FL", "gemini-3.5-flash-lite": "Gemini 3.5 FL"}
TBN = {"clinical": "Clinical", "ops": "Operations", "mlops": "MLOps"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def pair_label(o, n):
    return f"{NICE.get(o, o)} → {NICE.get(n, n)}"


def fig_decomposition(trans, out):
    """Mean change per pair x testbed, split into gains and three kinds of loss."""
    rows = [r for r in trans]
    tbs = [t for t in ("clinical", "ops", "mlops") if any(r["testbed"] == t for r in rows)]
    fig, axes = plt.subplots(1, len(tbs), figsize=(3.0 * len(tbs) + 1.6, 3.3), sharey=True)
    axes = [axes] if len(tbs) == 1 else axes
    pairs = []
    for r in rows:
        k = (r["old"], r["new"])
        if k not in pairs:
            pairs.append(k)
    for ax, tb in zip(axes, tbs):
        y = list(range(len(pairs)))[::-1]
        for yi, (o, n) in zip(y, pairs):
            r = next((x for x in rows if x["testbed"] == tb and x["old"] == o and x["new"] == n), None)
            if not r:
                ax.text(0.02, yi, "not run (free-tier quota)", va="center", ha="left", color=INK2, fontsize=7,
                        transform=ax.get_yaxis_transform())
                continue
            left = 0.0
            for val, col in ((r["loss_reliable_to_flaky"], ORANGE), (r["loss_reliable_break"], VIOLET),
                             (r["loss_other"], GRAY)):
                ax.barh(yi, val, left=left, color=col, height=0.62, edgecolor="white", linewidth=1)
                left += val
            ax.barh(yi, r["gains"], left=0, color=BLUE, height=0.62, edgecolor="white", linewidth=1)
            ax.plot([r["mean_change"]], [yi], marker="D", color=INK, markersize=4, zorder=5)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_title(TBN[tb], fontsize=9, color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("change in success rate (share of tasks)")
    axes[0].set_yticks(list(range(len(pairs)))[::-1])
    axes[0].set_yticklabels([pair_label(o, n) for o, n in pairs])
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(color=BLUE, label="gains"), Patch(color=ORANGE, label="loss: always-pass → flaky"),
               Patch(color=VIOLET, label="loss: always-pass → always-fail"), Patch(color=GRAY, label="loss: other tasks"),
               Line2D([], [], marker="D", color=INK, linestyle="", markersize=4, label="net change")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.savefig(out / "fig_decomposition.png")
    plt.close(fig)


def fig_noise(conf, out):
    """Single-run regressions per testbed: how many vanish on re-run vs persist."""
    agg = {}
    for c in conf["cells"]:
        a = agg.setdefault(c["testbed"], [0, 0, 0])
        a[0] += c["single_run_regressions"]
        a[1] += c["recovered_on_rerun"]
        a[2] += c["RNF"]
    tbs = [t for t in ("clinical", "ops", "mlops") if t in agg]
    fig, ax = plt.subplots(figsize=(4.2, 1.9))
    for i, tb in enumerate(tbs):
        tot, rec, rnf = agg[tb]
        persist = tot - rec
        ax.barh(i, rec, color=BLUE, height=0.55, edgecolor="white")
        ax.barh(i, persist, left=rec, color=ORANGE, height=0.55, edgecolor="white")
        ax.text(tot + 0.6, i, f"{rec}/{tot} vanish on re-run", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(tbs)))
    ax.set_yticklabels([TBN[t] for t in tbs])
    ax.invert_yaxis()
    ax.set_xlabel("single-run regressions (old model 3/3, new model fails run 1)")
    ax.set_xlim(0, max(a[0] for a in agg.values()) * 1.45)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="passes again in run 2 or 3"), Patch(color=ORANGE, label="fails again")],
              frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)
    fig.savefig(out / "fig_noise.png")
    plt.close(fig)


def fig_reliability(erosion, out):
    """pass^3 (share of tasks passed in all three runs), old vs new, per pair x testbed."""
    tbs = [t for t in ("clinical", "ops", "mlops") if any(e["testbed"] == t for e in erosion)]
    fig, axes = plt.subplots(1, len(tbs), figsize=(3.0 * len(tbs) + 1.6, 3.1), sharey=True)
    axes = [axes] if len(tbs) == 1 else axes
    pairs = []
    for e in erosion:
        if (e["old"], e["new"]) not in pairs:
            pairs.append((e["old"], e["new"]))
    for ax, tb in zip(axes, tbs):
        for yi, (o, n) in zip(list(range(len(pairs)))[::-1], pairs):
            e = next((x for x in erosion if x["testbed"] == tb and x["old"] == o and x["new"] == n), None)
            if not e:
                ax.text(0.5, yi, "not run (free-tier quota)", va="center", ha="center", color=INK2, fontsize=7)
                continue
            ax.plot([e["p3_old"], e["p3_new"]], [yi, yi], color=GRID, linewidth=2.2, zorder=1)
            ax.scatter([e["p3_old"]], [yi], color=GRAY, s=26, zorder=3)
            ax.scatter([e["p3_new"]], [yi], color=ORANGE if e.get("sig") else BLUE, s=30, zorder=4)
            if e.get("sig"):
                ax.text(max(e["p3_old"], e["p3_new"]) + 0.03, yi, "*", va="center", color=INK, fontsize=10)
        ax.set_xlim(0, 1.05)
        ax.set_title(TBN[tb] + (" (exploratory)" if tb == "clinical" else " (confirmatory)"), fontsize=9)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("tasks passed in all 3 runs")
    axes[0].set_yticks(list(range(len(pairs)))[::-1])
    axes[0].set_yticklabels([pair_label(o, n) for o, n in pairs])
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", color=GRAY, linestyle="", label="old model"),
                        Line2D([], [], marker="o", color=BLUE, linestyle="", label="new model"),
                        Line2D([], [], marker="o", color=ORANGE, linestyle="", label="new model, significant erosion (*)")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.14))
    fig.savefig(out / "fig_reliability.png")
    plt.close(fig)


def fig_overreach(conf, out):
    pr = conf["H3"]["pairs"]
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for yi, r in zip(list(range(len(pr)))[::-1], pr):
        ax.plot([r["overreach_old"], r["overreach_new"]], [yi, yi], color=GRID, linewidth=2.2)
        ax.scatter([r["overreach_old"]], [yi], color=GRAY, s=26, zorder=3)
        ax.scatter([r["overreach_new"]], [yi], color=BLUE, s=30, zorder=4)
        if r.get("p_holm", 1) < 0.05:
            ax.text(max(r["overreach_old"], r["overreach_new"]), yi + 0.28, "*", ha="center", va="center", fontsize=10)
    ax.set_yticks(list(range(len(pr)))[::-1])
    ax.set_yticklabels([pair_label(r["old"], r["new"]) + (f"  (n={r['n_tasks']})" if r["n_tasks"] != pr[0]["n_tasks"] else "") for r in pr])
    ax.set_xlabel("unlicensed-write rate per task (mean of 3 runs)")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color=GRAY, linestyle="", label="old"),
                       Line2D([], [], marker="o", color=BLUE, linestyle="", label="new")],
              frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)
    fig.savefig(out / "fig_overreach.png")
    plt.close(fig)


def parse_erosion(log_text):
    out = []
    pat = re.compile(r"^\s+(\w+)\s+(\S+)->(\S+)\s+pass\^3 old ([\d.]+) new ([\d.]+)\s+net loss ([+-]\d+)\s+p=([\d.]+)")
    for line in log_text.splitlines():
        m = pat.match(line)
        if m:
            out.append({"testbed": m.group(1), "old": m.group(2), "new": m.group(3), "p3_old": float(m.group(4)),
                        "p3_new": float(m.group(5)), "net_loss": int(m.group(6)), "p": float(m.group(7))})
    return out


def holm_flags(rows):
    by = {}
    for r in rows:
        by.setdefault(r["testbed"], []).append(r)
    for tb, rs in by.items():
        ps = sorted(range(len(rs)), key=lambda i: rs[i]["p"])
        m, run = len(rs), 0.0
        for rank, i in enumerate(ps):
            run = max(run, (m - rank) * rs[i]["p"])
            rs[i]["p_holm"] = min(1.0, run)
            rs[i]["sig"] = rs[i]["p_holm"] < 0.05
    return rows


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    conf = json.loads((src / "confirmatory.json").read_text())
    trans = json.loads(next(src.glob("transitions_clinical-main_ops-main50_mlops-main40.json")).read_text())
    erosion = holm_flags(parse_erosion((src / "analysis_log.txt").read_text()))
    (out / "erosion.json").write_text(json.dumps(erosion, indent=1))
    fig_decomposition(trans, out)
    fig_noise(conf, out)
    fig_reliability(erosion, out)
    fig_overreach(conf, out)
    print("figures written to", out)
