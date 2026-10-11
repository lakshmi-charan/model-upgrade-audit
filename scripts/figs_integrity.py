"""Figure: clinical upgrade effect (mean change in success) on contaminated vs clean data.
Usage: python paper/figs_integrity.py <clean dir> <contaminated dir> <out dir>"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figs import NICE, BLUE, ORANGE, GRAY, INK, INK2, GRID, pair_label
from matplotlib.lines import Line2D

clean, cont, out = (Path(a) for a in sys.argv[1:4])
C = {(c["old"], c["new"]): c for c in json.loads((clean / "confirmatory.json").read_text())["cells"] if c["testbed"] == "clinical"}
K = {(c["old"], c["new"]): c for c in json.loads((cont / "confirmatory.json").read_text())["cells"] if c["testbed"] == "clinical"}
pairs = [p for p in C if p in K]
fig, ax = plt.subplots(figsize=(5.2, 2.7))
for yi, p in zip(range(len(pairs))[::-1], pairs):
    k, c = K[p], C[p]
    ax.plot([k["delta"], c["delta"]], [yi, yi], color=GRID, linewidth=2.2, zorder=1)
    ax.errorbar([c["delta"]], [yi], xerr=[[c["delta"] - c["delta_ci"][0]], [c["delta_ci"][1] - c["delta"]]],
                fmt="o", color=BLUE, ms=6, capsize=2, zorder=4, elinewidth=1)
    ax.scatter([k["delta"]], [yi], color=ORANGE, s=34, zorder=3, marker="s")
ax.axvline(0, color=INK2, linewidth=0.8)
ax.set_yticks(list(range(len(pairs)))[::-1])
ax.set_yticklabels([pair_label(*p) for p in pairs])
ax.set_xlabel("change in clinical success rate (new − old)")
ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
ax.legend(handles=[Line2D([], [], marker="s", color=ORANGE, linestyle="", label="during the outage (63% of episodes affected)"),
                   Line2D([], [], marker="o", color=BLUE, linestyle="", label="clean re-run, with 95% CI")],
          frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=1, fontsize=7.5)
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fig_integrity.png"); plt.close(fig)
print("ok")
