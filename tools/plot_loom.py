"""Figure for a loom run: per-column probe traces (loom-facing vs control columns) and population
rates of the loom pathway. usage: plot_loom.py <tag> <out.png>   (tag e.g. loom_flyvis_g120_a90)"""
import sys, pathlib
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[1]
tag, out = sys.argv[1], sys.argv[2]
pr = pd.read_csv(ROOT / f"recordings/{tag}_probe.csv"); po = pd.read_csv(ROOT / f"recordings/{tag}.csv")
t0 = pr.t_ms[pr.threat == 1].min(); t1 = pr.t_ms[pr.threat == 1].max()
win = lambda d: d[(d.t_ms > t0 - 2000) & (d.t_ms < t1 + 600)]
pr, po = win(pr), win(po)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
cols = {"T4a": "#1f77b4", "T4b": "#2ca02c", "T4c": "#9467bd", "T4d": "#8c564b",
        "T5a": "#ff7f0e", "T5b": "#bcbd22", "T5c": "#e377c2", "T5d": "#d62728"}
for k, grp in enumerate(["loom", "ctrl"]):
    for t, c in cols.items():
        ax[k].plot((pr.t_ms - t0) / 1000, pr[f"{grp}:{t}"], color=c, lw=1.4, ls="-" if t[1] == "4" else "--", label=t)
    ax[k].set_title(("12 columns facing the predator" if grp == "loom" else "10 control columns (straight ahead)"))
    ax[k].set_ylabel("rate (Hz)")
for name, c in (("LPLC2", "#d62728"), ("LC4", "#1f77b4"), ("DNp01", "k")):
    ax[2].plot((po.t_ms - t0) / 1000, po[f"type:{name}_L"], color=c, lw=1.6, label=f"{name} (left)")
ax[2].set_title("loom-detector & giant-fibre populations")
for a in ax:
    a.axvspan(0, (t1 - t0) / 1000, color="0.9", zorder=0); a.set_xlabel("time from predator launch (s)")
ax[0].legend(ncol=2, fontsize=8); ax[2].legend(fontsize=8)
fig.suptitle(f"{tag}: FlyVis graded optic lobe -> real FlyWire T4/T5 -> spiking FlyWire LPLC2/LC4/giant fibre "
             f"(grey = predator approaching)", fontsize=10)
fig.tight_layout(); fig.savefig(ROOT / out, dpi=110)
print("wrote", out)
