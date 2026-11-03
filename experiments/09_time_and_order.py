"""
09 -- what the current model does not encode: timing and order.

Panel A: gaps between a patient's consecutive events, against the exponential
         that a Poisson (constant-rate) model implies. 91% of gaps are under a
         day, so events arrive in visits, not as a memoryless process.
Panel B: what tends to follow what, at concept-group level, as a lift over the
         independence benchmark. Order carries 0.52 bits of information.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG, RES = "figs", "results"
plt.rcParams.update({"figure.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})

ev = pd.read_csv("data/events_dated.csv", parse_dates=["event_date"])
ev = ev.sort_values(["person_id", "event_date"])
ev["prev"] = ev.groupby("person_id")["event_date"].shift()
g = ((ev.event_date - ev.prev).dt.total_seconds() / 86400).dropna()
g = g[g >= 0]
rate = 1.0 / g.mean()

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))

ax = axes[0]
edges = [0, 1, 7, 30, 180, 1e9]
labels = ["same day", "1-7 d", "7-30 d", "1-6 mo", "6 mo+"]
obs = np.array([((g >= edges[i]) & (g < edges[i+1])).mean()
                for i in range(len(labels))])
# what a constant-rate Poisson process with the same mean would give
mu = g.mean()
exp_ = np.array([np.exp(-edges[i]/mu) - np.exp(-edges[i+1]/mu)
                 for i in range(len(labels))])
xpos = np.arange(len(labels))
ax.bar(xpos - 0.2, 100*obs, width=0.4, color="#C1440E", label="observed")
ax.bar(xpos + 0.2, 100*exp_, width=0.4, color="#9aa5b1",
       label=f"constant-rate model (mean {mu:.0f} d)")
ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("% of gaps")
ax.set_title("A. Events arrive in bursts, not at a constant rate", fontsize=9)
ax.legend(fontsize=7.5, frameon=False)

ax = axes[1]
lift = np.load(f"{RES}/transition_lift.npy")
M = np.load(f"{RES}/transition_M.npy")
order = np.argsort(-M.sum(1))
v = 3.0
im = ax.imshow(lift[np.ix_(order, order)], cmap="RdBu_r", vmin=-v, vmax=v,
               interpolation="nearest")
ax.set_xlabel("concept group of the next visit")
ax.set_ylabel("concept group of this visit")
ax.set_title("B. What follows what (log$_2$ lift)\norder carries 0.52 bits", fontsize=9)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.set_label("log$_2$ lift", fontsize=7)
cb.ax.tick_params(labelsize=6)
fig.tight_layout()
fig.savefig(f"{FIG}/fig18_time_order.png", bbox_inches="tight")
print("wrote", f"{FIG}/fig18_time_order.png")
