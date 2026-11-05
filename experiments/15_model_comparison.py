"""15 -- fixed-effects vs visit-anchored: spectral fidelity against parameter count."""
from __future__ import annotations
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
FIG, RES = "figs", "results"

# visit model: measured in experiments/14
visit = [(7_850, 0.480, 7), (15_900, 0.476, 15), (32_600, 0.486, 25),
         (68_400, 0.467, 42), (107_400, 0.464, 51), (195_000, 0.464, 58)]
# fixed effects: block model and block+rank-15 factors (experiments/12 and earlier).
# these counts EXCLUDE the N per-patient propensities, shown separately below.
fixed = [(1_058 + 10_000, 0.413, 8), (1_058 + 11_595 + 10_000, 0.366, 75)]
real_rank = 81

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
ax = axes[0]
ax.plot([v[0] for v in visit], [v[1] for v in visit], "o-", ms=5,
        color="#C1440E", label="visit-anchored (no N dependence)")
ax.plot([f[0] for f in fixed], [f[1] for f in fixed], "s--", ms=6,
        color="#2C6E9B", label="fixed effects (includes N=10,000)")
ax.set_xscale("log")
ax.set_xlabel("model parameters")
ax.set_ylabel("spectral distance (KS)")
ax.set_title("A. Fidelity against parameter count", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="upper right")

ax = axes[1]
ax.plot([v[0] for v in visit], [v[2] for v in visit], "o-", ms=5, color="#C1440E")
ax.plot([f[0] for f in fixed], [f[2] for f in fixed], "s--", ms=6, color="#2C6E9B")
ax.axhline(real_rank, color="k", ls=":", lw=1.0)
ax.text(8e3, real_rank + 2, f"real rank = {real_rank}", fontsize=7.5)
ax.set_xscale("log")
ax.set_xlabel("model parameters")
ax.set_ylabel("effective rank (singular values > 0.1)")
ax.set_title("B. Rank recovered", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG}/fig25_model_comparison.png", bbox_inches="tight")
print("wrote", f"{FIG}/fig25_model_comparison.png")
json.dump(dict(visit=visit, fixed=fixed, real_rank=real_rank),
          open(f"{RES}/model_comparison.json", "w"), indent=1)
