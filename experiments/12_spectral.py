"""
12 -- spectral description and comparison.

Two uses of spectra here:

  * per subject: each patient has a small co-occurrence graph (concepts as nodes,
    edges when two concepts appear in the same encounter). Its normalised
    Laplacian spectrum is a fingerprint.
  * whole CDM: the singular values of the normalised incidence matrix, compared
    with the synthetic cohort's using the cumulative spectral distance.

The second is the more informative one, and it does not flatter the model.
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG, RES = "figs", "results"
plt.rcParams.update({"figure.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})

z = np.load(f"{RES}/cdm_spectrum.npz")
s_real, s_syn = z["real"], z["syn"]

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))

ax = axes[0]
k = np.arange(1, 41)
ax.plot(k, s_real[:40], "o-", ms=3, color="#1f4e79", label="Synthea")
ax.plot(k, s_syn[:40], "s-", ms=3, color="#c1440e", label="synthetic")
ax.set_xlabel("rank")
ax.set_ylabel("singular value of normalised incidence")
ax.set_title("A. Synthea's matrix is higher-rank\nthan the model produces", fontsize=9)
ax.legend(fontsize=7.5, frameon=False)

ax = axes[1]
grid = np.linspace(0, max(s_real.max(), s_syn.max()), 300)
def ccdf(s):
    return np.searchsorted(np.sort(s), grid, side="right") / len(s)
ax.plot(grid, ccdf(s_real), color="#1f4e79", lw=1.6, label="real")
ax.plot(grid, ccdf(s_syn), color="#c1440e", lw=1.6, label="synthetic")
ks = float(np.abs(ccdf(s_real) - ccdf(s_syn)).max())
ax.set_xlabel("singular value")
ax.set_ylabel("cumulative fraction")
ax.set_title(f"B. Cumulative spectral distance: KS = {ks:.2f}", fontsize=9)
ax.legend(fontsize=7.5, frameon=False)

fig.tight_layout()
fig.savefig(f"{FIG}/fig23_spectrum.png", bbox_inches="tight")
print(f"KS = {ks:.3f}")
print("real top-10:", np.round(s_real[:10], 2))
print("synth top-10:", np.round(s_syn[:10], 2))
