"""
08 -- what the fit actually does, in one picture.

Three panels, all from the Synthea slice:
  A  before fitting, the model expects every patient to have the same number of
     records; after fitting, the expectation tracks what each patient actually
     has
  B  each pass over the data strictly improves the fit, then flattens
  C  the same for concepts
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import topocdm as t
from topocdm.io import load_incidence
from topocdm.scalable import fit_streaming, array_chunks

FIG, RES = "figs", "results"
plt.rcParams.update({"figure.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
C = dict(ink="#1f2933", blue="#2C6E9B", orange="#C1440E", grey="#9aa5b1")

d, groups, cv = load_incidence()
G, H = 15, int(d.concept_group.max()) + 1

# --- before: every patient identical, every concept equal
T = d.T
rowsum = np.asarray(d.X.sum(axis=1)).ravel()
colsum = np.asarray(d.X.sum(axis=0)).ravel()
row_before = T * 1.0                    # u=1, sum(lambda)=1
col_before = np.full(d.c, d.total / d.c)

# --- fit
st = fit_streaming(array_chunks(d, 2000), n=d.n, c=d.c,
                   concept_group=d.concept_group, n_groups=G,
                   n_epochs=25, init_sample=4000)
p = st.params
lam_h = np.bincount(p.concept_group, weights=p.lam, minlength=H)
blam = (p.B @ lam_h)[p.patient_group]
row_after = T * p.u * blam
col_after = p.lam * np.bincount(p.concept_group, weights=np.ones(d.c),
                                minlength=H)[p.concept_group]

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))

ax = axes[0]
ax.scatter(row_before, rowsum, s=3, alpha=0.3, color=C["grey"],
           edgecolor="none", label="before fitting")
ax.scatter(row_after, rowsum, s=3, alpha=0.35, color=C["blue"],
           edgecolor="none", label="after fitting")
lim = [0, np.percentile(rowsum, 99.5)]
ax.plot(lim, lim, "k:", lw=0.9)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("records the model expects for this patient")
ax.set_ylabel("records the patient actually has")
ax.set_title("A. The model learns each patient's volume", fontsize=9)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")

ax = axes[1]
h = np.array(st.history)
ax.plot(np.arange(1, len(h) + 1), h, "o-", color=C["orange"], ms=4)
ax.set_xlabel("pass over the data")
ax.set_ylabel("log-likelihood")
ax.set_title("B. Each pass improves the fit, then it stops", fontsize=9)

fig.tight_layout()
fig.savefig(f"{FIG}/fig17_fitting.png", bbox_inches="tight")
json.dump(dict(passes=int(len(h)), loglik_start=float(h[0]), loglik_end=float(h[-1]),
               seconds=None), open(f"{RES}/fitting_loop.json", "w"), indent=1)
print(f"passes {len(h)}  loglik {h[0]:,.0f} -> {h[-1]:,.0f}")
print(f"corr(expected, actual) per patient: before "
      f"{np.corrcoef(row_before, rowsum)[0,1]:.3f}  after "
      f"{np.corrcoef(row_after, rowsum)[0,1]:.3f}")
