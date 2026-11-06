"""
10 -- seasonality and the visit chain.

The ordering metric matters. Comparing the *modal* concept group of consecutive
visits dilutes the signal, because a visit carries several groups at once. The
right measure is the presence transition: P(group b at visit v+1 | group a at
visit v), over the full H x H matrix.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topocdm.io import load_incidence
from topocdm.scalable import array_chunks, fit_streaming
from topocdm.temporal import fit_seasonality, seasonal_residual
from topocdm.visits import (
    assign_to_visits,
    assign_with_presence,
    fit_presence_model,
    fit_visit_model,
    sample_presence,
)

FIG, RES = "figs", "results"
plt.rcParams.update(
    {
        "figure.dpi": 160,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

d, groups, cv = load_incidence()
H = int(d.concept_group.max()) + 1
cg_of = dict(zip(d.concept_ids, d.concept_group))
ev = pd.read_csv("data/events_dated.csv", parse_dates=["event_date"])
vi = pd.read_csv("data/visits.csv", parse_dates=["visit_date"])
pe = pd.read_csv("data/periods.csv", parse_dates=["start_date", "end_date"])

st = fit_streaming(
    array_chunks(d, 2000),
    n=d.n,
    c=d.c,
    concept_group=d.concept_group,
    n_groups=15,
    n_epochs=8,
    init_sample=4000,
)

years = np.arange(2016, 2026)
months, tau, exp_hm, obs_hm = fit_seasonality(
    ev, pe, st.params, years, d.concept_group, d.concept_ids
)
seasonal, resid = seasonal_residual(tau, months)
print(f"tau {tau.shape}, seasonal {seasonal.shape}, residual max {resid.max():.1f}")

# ---------------- generator comparison on the presence-transition metric
old = fit_visit_model(vi, ev, cg_of, H, concept_group=d.concept_group)
new = fit_presence_model(ev, cg_of, H, d.concept_group)
rng = np.random.default_rng(0)
sample = rng.choice(d.n, size=1200, replace=False)
vd = vi.groupby("person_id")["visit_date"].apply(lambda s: np.sort(s.to_numpy()))


def presence_matrix(df):
    df = df.copy()
    df["grp"] = df.concept_id.map(cg_of)
    df = df.dropna(subset=["grp"])
    out = {}
    for (pid, dt), g in df.groupby(["person_id", "event_date"]):
        out.setdefault(pid, []).append((dt, set(g.grp.astype(int))))
    return out


def transitions(pm):
    C = np.zeros((H, H))
    N = np.zeros(H)
    for _pid, vs in pm.items():
        vs.sort()
        for (_, s1), (_, s2) in zip(vs, vs[1:]):
            for a in s1:
                N[a] += 1
                for b in s2:
                    C[a, b] += 1
    return C / np.maximum(N[:, None], 1)


def build(use_presence):
    rows = []
    for i in sample:
        pid = int(d.person_ids[i])
        if pid not in vd.index:
            continue
        dates = vd.loc[pid]
        counts = np.asarray(d.X[i].todense()).ravel()
        if use_presence:
            pres = sample_presence(len(dates), new, rng)
            pairs = assign_with_presence(counts, pres, new, rng)
        else:
            pairs = assign_to_visits(counts, len(dates), old, rng)
        for v, j in pairs:
            rows.append((pid, int(d.concept_ids[j]), dates[v]))
    return pd.DataFrame(rows, columns=["person_id", "concept_id", "event_date"])


Tr = transitions(presence_matrix(ev))
iu = np.triu_indices(H, 1)
res = {}
for name, use in [("group-sequence", False), ("presence chains", True)]:
    T = transitions(presence_matrix(build(use)))
    res[name] = (
        float(np.corrcoef(Tr.ravel(), T.ravel())[0, 1]),
        float(np.corrcoef(Tr[iu], T[iu])[0, 1]),
        T,
    )
    print(f"{name:18s} presence-transition r all={res[name][0]:.3f} offdiag={res[name][1]:.3f}")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

ax = axes[0]
moy = np.array([m.month for m in months])
sel = np.argsort(-seasonal.max(axis=1))[:4]
for k in sel:
    ax.plot(np.arange(1, 13), seasonal[k], "o-", ms=3, lw=1.2, label=groups[k][:26])
ax.axhline(1, color="k", ls=":", lw=0.8)
ax.set_xticks(range(1, 13))
ax.set_xlabel("month of year")
ax.set_ylabel(r"seasonal multiplier")
ax.set_title("A. Seasonal shape by concept group", fontsize=9)
ax.legend(fontsize=6, frameon=False, ncol=2)

ax = axes[1]
idx2020 = [k for k, m in enumerate(months) if m.year in (2020, 2021)]
im = ax.imshow(
    np.log2(np.maximum(resid[:, idx2020], 1e-3)),
    aspect="auto",
    cmap="RdBu_r",
    vmin=-2,
    vmax=2,
    interpolation="nearest",
)
ax.set_xlabel("month, 2020-2021")
ax.set_ylabel("concept group")
ax.set_title("B. Deviation from the seasonal norm\n(COVID-era residual)", fontsize=9)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.set_label(r"$\log_2$ residual", fontsize=7)
cb.ax.tick_params(labelsize=6)

ax = axes[2]
ax.scatter(
    Tr[iu],
    res["presence chains"][2][iu],
    s=14,
    alpha=0.7,
    color="#4E8B4A",
    edgecolor="none",
    label="presence chains",
)
ax.scatter(
    Tr[iu],
    res["group-sequence"][2][iu],
    s=14,
    alpha=0.45,
    color="#9aa5b1",
    edgecolor="none",
    label="old generator",
)
lim = [Tr[iu].min() - 0.02, Tr[iu].max() + 0.02]
ax.plot(lim, lim, "k:", lw=0.9)
ax.set_xlabel("real presence transition")
ax.set_ylabel("simulated")
ax.set_title(
    f"C. Ordering: r={res['presence chains'][1]:.2f} (was {res['group-sequence'][1]:.2f})",
    fontsize=9,
)
ax.legend(fontsize=7, frameon=False)

fig.tight_layout()
fig.savefig(f"{FIG}/fig19_visit_chain.png", bbox_inches="tight")
print("wrote", f"{FIG}/fig19_visit_chain.png")
