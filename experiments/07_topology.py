"""
07 -- what the topology is, and what co-occurrence cannot see.

Three figures, all from the real Synthea CDM:

  fig14  the bipartite graph (what an edge is) and its concept projection
  fig15  the model's three objects: patient propensity, concept intensity,
         block affinity
  fig16  associations that survive adjustment for patient propensity, and
         associations that do not

Run: MPLCONFIGDIR=./.mplconfig .venv/bin/python experiments/07_topology.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topocdm.io import load_incidence
from topocdm.scalable import array_chunks, fit_streaming

FIG, RES = "figs", "results"
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)
plt.rcParams.update(
    {
        "figure.dpi": 160,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.5,
    }
)
C = {
    "ink": "#1f2933",
    "blue": "#2C6E9B",
    "orange": "#C1440E",
    "green": "#4E8B4A",
    "grey": "#9aa5b1",
    "red": "#B4443C",
}


def mantel_haenszel(X, strata):
    """Cochran-Mantel-Haenszel OR for every concept pair, stratified by
    patient propensity. This is the fair version of a co-occurrence model:
    it asks 'is the association still there within propensity strata?'"""
    B = X.toarray() if sp.issparse(X) else np.asarray(X)
    B = (B > 0).astype(np.float64)
    nc = B.shape[1]
    num = np.zeros((nc, nc))
    den = np.zeros((nc, nc))
    for s in np.unique(strata):
        m = strata == s
        if m.sum() < 5:
            continue
        Bs = B[m]
        n = Bs.sum(axis=0)  # concept totals in stratum
        both = Bs.T @ Bs  # pairs
        N = float(m.sum())
        a = both
        b = n[None, :] - both
        c = n[:, None] - both
        d = N - n[:, None] - n[None, :] + both
        num += a * d / N
        den += b * c / N
    with np.errstate(divide="ignore", invalid="ignore"):
        or_ = np.where(den > 0, num / den, np.nan)
    return or_


def main():
    d, groups, cv = load_incidence()
    X = d.X
    print(f"{d.n:,} patients x {d.c:,} concepts, {d.total:,} events")

    st = fit_streaming(
        array_chunks(d, 2000),
        n=d.n,
        c=d.c,
        concept_group=d.concept_group,
        n_groups=15,
        n_epochs=8,
        init_sample=4000,
    )
    p = st.params
    print(f"fitted: {len(np.unique(p.patient_group))} patient groups")

    # ---------------------------------------------------------------- fig 14
    prev = np.asarray((X > 0).mean(axis=0)).ravel()
    # intermediate prevalence: the very common concepts are in nearly every
    # patient and make an uninformative picture
    mid = np.where((prev > 0.04) & (prev < 0.45))[0]
    keep = mid[np.argsort(-prev[mid])][:34]
    # random patients, then ordered by how many of these they have: shows real
    # sparsity instead of the densest-patient selection artefact
    rng = np.random.default_rng(0)
    cand = rng.choice(d.n, size=160, replace=False)
    sub = np.asarray((X[cand][:, keep] > 0).sum(axis=1)).ravel()
    cand = cand[np.argsort(sub)]
    M = (X[cand][:, keep] > 0).toarray()
    M = M[::-1]

    fig = plt.figure(figsize=(11, 4.4))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(M, aspect="auto", cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xlabel("concept (34 with 4-45% prevalence)")
    ax.set_ylabel("160 randomly sampled patients\n(ordered by how many concepts they have)")
    ax.set_title("An edge: one patient, one concept\n(each black dot is a record)", fontsize=9)
    ax.grid(False)

    ax = fig.add_subplot(1, 3, 2)
    Bk = sp.csr_matrix((X[:, keep] > 0).astype(np.float64))
    P = np.asarray((Bk.T @ Bk).todense()) / d.n
    ax.imshow(P, cmap="Blues", interpolation="nearest", vmin=0, vmax=P.max())
    ax.set_title("The projection a co-occurrence model\nuses: concept x concept", fontsize=9)
    ax.set_xlabel("concept")
    ax.set_ylabel("concept")
    ax.grid(False)

    ax = fig.add_subplot(1, 3, 3)
    deg = np.asarray(X.sum(axis=1)).ravel()
    ndeg = np.asarray((X > 0).sum(axis=1)).ravel()
    for arr, col, lab in [
        (deg, C["blue"], "records per patient"),
        (ndeg, C["orange"], "distinct concepts per patient"),
    ]:
        a = np.sort(arr)[::-1]
        a = a[a > 0]
        ax.plot(a, np.arange(1, a.size + 1) / a.size, color=col, lw=1.4, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("degree")
    ax.set_ylabel(r"$P(K \geq k)$")
    ax.set_title(
        "The topology the projection discards:\nhundreds vs thousands of edges", fontsize=9
    )
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig14_edge.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- fig 15
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

    ax = axes[0]
    u = p.u
    a = np.sort(u)[::-1]
    ax.plot(a, np.arange(1, a.size + 1) / a.size, color=C["blue"], lw=1.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"patient propensity $u_i$")
    ax.set_ylabel(r"$P(U \geq u)$")
    ax.set_title(r"$u_i$: how many edges a patient gets", fontsize=9)

    ax = axes[1]
    lam = p.lam
    idx = np.argsort(-lam)
    ax.plot(lam[idx] / lam.sum(), color=C["orange"], lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("concept rank")
    ax.set_ylabel(r"$\lambda_j$ (share)")
    ax.set_title(r"$\lambda_j$: how often a concept occurs", fontsize=9)

    ax = axes[2]
    B = np.array(p.B)
    inv = np.argsort(-np.asarray(p.B).sum(axis=0))
    im = ax.imshow(
        np.log2(B[:, inv]),
        cmap="RdBu_r",
        vmin=-np.abs(np.log2(B)).max(),
        vmax=np.abs(np.log2(B)).max(),
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xlabel("concept group")
    ax.set_ylabel("patient group")
    ax.set_title(r"$B[g,h]$: which patient groups accumulate which", fontsize=9)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"$\log_2$ affinity", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig15_three_objects.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- fig 16
    # restrict to concepts with enough prevalence for a stable OR
    sel = np.where((prev > 0.02) & (prev < 0.7))[0]
    sel = sel[np.argsort(-prev[sel])][:45]
    Xs = X[:, sel]
    ratio = np.asarray(Xs.sum(axis=1)).ravel()
    ratio = ratio / max(np.median(ratio), 1e-9)
    q = np.quantile(ratio, np.linspace(0, 1, 6))
    strata = np.clip(np.digitize(ratio, q[1:-1]), 0, 4)

    mh = mantel_haenszel(Xs, strata)
    # raw OR
    Bs = sp.csr_matrix((Xs > 0).astype(np.float64))
    n = np.asarray(Bs.sum(axis=0)).ravel()
    both = np.asarray((Bs.T @ Bs).todense())
    N = float(Bs.shape[0])
    a = both
    b = n[None, :] - both
    c = n[:, None] - both
    dd = N - n[:, None] - n[None, :] + both
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = (a * dd) / (b * c)
    iu = np.triu_indices(len(sel), 1)
    r_, m_ = raw[iu], mh[iu]
    both_ = both[iu]
    # require both cells to be well populated, or the OR is a separation
    # artefact rather than an association
    ok = np.isfinite(r_) & np.isfinite(m_) & (r_ > 0) & (m_ > 0) & (both_ >= 25) & (r_ < 2**8)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    ax = axes[0]
    ax.scatter(
        np.log2(r_[ok]), np.log2(m_[ok]), s=10, alpha=0.55, color=C["blue"], edgecolor="none"
    )
    lim = [-2, np.log2(r_[ok].max()) + 0.5]
    ax.plot(lim, lim, "k:", lw=0.9)
    ax.axhline(0, color=C["green"], lw=0.8, ls="--")
    ax.set_xlabel(r"$\log_2$ odds ratio, raw co-occurrence")
    ax.set_ylabel(r"$\log_2$ odds ratio, within propensity strata")
    ax.set_title("Most 'co-occurrence' is propensity", fontsize=9)

    ax = axes[1]
    drop = np.log2(r_[ok]) - np.log2(m_[ok])
    ax.hist(drop, bins=30, color=C["orange"], alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="k", ls=":", lw=0.9)
    ax.set_xlabel(r"$\log_2$ OR lost to propensity adjustment")
    ax.set_ylabel("concept pairs")
    frac = float((drop > 1).mean())
    ax.set_title(f"{frac * 100:.0f}% of pairs lose >half their association", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig16_adjustment.png", bbox_inches="tight")
    plt.close(fig)

    json.dump(
        {
            "n_pairs": int(ok.sum()),
            "median_raw_or": float(np.median(r_[ok])),
            "median_adjusted_or": float(np.median(m_[ok])),
            "frac_losing_half": frac,
            "frac_adjusted_below_1_2": float((m_[ok] < 1.2).mean()),
            "frac_raw_above_2": float((r_[ok] > 2).mean()),
        },
        open(f"{RES}/topology.json", "w"),
        indent=1,
    )
    print(
        f"\npairs {ok.sum()}  median raw OR {np.median(r_[ok]):.2f}  "
        f"median adjusted OR {np.median(m_[ok]):.2f}"
    )
    print(
        f"{frac * 100:.0f}% of pairs lose more than half their association "
        f"after adjusting for patient propensity"
    )


if __name__ == "__main__":
    main()
