"""
01 -- parameter recovery and model selection on simulated CDMs.

Two questions, both answerable only because the truth is known:

  1. Does the fit recover the parameters that generated the data? Not "does the
     likelihood improve" -- that is trivially true -- but do the latent patient
     groups come back, and does the fitted mean matrix match the true one?

  2. Does model selection pick the right number of patient groups? If AIC
     cannot tell a one-block world from a four-block world, the whole
     inferential programme is dead in the water.

Run: MPLCONFIGDIR=./.mplconfig .venv/bin/python experiments/01_recovery.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import topocdm as t

FIG, RES = "figs", "results"
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)
plt.rcParams.update({"figure.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
C = dict(blue="#2C6E9B", orange="#D9822B", green="#4E8B4A", red="#B4443C")


def group_accuracy(est, true, G):
    M = np.array([[((est == g) & (true == k)).sum() for k in range(G)]
                  for g in range(G)])
    ri, ci = linear_sum_assignment(-M)
    return float(M[ri, ci].sum() / M.sum()), ri, ci


def main():
    cfg = t.SimConfig(n=6000, c=80, n_concept_groups=5, n_patient_groups=4,
                      block_amplification=5.0, block_noise=0.05, seed=101)
    data, true = t.simulate(cfg)
    mu_true = t.mean_matrix(true, data)

    print(f"simulated: N={data.n} C={data.c} events={data.total} "
          f"({data.total / data.T.sum():.2f}/person-year)")

    rows = []
    for G in [1, 2, 3, 4, 5, 6, 8]:
        r = t.fit(data, n_patient_groups=G, n_restarts=3, seed=G)
        mu = t.mean_matrix(r.params, data)
        row = dict(G=G, loglik=r.loglik, aic=t.aic_mixture(r.params, data),
                   bic=t.bic_mixture(r.params, data),
                   mu_r=float(np.corrcoef(mu_true.ravel(), mu.ravel())[0, 1]),
                   iters=r.n_iter, converged=bool(r.converged))
        if G > 1:
            row["group_acc"], _, _ = group_accuracy(
                r.params.patient_group, true.patient_group, G)
        rows.append(row)
        print(f"  G={G}: loglik {r.loglik:10.1f}  AIC {row['aic']:10.1f}  "
              f"mu_r {row['mu_r']:.3f}  "
              f"acc {row.get('group_acc', float('nan')):.3f}")

    best_aic = min(rows, key=lambda x: x["aic"])
    best_bic = min(rows, key=lambda x: x["bic"])
    print(f"\ntrue G = {cfg.n_patient_groups}; "
          f"AIC picks {best_aic['G']}, BIC picks {best_bic['G']}")

    json.dump(dict(rows=rows, true_G=cfg.n_patient_groups,
                   aic_G=best_aic["G"], bic_G=best_bic["G"]),
              open(f"{RES}/recovery.json", "w"), indent=1)

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    Gs = [r["G"] for r in rows]

    ax = axes[0]
    ax.plot(Gs, [r["aic"] for r in rows], "o-", color=C["blue"], ms=4)
    ax.axvline(cfg.n_patient_groups, color=C["red"], ls="--", lw=1.0)
    ax.set_xlabel("number of patient groups fitted")
    ax.set_ylabel("AIC (mixture likelihood)")
    ax.set_title("Model selection finds the truth", fontsize=9)

    ax = axes[1]
    acc = [(r["G"], r["group_acc"]) for r in rows if "group_acc" in r]
    ax.plot([a[0] for a in acc], [a[1] for a in acc], "o-", color=C["green"], ms=4)
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.axvline(cfg.n_patient_groups, color=C["red"], ls="--", lw=1.0)
    ax.set_xlabel("number of patient groups fitted")
    ax.set_ylabel("group recovery accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Latent groups are identified", fontsize=9)

    ax = axes[2]
    ax.plot(Gs, [r["mu_r"] for r in rows], "o-", color=C["orange"], ms=4)
    ax.axvline(cfg.n_patient_groups, color=C["red"], ls="--", lw=1.0)
    ax.set_xlabel("number of patient groups fitted")
    ax.set_ylabel(r"corr($\hat\mu$, $\mu_{true}$)")
    ax.set_title("Mean structure is recovered", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig8_recovery.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG}/fig8_recovery.png")


if __name__ == "__main__":
    main()
