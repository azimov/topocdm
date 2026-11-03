"""
02 -- fit, generate, and validate against held-out patients.

This is the experiment the empirical case rests on. The claim to test:

    a topology model fitted to a CDM generates synthetic patients whose
    prevalence and incidence match held-out real patients, and whose joint
    structure (codes per patient, co-occurrence, concentration) is right in a
    way that a marginals-only model cannot manage.

Run twice: once on a world that has block structure, and once on a null world
that has none. The second run is the specificity test -- a method that finds
structure in a world without any has not measured anything.

Run: MPLCONFIGDIR=./.mplconfig .venv/bin/python experiments/02_validate.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import topocdm as t
from topocdm.evaluate import predictive_loglik_nb
from topocdm.fit import fit_propensity_nb
from topocdm.model import Params

FIG, RES = "figs", "results"
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)
plt.rcParams.update({"figure.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
C = dict(blue="#2C6E9B", orange="#D9822B", green="#4E8B4A",
         red="#B4443C", purple="#6B5B95", grey="#8A8A8A")


def marginals_only(data: t.CDMIncidence) -> Params:
    """The baseline: concept intensities, everyone identical. This is the model
    that any marginal-fidelity check implicitly assumes."""
    lam = data.X.sum(axis=0) / max(data.T.sum(), 1e-9)
    return Params(u=np.ones(data.n), lam=lam, B=None,
                  patient_group=np.zeros(data.n, dtype=int),
                  concept_group=data.concept_group.copy(), n_groups=1)


def run_world(name, cfg, G_max=6):
    data, true = t.simulate(cfg)
    rng = np.random.default_rng(0)
    idx = rng.permutation(data.n)
    n_tr = int(0.75 * data.n)
    tr, te = data.subset_patients(idx[:n_tr]), data.subset_patients(idx[n_tr:])

    print(f"\n=== {name} ===")
    print(f"  train N={tr.n} test N={te.n} events={data.total}")

    # model selection on the training half
    fits = {}
    for G in range(1, G_max + 1):
        fits[G] = t.fit(tr, n_patient_groups=G, n_restarts=2, seed=G)
    aics = {G: t.aic_mixture(f.params, tr) for G, f in fits.items()}
    G_hat = min(aics, key=aics.get)
    print("  AIC by G:", {g: int(v) for g, v in aics.items()}, "-> G =", G_hat,
          f"(true {cfg.n_patient_groups})")

    topo = fits[G_hat]
    marg = marginals_only(tr)

    # generate fresh patients, holding the test patients' person-time
    template = te
    out = {}
    for label, p in [("topology", topo.params), ("marginals", marg)]:
        k = fit_propensity_nb(tr, p).params["shape"] if p.B is not None or p.u.std() > 0 else 1e4
        dist = fit_propensity_nb(tr, p)
        syn = t.sample_cdm(p, template, patient_dist=dist, mode="fresh",
                           rng=np.random.default_rng(1))
        cmp = t.compare(te, syn)
        held = predictive_loglik_nb(te, p, k, marginalise_groups=(p.B is not None))
        out[label] = dict(cmp=cmp, heldout=held, k=k)
        print(f"  {label:10s} prev r={cmp['prevalence']['r']:.4f} "
              f"slope={cmp['prevalence']['slope']:.3f} | inc r={cmp['incidence']['r']:.4f} "
              f"slope={cmp['incidence']['slope']:.3f} | deg {np.round(cmp['degree_mean'],2)} "
              f"L1={cmp['degree_l1']:.3f} | cooc r={cmp['cooccurrence_r']:.3f} "
              f"| gini {np.round(cmp['gini_events'],3)} | heldout/event {held:.4f}")

    return dict(name=name, G_hat=G_hat, G_true=cfg.n_patient_groups,
                aics={str(g): float(v) for g, v in aics.items()},
                results={k: dict(prevalence=v["cmp"]["prevalence"],
                                 incidence=v["cmp"]["incidence"],
                                 degree_mean=v["cmp"]["degree_mean"],
                                 degree_p99=v["cmp"]["degree_p99"],
                                 degree_l1=v["cmp"]["degree_l1"],
                                 cooccurrence_r=v["cmp"]["cooccurrence_r"],
                                 gini=v["cmp"]["gini_events"],
                                 heldout=v["heldout"])
                         for k, v in out.items()})


def fig_validate(res, path=f"{FIG}/fig9_validate.png"):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    worlds = [r for r in res]
    labels = ["has structure", "null world"]
    xs = np.arange(len(worlds))

    ax = axes[0]
    w = 0.36
    for i, (label, col) in enumerate([("topology", C["blue"]),
                                     ("marginals", C["orange"])]):
        vals = [r["results"][label]["prevalence"]["slope"] for r in worlds]
        ax.bar(xs + (i - 0.5) * w, vals, width=w, color=col, label=label)
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("prevalence calibration slope")
    ax.set_title("Marginally, both are fine", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for i, (label, col) in enumerate([("topology", C["blue"]),
                                     ("marginals", C["orange"])]):
        vals = [r["results"][label]["cooccurrence_r"] for r in worlds]
        bars = ax.bar(xs + (i - 0.5) * w, vals, width=w, color=col, label=label)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 0.002, f"{v:.3f}",
                    ha="center", fontsize=6.5)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("co-occurrence correlation")
    ax.set_title("Jointly, only one is", fontsize=9)
    ax.set_ylim(0.9, 1.0)

    ax = axes[2]
    for i, (label, col) in enumerate([("topology", C["blue"]),
                                     ("marginals", C["orange"])]):
        vals = [r["results"][label]["heldout"] for r in worlds]
        ax.bar(xs + (i - 0.5) * w, vals, width=w, color=col, label=label)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("held-out log-lik per event")
    ax.set_title("Predicting a new patient", fontsize=9)
    for i, (label, col) in enumerate([("topology", C["blue"]),
                                     ("marginals", C["orange"])]):
        vals = [r["results"][label]["heldout"] for r in worlds]
        for x_, v in zip(xs + (i - 0.5) * w, vals):
            ax.text(x_, v - 0.08, f"{v:.2f}", ha="center", fontsize=6.5)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    base = dict(n=8000, c=80, n_concept_groups=5, n_patient_groups=4,
                sigma_u=1.2, seed=202)
    structured = t.SimConfig(**{**base, "block_amplification": 5.0,
                                "block_noise": 0.05})
    nullworld = t.SimConfig(**{**base, "block_amplification": 0.0,
                               "block_noise": 1e-9})

    res = [run_world("structured", structured),
           run_world("null", nullworld)]
    json.dump(res, open(f"{RES}/validate.json", "w"), indent=1)
    fig_validate(res)
    print(f"\nwrote {FIG}/fig9_validate.png")


if __name__ == "__main__":
    main()
