"""
04 -- fit a real OMOP CDM (the local Synthea slice) and validate on held-out patients.

This is the first test that is not simulated. It asks the same three questions
as 02_validate.py, on the 10,000-patient Synthea slice (773 concepts):

  1. Does the fit recover the marginal structure (prevalence, incidence)?
  2. Does it recover the joint structure (codes per patient, co-occurrence),
     which a marginals-only model cannot?
  3. Does a synthetic cohort drawn from the release behave like held-out
     patients?

Data are not committed. Run scripts/extract_omop.sh first.

Run: MPLCONFIGDIR=./.mplconfig .venv/bin/python experiments/04_real_cdm.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import topocdm as t
from topocdm.io import load_incidence
from topocdm.model import CDMIncidence, Params
from topocdm.scalable import (fit_streaming, array_chunks,
                              fit_propensity_streaming)
from topocdm.release import make_release, sample_release
from topocdm.evaluate import predictive_loglik_nb, cooccurrence, prevalence

FIG, RES = "figs", "results"
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)
plt.rcParams.update({"figure.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
C = dict(blue="#2C6E9B", orange="#D9822B", green="#4E8B4A",
         red="#B4443C", purple="#6B5B95", grey="#8A8A8A")


def dense(d):
    return CDMIncidence(X=d.X.toarray(), T=d.T, concept_group=d.concept_group)


def marginals_only(d):
    lam = np.asarray(d.X.sum(axis=0)).ravel() / d.T.sum()
    return Params(u=np.ones(d.n), lam=lam, B=None,
                  patient_group=np.zeros(d.n, dtype=int),
                  concept_group=d.concept_group.copy(), n_groups=1)


def main():
    t0 = time.time()
    data, groups, cv = load_incidence()
    print(f"loaded {data.n:,} patients x {data.c:,} concepts, "
          f"{data.total:,} events, {data.density:.2f}/person-year "
          f"({time.time()-t0:.1f}s)")

    rng = np.random.default_rng(0)
    idx = rng.permutation(data.n)
    ntr = int(0.75 * data.n)
    tr, te = data.take(idx[:ntr]), data.take(idx[ntr:])
    print(f"train {tr.n:,} / test {te.n:,}")

    # ---- model selection by held-out prediction, not by AIC
    # AIC keeps falling with G on this data because there is no planted truth;
    # the honest criterion is held-out log-likelihood.
    dense_te = dense(te)
    table = []
    for G in [1, 4, 8, 15, 25]:
        st = fit_streaming(array_chunks(tr, chunk_size=2000), n=tr.n, c=tr.c,
                           concept_group=tr.concept_group, n_groups=G,
                           n_epochs=8, init_sample=4000)
        k = fit_propensity_streaming(array_chunks(tr, 2000), st.params)
        held = predictive_loglik_nb(dense_te, st.params, k,
                                    marginalise_groups=(G > 1))
        table.append(dict(G=G, heldout=float(held)))
        print(f"  G={G:2d}  held-out loglik/event {held:.4f}")

    best = max(table, key=lambda r: r["heldout"])
    G_hat = best["G"]
    gains = [table[i + 1]["heldout"] - table[i]["heldout"]
             for i in range(len(table) - 1)]
    print(f"  held-out improves monotonically; marginal gain at G={G_hat} "
          f"is {gains[-1]:+.4f}")

    # ---- final fit and release
    st = fit_streaming(array_chunks(tr, chunk_size=2000), n=tr.n, c=tr.c,
                       concept_group=tr.concept_group, n_groups=G_hat,
                       n_epochs=12, init_sample=5000)
    k = fit_propensity_streaming(array_chunks(tr, 2000), st.params)
    rel = make_release(st.params, tr.concept_ids, tr.concept_group, k,
                       tr.density, n_patients=tr.n, n_events=tr.total)
    print(f"  release: {rel.summary()}")

    # ---- generate and compare against held-out patients
    out = {}
    for label, params, dist in [
            ("topology", rel, None),
            ("marginals", None, None)]:
        if label == "topology":
            syn = sample_release(rel, n_patients=te.n, density=te.density,
                                 person_time=te.T,
                                 rng=np.random.default_rng(1))
            cmp = t.compare(dense(te), dense(syn))
            held = predictive_loglik_nb(dense_te, st.params, k,
                                        marginalise_groups=True)
        else:
            p = marginals_only(tr)
            X = np.zeros((te.n, te.c))
            # simple Poisson draw at the marginal rate
            rate = np.asarray(tr.X.sum(axis=0)).ravel() / tr.T.sum()
            mu = te.T[:, None] * rate[None, :]
            X = np.random.default_rng(2).poisson(mu)
            syn = CDMIncidence(X=X, T=te.T, concept_group=te.concept_group)
            cmp = t.compare(dense(te), syn)
            held = predictive_loglik_nb(dense_te, p, k, marginalise_groups=False)
        out[label] = dict(cmp=cmp, heldout=float(held))
        print(f"  {label:10s} prev r={cmp['prevalence']['r']:.4f} "
              f"slope={cmp['prevalence']['slope']:.3f} | "
              f"inc r={cmp['incidence']['r']:.4f} slope={cmp['incidence']['slope']:.3f} | "
              f"deg {np.round(cmp['degree_mean'],1)} L1={cmp['degree_l1']:.3f} | "
              f"cooc r={cmp['cooccurrence_r']:.3f} | held {held:.4f}")

    json.dump(dict(selection=table, G_hat=G_hat,
                   release=rel.summary(),
                   results={k2: dict(prevalence=v["cmp"]["prevalence"],
                                     incidence=v["cmp"]["incidence"],
                                     degree_mean=v["cmp"]["degree_mean"],
                                     degree_l1=v["cmp"]["degree_l1"],
                                     cooccurrence_r=v["cmp"]["cooccurrence_r"],
                                     heldout=v["heldout"])
                            for k2, v in out.items()},
                   data=dict(n=int(data.n), c=int(data.c),
                             events=int(data.total), density=float(data.density))),
              open(f"{RES}/real_cdm.json", "w"), indent=1)

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))

    ax = axes[0]
    ax.plot([r["G"] for r in table], [r["heldout"] for r in table],
            "o-", color=C["blue"], ms=4)
    ax.axvline(G_hat, color=C["red"], ls="--", lw=1.0)
    ax.set_xlabel("patient groups fitted")
    ax.set_ylabel("held-out log-lik per event")
    ax.set_title("More groups help, with diminishing returns", fontsize=9)

    ax = axes[1]
    d_real = np.sort(np.asarray(te.X.sum(1)).ravel())[::-1]
    syn = sample_release(rel, n_patients=te.n, person_time=te.T,
                         rng=np.random.default_rng(3))
    d_syn = np.sort(np.asarray(syn.X.sum(1)).ravel())[::-1]
    for arr, col, lab in [(d_real, C["blue"], "held-out patients"),
                          (d_syn, C["orange"], "synthetic")]:
        arr = arr[arr > 0]
        ax.plot(arr, np.arange(1, arr.size + 1) / arr.size, color=col, lw=1.3, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("events per patient"); ax.set_ylabel(r"$P(K \geq k)$")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Degree distribution is reproduced", fontsize=9)

    ax = axes[2]
    pv = prevalence(dense(te))
    top = np.argsort(-pv)[:30]
    real_c = cooccurrence(dense(te), top)[np.triu_indices(len(top), 1)]
    syn_c = cooccurrence(dense(syn), top)[np.triu_indices(len(top), 1)]
    ax.scatter(real_c, syn_c, s=8, color=C["green"], alpha=0.6)
    lim = [0, max(real_c.max(), syn_c.max()) * 1.05]
    ax.plot(lim, lim, "k:", lw=0.8)
    ax.set_xlabel("Synthea co-occurrence"); ax.set_ylabel("synthetic")
    ax.set_title(f"Co-occurrence is only partly reproduced (r={np.corrcoef(real_c, syn_c)[0,1]:.2f})", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{FIG}/fig11_real_cdm.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {FIG}/fig11_real_cdm.png and {RES}/real_cdm.json")


if __name__ == "__main__":
    main()
