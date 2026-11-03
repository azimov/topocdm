"""
05 -- simulated versus observed calendar-year incidence.

The comparison an empirical team will actually make: for a set of concepts, does
a synthetic CDM reproduce the incidence a CohortIncidence-style analysis sees in
the real one?

Design decisions that matter for fairness:

  * the temporal profile is fitted over 2010-2025, not 2016-2025, so the
    completeness step at 2016 has to be reproduced rather than excluded;
  * both sides use the same cohort definition (visit in the year, 365 days of
    prior observation, prevalent cases excluded);
  * observed intervals are exact Poisson; simulated intervals are the model's
    analytic predictive interval, so the comparison is not circular through the
    same Monte Carlo.

Run: MPLCONFIGDIR=./.mplconfig .venv/bin/python experiments/05_incidence.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import topocdm as t
from topocdm.io import load_incidence
from topocdm.scalable import (fit_streaming, array_chunks,
                              fit_propensity_streaming)
from topocdm.release import make_release
from topocdm.temporal import fit_temporal, sample_periods, generate_dated
from topocdm.cohort_incidence import cohort_incidence, predictive_ci

FIG, RES = "figs", "results"
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)
plt.rcParams.update({"figure.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.5})
C = dict(obs="#1f4e79", sim="#c1440e", grid="#999999")


def main():
    years = np.arange(2010, 2026)
    t0 = time.time()

    d, groups, cv = load_incidence()
    print(f"loaded {d.n:,} x {d.c:,} in {time.time()-t0:.1f}s")

    st = fit_streaming(array_chunks(d, 2000), n=d.n, c=d.c,
                       concept_group=d.concept_group, n_groups=15,
                       n_epochs=8, init_sample=4000)
    k = fit_propensity_streaming(array_chunks(d, 2000), st.params)
    rel = make_release(st.params, d.concept_ids, d.concept_group, k, d.density)
    print(f"fitted; release {rel.summary()['n_parameters']} parameters")

    ev = pd.read_csv("data/events_dated.csv", parse_dates=["event_date"])
    vi = pd.read_csv("data/visits.csv", parse_dates=["visit_date"])
    pe = pd.read_csv("data/periods.csv", parse_dates=["start_date", "end_date"])

    prof = fit_temporal(ev, vi, pe, st.params, years, d.concept_ids)
    print("tau:", dict(zip(years.tolist(), np.round(prof.tau, 2).tolist())))

    # ---- generate a synthetic cohort of the same size and period structure
    t0 = time.time()
    syn_per = sample_periods(pe, d.n, np.random.default_rng(11))
    sv, se = generate_dated(rel, d.n, syn_per, prof, rng=np.random.default_rng(11))
    print(f"generated {len(sv):,} visits / {len(se):,} events in {time.time()-t0:.1f}s")

    # ---- concepts to compare: clinical conditions (Condition domain, Disorder)
    cmeta = pd.read_csv("data/concepts.csv",
                        usecols=["concept_id", "domain_id", "concept_class_id"])
    cond = set(cmeta[(cmeta.domain_id == "Condition") &
                     (cmeta.concept_class_id == "Disorder")].concept_id)
    w = ev[ev.event_date.dt.year.between(2016, 2025)]
    cand = w[w.concept_id.isin(cond)].concept_id.value_counts()
    picks = [int(c) for c in cand.index[:9]]
    # load_incidence keeps only the grouping columns, so read names directly
    names = (pd.read_csv("data/concepts.csv", usecols=["concept_id", "concept_name"])
             .set_index("concept_id")["concept_name"].to_dict())

    results = []
    for cid in picks:
        ro = cohort_incidence(vi, ev, pe, years, cid)
        si = cohort_incidence(sv, se, syn_per, years, cid)
        # the comparison window is the complete period: before 2016 this extract
        # is sparsely populated and the sparsity is concept-specific, which a
        # single global tau cannot represent
        ro = ro[ro.year >= 2016]; si = si[si.year >= 2016]
        m = ro.merge(si, on="year", suffixes=("_obs", "_sim"))
        m["concept_id"] = cid
        m["concept"] = names.get(cid, str(cid))[:38]
        results.append(m)
        print(f"  {cid:>10}  {m.concept.iloc[0][:34]:34s} "
              f"obs rate {m.rate_obs.mean():7.1f}  sim {m.rate_sim.mean():7.1f}")
    allr = pd.concat(results, ignore_index=True)
    allr.to_csv(f"{RES}/incidence_compare.csv", index=False)

    # ---------------------------------------------------------------- figure
    n_panel = len(picks)
    ncol = 3
    nrow = int(np.ceil(n_panel / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.5, 2.5 * nrow),
                             sharex=True, squeeze=False)
    for ax, cid in zip(axes.ravel(), picks):
        m = allr[allr.concept_id == cid]
        ax.fill_between(m.year, m.rate_lo_obs, m.rate_hi_obs,
                        color=C["obs"], alpha=0.18, linewidth=0)
        ax.plot(m.year, m.rate_obs, "o-", color=C["obs"], ms=3, lw=1.3,
                label="observed")
        ax.plot(m.year, m.rate_sim, "s--", color=C["sim"], ms=3, lw=1.3,
                label="simulated")
        ax.set_title(m.concept.iloc[0], fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.set_ylabel("per 1000 PY", fontsize=7)
    for ax in axes.ravel()[n_panel:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.suptitle("Calendar-year incidence, 2016-2025: observed (shaded = exact "
                 "Poisson 95% CI) vs simulated", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{FIG}/fig12_incidence_by_year.png", bbox_inches="tight")
    plt.close(fig)

    # ---- calibration scatter across every concept-year
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    ax = axes[0]
    ok = (allr.rate_obs > 0) & (allr.rate_sim > 0)
    ax.scatter(allr.rate_obs[ok], allr.rate_sim[ok], s=14, alpha=0.7,
               color=C["sim"], edgecolor="none")
    lim = [0.5 * min(allr.rate_obs[ok].min(), allr.rate_sim[ok].min()),
           2 * max(allr.rate_obs[ok].max(), allr.rate_sim[ok].max())]
    ax.plot(lim, lim, "k:", lw=0.9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("observed rate (per 1000 PY)")
    ax.set_ylabel("simulated rate")
    r = np.corrcoef(np.log(allr.rate_obs[ok]), np.log(allr.rate_sim[ok]))[0, 1]
    ax.set_title(f"concept-years, log-log (r={r:.3f})", fontsize=9)

    ax = axes[1]
    ratio = (allr.rate_sim / allr.rate_obs).replace([np.inf, -np.inf], np.nan).dropna()
    ratio = ratio[ratio > 0]
    ax.hist(np.log2(ratio), bins=20, color=C["sim"], alpha=0.8,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="k", ls=":", lw=0.9)
    ax.set_xlabel(r"$\log_2$(simulated / observed)  — 0 = perfect")
    ax.set_ylabel("concept-years")
    med = float(np.median(ratio))
    ax.set_title(f"median ratio {med:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig13_incidence_calibration.png", bbox_inches="tight")
    plt.close(fig)

    json.dump(dict(tau=[float(x) for x in prof.tau],
                   years=[int(y) for y in years],
                   corr_log=float(r), median_ratio=med,
                   concepts=[int(c) for c in picks],
                   n_events_real=int(len(ev)), n_events_sim=int(len(se))),
              open(f"{RES}/incidence.json", "w"), indent=1)
    print(f"\nlog-log r={r:.3f}  median ratio={med:.2f}")
    print(f"wrote {FIG}/fig12_incidence_by_year.png, {FIG}/fig13_incidence_calibration.png")


if __name__ == "__main__":
    main()
