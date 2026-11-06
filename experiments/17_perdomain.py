"""
17 -- the per-domain temporal model: fit, generate, and compare.

This is the model of docs/MODEL_V2.md section 2:

  * interval processes for condition and drug (start + duration)
  * point processes for measurement, observation and procedure
  * an absorbing death process
  * one shared patient heterogeneity term u_i ~ Gamma(k)
  * a rank-K cross-domain coupling exp(z_i . beta_j), which is RECORDING
    DEPENDENCE, not clinical or causal dependence

It reports, in one place and at the same level of prominence:
  * the level check (generated totals against the source totals)
  * prevalence / incidence / degree / co-occurrence agreement
  * the whole-CDM spectrum (the diagnostic that disagrees with the summaries)
  * the interval tables the source leaves empty (condition_era, drug_era)

Run: MPLCONFIGDIR=./.mplconfig .venv/bin/python experiments/17_perdomain.py
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

from topocdm import perdomain as pdm
from topocdm.evaluate import compare, prevalence
from topocdm.model import CDMIncidence

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
        "grid.alpha": 0.22,
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
    "purple": "#6B5B95",
}


def _matrix_for(records, periods, concept_ids):
    """Build an N x C matrix on a fixed concept set."""
    persons = np.sort(periods["person_id"].unique())
    prow = {int(p): i for i, p in enumerate(persons)}
    ccol = {int(c): j for j, c in enumerate(concept_ids)}
    ev = records[records["concept_id"].isin(set(concept_ids))]
    rows = ev["person_id"].map(prow).to_numpy(dtype=np.int64)
    cols = ev["concept_id"].map(ccol).to_numpy(dtype=np.int64)
    from scipy import sparse

    X = sparse.csr_matrix((np.ones(len(ev)), (rows, cols)), shape=(len(persons), len(concept_ids)))
    X.sum_duplicates()
    return X


def main():
    t0 = time.time()
    events, periods = pdm.load_extract()
    print(f"loaded {len(events):,} records, {len(periods):,} persons")

    fit = pdm.fit_perdomain(events, periods, K=20)
    print(
        f"fitted in {time.time() - t0:.1f}s: k={fit.k:.2f}, "
        f"propensity_tmean={fit.propensity_tmean:.2f}, "
        f"death={fit.death['frac']:.3f}, n_params={fit.n_parameters:,}"
    )

    # ---- generate a cohort on resampled follow-up ----
    rng = np.random.default_rng(1)
    syn_periods = pdm.sample_periods(periods, len(periods), rng)
    tables = pdm.generate_cohort(fit, len(syn_periods), syn_periods, seed=2)

    # ---- level check (before any interpretation) ----
    level = {}
    for table, dom in [
        ("condition_occurrence", "condition"),
        ("drug_exposure", "drug"),
        ("procedure_occurrence", "procedure"),
        ("measurement", "measurement"),
        ("observation", "observation"),
        ("visit_occurrence", "visit"),
    ]:
        gen = len(tables[table])
        obs = fit.n_events[dom]
        level[dom] = {"observed": obs, "generated": gen, "ratio": gen / max(obs, 1)}
    level["death"] = {
        "observed": fit.meta["n_deaths"],
        "generated": len(tables["death"]),
        "ratio": len(tables["death"]) / max(fit.meta["n_deaths"], 1),
    }
    print("\nlevel check (generated / observed):")
    for d, v in level.items():
        print(f"  {d:12s} {v['generated']:>9,} / {v['observed']:>9,}  ratio {v['ratio']:.2f}")

    # ---- summary comparison on the 773-concept clinical matrix ----
    X_src, concept_ids, persons, T_src = pdm.clinical_incidence(events, periods)
    syn_records = pdm.tables_to_records(tables)
    X_syn = _matrix_for(syn_records, syn_periods, concept_ids)
    T_syn = np.maximum(
        (syn_periods["end_date"] - syn_periods["start_date"]).dt.days.to_numpy(float) / 365.25,
        1 / 365.25,
    )

    src = CDMIncidence(
        X=X_src.toarray(), T=T_src, concept_group=np.zeros(len(concept_ids), dtype=int)
    )
    syn = CDMIncidence(
        X=X_syn.toarray(), T=T_syn, concept_group=np.zeros(len(concept_ids), dtype=int)
    )
    stats = compare(src, syn)

    # ---- spectral comparison (degree-normalised incidence) ----
    s_src = pdm.spectrum(X_src)
    s_syn = pdm.spectrum(X_syn)
    ks = pdm.spectral_ks(s_src, s_syn)
    rank_src = pdm.effective_rank(s_src)
    rank_syn = pdm.effective_rank(s_syn)
    print(
        f"\nsummary: prevalence r={stats['prevalence']['r']:.3f} "
        f"(slope {stats['prevalence']['slope']:.2f}), "
        f"incidence r={stats['incidence']['r']:.3f}, "
        f"degree L1={stats['degree_l1']:.3f}, "
        f"co-occurrence r={stats['cooccurrence_r']:.3f}"
    )
    print(f"spectrum: KS={ks:.3f}, rank>0.1 source {rank_src} / synthetic {rank_syn}")

    # ---- interval tables the source leaves empty ----
    era = {}
    for name, scol, ecol in (
        ("condition_era", "condition_era_start_date", "condition_era_end_date"),
        ("drug_era", "drug_era_start_date", "drug_era_end_date"),
    ):
        df = tables[name]
        era[name] = {
            "rows": int(len(df)),
            "median_days": float(
                (pd.to_datetime(df[ecol]) - pd.to_datetime(df[scol])).dt.days.median()
            )
            if len(df)
            else 0.0,
        }

    # ---- persist ----
    out = {
        "fitted": {
            "k": float(fit.k),
            "propensity_tmean": float(fit.propensity_tmean),
            "death_frac": float(fit.death["frac"]),
            "n_parameters": int(fit.n_parameters),
            "K": int(fit.meta["K"]),
            "n_atoms": int(fit.n_concepts),
        },
        "level": level,
        "summary": stats,
        "spectrum": {
            "ks": ks,
            "rank_source": rank_src,
            "rank_synthetic": rank_syn,
            "top10_source": s_src[:10].tolist(),
            "top10_synthetic": s_syn[:10].tolist(),
        },
        "era": era,
        "durations": {
            d: {
                "pi0": float(fit.duration[d]["pi0"]),
                "mu": fit.duration[d]["mu"].tolist(),
                "s": fit.duration[d]["s"].tolist(),
                "observed_median": fit.duration[d]["observed_median"],
            }
            for d in pdm.INTERVAL_DOMAINS
        },
    }
    json.dump(
        out,
        open(f"{RES}/perdomain.json", "w"),
        indent=1,
        default=lambda o: float(o) if isinstance(o, np.floating) else str(o),
    )

    # ---- figure: levels, summary, spectrum all side by side ----
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))

    ax = axes[0]
    doms = list(level)
    ratios = [level[d]["ratio"] for d in doms]
    ax.bar(
        doms, ratios, color=[C["green"] if 0.9 <= r <= 1.1 else C["red"] for r in ratios], width=0.6
    )
    ax.axhline(1.0, color="k", ls=":", lw=0.9)
    ax.set_ylim(0, max(1.35, max(ratios) * 1.05))
    ax.set_ylabel("generated / source")
    ax.set_title("Level check first", fontsize=9)
    ax.tick_params(axis="x", rotation=35)

    ax = axes[1]
    pv = prevalence(src)
    pvs = prevalence(syn)
    ax.scatter(pv, pvs, s=8, alpha=0.5, color=C["blue"], edgecolor="none")
    lim = [0, max(pv.max(), pvs.max()) * 1.05]
    ax.plot(lim, lim, "k:", lw=0.8)
    ax.set_xlabel("source prevalence")
    ax.set_ylabel("synthetic prevalence")
    ax.set_title(
        f"Prevalence r={stats['prevalence']['r']:.2f}; "
        f"co-occurrence r={stats['cooccurrence_r']:.2f}",
        fontsize=9,
    )

    ax = axes[2]
    k = np.arange(1, 41)
    ax.plot(k, s_src[:40], "o-", ms=2.8, color=C["blue"], label="source")
    ax.plot(k, s_syn[:40], "s-", ms=2.8, color=C["orange"], label="synthetic")
    ax.set_xlabel("rank")
    ax.set_ylabel("singular value")
    ax.set_title(f"Spectrum KS={ks:.2f}; rank {rank_src} vs {rank_syn}", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(f"{FIG}/fig27_perdomain.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {FIG}/fig27_perdomain.png and {RES}/perdomain.json")
    print(
        f"interval tables: condition_era {era['condition_era']['rows']:,}, "
        f"drug_era {era['drug_era']['rows']:,} (source: empty)"
    )


if __name__ == "__main__":
    main()
