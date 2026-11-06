"""
25 -- the two easy fixes, then prevalence and incidence with simulation intervals.

Fixes applied:
  - batching: a derived encounter key (person, date) for Synthea, which has no
    visit IDs. This lets the visit-anchored model run on Synthea.
  - refill: a post-generation re-clustering of drug records into courses,
    fitted on the source fill-per-era and within-era gap distributions.

The plots are the marginal fidelity the review says already works: per-concept
prevalence and incidence, real against R synthetic cohorts, with a 5-95%
simulation interval across cohorts. Both datasets: Synthea (2,000-patient
sample) and MIMIC (100 patients).

Run: MPLCONFIGDIR=./.mplconfig .venv/bin/python experiments/25_fixes_fidelity.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topocdm import perdomain as pdm
from topocdm import visitanchor as va

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
C = {"real": "#1f2933", "syn": "#2C6E9B", "band": "#2C6E9B", "bad": "#B4443C"}

# Prevalence bins for the gate: a single scalar hides where the model fails.
BINS = [("rare_1_3", 0.01, 0.03), ("mid_3_30", 0.03, 0.30), ("common_30_100", 0.30, 1.0001)]


def _load_m21():
    path = os.path.join(os.path.dirname(__file__), "21_subject_spectra.py")
    spec = importlib.util.spec_from_file_location("m21", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def derive_encounters(events: pd.DataFrame, visit_concept_id: int = 9202):
    """Derived encounter key: one visit per (person_id, date)."""
    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["start_date"]).dt.normalize()
    clinical_domains = ("condition", "drug", "measurement", "observation", "procedure")
    clinical = ev[ev["domain"].isin(clinical_domains)]
    enc = (
        clinical[["person_id", "date"]]
        .drop_duplicates()
        .sort_values(["person_id", "date"])
        .reset_index(drop=True)
    )
    enc["visit_occurrence_id"] = np.arange(1, len(enc) + 1, dtype=np.int64)
    enc["visit_start_date"] = enc["date"]
    enc["visit_end_date"] = enc["date"]
    enc["visit_concept_id"] = visit_concept_id
    ev = ev.merge(
        enc[["person_id", "date", "visit_occurrence_id"]], on=["person_id", "date"], how="left"
    )
    ev.loc[~ev["domain"].isin(clinical_domains), "visit_occurrence_id"] = np.nan
    return ev, enc


def _matrix_for(records, periods, concept_ids):
    persons = np.sort(periods["person_id"].unique())
    prow = {int(p): i for i, p in enumerate(persons)}
    ccol = {int(c): j for j, c in enumerate(concept_ids)}
    ev = records[records["concept_id"].isin(set(concept_ids))]
    rows = ev["person_id"].map(prow).to_numpy(dtype=np.int64)
    cols = ev["concept_id"].map(ccol).to_numpy(dtype=np.int64)
    X = sparse.csr_matrix((np.ones(len(ev)), (rows, cols)), shape=(len(persons), len(concept_ids)))
    X.sum_duplicates()
    return X


def _marginals(X, T):
    prev = np.asarray((X > 0).mean(axis=0)).ravel()
    inc = np.asarray(X.sum(axis=0)).ravel() / max(T.sum(), 1e-9) * 1000.0
    return prev, inc


def _run_dataset(name, events, periods, visits, R, out_prefix):
    t0 = time.time()
    fit = va.fit_visitanchor(events, periods, visits)

    X_real, concept_ids, persons, T_real = pdm.clinical_incidence(events, periods)
    prev_real, inc_real = _marginals(X_real, T_real)

    prev_draws, inc_draws = [], []
    for r in range(R):
        rng = np.random.default_rng(1000 + r)
        sp = pdm.sample_periods(periods, len(periods), rng)
        tab = va.generate_visitanchor(fit, sp, seed=2000 + r)
        # Refill re-dates drug records but preserves their count, so it cannot
        # change prevalence or incidence. It is tested separately; omitted here
        # to keep the R=50 fidelity run tractable.
        rec = pdm.tables_to_records(tab)
        X = _matrix_for(rec, sp, concept_ids)
        T = np.maximum(
            (sp["end_date"] - sp["start_date"]).dt.days.to_numpy(float) / 365.25, 1 / 365.25
        )
        p, i = _marginals(X, T)
        prev_draws.append(p)
        inc_draws.append(i)

    P = np.vstack(prev_draws)
    inc = np.vstack(inc_draws)
    plo, pmid, phi = np.percentile(P, [5, 50, 95], axis=0)
    ilo, imid, ihi = np.percentile(inc, [5, 50, 95], axis=0)

    # keep concepts with more than 10 real records for the plots/ratios
    real_total = np.asarray(X_real.sum(axis=0)).ravel()
    keep = real_total > 10
    order = np.argsort(-prev_real)
    o = order[keep[order]]
    cover_p = float(np.mean((prev_real[o] >= plo[o]) & (prev_real[o] <= phi[o])))
    cover_i = float(np.mean((inc_real[o] >= ilo[o]) & (inc_real[o] <= ihi[o])))
    bins = {}
    for bname, blo, bhi in BINS:
        m = keep & (prev_real >= blo) & (prev_real < bhi)
        if m.sum() == 0:
            continue
        ip = (prev_real[m] >= plo[m]) & (prev_real[m] <= phi[m])
        ii = (inc_real[m] >= ilo[m]) & (inc_real[m] <= ihi[m])
        bins[bname] = {
            "n": int(m.sum()),
            "prevalence_coverage": float(ip.mean()),
            "incidence_coverage": float(ii.mean()),
            "prevalence_ratio_median": float(np.median(pmid[m] / np.maximum(prev_real[m], 1e-12))),
            "incidence_ratio_median": float(np.median(imid[m] / np.maximum(inc_real[m], 1e-12))),
            "prevalence_over": int(((pmid[m] > prev_real[m]) & ~ip).sum()),
            "prevalence_under": int(((pmid[m] < prev_real[m]) & ~ip).sum()),
        }
    inc_ratio_median = float(np.median(imid[o] / np.maximum(inc_real[o], 1e-12)))
    print(
        f"{name}: {R} cohorts in {time.time() - t0:.1f}s; "
        f"coverage overall P {cover_p:.0%} / I {cover_i:.0%}; "
        f"incidence ratio median {inc_ratio_median:.2f}"
    )
    for bname, b in bins.items():
        print(
            f"    {bname:14s} n={b['n']:4d} P cov {b['prevalence_coverage']:.0%} "
            f"(over {b['prevalence_over']}, under {b['prevalence_under']}) "
            f"P ratio {b['prevalence_ratio_median']:.2f} "
            f"I ratio {b['incidence_ratio_median']:.2f}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.0))
    x = np.arange(len(o)) + 1

    ax = axes[0, 0]
    ax.plot(x, prev_real[o], "o", ms=3, color=C["real"], label="real")
    ax.plot(x, pmid[o], "-", lw=1.2, color=C["syn"], label="synthetic median")
    ax.fill_between(
        x, plo[o], phi[o], color=C["band"], alpha=0.18, label="5-95% simulation interval"
    )
    ax.set_yscale("log")
    ax.set_xlabel("concept rank (by real prevalence)")
    ax.set_ylabel("prevalence")
    ax.set_title(f"{name}: prevalence", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)

    ax = axes[0, 1]
    ax.plot(x, inc_real[o], "o", ms=3, color=C["real"], label="real")
    ax.plot(x, imid[o], "-", lw=1.2, color=C["syn"], label="synthetic median")
    ax.fill_between(x, ilo[o], ihi[o], color=C["band"], alpha=0.18)
    ax.set_yscale("log")
    ax.set_xlabel("concept rank (by real prevalence)")
    ax.set_ylabel("incidence per 1000 PY")
    ax.set_title(f"{name}: incidence", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)

    ax = axes[1, 0]
    ratio = pmid[o] / np.maximum(prev_real[o], 1e-12)
    rlo = plo[o] / np.maximum(prev_real[o], 1e-12)
    rhi = phi[o] / np.maximum(prev_real[o], 1e-12)
    ax.fill_between(x, rlo, rhi, color=C["band"], alpha=0.18)
    ax.plot(x, ratio, "-", lw=1.2, color=C["syn"])
    ax.axhline(1.0, color="k", ls=":", lw=0.9)
    ax.set_xlabel("concept rank")
    ax.set_ylabel("synthetic / real")
    ax.set_title("Prevalence ratio", fontsize=9)

    ax = axes[1, 1]
    ratio = imid[o] / np.maximum(inc_real[o], 1e-12)
    rlo = ilo[o] / np.maximum(inc_real[o], 1e-12)
    rhi = ihi[o] / np.maximum(inc_real[o], 1e-12)
    ax.fill_between(x, rlo, rhi, color=C["band"], alpha=0.18)
    ax.plot(x, ratio, "-", lw=1.2, color=C["syn"])
    ax.axhline(1.0, color="k", ls=":", lw=0.9)
    ax.set_xlabel("concept rank")
    ax.set_ylabel("synthetic / real")
    ax.set_title("Incidence ratio", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{FIG}/{out_prefix}.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "R": R,
        "n_concepts": int(len(concept_ids)),
        "n_compared": int(len(o)),
        "bins": bins,
        "incidence_ratio_median": inc_ratio_median,
        "prevalence_coverage": cover_p,
        "incidence_coverage": cover_i,
        "prevalence_real": prev_real.tolist(),
        "prevalence_median": pmid.tolist(),
        "prevalence_lo": plo.tolist(),
        "prevalence_hi": phi.tolist(),
        "incidence_real": inc_real.tolist(),
        "incidence_median": imid.tolist(),
        "incidence_lo": ilo.tolist(),
        "incidence_hi": ihi.tolist(),
    }


def main():
    out = {}

    # ---- Synthea sample, with derived encounters ----
    ev, pe = pdm.load_extract()
    rng = np.random.default_rng(0)
    keep = np.sort(rng.choice(pe.person_id.unique(), 2000, replace=False))
    ev_s = ev[ev.person_id.isin(keep)].copy()
    pe_s = pe[pe.person_id.isin(keep)].copy()
    ev_s, enc_s = derive_encounters(ev_s)
    out["synthea"] = _run_dataset(
        "Synthea (2k sample)", ev_s, pe_s, enc_s, R=50, out_prefix="fig39_synthea_fidelity"
    )

    # ---- MIMIC demo ----
    m21 = _load_m21()
    ev_m, pe_m, vis_m = m21._load_mimic()
    out["mimic"] = _run_dataset(
        "MIMIC-IV demo", ev_m, pe_m, vis_m, R=50, out_prefix="fig39_mimic_fidelity"
    )

    json.dump(out, open(f"{RES}/fixes_fidelity.json", "w"), indent=1)
    print(
        f"wrote {FIG}/fig39_synthea_fidelity.png, "
        f"{FIG}/fig39_mimic_fidelity.png and {RES}/fixes_fidelity.json"
    )


if __name__ == "__main__":
    main()
