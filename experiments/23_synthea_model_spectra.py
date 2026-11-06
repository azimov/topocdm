"""
23 -- model fitted to Synthea, synthetic subject spectra vs Synthea.

This is the apples-to-apples comparison the earlier figure was not: the model
is fitted to the Synthea slice, a synthetic cohort is generated from that fit,
and the per-subject spectra of the generated patients are compared with those
of the Synthea patients themselves.

Object: the normalised Laplacian spectrum of one patient's condition-drug
same-day co-occurrence graph (as in experiments 21 and 22).

Synthea has no visit_occurrence_id, so the visit-anchored model cannot be
fitted without inventing a visit. This uses the per-domain model, which is the
model that fits this database directly.

Run: MPLCONFIGDIR=./.mplconfig .venv/bin/python experiments/23_synthea_model_spectra.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topocdm import perdomain as pdm

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
C = {"real": "#4E8B4A", "synthetic": "#C1440E", "ink": "#1f2933"}


def _load_m21():
    path = os.path.join(os.path.dirname(__file__), "21_subject_spectra.py")
    spec = importlib.util.spec_from_file_location("m21", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scalars(d):
    return {
        "n_nodes": np.array([len(e) for e in d.values()]),
        "lambda2": np.array([e[1] for e in d.values() if len(e) > 1]),
        "lambda_max": np.array([e[-1] for e in d.values()]),
    }


def _summary(d):
    s = _scalars(d)
    return {
        "n": int(len(d)),
        "nodes": float(np.median(s["n_nodes"])),
        "lambda2": float(np.median(s["lambda2"])),
        "lambda_max": float(np.median(s["lambda_max"])),
    }


def main():
    t0 = time.time()
    m21 = _load_m21()
    events, periods = pdm.load_extract()
    print(f"Synthea: {len(events):,} records, {periods.person_id.nunique():,} patients")

    fit = pdm.fit_perdomain(events, periods, K=20)
    print(f"fitted in {time.time() - t0:.1f}s: k={fit.k:.2f}, n_params={fit.n_parameters:,}")

    sp = pdm.sample_periods(periods, len(periods), np.random.default_rng(1))
    tables = pdm.generate_cohort(fit, len(sp), sp, seed=2)
    synthetic = pdm.tables_to_records(tables)
    print(f"generated {len(synthetic):,} records for {len(sp):,} patients")

    real = m21.subject_spectra(events)
    syn = m21.subject_spectra(synthetic)
    n_pat = periods.person_id.nunique()
    cov_r, cov_s = len(real) / n_pat, len(syn) / n_pat
    print(
        f"subject spectra: {len(real):,} Synthea ({cov_r:.1%}), "
        f"{len(syn):,} synthetic ({cov_s:.1%})"
    )

    rs, ss = _summary(real), _summary(syn)
    for name, s in [("Synthea", rs), ("synthetic", ss)]:
        print(
            f"  {name:10s} n={s['n']:5d} nodes {s['nodes']:5.1f} "
            f"l2 {s['lambda2']:.3f} lmax {s['lambda_max']:.3f}"
        )

    q = np.linspace(0, 1, 50)
    R = np.array([m21._interp(e, q) for e in real.values()])
    S = np.array([m21._interp(e, q) for e in syn.values()])
    l2_dist = float(np.sqrt(np.mean((R.mean(0) - S.mean(0)) ** 2)))
    ks = float(
        stats.ks_2samp(
            np.concatenate(list(real.values())), np.concatenate(list(syn.values()))
        ).statistic
    )
    print(f"  mean-spectrum L2 {l2_dist:.3f}; pooled KS {ks:.3f}")

    # ---- examples: Synthea vs its own model's synthetic ----
    ex_r = m21._pick(real, 4, np.random.default_rng(11), lo=5, hi=100)
    ex_s = m21._pick(syn, 4, np.random.default_rng(12), lo=5, hi=100)
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 5.0), sharey=True)
    for row, (examples, col, lab) in enumerate(
        [(ex_r, C["real"], "Synthea"), (ex_s, C["synthetic"], "topocdm synthetic (fit to Synthea)")]
    ):
        for j, (pid, ev) in enumerate(examples):
            ax = axes[row, j]
            x = np.linspace(0, 1, len(ev))
            ax.plot(x, ev, color=col, lw=1.4)
            ax.set_title(f"{lab}\np{pid}, n={len(ev)}", fontsize=8)
            ax.set_xlabel("rank fraction")
            if j == 0:
                ax.set_ylabel("normalised Laplacian eigenvalue")
    fig.suptitle("Subject spectra, same dataset: Synthea vs a model fitted to Synthea", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig35_synthea_model_spectra_examples.png", bbox_inches="tight")
    plt.close(fig)

    # ---- ensemble ----
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))
    ax = axes[0]
    for arr, col, lab in [
        (R, C["real"], f"Synthea (n={len(R):,})"),
        (S, C["synthetic"], f"synthetic (n={len(S):,})"),
    ]:
        ax.plot(q, arr.mean(0), color=col, lw=1.5, label=lab)
        ax.fill_between(
            q, np.percentile(arr, 5, axis=0), np.percentile(arr, 95, axis=0), color=col, alpha=0.15
        )
    ax.set_xlabel("rank fraction")
    ax.set_ylabel("normalised Laplacian eigenvalue")
    ax.set_title(
        f"A. Mean spectrum, conditional on a graph\n(L2 {l2_dist:.2f}, KS {ks:.2f})", fontsize=9
    )
    ax.legend(fontsize=6.5, frameon=False)

    ax = axes[1]
    bins = np.linspace(0, 1.0, 25)
    for arr, col, lab in [
        (_scalars(real)["lambda2"], C["real"], "Synthea"),
        (_scalars(syn)["lambda2"], C["synthetic"], "synthetic"),
    ]:
        ax.hist(arr, bins=bins, density=True, alpha=0.5, color=col, label=lab)
    ax.set_xlabel(r"algebraic connectivity $\lambda_2$")
    ax.set_ylabel("density")
    ax.set_title("B. Second eigenvalue", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)

    ax = axes[2]
    vals = [cov_r * 100, cov_s * 100]
    ax.bar(["Synthea", "synthetic"], vals, color=[C["real"], C["synthetic"]], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("patients with a subject graph (%)")
    ax.set_title("C. Coverage", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{FIG}/fig36_synthea_model_spectra_ensemble.png", bbox_inches="tight")
    plt.close(fig)

    json.dump(
        {
            "real": rs,
            "synthetic": ss,
            "l2_mean_spectrum": l2_dist,
            "ks_pooled": ks,
            "n_patients": int(n_pat),
            "coverage_real": float(cov_r),
            "coverage_syn": float(cov_s),
            "mean_spectrum_real": R.mean(0).tolist(),
            "mean_spectrum_syn": S.mean(0).tolist(),
            "fitted": {
                "k": float(fit.k),
                "n_parameters": int(fit.n_parameters),
                "K": int(fit.meta["K"]),
            },
        },
        open(f"{RES}/synthea_model_spectra.json", "w"),
        indent=1,
    )
    print(
        f"wrote {FIG}/fig35_synthea_model_spectra_examples.png, "
        f"{FIG}/fig36_synthea_model_spectra_ensemble.png and "
        f"{RES}/synthea_model_spectra.json"
    )


if __name__ == "__main__":
    main()
