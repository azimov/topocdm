"""
24 -- per-subject spectra over ALL clinical concepts.

This is the object as originally specified: a patient's same-day co-occurrence
graph over every clinical concept they have (condition, drug, measurement,
observation, procedure), normalised Laplacian spectrum.

The condition-drug restriction in experiments 21-23 was a derived diagnostic to
avoid a near-complete graph; it was not the requested object. This restores the
full concept set and shows what the spectra actually look like, for both
datasets, real versus synthetic from a model fitted to the same dataset.

Run: MPLCONFIGDIR=./.mplconfig .venv/bin/python experiments/24_subject_spectra_all.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np

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
C = {
    "mimic": "#2C6E9B",
    "mimic_syn": "#C1440E",
    "synthea": "#4E8B4A",
    "synthea_syn": "#8A6D3B",
    "ink": "#1f2933",
}


def _load_m21():
    path = os.path.join(os.path.dirname(__file__), "21_subject_spectra.py")
    spec = importlib.util.spec_from_file_location("m21", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # all clinical concepts: the object as originally specified
    mod.GRAPH_DOMAINS = ("condition", "drug", "measurement", "observation", "procedure")
    return mod


def _summary(d):
    n = np.array([len(e) for e in d.values()])
    l2 = np.array([e[1] for e in d.values() if len(e) > 1])
    lm = np.array([e[-1] for e in d.values()])
    near = np.array([(np.abs(e - 1) < 0.1).mean() for e in d.values()])
    return {
        "n": int(len(d)),
        "nodes": float(np.median(n)),
        "lambda2": float(np.median(l2)),
        "lambda_max": float(np.median(lm)),
        "frac_near_1": float(np.median(near)),
    }


def _examples(ax_row, examples, col, lab):
    for j, (pid, ev) in enumerate(examples):
        ax = ax_row[j]
        ax.plot(np.linspace(0, 1, len(ev)), ev, color=col, lw=1.4)
        ax.set_title(f"{lab}\np{pid}, n={len(ev)}", fontsize=8)
        ax.set_xlabel("rank fraction")
        if j == 0:
            ax.set_ylabel("normalised Laplacian eigenvalue")


def main():
    t0 = time.time()
    m21 = _load_m21()

    # ---- MIMIC: observed vs visit-anchored synthetic ----
    ev_m, pe_m, visits = m21._load_mimic()
    fit_m = va.fit_visitanchor(ev_m, pe_m, visits)
    sp_m = pdm.sample_periods(pe_m, len(pe_m), np.random.default_rng(1))
    tab_m = va.generate_visitanchor(fit_m, sp_m, seed=2)
    R_m = m21.subject_spectra(ev_m)
    S_m = m21.subject_spectra(pdm.tables_to_records(tab_m))

    # ---- Synthea: observed vs per-domain synthetic ----
    ev_s, pe_s = pdm.load_extract()
    fit_s = pdm.fit_perdomain(ev_s, pe_s, K=20)
    sp_s = pdm.sample_periods(pe_s, len(pe_s), np.random.default_rng(3))
    tab_s = pdm.generate_cohort(fit_s, len(sp_s), sp_s, seed=4)
    R_s = m21.subject_spectra(ev_s)
    S_s = m21.subject_spectra(pdm.tables_to_records(tab_s))
    print(f"built in {time.time() - t0:.1f}s")

    sources = [
        ("MIMIC observed", R_m, C["mimic"]),
        ("MIMIC synthetic", S_m, C["mimic_syn"]),
        ("Synthea observed", R_s, C["synthea"]),
        ("Synthea synthetic", S_s, C["synthea_syn"]),
    ]
    for name, d, _ in sources:
        s = _summary(d)
        print(
            f"  {name:20s} n={s['n']:6d} nodes {s['nodes']:6.1f} "
            f"l2 {s['lambda2']:.3f} lmax {s['lambda_max']:.3f} "
            f"frac|e-1|<0.1 {s['frac_near_1']:.3f}"
        )

    # ---- example lines, one figure per dataset ----
    for dataset, real, syn, col_r, col_s in [
        ("mimic", R_m, S_m, C["mimic"], C["mimic_syn"]),
        ("synthea", R_s, S_s, C["synthea"], C["synthea_syn"]),
    ]:
        ex_r = m21._pick(real, 4, np.random.default_rng(11), lo=10, hi=600)
        ex_s = m21._pick(syn, 4, np.random.default_rng(12), lo=10, hi=600)
        fig, axes = plt.subplots(2, 4, figsize=(12.0, 5.0), sharey=True)
        _examples(axes[0], ex_r, col_r, f"{dataset} observed")
        _examples(axes[1], ex_s, col_s, f"{dataset} synthetic")
        fig.suptitle(f"All-concept subject spectra: {dataset}", fontsize=10)
        fig.tight_layout()
        fig.savefig(f"{FIG}/fig37_{dataset}_all_subject_spectra.png", bbox_inches="tight")
        plt.close(fig)

    # ---- ensemble: full spectrum + tail zooms ----
    q = np.linspace(0, 1, 60)
    interp = {name: np.array([m21._interp(e, q) for e in d.values()]) for name, d, _ in sources}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for name, _, col in sources:
        m = interp[name].mean(axis=0)
        axes[0].plot(q, m, color=col, lw=1.4, label=name)
    axes[0].set_xlabel("rank fraction")
    axes[0].set_ylabel("normalised Laplacian eigenvalue")
    axes[0].set_title("A. Full spectrum (flat near 1)", fontsize=9)
    axes[0].legend(fontsize=6, frameon=False)

    for ax, lo, hi, title in [
        (axes[1], 0.0, 0.15, "B. Low tail"),
        (axes[2], 0.85, 1.0, "C. High tail"),
    ]:
        m = (q >= lo) & (q <= hi)
        for name, _, col in sources:
            arr = np.percentile(interp[name], [5, 50, 95], axis=0)
            ax.plot(q[m], arr[1][m], color=col, lw=1.4, label=name)
            ax.fill_between(q[m], arr[0][m], arr[2][m], color=col, alpha=0.12)
        ax.set_xlabel("rank fraction")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig38_all_subject_spectra_ensemble.png", bbox_inches="tight")
    plt.close(fig)

    out = {
        "sources": {name: _summary(d) for name, d, _ in sources},
        "mean_spectra": {name: interp[name].mean(axis=0).tolist() for name, _, _ in sources},
    }
    json.dump(out, open(f"{RES}/all_subject_spectra.json", "w"), indent=1)
    print(
        f"wrote {FIG}/fig37_*_all_subject_spectra.png, "
        f"{FIG}/fig38_all_subject_spectra_ensemble.png and "
        f"{RES}/all_subject_spectra.json"
    )


if __name__ == "__main__":
    main()
