"""
22 -- the same subject spectra for the Synthea slice.

The object is identical to experiment 21: the normalised Laplacian spectrum of
one patient's condition-drug encounter graph (same-day co-occurrence).
`visit_occurrence_id` is empty in the Synthea load, so the encounter is the
date, which is how the MIMIC spectra were built too.

Three sources are compared:

    MIMIC patients            observed de-identified EHR (ICU)
    topocdm synthetic         visit-anchored model fitted to MIMIC
    Synthea patients          a rule-based patient simulator

Run: MPLCONFIGDIR=./.mplconfig .venv/bin/python experiments/22_synthea_spectra.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topocdm import perdomain as pdm
from topocdm import visitanchor as va

FIG, RES = "figs", "results"
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)
plt.rcParams.update({"figure.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.2,
                     "grid.linewidth": 0.5})
C = dict(mimic="#2C6E9B", synthea="#4E8B4A", synthetic="#C1440E",
         ink="#1f2933", grey="#9aa5b1")


def _load_m21():
    path = os.path.join(os.path.dirname(__file__), "21_subject_spectra.py")
    spec = importlib.util.spec_from_file_location("m21", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scalars(d):
    return dict(n_nodes=np.array([len(e) for e in d.values()]),
                lambda2=np.array([e[1] for e in d.values() if len(e) > 1]),
                lambda_max=np.array([e[-1] for e in d.values()]))


def _summary(d):
    s = _scalars(d)
    return dict(n=int(len(d)),
                nodes=float(np.median(s["n_nodes"])),
                lambda2=float(np.median(s["lambda2"])),
                lambda_max=float(np.median(s["lambda_max"])))


def main():
    m21 = _load_m21()
    rng = np.random.default_rng(0)

    # ---- Synthea ----
    ev_s, pe_s = pdm.load_extract()
    syn_s = m21.subject_spectra(ev_s)
    print(f"Synthea: {len(syn_s)} subject graphs of "
          f"{ev_s.person_id.nunique()} patients")

    # ---- MIMIC and the visit-anchored synthetic ----
    ev_m, pe_m, visits = m21._load_mimic()
    fit = va.fit_visitanchor(ev_m, pe_m, visits)
    sp = pdm.sample_periods(pe_m, len(pe_m), np.random.default_rng(1))
    tables = va.generate_visitanchor(fit, sp, seed=2)
    syn_m = m21.subject_spectra(ev_m)
    syn_t = m21.subject_spectra(pdm.tables_to_records(tables))
    print(f"MIMIC: {len(syn_m)}; topocdm synthetic: {len(syn_t)}")

    sources = [("MIMIC (observed)", syn_m, C["mimic"]),
               ("topocdm synthetic", syn_t, C["synthetic"]),
               ("Synthea (simulator)", syn_s, C["synthea"])]
    for name, d, _ in sources:
        s = _summary(d)
        print(f"  {name:20s} n={s['n']:4d} nodes {s['nodes']:5.1f} "
              f"l2 {s['lambda2']:.3f} lmax {s['lambda_max']:.3f}")

    # ---- examples: Synthea vs MIMIC ----
    ex_s = m21._pick(syn_s, 4, np.random.default_rng(11), lo=5, hi=100)
    ex_m = m21._pick(syn_m, 4, np.random.default_rng(7), lo=30, hi=250)
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 5.0), sharey=True)
    for row, (examples, col, lab) in enumerate([
            (ex_s, C["synthea"], "Synthea"),
            (ex_m, C["mimic"], "MIMIC")]):
        for j, (pid, ev) in enumerate(examples):
            ax = axes[row, j]
            x = np.linspace(0, 1, len(ev))
            ax.plot(x, ev, color=col, lw=1.4)
            ax.set_title(f"{lab} p{pid}\nn={len(ev)}", fontsize=8)
            ax.set_xlabel("rank fraction")
            if j == 0:
                ax.set_ylabel("normalised Laplacian eigenvalue")
    fig.suptitle("Condition-drug subject spectra: simulator against observed "
                 "data", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig33_synthea_subject_spectra_examples.png",
                bbox_inches="tight")
    plt.close(fig)

    # ---- ensemble: three sources ----
    q = np.linspace(0, 1, 50)
    interp = {name: np.array([m21._interp(e, q) for e in d.values()])
              for name, d, _ in sources}
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))

    ax = axes[0]
    for name, d, col in sources:
        arr = interp[name]
        ax.plot(q, arr.mean(axis=0), color=col, lw=1.5, label=name)
        ax.fill_between(q, np.percentile(arr, 5, axis=0),
                        np.percentile(arr, 95, axis=0), color=col, alpha=0.13)
    ax.set_xlabel("rank fraction")
    ax.set_ylabel("normalised Laplacian eigenvalue")
    ax.set_title("A. Mean spectrum", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)

    ax = axes[1]
    bins = np.linspace(0, 1.0, 25)
    for name, d, col in sources:
        x = _scalars(d)["lambda2"]
        ax.hist(x, bins=bins, density=True, alpha=0.45, color=col, label=name)
    ax.set_xlabel(r"algebraic connectivity $\lambda_2$")
    ax.set_ylabel("density")
    ax.set_title("B. Second eigenvalue", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)

    ax = axes[2]
    bins = np.linspace(0.9, 2.02, 25)
    for name, d, col in sources:
        x = _scalars(d)["lambda_max"]
        ax.hist(x, bins=bins, density=True, alpha=0.45, color=col, label=name)
    ax.set_xlabel(r"largest eigenvalue $\lambda_n$")
    ax.set_ylabel("density")
    ax.set_title("C. Largest eigenvalue", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)

    fig.tight_layout()
    fig.savefig(f"{FIG}/fig34_subject_spectra_three_sources.png",
                bbox_inches="tight")
    plt.close(fig)

    # ---- summary JSON ----
    def dist(a, b):
        A = np.concatenate(list(a.values()))
        B = np.concatenate(list(b.values()))
        return float(stats.ks_2samp(A, B).statistic)

    out = dict(
        sources={name: _summary(d) for name, d, _ in sources},
        ks=dict(synthea_vs_mimic=dist(syn_s, syn_m),
                synthea_vs_synthetic=dist(syn_s, syn_t),
                mimic_vs_synthetic=dist(syn_m, syn_t)),
        mean_spectra={name: interp[name].mean(axis=0).tolist()
                      for name, _, _ in sources},
    )
    json.dump(out, open(f"{RES}/synthea_subject_spectra.json", "w"), indent=1)
    print(json.dumps(out["ks"], indent=1))
    print(f"wrote {FIG}/fig33_synthea_subject_spectra_examples.png, "
          f"{FIG}/fig34_subject_spectra_three_sources.png and "
          f"{RES}/synthea_subject_spectra.json")


if __name__ == "__main__":
    main()
