"""
03 -- does the streaming fit scale, and does subsampling change the answer?

Two claims to test:

  1. A CDM with millions of events can be fitted without materialising the
     patient x concept matrix, in time and memory that fit in one process.
  2. A random patient subsample gives essentially the same parameters as the
     full data. This is what makes the sampling strategy safe: the release
     contains no per-patient terms, so the sample is used only to estimate
     global quantities.

Run: MPLCONFIGDIR=./.mplconfig .venv/bin/python experiments/03_scale.py
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topocdm.release import make_release, sample_release
from topocdm.scalable import array_chunks, fit_propensity_streaming, fit_streaming, simulate_sparse

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
        "grid.alpha": 0.25,
    }
)
C = {"blue": "#2C6E9B", "orange": "#D9822B", "green": "#4E8B4A", "grey": "#8A8A8A"}


def main():
    print("generating 200k x 1000 CDM ...")
    data, truth = simulate_sparse(n=200_000, c=1000, seed=1)
    print(
        f"  {data.n:,} patients, {data.c:,} concepts, {data.total:,} events, "
        f"density {data.density:.2f}/py, nnz {data.X.nnz:,}"
    )
    print(f"  a dense matrix would be {data.n * data.c * 8 / 1e9:.2f} GB")

    rows = []
    ref_lam = None
    for frac in [1.0, 0.5, 0.25, 0.10, 0.05]:
        t0 = time.time()
        st = fit_streaming(
            array_chunks(data, chunk_size=5000, sample_frac=frac),
            n=data.n,
            c=data.c,
            concept_group=data.concept_group,
            n_groups=5,
            n_epochs=6,
            init_sample=5000,
        )
        dt = time.time() - t0
        lam = st.params.lam / st.params.lam.sum()
        if ref_lam is None:
            ref_lam = lam
        rows.append(
            {
                "frac": frac,
                "seconds": dt,
                "epochs": st.epochs,
                "lam_l1_vs_full": float(np.abs(lam - ref_lam).sum()),
                "patients_seen": st.n_patients_seen,
            }
        )
        print(f"  frac {frac:4.2f}: {dt:6.1f}s  lambda L1 vs full {rows[-1]['lam_l1_vs_full']:.4f}")

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9

    # the release: how many numbers actually leave
    k = fit_propensity_streaming(array_chunks(data, 5000), st.params)
    rel = make_release(
        st.params,
        data.concept_ids,
        data.concept_group,
        k,
        data.density,
        n_patients=data.n,
        n_events=data.total,
    )
    summary = rel.summary()
    print(f"\nrelease summary: {summary}")

    # density is a generator parameter
    dens = []
    for d in [2.0, 8.0, 20.0]:
        syn = sample_release(rel, n_patients=5000, years=5, density=d, rng=np.random.default_rng(1))
        dens.append({"target": d, "realised": syn.density, "events": syn.total})

    out = {
        "scale": rows,
        "peak_rss_gb": peak,
        "release": summary,
        "density": dens,
        "data": {
            "n": data.n,
            "c": data.c,
            "events": data.total,
            "nnz": int(data.X.nnz),
            "dense_gb": data.n * data.c * 8 / 1e9,
        },
    }
    json.dump(out, open(f"{RES}/scale.json", "w"), indent=1)

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))

    ax = axes[0]
    labels = ["dense incidence\nmatrix", "actual peak\nRSS of the fit"]
    vals = [out["data"]["dense_gb"], out["peak_rss_gb"]]
    bars = ax.bar(labels, vals, color=[C["grey"], C["blue"]], width=0.55)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.04, f"{v:.2f} GB", ha="center", fontsize=8)
    ax.set_ylabel("memory (GB)")
    ax.set_title(f"{data.total / 1e6:.1f}M events, {data.n / 1e3:.0f}k patients", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.25)

    ax = axes[1]
    ax.plot(
        [r["frac"] for r in rows],
        [r["lam_l1_vs_full"] for r in rows],
        "o-",
        color=C["orange"],
        ms=4,
    )
    ax.set_xlabel("fraction of patients used")
    ax.set_ylabel(r"$L_1$ distance of $\lambda$ from full fit")
    ax.set_title("Subsampling barely moves the answer", fontsize=9)

    ax = axes[2]
    ax.plot([d["target"] for d in dens], [d["realised"] for d in dens], "o", color=C["green"], ms=6)
    lim = [0, max(d["target"] for d in dens) * 1.1]
    ax.plot(lim, lim, "k:", lw=0.8)
    ax.set_xlabel("requested density (events / person-year)")
    ax.set_ylabel("realised density")
    ax.set_title("Density is a generator parameter", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{FIG}/fig10_scale.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {FIG}/fig10_scale.png and {RES}/scale.json")


if __name__ == "__main__":
    main()
