"""
26 -- held-out test of the per-concept prevalence layer.

The model is fitted on 80% of patients, the synthetic cohort is generated for
the held-out 20%, and prevalence/incidence are scored against those unseen
patients. Before/after the hurdle (topocdm.hurdle): the layer is calibrated on
the training split only, so any improvement is not in-sample.

Run: MPLCONFIGDIR=./.mplconfig .venv/bin/python experiments/26_hurdle.py
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

from topocdm import hurdle
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
C = {"real": "#1f2933", "before": "#9aa5b1", "after": "#2C6E9B"}
BINS = [("rare_1_3", 0.01, 0.03), ("mid_3_30", 0.03, 0.30), ("common_30_100", 0.30, 1.0001)]


def derive_encounters(events):
    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["start_date"]).dt.normalize()
    cd = ("condition", "drug", "measurement", "observation", "procedure")
    clin = ev[ev["domain"].isin(cd)]
    enc = (
        clin[["person_id", "date"]]
        .drop_duplicates()
        .sort_values(["person_id", "date"])
        .reset_index(drop=True)
    )
    enc["visit_occurrence_id"] = np.arange(1, len(enc) + 1)
    enc["visit_start_date"] = enc["date"]
    enc["visit_end_date"] = enc["date"]
    enc["visit_concept_id"] = 9202
    ev = ev.merge(
        enc[["person_id", "date", "visit_occurrence_id"]], on=["person_id", "date"], how="left"
    )
    ev.loc[~ev["domain"].isin(cd), "visit_occurrence_id"] = np.nan
    return ev, enc


def matrix_on(records, periods, concept_ids):
    persons = np.sort(periods["person_id"].unique())
    prow = {int(p): i for i, p in enumerate(persons)}
    ccol = {int(c): j for j, c in enumerate(concept_ids)}
    ev = records[records["concept_id"].isin(set(concept_ids))]
    rows = ev["person_id"].map(prow).to_numpy(dtype=np.int64)
    cols = ev["concept_id"].map(ccol).to_numpy(dtype=np.int64)
    X = sparse.csr_matrix((np.ones(len(ev)), (rows, cols)), shape=(len(persons), len(concept_ids)))
    X.sum_duplicates()
    return X.toarray()


def coverage(real, draws, lo, hi):
    d = np.vstack(draws)
    p5, p95 = np.percentile(d, [5, 95], axis=0)
    inside = (real >= p5) & (real <= p95)
    return inside, p5, p95


def run(name, events, periods, visits, R, prefix):
    t0 = time.time()
    persons = np.sort(periods["person_id"].unique())
    rng = np.random.default_rng(0)
    perm = rng.permutation(persons)
    ntr = int(0.8 * len(persons))
    train = set(perm[:ntr])
    test = set(perm[ntr:])
    ev_tr = events[events["person_id"].isin(train)]
    ev_te = events[events["person_id"].isin(test)]
    pe_tr = periods[periods["person_id"].isin(train)]
    pe_te = periods[periods["person_id"].isin(test)]
    vi_tr = visits[visits["person_id"].isin(train)]

    X_all, concept_ids, _, _ = pdm.clinical_incidence(events, periods)
    X_tr = matrix_on(ev_tr, pe_tr, concept_ids)
    X_te = matrix_on(ev_te, pe_te, concept_ids)
    (X_tr > 0).mean(0)
    prev_te = (X_te > 0).mean(0)
    T_te = np.maximum(
        (pe_te["end_date"] - pe_te["start_date"]).dt.days.to_numpy(float) / 365.25, 1 / 365.25
    )
    X_te.sum(0) / T_te.sum() * 1000.0

    fit = va.fit_visitanchor(ev_tr, pe_tr, vi_tr)

    before, after = [], []
    for r in range(R):
        sp = pdm.sample_periods(pe_te, len(pe_te), np.random.default_rng(1000 + r))
        tab = va.generate_visitanchor(fit, sp, seed=2000 + r)
        Xs = matrix_on(pdm.tables_to_records(tab), sp, concept_ids)
        before.append((Xs > 0).mean(0))
        pi_r = hurdle.sample_prevalence(X_tr, np.random.default_rng(3000 + r))
        after.append((hurdle.apply_prevalence(Xs, pi_r) > 0).mean(0))

    # evaluate on concepts the training split could know about
    train_total = np.asarray(X_tr.sum(axis=0)).ravel()
    test_total = np.asarray(X_te.sum(axis=0)).ravel()
    keep = (train_total > 10) & (test_total > 10)
    idx = np.where(keep)[0]
    out = {
        "name": name,
        "R": R,
        "n_train": len(train),
        "n_test": len(test),
        "n_concepts": int(len(concept_ids)),
        "n_compared": int(len(idx)),
        "bins": {},
    }
    print(
        f"{name}: train {len(train)} / test {len(test)}, "
        f"{len(idx)} concepts compared, {R} cohorts in {time.time() - t0:.1f}s"
    )
    for bname, lo, hi in BINS:
        m = idx[(prev_te[idx] >= lo) & (prev_te[idx] < hi)]
        if len(m) == 0:
            continue
        ib, _, _ = coverage(prev_te[m], [b[m] for b in before], None, None)
        ia, _, _ = coverage(prev_te[m], [a[m] for a in after], None, None)
        out["bins"][bname] = {
            "n": int(len(m)),
            "before": float(ib.mean()),
            "after": float(ia.mean()),
            "before_ratio": float(
                np.median(np.vstack([b[m] for b in before]).mean(0) / np.maximum(prev_te[m], 1e-12))
            ),
            "after_ratio": float(
                np.median(np.vstack([a[m] for a in after]).mean(0) / np.maximum(prev_te[m], 1e-12))
            ),
        }
        print(
            f"  {bname:14s} n={len(m):4d} prevalence coverage "
            f"{ib.mean():.0%} -> {ia.mean():.0%}; "
            f"ratio {out['bins'][bname]['before_ratio']:.2f} -> "
            f"{out['bins'][bname]['after_ratio']:.2f}"
        )
    # overall
    ib, _, _ = coverage(prev_te[idx], [b[idx] for b in before], None, None)
    ia, _, _ = coverage(prev_te[idx], [a[idx] for a in after], None, None)
    out["overall"] = {"before": float(ib.mean()), "after": float(ia.mean())}
    print(f"  overall coverage {ib.mean():.0%} -> {ia.mean():.0%}")
    return out


def main():
    out = {}
    ev, pe = pdm.load_extract()
    rng = np.random.default_rng(1)
    keep = np.sort(rng.choice(pe.person_id.unique(), 2000, replace=False))
    ev_s = ev[ev.person_id.isin(keep)].copy()
    pe_s = pe[pe.person_id.isin(keep)].copy()
    ev_s, enc_s = derive_encounters(ev_s)
    out["synthea"] = run("Synthea 2k", ev_s, pe_s, enc_s, R=20, prefix="fig40_synthea_hurdle")

    spec = importlib.util.spec_from_file_location(
        "m21", os.path.join(os.path.dirname(__file__), "21_subject_spectra.py")
    )
    m21 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m21)
    ev_m, pe_m, vis_m = m21._load_mimic()
    out["mimic"] = run("MIMIC demo", ev_m, pe_m, vis_m, R=50, prefix="fig40_mimic_hurdle")

    # figure: before/after by bin
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), sharey=True)
    for ax, key in zip(axes, ["synthea", "mimic"]):
        d = out[key]
        bins = list(d["bins"])
        x = np.arange(len(bins))
        b = [d["bins"][k]["before"] for k in bins]
        a = [d["bins"][k]["after"] for k in bins]
        ax.bar(x - 0.2, b, width=0.4, color=C["before"], label="before")
        ax.bar(x + 0.2, a, width=0.4, color=C["after"], label="after")
        ax.axhline(0.8, color="k", ls=":", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace("_", "\n") for k in bins], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{d['name']}: held-out prevalence coverage", fontsize=9)
        ax.set_ylabel("coverage")
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig40_hurdle_coverage.png", bbox_inches="tight")
    plt.close(fig)
    json.dump(out, open(f"{RES}/hurdle.json", "w"), indent=1)
    print(f"wrote {FIG}/fig40_hurdle_coverage.png and {RES}/hurdle.json")


if __name__ == "__main__":
    main()
