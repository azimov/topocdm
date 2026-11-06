"""
20 -- MIMIC: interval structure, visit anchoring, and predictive checks.

The per-domain model draws event starts independently in time. MIMIC does not
look like that: 99.4% of consecutive events within a patient are on the same
day, because events are recorded in encounters. This experiment adds the visit
anchor (topocdm.visitanchor) and compares the two generators with a predictive
check rather than a single realisation.

The check is the point. A model defines a distribution over datasets given N
and the observation periods; one draw can be above or below the observed data.
So we draw R cohorts from each model, compute the same dataset-level statistic
on each, and ask where the real MIMIC value falls. A model is adequate when the
real value sits inside the model's predictive distribution, not when one
arbitrary draw happens to match.

Run: MPLCONFIGDIR=./.mplconfig .venv/bin/python experiments/20_mimic_visitanchor.py
"""

from __future__ import annotations

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

DATA = os.environ.get("MIMIC_DIR", "data/mimic_eunomia")
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
    "blue": "#2C6E9B",
    "orange": "#C1440E",
    "green": "#4E8B4A",
    "grey": "#9aa5b1",
    "red": "#B4443C",
    "purple": "#6B5B95",
}

TABLE_DOMAIN = [
    ("condition_occurrence", "condition"),
    ("drug_exposure", "drug"),
    ("procedure_occurrence", "procedure"),
    ("measurement", "measurement"),
    ("observation", "observation"),
    ("visit_occurrence", "visit"),
]
ERA = [
    ("condition_era", "condition_era_start_date", "condition_era_end_date"),
    ("drug_era", "drug_era_start_date", "drug_era_end_date"),
]


def load_mimic(d: str):
    def rd(t, cols, dates=()):
        return pd.read_csv(
            os.path.join(d, t + ".csv"), usecols=cols, parse_dates=list(dates), low_memory=False
        )

    cc = rd(
        "condition_occurrence",
        [
            "person_id",
            "condition_concept_id",
            "condition_start_date",
            "condition_end_date",
            "visit_occurrence_id",
        ],
        ["condition_start_date", "condition_end_date"],
    )
    de = rd(
        "drug_exposure",
        [
            "person_id",
            "drug_concept_id",
            "drug_exposure_start_date",
            "drug_exposure_end_date",
            "visit_occurrence_id",
        ],
        ["drug_exposure_start_date", "drug_exposure_end_date"],
    )
    vo = rd(
        "visit_occurrence",
        [
            "person_id",
            "visit_concept_id",
            "visit_start_date",
            "visit_end_date",
            "visit_occurrence_id",
        ],
        ["visit_start_date", "visit_end_date"],
    )
    po = rd(
        "procedure_occurrence",
        ["person_id", "procedure_concept_id", "procedure_date", "visit_occurrence_id"],
        ["procedure_date"],
    )
    me = rd(
        "measurement",
        ["person_id", "measurement_concept_id", "measurement_date", "visit_occurrence_id"],
        ["measurement_date"],
    )
    ob = rd(
        "observation",
        ["person_id", "observation_concept_id", "observation_date", "visit_occurrence_id"],
        ["observation_date"],
    )
    dd = rd("death", ["person_id", "death_date"], ["death_date"])
    op = rd(
        "observation_period",
        ["person_id", "observation_period_start_date", "observation_period_end_date"],
        ["observation_period_start_date", "observation_period_end_date"],
    )

    def frame(df, dom, cc_, sc_, ec_):
        return pd.DataFrame(
            {
                "person_id": df.person_id,
                "domain": dom,
                "concept_id": df[cc_],
                "start_date": df[sc_],
                "end_date": df[ec_],
                "visit_occurrence_id": df["visit_occurrence_id"],
            }
        )

    events = pd.concat(
        [
            frame(
                cc,
                "condition",
                "condition_concept_id",
                "condition_start_date",
                "condition_end_date",
            ),
            frame(
                de, "drug", "drug_concept_id", "drug_exposure_start_date", "drug_exposure_end_date"
            ),
            frame(vo, "visit", "visit_concept_id", "visit_start_date", "visit_end_date"),
            frame(po, "procedure", "procedure_concept_id", "procedure_date", "procedure_date"),
            frame(
                me, "measurement", "measurement_concept_id", "measurement_date", "measurement_date"
            ),
            frame(
                ob, "observation", "observation_concept_id", "observation_date", "observation_date"
            ),
            pd.DataFrame(
                {
                    "person_id": dd.person_id,
                    "domain": "death",
                    "concept_id": 0,
                    "start_date": dd.death_date,
                    "end_date": pd.NaT,
                    "visit_occurrence_id": np.nan,
                }
            ),
        ],
        ignore_index=True,
    )
    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.to_datetime(events["end_date"])
    periods = op.rename(
        columns={
            "observation_period_start_date": "start_date",
            "observation_period_end_date": "end_date",
        }
    )
    return events, periods, vo


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


def _gap_stats(records):
    """Same-patient consecutive gaps: fraction 0 days, and <=30 days."""
    g = []
    for _, x in records.groupby("person_id"):
        d = np.sort(pd.to_datetime(x["start_date"]).values)
        if len(d) > 1:
            g.extend(np.diff(d).astype("timedelta64[D]").astype(int))
    g = np.asarray(g, dtype=int)
    return float((g == 0).mean()) if len(g) else np.nan


def _merge_stats(records, domain):
    """Fraction of consecutive same-concept gaps within 30 days (era merging)."""
    sub = records[records["domain"] == domain]
    g = []
    for _, x in sub.groupby(["person_id", "concept_id"]):
        d = np.sort(pd.to_datetime(x["start_date"]).values)
        if len(d) > 1:
            g.extend(np.diff(d).astype("timedelta64[D]").astype(int))
    g = np.asarray(g, dtype=int)
    return float((g <= 30).mean()) if len(g) else np.nan


def _era_rows(tables, name):
    df = tables.get(name)
    return int(len(df)) if df is not None else 0


def dataset_stats(records, tables, periods, concept_ids):
    stats = {f"events_{dom}": len(tables.get(table, [])) for table, dom in TABLE_DOMAIN}
    stats["gap_zero"] = _gap_stats(records)
    stats["merge_condition"] = _merge_stats(records, "condition")
    stats["merge_drug"] = _merge_stats(records, "drug")
    for name, _, _ in ERA:
        stats[f"{name}_rows"] = _era_rows(tables, name)
    X = _matrix_for(records, periods, concept_ids)
    B = X > 0
    stats["degree_mean"] = float(np.asarray(B.sum(axis=1)).ravel().mean())
    stats["prev_mean"] = float(np.asarray(B.mean(axis=0)).ravel().mean())
    stats["prev_sd"] = float(np.asarray(B.mean(axis=0)).ravel().std())
    stats["rank"] = int(pdm.effective_rank(pdm.spectrum(X)))
    return stats


def summarise(draws, real):
    """Median, 5-95% interval, and the percentile of the real value."""
    out = {}
    for k in real:
        x = np.asarray([d[k] for d in draws], dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        lo, hi = np.percentile(x, [5, 95])
        pct = float((x <= real[k]).mean())
        out[k] = {
            "real": float(real[k]),
            "median": float(np.median(x)),
            "lo": float(lo),
            "hi": float(hi),
            "pct": float(pct),
            "inside": bool(lo <= real[k] <= hi),
        }
    return out


def main():
    R = int(os.environ.get("R", "50"))
    t0 = time.time()
    events, periods, visits = load_mimic(DATA)
    print(
        f"MIMIC-IV demo: {periods.person_id.nunique()} patients, "
        f"{len(events):,} records, {len(visits):,} visits"
    )

    pd_fit = pdm.fit_perdomain(events, periods, K=5)
    va_fit = va.fit_visitanchor(events, periods, visits)
    print(f"per-domain fit {time.time() - t0:.1f}s; visit-anchor fit done")

    X_real, concept_ids, _, _ = pdm.clinical_incidence(events, periods)
    real_ce = pd.read_csv(
        os.path.join(DATA, "condition_era.csv"),
        parse_dates=["condition_era_start_date", "condition_era_end_date"],
    )
    real_de = pd.read_csv(
        os.path.join(DATA, "drug_era.csv"), parse_dates=["drug_era_start_date", "drug_era_end_date"]
    )
    real_tables = {t: events[events["domain"] == d] for t, d in TABLE_DOMAIN}
    real_tables["condition_era"] = real_ce
    real_tables["drug_era"] = real_de
    real = dataset_stats(events, real_tables, periods, concept_ids)
    print("\nreal dataset statistics:")
    for k, v in real.items():
        print(f"  {k:24s} {v:,.3f}" if isinstance(v, float) else f"  {k:24s} {v:,.0f}")

    def run_model(name, gen):
        draws = []
        for r in range(R):
            rng = np.random.default_rng(1000 + r)
            sp = pdm.sample_periods(periods, len(periods), rng)
            tables = gen(sp, seed=2000 + r)
            draws.append(dataset_stats(pdm.tables_to_records(tables), tables, sp, concept_ids))
        print(f"  {name}: {R} cohorts in {time.time() - t0:.1f}s")
        return draws

    draws_pd = run_model(
        "per-domain", lambda sp, seed: pdm.generate_cohort(pd_fit, len(sp), sp, seed=seed)
    )
    draws_va = run_model(
        "visit-anchor", lambda sp, seed: va.generate_visitanchor(va_fit, sp, seed=seed)
    )

    sum_pd = summarise(draws_pd, real)
    sum_va = summarise(draws_va, real)

    keys = [
        "events_condition",
        "events_drug",
        "events_measurement",
        "gap_zero",
        "merge_condition",
        "merge_drug",
        "condition_era_rows",
        "drug_era_rows",
        "degree_mean",
        "rank",
    ]
    print("\npredictive check (real value vs model output; 5-95% interval):")
    print(f"  {'statistic':22s} {'real':>9s}  {'per-domain':>22s}  {'visit-anchor':>22s}")
    for k in keys:
        a, b = sum_pd.get(k), sum_va.get(k)
        if not a or not b:
            continue
        print(
            f"  {k:22s} {a['real']:>9.3f}  "
            f"{a['median']:>8.2f} [{a['lo']:.2f},{a['hi']:.2f}] "
            f"{'in' if a['inside'] else 'OUT':>3s}  "
            f"{b['median']:>8.2f} [{b['lo']:.2f},{b['hi']:.2f}] "
            f"{'in' if b['inside'] else 'OUT':>3s}"
        )

    out = {
        "R": R,
        "real": real,
        "per_domain": {k: sum_pd[k] for k in sum_pd},
        "visit_anchor": {k: sum_va[k] for k in sum_va},
    }
    json.dump(
        out,
        open(f"{RES}/mimic_predictive.json", "w"),
        indent=1,
        default=lambda o: float(o) if isinstance(o, np.floating) else str(o),
    )

    # ---- figure: the three failures, before and after ----
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0))

    def panel(ax, key, title, xlabel, log=False):
        for draws, col, lab in [
            (draws_pd, C["grey"], "per-domain"),
            (draws_va, C["orange"], "visit-anchor"),
        ]:
            x = np.asarray([d[key] for d in draws], dtype=float)
            x = x[np.isfinite(x)]
            ax.hist(x, bins=20, alpha=0.55, color=col, label=lab, edgecolor="white", linewidth=0.3)
        ax.axvline(real[key], color="k", ls="--", lw=1.1, label="MIMIC")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        if log:
            ax.set_xscale("log")
        ax.legend(fontsize=6.5, frameon=False)

    panel(axes[0, 0], "gap_zero", "A. Consecutive events on the same day", "fraction of gaps = 0 d")
    panel(axes[0, 1], "condition_era_rows", "B. Condition eras", "era rows")
    panel(axes[1, 0], "drug_era_rows", "C. Drug eras", "era rows")
    panel(axes[1, 1], "events_measurement", "D. Measurement count (level)", "rows")

    fig.tight_layout()
    fig.savefig(f"{FIG}/fig30_mimic_predictive.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {FIG}/fig30_mimic_predictive.png and {RES}/mimic_predictive.json")


if __name__ == "__main__":
    main()
