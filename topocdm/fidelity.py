"""
topocdm.fidelity -- does the fitted model reproduce prevalence and incidence?

This is the check that decides whether the model has learned anything beyond
the marginals. It takes the events and periods the fit was produced from,
generates R synthetic cohorts from the fitted parameters, and compares the
per-concept prevalence and incidence of the real cohort against the synthetic
ones.

Two properties of the design matter for the answer being honest:

- **The real side is held out from generation, not from fitting.** The fit's
  per-patient terms (`u`, `z`) are diagnostics only; generation draws new
  patients from `N(z_mean, z_cov)` and a fresh Gamma propensity. So a synthetic
  cohort is a draw from the population model, and the comparison is real.
  Patients that were *not* fitted (if any) can be excluded via `fit_person_ids`.

- **A single scalar hides where the model fails.** Coverage is reported over
  all well-supported concepts, but also split into prevalence bins. The known
  failure mode is concentrated in the common bin, so a pooled number would
  hide it.

Prevalence is the fraction of patients with at least one record of a concept.
Incidence is events per 1000 person-years. Both are computed on the mapped
clinical domains only (condition, drug, procedure, measurement, observation),
which is the object the per-domain model is fitted to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from topocdm import perdomain as pdm

#: prevalence bins for the gate: a pooled scalar hides the common-bin failure
BINS = (
    ("rare_1_3", 0.01, 0.03),
    ("mid_3_30", 0.03, 0.30),
    ("common_30_100", 0.30, 1.0001),
)

#: concepts with at most this many real records are excluded from ratios:
#: a concept seen once has prevalence 1/10000 and any ratio is noise
MIN_REAL_RECORDS = 10

#: coverage below this on the pooled concepts is a fail
COVERAGE_FLOOR = 0.90


def _matrix(records: pd.DataFrame, periods: pd.DataFrame, concept_ids: np.ndarray):
    """(N, C) count matrix and person-years, on a fixed concept vocabulary."""
    persons = np.sort(periods["person_id"].unique())
    prow = {int(p): i for i, p in enumerate(persons)}
    ccol = {int(c): j for j, c in enumerate(concept_ids)}
    ev = records[records["domain"].isin(pdm.CLINICAL_DOMAINS) & records["concept_id"].isin(ccol)]
    rows = ev["person_id"].map(prow)
    cols = ev["concept_id"].map(ccol)
    ok = rows.notna() & cols.notna()
    X = sparse.csr_matrix(
        (np.ones(int(ok.sum())), (rows[ok].to_numpy(np.int64), cols[ok].to_numpy(np.int64))),
        shape=(len(persons), len(concept_ids)),
    )
    X.sum_duplicates()
    T = (periods["end_date"] - periods["start_date"]).dt.days.to_numpy(float) / 365.25
    return X, np.maximum(T, 1 / 365.25)


def _marginals(X, T):
    prev = np.asarray((X > 0).mean(axis=0)).ravel()
    inc = np.asarray(X.sum(axis=0)).ravel() / max(T.sum(), 1e-9) * 1000.0
    return prev, inc


def _calib(y_real: np.ndarray, y_syn: np.ndarray) -> dict:
    """Calibration of synthetic against real values."""
    m = np.isfinite(y_real) & np.isfinite(y_syn)
    if m.sum() < 3:
        return {"r": float("nan"), "slope": float("nan"), "med_ratio": float("nan")}
    slope = float(np.polyfit(y_syn[m], y_real[m], 1)[0])
    ratio = y_real[m] / np.maximum(y_syn[m], 1e-12)
    return {
        "r": float(np.corrcoef(y_real[m], y_syn[m])[0, 1]),
        "slope": slope,
        "med_ratio": float(np.median(ratio)),
    }


def fidelity_report(
    fit,
    events: pd.DataFrame,
    periods: pd.DataFrame,
    *,
    n_draws: int = 20,
    held_out_person_ids=None,
    seed: int = 0,
    min_real_records: int = MIN_REAL_RECORDS,
) -> dict:
    """Compare real prevalence/incidence against R generated cohorts.

    `events` and `periods` are the frames the fit was produced from. If
    `held_out_person_ids` is given, the real side is restricted to those
    patients; otherwise it is the whole extract.
    """
    if n_draws < 2:
        raise ValueError("n_draws must be >= 2 to form a simulation interval")

    real_events, real_periods = events, periods
    if held_out_person_ids is not None:
        keep = np.asarray(held_out_person_ids)
        real_events = events[events["person_id"].isin(keep)].copy()
        real_periods = periods[periods["person_id"].isin(keep)].copy()
        if real_periods["person_id"].nunique() == 0:
            raise ValueError("held-out set contains no patients")

    X_real, concept_ids, persons, T_real = pdm.clinical_incidence(real_events, real_periods)
    prev_real, inc_real = _marginals(X_real, T_real)

    prev_draws, inc_draws = [], []
    n_synthetic = 0
    for r in range(n_draws):
        rng = np.random.default_rng(seed + r)
        sp = pdm.sample_periods(real_periods, len(real_periods), rng)
        tables = pdm.generate_cohort(fit, len(sp), sp, seed=seed + 10_000 + r)
        n_synthetic = int(len(sp))
        rec = pdm.tables_to_records(tables)
        X_s, T_s = _matrix(rec, sp, concept_ids)
        p, i = _marginals(X_s, T_s)
        prev_draws.append(p)
        inc_draws.append(i)

    P = np.vstack(prev_draws)
    inc_d = np.vstack(inc_draws)
    plo, pmid, phi = np.percentile(P, [5, 50, 95], axis=0)
    ilo, imid, ihi = np.percentile(inc_d, [5, 50, 95], axis=0)

    support = np.asarray(X_real.sum(axis=0)).ravel() > min_real_records
    if support.sum() < 1:
        raise ValueError(
            f"no concepts with more than {min_real_records} real records; "
            "cannot form a fidelity report"
        )

    def _bins() -> dict[str, dict]:
        out: dict[str, dict] = {}
        for name, lo, hi in BINS:
            m = support & (prev_real >= lo) & (prev_real < hi)
            if m.sum() == 0:
                out[name] = {"n": 0}
                continue
            ip = (prev_real[m] >= plo[m]) & (prev_real[m] <= phi[m])
            ii = (inc_real[m] >= ilo[m]) & (inc_real[m] <= ihi[m])
            out[name] = {
                "n": int(m.sum()),
                "prevalence_coverage": float(ip.mean()),
                "incidence_coverage": float(ii.mean()),
                "prevalence_ratio_median": float(
                    np.median(pmid[m] / np.maximum(prev_real[m], 1e-12))
                ),
                "incidence_ratio_median": float(
                    np.median(imid[m] / np.maximum(inc_real[m], 1e-12))
                ),
                "prevalence_over": int(((pmid[m] > prev_real[m]) & ~ip).sum()),
                "prevalence_under": int(((pmid[m] < prev_real[m]) & ~ip).sum()),
            }
        return out

    cover_p = float(
        np.mean((prev_real[support] >= plo[support]) & (prev_real[support] <= phi[support]))
    )
    cover_i = float(
        np.mean((inc_real[support] >= ilo[support]) & (inc_real[support] <= ihi[support]))
    )

    binned = _bins()
    common = binned.get("common_30_100", {})
    gates = {
        "pooled_prevalence_coverage": cover_p,
        "pooled_incidence_coverage": cover_i,
        "coverage_floor": COVERAGE_FLOOR,
        "passes_pooled": bool(cover_p >= COVERAGE_FLOOR),
        "passes_common_bin": bool(common.get("prevalence_coverage", 0.0) >= COVERAGE_FLOOR),
    }

    return {
        "n_draws": n_draws,
        "seed": seed,
        "n_patients_real": int(len(persons)),
        "n_patients_synthetic": int(n_synthetic),
        "n_concepts": int(len(concept_ids)),
        "n_concepts_scored": int(support.sum()),
        "min_real_records": int(min_real_records),
        "coverage": {"prevalence": cover_p, "incidence": cover_i},
        "ratios": {
            "prevalence": _calib(prev_real[support], pmid[support]),
            "incidence": _calib(inc_real[support], imid[support]),
        },
        "bins": binned,
        "gates": gates,
        "concepts": {
            "concept_id": [int(c) for c in concept_ids],
            "supported": [bool(s) for s in support],
            "prevalence_real": prev_real.tolist(),
            "prevalence_median": pmid.tolist(),
            "prevalence_lo": plo.tolist(),
            "prevalence_hi": phi.tolist(),
            "incidence_real": inc_real.tolist(),
            "incidence_median": imid.tolist(),
            "incidence_lo": ilo.tolist(),
            "incidence_hi": ihi.tolist(),
        },
    }
