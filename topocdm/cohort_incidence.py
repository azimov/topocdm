"""
topocdm.cohort_incidence -- calendar-year incidence, the CohortIncidence way.

Implements the minimal version of the OHDSI CohortIncidence computation so that
a real CDM and a synthetic one can be compared on the statistic that actually
matters:

    cohort  = persons with at least one visit in calendar year y
    entry   = first visit in the year
    outcome = first occurrence of the target concept on or after entry
    PT      = entry -> min(outcome, year end, observation end)
    IR      = outcomes / PT, per 1000 person-years

Two kinds of interval are provided, and the difference between them is the
point:

  * `poisson_ci`    -- the sampling interval for the *observed* count. This is
                       what you get from one database and it is what
                       CohortIncidence reports.

  * `predictive_ci` -- the interval implied by the fitted model for a *new*
                       database with the same observation process. This is the
                       quantity that answers "would another database with the
                       same structure have shown a different rate?", and it is
                       available analytically without repeated simulation.

The distinction matters for the question that motivated this: when two data
sources disagree on a rate, the poisson interval only tells you whether each
source's own count is precise. The predictive interval tells you whether the
difference exceeds what the observation process alone would produce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def cohort_incidence(visits: pd.DataFrame, events: pd.DataFrame,
                     periods: pd.DataFrame, years, concept_id: int,
                     min_prior_days: int = 365) -> pd.DataFrame:
    """Incidence of one concept, by calendar year, in the CohortIncidence style."""
    v = visits.copy()
    v["year"] = pd.to_datetime(v["visit_date"]).dt.year
    e = events[events["concept_id"] == concept_id].copy()
    if len(e):
        e["event_date"] = pd.to_datetime(e["event_date"])
        first_ev = e.groupby("person_id")["event_date"].min()
        ev_persons = first_ev.index.to_numpy()
        ev_dates = first_ev.to_numpy().astype("datetime64[ns]")
        ev_lookup = dict(zip(ev_persons, ev_dates))
    else:
        ev_lookup = {}

    per_end = periods.set_index("person_id")["end_date"]
    per_start = periods.set_index("person_id")["start_date"]
    never = np.datetime64("2262-01-01", "ns")

    out = []
    for y in years:
        py = v[v["year"] == y]
        if py.empty:
            continue
        entry = py.groupby("person_id")["visit_date"].min()
        persons = entry.index.to_numpy()
        start = entry.to_numpy().astype("datetime64[ns]")

        # CohortIncidence-style eligibility: require prior observation, and
        # drop prevalent cases so the cohort is incident.
        obs_start = per_start.loc[persons].to_numpy().astype("datetime64[ns]")
        prior_days = (start - obs_start) / np.timedelta64(1, "D")
        eligible = prior_days >= min_prior_days
        ev_all = np.array([ev_lookup.get(p, never) for p in persons],
                          dtype="datetime64[ns]")
        eligible &= ev_all >= start
        persons, start = persons[eligible], start[eligible]

        y_end = np.datetime64(f"{int(y) + 1}-01-01", "ns")
        obs_end = per_end.loc[persons].to_numpy().astype("datetime64[ns]")
        limit = np.minimum(np.full(len(persons), y_end), obs_end)

        ev_date = np.array([ev_lookup.get(p, never) for p in persons],
                           dtype="datetime64[ns]")
        if len(persons) == 0:
            continue
        # an outcome is a first occurrence *during follow-up*, not merely an
        # event that exists somewhere after entry
        outcome = (ev_date >= start) & (ev_date < limit)
        exit_date = np.where(outcome, np.minimum(ev_date, limit), limit)

        days = (exit_date - start) / np.timedelta64(1, "D")
        pt = np.clip(days.astype(float), 0.0, None) / 365.25
        n_out = int(outcome.sum())
        total_pt = float(pt.sum())
        lo, hi = poisson_ci(n_out)
        out.append(dict(year=int(y), persons=int(len(persons)), outcomes=n_out,
                        person_time=total_pt,
                        rate=1000 * n_out / max(total_pt, 1e-9),
                        rate_lo=1000 * lo / max(total_pt, 1e-9),
                        rate_hi=1000 * hi / max(total_pt, 1e-9)))
    return pd.DataFrame(out)


def poisson_ci(k: int, alpha: float = 0.05):
    """Exact (Garwood) interval for a Poisson count."""
    if k == 0:
        lo = 0.0
    else:
        lo = 0.5 * stats.chi2.ppf(alpha / 2, 2 * k)
    hi = 0.5 * stats.chi2.ppf(1 - alpha / 2, 2 * (k + 1))
    return float(lo), float(hi)


def predictive_ci(mean_events: float, person_time: float,
                  dispersion: float | None = None, alpha: float = 0.05,
                  scale: float = 1000.0, n_effective: float = 1.0):
    """Interval for the rate of a *new* database, from the fitted model.

    If outcomes are Poisson the count has variance equal to its mean. With the
    negative-binomial marginal induced by integrating patient propensity,

        Var(X_i) = mu_i + mu_i^2 / k

    Summing over a cohort of n patients whose means are roughly equal mu/n,

        Var(sum) = mu + mu^2 / (n k)

    so `n_effective` must be supplied: using the single-patient formula on a
    cohort sum over-states the interval by a factor of about sqrt(n). This is
    the analytic alternative to generating many synthetic databases and taking
    percentiles.
    """
    m = max(float(mean_events), 1e-9)
    n_eff = max(float(n_effective), 1e-9)
    var = m if dispersion is None else m + m ** 2 / (max(dispersion, 1e-9) * n_eff)
    sd = np.sqrt(var)
    rate = scale * m / person_time
    sd_rate = scale * sd / person_time
    z = stats.norm.ppf(1 - alpha / 2)
    return rate, rate - z * sd_rate, rate + z * sd_rate


def monte_carlo_ci(realisations, alpha: float = 0.05):
    """Percentile interval over repeated synthetic databases."""
    r = np.asarray([x["rate"] for x in realisations], dtype=float)
    return float(np.percentile(r, 100 * alpha / 2)), float(np.percentile(r, 100 * (1 - alpha / 2)))
