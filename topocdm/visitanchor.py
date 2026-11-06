"""
topocdm.visitanchor -- a visit-anchored interval model.

The per-domain model in `topocdm.perdomain` draws event starts independently in
time. On MIMIC that is the wrong object: 99.4% of consecutive events within a
patient are on the same day, because events are recorded in encounters, not as
a continuous stream. `visit_occurrence_id` is 100% populated there, so the
interval structure is observable rather than something to be invented.

This module models the encounter as the unit:

    visits per patient        an empirical distribution, sampled by bootstrap
    visit dates               uniform within the resampled observation period
    events per visit, domain  an empirical distribution, sampled by bootstrap
    concept, given domain     the empirical catalogue
    durations                 the per-domain quantile model (as in perdomain)

All events in a visit share the visit date. That is the whole interval fix: it
reproduces the same-day batching, and therefore the era merging, without
assuming a parametric gap law. As everywhere else in this project, the shapes
are estimated from the fitted database and are not claims that transfer.

The visit domain is the encounter itself: one row per visit, with a visit type
drawn from the observed distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import topocdm.perdomain as pdm

WITHIN_VISIT = ("condition", "drug", "measurement", "observation", "procedure")


@dataclass
class VisitAnchorFit:
    visit_counts: np.ndarray = None  # type: ignore[assignment]  # (n_observed,) visits per patient
    visit_durations: np.ndarray = None  # type: ignore[assignment]  # (n_observed,) days
    visit_types: np.ndarray = None  # type: ignore[assignment]  # visit_concept_id values
    visit_type_probs: np.ndarray = None  # type: ignore[assignment]
    counts_per_visit: dict = field(default_factory=dict)  # domain -> counts
    concept_ids: dict = field(default_factory=dict)
    concept_probs: dict = field(default_factory=dict)
    duration: dict = field(default_factory=dict)
    death: dict = field(default_factory=dict)
    n_events: dict = field(default_factory=dict)
    n_patients: int = 0
    n_visits: int = 0
    meta: dict = field(default_factory=dict)


def _empirical_counts(values: np.ndarray):
    """Quantile summary of a non-negative integer distribution.

    Kept as quantiles rather than a parametric family: on this database the
    per-visit counts are extremely heavy-tailed (measurements per visit reach
    26,425), and a Poisson or negative binomial would misplace the tail.
    """
    v = np.asarray(values, dtype=float)
    return np.quantile(v, np.linspace(0, 1, 101))


def fit_visitanchor(
    events: pd.DataFrame, periods: pd.DataFrame, visits: pd.DataFrame, seed: int = 0
) -> VisitAnchorFit:
    """Fit the visit-anchored model.

    events: person_id, domain, concept_id, start_date, end_date,
            visit_occurrence_id
    periods: person_id, start_date, end_date
    visits: person_id, visit_occurrence_id, visit_start_date, visit_end_date,
            visit_concept_id
    """
    persons = np.sort(periods["person_id"].unique())
    v = visits.drop_duplicates(["person_id", "visit_occurrence_id"]).copy()
    v["visit_start_date"] = pd.to_datetime(v["visit_start_date"])
    v["visit_end_date"] = pd.to_datetime(v["visit_end_date"])
    v = v.sort_values(["person_id", "visit_start_date"]).reset_index(drop=True)

    visit_counts = v.groupby("person_id").size().reindex(persons, fill_value=0)
    visit_days = (v["visit_end_date"] - v["visit_start_date"]).dt.days
    vt = v["visit_concept_id"].value_counts()

    fit = VisitAnchorFit(
        visit_counts=visit_counts.to_numpy(dtype=int),
        visit_durations=visit_days.to_numpy(dtype=float),
        visit_types=vt.index.to_numpy(dtype=np.int64),
        visit_type_probs=(vt / vt.sum()).to_numpy(dtype=float),
        n_patients=int(len(persons)),
        n_visits=int(len(v)),
    )

    key = pd.MultiIndex.from_frame(v[["person_id", "visit_occurrence_id"]])
    for d in WITHIN_VISIT:
        ev = events[(events["domain"] == d) & events["visit_occurrence_id"].notna()]
        cnt = ev.groupby(["person_id", "visit_occurrence_id"]).size().reindex(key, fill_value=0)
        fit.counts_per_visit[d] = cnt.to_numpy(dtype=int)
        fit.n_events[d] = int(len(ev))
        vc = ev["concept_id"].value_counts()
        fit.concept_ids[d] = vc.index.to_numpy(dtype=np.int64)
        fit.concept_probs[d] = (vc / vc.sum()).to_numpy(dtype=float)

    # durations and death: same estimators as the per-domain model
    for d in pdm.INTERVAL_DOMAINS:
        fit.duration[d] = pdm.fit_durations(events, d)

    ob_start = (
        periods.set_index("person_id").loc[persons, "start_date"].to_numpy(dtype="datetime64[D]")
    )
    ob_end = periods.set_index("person_id").loc[persons, "end_date"].to_numpy(dtype="datetime64[D]")
    death = events[events["domain"] == "death"]
    died = death.set_index("person_id")["start_date"].to_dict()
    follow = (ob_end - ob_start) / np.timedelta64(1, "D")
    frac_times = []
    for i, p in enumerate(persons):
        if int(p) in died:
            dd = np.datetime64(died[int(p)], "D")
            t = (dd - ob_start[i]) / np.timedelta64(1, "D")
            frac_times.append(float(np.clip(t / max(follow[i], 1.0), 0.0, 1.0)))
    fit.death = {
        "frac": float(len(frac_times) / len(persons)),
        "frac_times": np.asarray(frac_times, dtype=float),
    }
    return fit


def _sample_counts(pool: np.ndarray, n: int, rng) -> np.ndarray:
    """Bootstrap n counts from an observed count distribution."""
    return rng.choice(np.asarray(pool, dtype=np.int64), size=n)


def generate_visitanchor(fit: VisitAnchorFit, periods: pd.DataFrame, seed: int = 0) -> dict:
    """Generate a visit-anchored cohort and its OMOP tables.

    Every event is attached to a visit and shares its date. Durations are
    drawn from the per-domain quantile model; death truncates.
    """
    rng = np.random.default_rng(seed)
    n = len(periods)
    s = periods["start_date"].to_numpy(dtype="datetime64[D]")
    e = periods["end_date"].to_numpy(dtype="datetime64[D]")

    n_visits = _sample_counts(fit.visit_counts, n, rng)
    rows = []  # (person_id, domain, concept_id, start_date)
    for i in range(n):
        k = int(n_visits[i])
        if k <= 0:
            continue
        span = max(int((e[i] - s[i]) / np.timedelta64(1, "D")), 1)
        off = np.sort(rng.integers(0, span, size=k))
        dates = s[i] + off.astype("timedelta64[D]")
        for day in dates:
            # the encounter itself
            vt = int(rng.choice(fit.visit_types, p=fit.visit_type_probs))
            rows.append((i + 1, "visit", vt, day))
            # its contents
            for d in WITHIN_VISIT:
                cnt = int(_sample_counts(fit.counts_per_visit[d], 1, rng)[0])
                if cnt <= 0:
                    continue
                cids = rng.choice(fit.concept_ids[d], size=cnt, p=fit.concept_probs[d])
                for c in cids:
                    rows.append((i + 1, d, int(c), day))

    events = pd.DataFrame(rows, columns=["person_id", "domain", "concept_id", "start_date"])
    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.NaT

    # durations for interval domains, clipped to the observation period
    ob_end_map = periods.set_index("person_id")["end_date"]
    for d in pdm.INTERVAL_DOMAINS:
        m = events["domain"] == d
        if not m.any():
            continue
        dur = pdm.sample_durations(fit.duration[d], int(m.sum()), rng)
        starts = events.loc[m, "start_date"].to_numpy(dtype="datetime64[D]")
        obe = pd.to_datetime(events.loc[m, "person_id"].map(ob_end_map)).to_numpy(
            dtype="datetime64[D]"
        )
        max_dur = np.maximum((obe - starts) / np.timedelta64(1, "D"), 0)
        dur_days = np.minimum(dur.astype(float), max_dur).astype(np.int64)
        ends = starts + dur_days.astype("timedelta64[D]")
        events.loc[m, "end_date"] = ends.astype("datetime64[ns]")

    # death: observed mortality fraction and within-window timing
    follow = (e - s) / np.timedelta64(1, "D")
    died = rng.random(n) < fit.death["frac"]
    if died.any() and len(fit.death["frac_times"]):
        frac = rng.choice(fit.death["frac_times"], size=int(died.sum()))
        t_days = (frac * follow[died]).astype("timedelta64[D]")
    else:
        t_days = np.array([], dtype="timedelta64[D]")
    death_date = s[died] + t_days
    death = pd.DataFrame(
        {"person_id": np.arange(1, n + 1)[died], "death_date": pd.to_datetime(death_date)}
    )

    if len(death):
        dmap = death.set_index("person_id")["death_date"]
        dd = pd.to_datetime(events["person_id"].map(dmap)).to_numpy(dtype="datetime64[ns]")
        start_ns = events["start_date"].to_numpy(dtype="datetime64[ns]")
        keep = np.isnat(dd) | (start_ns <= dd)
        events = events[keep].copy()

    return pdm.tables_from_events(events, periods, death)
