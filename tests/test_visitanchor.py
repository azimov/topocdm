"""Tests for the visit-anchored interval model (topocdm.visitanchor)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from topocdm import perdomain as pdm
from topocdm import visitanchor as va


def _synthetic(n=300, seed=0):
    """A small visit-anchored CDM: every event belongs to a visit."""
    rng = np.random.default_rng(seed)
    periods = pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1),
            "start_date": pd.Timestamp("2020-01-01"),
            "end_date": pd.Timestamp("2024-12-31"),
        }
    )
    visits, events = [], []
    vid = 0
    for i in range(n):
        for _ in range(3):
            vid += 1
            day = rng.integers(0, 5 * 365 - 1)
            s = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(day))
            e = s + pd.Timedelta(days=int(rng.integers(0, 5)))
            visits.append((i + 1, vid, s, e, 9202))
            for dom, concepts in [
                ("condition", [100, 101]),
                ("drug", [200, 201]),
                ("measurement", [300, 301, 302]),
            ]:
                k = int(rng.poisson(2.0))
                for _ in range(k):
                    c = int(rng.choice(concepts))
                    dur = (
                        pd.Timedelta(days=int(rng.integers(0, 30)))
                        if dom in ("condition", "drug")
                        else pd.NaT
                    )
                    events.append((i + 1, dom, c, s, s + dur if dur is not pd.NaT else pd.NaT, vid))
    events = pd.DataFrame(
        events,
        columns=[
            "person_id",
            "domain",
            "concept_id",
            "start_date",
            "end_date",
            "visit_occurrence_id",
        ],
    )
    visits = pd.DataFrame(
        visits,
        columns=[
            "person_id",
            "visit_occurrence_id",
            "visit_start_date",
            "visit_end_date",
            "visit_concept_id",
        ],
    )
    return events, periods, visits


def test_fit_generate_and_interval_structure():
    events, periods, visits = _synthetic()
    fit = va.fit_visitanchor(events, periods, visits)
    assert fit.n_visits == len(visits)
    assert fit.visit_counts.mean() == 3.0
    assert set(fit.counts_per_visit) == set(va.WITHIN_VISIT)

    sp = pdm.sample_periods(periods, len(periods), np.random.default_rng(1))
    tables = va.generate_visitanchor(fit, sp, seed=2)
    for name in (
        "condition_occurrence",
        "drug_exposure",
        "measurement",
        "visit_occurrence",
        "condition_era",
        "drug_era",
    ):
        assert name in tables
    assert len(tables["condition_occurrence"]) > 0
    assert len(tables["condition_era"]) > 0

    records = pdm.tables_to_records(tables)
    gaps = []
    for _, x in records.groupby("person_id"):
        d = np.sort(pd.to_datetime(x["start_date"]).values)
        if len(d) > 1:
            gaps.extend(np.diff(d).astype("timedelta64[D]").astype(int))
    # events are visit-anchored, so almost every consecutive gap is zero
    assert np.mean(np.asarray(gaps) == 0) > 0.8


def test_levels_are_in_the_right_ballpark():
    events, periods, visits = _synthetic(n=400, seed=3)
    fit = va.fit_visitanchor(events, periods, visits)
    sp = pdm.sample_periods(periods, len(periods), np.random.default_rng(4))
    tables = va.generate_visitanchor(fit, sp, seed=5)
    for table, dom in [
        ("condition_occurrence", "condition"),
        ("drug_exposure", "drug"),
        ("measurement", "measurement"),
    ]:
        assert 0.4 < len(tables[table]) / max(fit.n_events[dom], 1) < 2.5


def test_events_share_the_visit_date():
    events, periods, visits = _synthetic(n=100, seed=7)
    fit = va.fit_visitanchor(events, periods, visits)
    sp = pdm.sample_periods(periods, len(periods), np.random.default_rng(8))
    tables = va.generate_visitanchor(fit, sp, seed=9)
    v = tables["visit_occurrence"]
    c = tables["condition_occurrence"]
    # every condition date must be one of that patient's visit dates
    vd = set(zip(v["person_id"], pd.to_datetime(v["visit_start_date"]).dt.normalize()))
    cd = set(zip(c["person_id"], pd.to_datetime(c["condition_start_date"]).dt.normalize()))
    assert cd.issubset(vd)
