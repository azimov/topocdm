"""Tests for the fidelity report (topocdm.fidelity)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from topocdm import perdomain as pdm
from topocdm.fidelity import fidelity_report


def _cohort(n=120, seed=0):
    """A small extract with several prevalence levels, including a common one."""
    rng = np.random.default_rng(seed)
    rows = []
    # concept prevalence is set by the number of draws per patient
    plan = {"condition": [1, 2, 3], "drug": [10, 11], "measurement": [20, 21]}
    for dom, concepts in plan.items():
        for i in range(1, n + 1):
            for _ in range(rng.integers(1, 6)):
                c = int(rng.choice(concepts))
                s = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 900)))
                e = s + pd.Timedelta(days=int(rng.integers(0, 300)))
                rows.append((i, dom, c, s, e))
    events = pd.DataFrame(
        rows, columns=["person_id", "domain", "concept_id", "start_date", "end_date"]
    )
    periods = pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1),
            "start_date": pd.Timestamp("2020-01-01"),
            "end_date": pd.Timestamp("2022-12-31"),
        }
    )
    return events, periods


@pytest.fixture(scope="module")
def fitted():
    events, periods = _cohort()
    fit = pdm.fit_perdomain(events, periods, K=3)
    return fit, events, periods


def test_report_has_the_expected_shape(fitted):
    fit, events, periods = fitted
    rep = fidelity_report(fit, events, periods, n_draws=3, min_real_records=1)

    assert rep["n_draws"] == 3
    assert rep["n_patients_real"] == 120
    assert rep["n_patients_synthetic"] == 120
    n = rep["n_concepts"]
    assert rep["n_concepts_scored"] > 0
    # every concept keeps a full set of columns, supported or not
    for key in (
        "concept_id",
        "supported",
        "prevalence_real",
        "prevalence_median",
        "prevalence_lo",
        "prevalence_hi",
        "incidence_real",
        "incidence_median",
        "incidence_lo",
        "incidence_hi",
    ):
        assert len(rep["concepts"][key]) == n


def test_intervals_bracket_the_median(fitted):
    fit, events, periods = fitted
    rep = fidelity_report(fit, events, periods, n_draws=5, min_real_records=1)
    c = rep["concepts"]
    lo = np.array(c["prevalence_lo"])
    mid = np.array(c["prevalence_median"])
    hi = np.array(c["prevalence_hi"])
    assert np.all(lo <= mid + 1e-12)
    assert np.all(mid <= hi + 1e-12)
    # prevalence is a fraction and incidence is non-negative
    assert np.all((np.array(c["prevalence_real"]) >= 0) & (np.array(c["prevalence_real"]) <= 1))
    assert np.all(np.array(c["incidence_real"]) >= 0)


def test_report_is_json_serialisable(fitted):
    fit, events, periods = fitted
    rep = fidelity_report(fit, events, periods, n_draws=3, min_real_records=1)
    # no numpy scalars or arrays survive into the report
    json.dumps(rep)


def test_gates_are_booleans(fitted):
    fit, events, periods = fitted
    rep = fidelity_report(fit, events, periods, n_draws=3, min_real_records=1)
    g = rep["gates"]
    assert isinstance(g["passes_pooled"], bool)
    assert isinstance(g["passes_common_bin"], bool)
    assert 0.0 <= g["pooled_prevalence_coverage"] <= 1.0
    assert 0.0 <= g["pooled_incidence_coverage"] <= 1.0


def test_held_out_patients_are_restricted(fitted):
    fit, events, periods = fitted
    held = np.arange(1, 41)
    rep = fidelity_report(
        fit, events, periods, n_draws=3, held_out_person_ids=held, min_real_records=1
    )
    assert rep["n_patients_real"] == 40


def test_held_out_set_must_not_be_empty(fitted):
    fit, events, periods = fitted
    with pytest.raises(ValueError, match="no patients"):
        fidelity_report(fit, events, periods, n_draws=3, held_out_person_ids=[999_999])


def test_too_few_draws_is_rejected(fitted):
    fit, events, periods = fitted
    with pytest.raises(ValueError, match="n_draws"):
        fidelity_report(fit, events, periods, n_draws=1)


def test_fast_path_matches_full_pipeline(fitted):
    """Skipping the era build must not change prevalence or incidence.

    Era merging only re-dates and merges intervals; it cannot add or remove a
    (patient, concept) presence. The fast path is what makes a 1M-patient
    report tractable, so it has to agree with the full one exactly.
    """
    fit, events, periods = fitted
    fast = fidelity_report(fit, events, periods, n_draws=3, min_real_records=1)
    full = fidelity_report(fit, events, periods, n_draws=3, min_real_records=1, full_pipeline=True)

    for key in ("prevalence_median", "incidence_median"):
        assert np.allclose(fast["concepts"][key], full["concepts"][key])
    assert fast["coverage"] == full["coverage"]
    assert fast["full_pipeline"] is False
    assert full["full_pipeline"] is True


def test_progress_callback_is_called(fitted):
    fit, events, periods = fitted
    seen = []
    fidelity_report(
        fit,
        events,
        periods,
        n_draws=3,
        min_real_records=1,
        progress=lambda done, total, elapsed: seen.append((done, total)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_multiple_periods_per_patient_do_not_inflate_the_cohort(fitted):
    """A patient with two observation periods is still one patient (#4).

    Sizing the synthetic cohort by period rows rather than unique patients made
    every draw 1.55x too large on a real extract, inflating synthetic
    prevalence by the same factor. Person-time must also be totalled per
    person, not per period row, or the incidence denominator is wrong too.
    """
    events, periods = _cohort()
    # give every other patient a second, disjoint observation period
    second = periods[periods["person_id"] % 2 == 0].copy()
    second["start_date"] = pd.Timestamp("2023-01-01")
    second["end_date"] = pd.Timestamp("2023-12-31")
    two = pd.concat([periods, second], ignore_index=True)
    assert len(two) > two["person_id"].nunique()  # the shape that broke it

    fit = pdm.fit_perdomain(events, periods, K=3)
    rep = fidelity_report(fit, events, two, n_draws=3, min_real_records=1)

    assert rep["n_patients_real"] == two["person_id"].nunique()
    assert rep["n_patients_synthetic"] == rep["n_patients_real"]

    # real person-time is the period total, not one year per period row
    persons = np.sort(two["person_id"].unique())
    _, _, T = pdm._person_windows(two, persons)
    expect = (365 * 2 + 365) / 365.25  # odd patients: 2020-2022; even: + 2023
    assert T.mean() == pytest.approx((expect + expect + 1.0) / 2, rel=0.02)


def _counts_from_events(events, periods):
    """The aggregates `db.count_events` would return for this extract."""
    ev = events[events["domain"].isin(pdm.CLINICAL_DOMAINS) & (events["concept_id"] != 0)]
    return (
        ev.groupby(["domain", "concept_id"])
        .agg(n_patients=("person_id", "nunique"), n_events=("person_id", "size"))
        .reset_index()
    )


def test_count_path_matches_event_path(fitted):
    """Aggregating the real side in SQL must not change the report (#5).

    `db.count_events` computes patients and record totals in the warehouse so
    a 100k-patient extract never has to be materialised in Python. The counts
    carry everything prevalence and incidence use, so the two paths must
    agree to the last digit.
    """
    fit, events, periods = fitted
    counts = _counts_from_events(events, periods)

    from_events = fidelity_report(fit, events, periods, n_draws=4, min_real_records=1)
    from_counts = fidelity_report(
        fit, None, periods, n_draws=4, min_real_records=1, real_counts=counts
    )

    for key in ("prev", "inc"):
        a = from_events["concepts"][f"prevalence_{'real' if key == 'prev' else 'real'}"]
        b = from_counts["concepts"][f"prevalence_{'real' if key == 'prev' else 'real'}"]
        assert np.allclose(a, b)
    for key in ("incidence_real", "prevalence_real", "prevalence_median", "incidence_median"):
        assert np.allclose(from_events["concepts"][key], from_counts["concepts"][key]), key

    assert from_events["n_patients_real"] == from_counts["n_patients_real"]
    assert from_events["n_patients_synthetic"] == from_counts["n_patients_synthetic"]
    assert from_events["n_concepts_scored"] == from_counts["n_concepts_scored"]
    for key in from_events["coverage"]:
        assert from_events["coverage"][key] == pytest.approx(from_counts["coverage"][key])
    for name, bin_ev in from_events["bins"].items():
        bin_ct = from_counts["bins"][name]
        for key, val in bin_ev.items():
            if isinstance(val, float):
                # prevalence is a sparse-matrix mean on one path and an integer
                # ratio on the other; differences are float noise, not logic
                assert val == pytest.approx(bin_ct[key], abs=1e-9), (name, key)
            else:
                assert val == bin_ct[key], (name, key)


def test_counts_need_events_for_held_out(fitted):
    """Held-out selection cannot be done from aggregates alone."""
    fit, events, periods = fitted
    counts = _counts_from_events(events, periods)
    with pytest.raises(ValueError, match="not separable by patient"):
        fidelity_report(
            fit, None, periods, n_draws=2, real_counts=counts, held_out_person_ids=[1, 2]
        )


def test_requires_events_or_counts(fitted):
    fit, _events, periods = fitted
    with pytest.raises(ValueError, match="pass either"):
        fidelity_report(fit, None, periods, n_draws=2)
