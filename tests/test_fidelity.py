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
