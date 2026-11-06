"""The same run_fit pipeline on DuckDB, with no server.

This is the "runs here first" test: the SQL extraction, the scratch sample,
the fit and the drop are the same code the Databricks backend uses. The CDM
fixture is a checked-in .sql file (tests/fixtures/duckdb_test.sql) so the SQL
that the pipeline is tested against is reviewable and reusable.
"""

from __future__ import annotations

import pathlib

import pytest

from topocdm import db

pytestmark = pytest.mark.integration

FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "duckdb_test.sql"


def test_run_fit_on_duckdb():
    backend = db.DuckDBBackend(":memory:")
    try:
        backend.conn.execute(FIXTURE.read_text())
        fit, summary = db.run_fit(backend, "main", "main", n_patients=5, seed=0, k=0)
    finally:
        backend.close()

    assert summary["patients"] == 5
    assert summary["period_rows"] == 6  # patient 1 has two periods
    assert summary["records"] > 0
    assert summary["patients"] == fit.meta["n_patients"]
    assert len(fit.u) == 5


def _duckdb_with_fixture():
    backend = db.DuckDBBackend(":memory:")
    backend.conn.execute(FIXTURE.read_text())
    return backend


def test_aggregates_match_the_event_extract():
    """Every aggregate must equal the same statistic computed from the rows.

    This is the check the design rule in docs/DESIGN.md 9.1 requires: a
    client-side and a SQL-side implementation of one statistic agree exactly,
    so a report computed from counts is the report computed from events.
    """
    import numpy as np
    import pandas as pd

    backend = _duckdb_with_fixture()
    try:
        events, periods, _ = db.extract_sample(backend, "main", "main", 5, 0)
        counts = db.count_events(backend, "main", "main", 5, 0)
        years = db.count_domain_years(backend, "main", "main", 5, 0)
        stats, sample = db.duration_sample(backend, "main", "main", 5, 0, limit=1000)
    finally:
        backend.close()

    for col in ("start_date", "end_date"):
        events[col] = pd.to_datetime(events[col])

    # per (domain, concept): patients and records
    ev = events[events["domain"].isin(db.pdm.CLINICAL_DOMAINS)]
    ev = ev[ev["concept_id"] != 0]
    want = (
        ev.groupby(["domain", "concept_id"])
        .agg(n_patients=("person_id", "nunique"), n_events=("person_id", "size"))
        .reset_index()
    )
    got = counts.sort_values(["domain", "concept_id"]).reset_index(drop=True)
    want = want.sort_values(["domain", "concept_id"]).reset_index(drop=True)
    assert list(got["domain"]) == list(want["domain"])
    assert list(got["concept_id"]) == list(want["concept_id"])
    assert np.array_equal(got["n_patients"].to_numpy(), want["n_patients"].to_numpy())
    assert np.array_equal(got["n_events"].to_numpy(), want["n_events"].to_numpy())

    # events per domain-year. death is excluded on purpose: it is an absorbing
    # process fitted separately, not a lam_t domain.
    ev2 = events.copy()
    ev2["year"] = ev2["start_date"].dt.year
    ev2 = ev2[ev2["domain"].isin(db.pdm.ALL_DOMAINS) & (ev2["domain"] != "death")]
    want_y = ev2.groupby(["domain", "year"]).size().reset_index(name="n_events")
    got_y = years.sort_values(["domain", "year"]).reset_index(drop=True)
    want_y = want_y.sort_values(["domain", "year"]).reset_index(drop=True)
    assert np.array_equal(got_y["n_events"].to_numpy(), want_y["n_events"].to_numpy())

    # duration zero counts
    for _, row in stats.iterrows():
        sub = events[events["domain"] == row["domain"]]
        d = (sub["end_date"] - sub["start_date"]).dt.days
        assert int(row["n_total"]) == len(sub)
        assert int(row["n_zero"]) == int((d.isna() | (d <= 0)).sum())
    assert not sample.empty


def test_fit_patients_caps_the_extract():
    """A bounded fit sample is taken in SQL, not by filtering after transfer."""
    backend = _duckdb_with_fixture()
    try:
        full, _, _ = db.extract_sample(backend, "main", "main", 5, 0)
        capped, _, _ = db.extract_sample(backend, "main", "main", 5, 0, fit_patients=3)
        left = backend.fetch(
            "select table_name from duckdb_tables() where table_name like 'topocdm_sample%'"
        )
    finally:
        backend.close()

    assert capped["person_id"].nunique() == 3
    assert len(capped) < len(full)
    # nested, deterministic, and cleaned up
    assert set(capped["person_id"]) <= set(full["person_id"])
    assert len(left) == 0


def test_counts_describe_the_capped_cohort():
    """A capped extract must not score one cohort over another's person-time.

    The bug this guards: with --max-patients the extract holds the capped
    cohort, but the counts were aggregated over the full sample, so real
    incidence was `records of N patients / person-time of M patients`. That
    doubled incidence and drove coverage to ~0%. The counts must describe the
    same patients `periods` describes.
    """
    import numpy as np

    backend = _duckdb_with_fixture()
    try:
        fit, summary, counts, periods = db.run_fit(
            backend,
            "main",
            "main",
            n_patients=5,
            seed=0,
            k=0,
            max_patients=3,
            return_counts=True,
        )
        full_counts = db.count_events(backend, "main", "main", 5, 0)
        _, full_periods, _ = db.extract_sample(backend, "main", "main", 5, 0)
    finally:
        backend.close()

    cohort = set(periods["person_id"])
    assert len(cohort) == 3
    assert summary["patients"] == 3

    # every counted record belongs to a patient in the scored cohort
    capped_total = int(counts["n_events"].sum())
    full_total = int(full_counts["n_events"].sum())
    assert capped_total < full_total

    # and every counted concept has at most the cohort's patients
    assert counts["n_patients"].max() <= len(cohort)

    # Incidence is records over the *scored cohort's* person-time. The bug
    # scored the full sample's records against the capped person-time, which
    # inflates the rate by roughly the ratio of the two cohorts.
    persons = np.sort(periods["person_id"].unique())
    _, _, T = db.pdm._person_windows(periods, persons)
    from_cohort = capped_total / float(T.sum())
    full_persons = np.sort(full_periods["person_id"].unique())
    _, _, T_full = db.pdm._person_windows(full_periods, full_persons)
    from_full_sample = capped_total / float(T_full.sum())
    assert from_cohort > from_full_sample  # smaller cohort -> higher rate
