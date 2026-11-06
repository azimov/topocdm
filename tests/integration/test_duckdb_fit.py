"""The same run_fit pipeline on DuckDB, with no server.

This is the "runs here first" test: the SQL extraction, the scratch sample,
the fit and the drop are the same code the Databricks backend uses.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from topocdm import db

pytestmark = pytest.mark.integration


def _fixture(conn):
    conn.execute("create table person(person_id bigint)")
    conn.executemany("insert into person values (?)",
                     [(i,) for i in range(1, 6)])

    conn.execute("""
        create table observation_period(
            person_id bigint,
            observation_period_start_date date,
            observation_period_end_date date)
    """)
    conn.executemany("insert into observation_period values (?, ?, ?)", [
        (1, date(2020, 1, 1), date(2020, 12, 31)),
        (1, date(2021, 1, 1), date(2022, 12, 31)),
        (2, date(2020, 1, 1), date(2022, 12, 31)),
        (3, date(2020, 1, 1), date(2022, 12, 31)),
        (4, date(2020, 1, 1), date(2022, 12, 31)),
        (5, date(2020, 1, 1), date(2022, 12, 31)),
    ])

    conn.execute("""
        create table condition_occurrence(
            person_id bigint, condition_concept_id integer,
            condition_start_date date, condition_end_date date)
    """)
    cond = []
    for p in range(1, 6):
        for g in range(20):
            s = date(2020, 1, 1) + timedelta(days=(g * 17) % 900)
            cond.append((p, 100 + g % 3, s, s + timedelta(days=30)))
    conn.executemany(
        "insert into condition_occurrence values (?, ?, ?, ?)", cond)

    conn.execute("""
        create table drug_exposure(
            person_id bigint, drug_concept_id integer,
            drug_exposure_start_date date, drug_exposure_end_date date)
    """)
    drug = []
    for p in range(1, 6):
        for g in range(20):
            s = date(2020, 1, 1) + timedelta(days=(g * 23) % 900)
            drug.append((p, 200 + g % 3, s, s + timedelta(days=60)))
    conn.executemany(
        "insert into drug_exposure values (?, ?, ?, ?)", drug)

    conn.execute("""
        create table visit_occurrence(
            person_id bigint, visit_concept_id integer,
            visit_start_date date, visit_end_date date)
    """)
    visits = []
    for p in range(1, 6):
        for g in range(5):
            s = date(2020, 1, 1) + timedelta(days=(g * 31) % 900)
            visits.append((p, 9201, s, s))
    conn.executemany(
        "insert into visit_occurrence values (?, ?, ?, ?)", visits)

    conn.execute("""
        create table procedure_occurrence(
            person_id bigint, procedure_concept_id integer, procedure_date date)
    """)
    proc = []
    for p in range(1, 6):
        for g in range(10):
            proc.append((p, 300 + g % 2,
                         date(2020, 1, 1) + timedelta(days=(g * 41) % 900)))
    conn.executemany(
        "insert into procedure_occurrence values (?, ?, ?)", proc)

    conn.execute("""
        create table measurement(
            person_id bigint, measurement_concept_id integer, measurement_date date)
    """)
    meas = []
    for p in range(1, 6):
        for g in range(15):
            meas.append((p, 400 + g % 2,
                         date(2020, 1, 1) + timedelta(days=(g * 43) % 900)))
    conn.executemany("insert into measurement values (?, ?, ?)", meas)

    conn.execute("""
        create table observation(
            person_id bigint, observation_concept_id integer, observation_date date)
    """)
    obs = []
    for p in range(1, 6):
        for g in range(10):
            obs.append((p, 500 + g % 2,
                        date(2020, 1, 1) + timedelta(days=(g * 47) % 900)))
    conn.executemany("insert into observation values (?, ?, ?)", obs)

    conn.execute("create table death(person_id bigint, death_date date)")
    conn.execute("insert into death values (5, date '2022-06-01')")


def test_run_fit_on_duckdb():
    backend = db.DuckDBBackend(":memory:")
    try:
        _fixture(backend.conn)
        fit, summary = db.run_fit(backend, "main", "main",
                                  n_patients=5, seed=0, k=0)
    finally:
        backend.close()

    assert summary["patients"] == 5
    assert summary["period_rows"] == 6          # patient 1 has two periods
    assert summary["records"] > 0
    assert summary["patients"] == fit.meta["n_patients"]
    assert len(fit.u) == 5
