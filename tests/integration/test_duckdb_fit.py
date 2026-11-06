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
