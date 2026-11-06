"""Shared fixtures for the integration tests.

The unit tests need no database. The integration tests run only when CDM_DSN
points at a live OMOP Postgres database; CI supplies one, and locally you can
use the OHDSI lab (demo_cdm has multiple observation periods per patient).
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def cdm_dsn():
    dsn = os.environ.get("CDM_DSN")
    if not dsn:
        pytest.skip("set CDM_DSN to a Postgres OMOP test database")
    return dsn


@pytest.fixture(scope="session")
def cdm_schema():
    return os.environ.get("CDM_SCHEMA", "demo_cdm")


@pytest.fixture(scope="session")
def cdm_results_schema():
    return os.environ.get("CDM_RESULTS_SCHEMA", "public")


@pytest.fixture(scope="session")
def cdm_conn(cdm_dsn):
    psycopg = pytest.importorskip("psycopg")
    conn = psycopg.connect(cdm_dsn)
    yield conn
    conn.close()
