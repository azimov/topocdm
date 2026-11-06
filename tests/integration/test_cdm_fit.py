"""The shared run_fit pipeline against a live OMOP Postgres database.

This is the same Python path the Databricks backend uses. demo_cdm has 2,694
patients and 5,343 observation periods, so it exercises the multiple-period
case that broke the Databricks fit (issue #1).
"""

from __future__ import annotations

import numpy as np
import pytest

from topocdm import db
from topocdm import perdomain as pdm

pytestmark = pytest.mark.integration

SAMPLE = 100


def test_run_fit_live_cdm(cdm_dsn, cdm_schema, cdm_results_schema):
    backend = db.PostgresBackend(cdm_dsn)
    try:
        fit, summary = db.run_fit(
            backend, cdm_schema, cdm_results_schema,
            n_patients=SAMPLE, seed=0, k=0)
    finally:
        backend.close()

    assert summary["patients"] > 0
    assert summary["records"] > 0
    assert summary["patients"] == fit.meta["n_patients"]
    assert summary["period_rows"] >= summary["patients"]
    assert len(fit.u) == summary["patients"]
    assert len(fit.eta) == fit.n_concepts
    assert np.isfinite(fit.u).all()
    assert np.isfinite(fit.eta).all()


def test_extract_sample_live_cdm(cdm_dsn, cdm_schema, cdm_results_schema):
    backend = db.PostgresBackend(cdm_dsn)
    try:
        events, periods, table = db.extract_sample(
            backend, cdm_schema, cdm_results_schema,
            n_patients=SAMPLE, seed=0)
    finally:
        backend.close()

    assert table is not None
    assert len(events) > 0
    n_patients = periods["person_id"].nunique()
    assert n_patients > 0
    assert len(periods) >= n_patients

    X, concept_ids, persons, T = pdm.clinical_incidence(events, periods)
    assert X.shape[0] == len(persons) == len(T)
    assert X.shape[1] == len(concept_ids)
    assert np.isfinite(T).all()
