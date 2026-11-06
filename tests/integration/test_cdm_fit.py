"""Fit the per-domain model against a live OMOP Postgres database.

This is the test that was missing: it executes the real extract SQL against a
real CDM, then runs the fit. demo_cdm has 2,694 patients and 5,343 observation
periods, so it exercises the multiple-observation-period case that broke the
Databricks path (issue #1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from topocdm import perdomain as pdm

pytestmark = pytest.mark.integration

SAMPLE = 100

EVENTS_SQL = """
with keep as (
    select person_id from {schema}.person order by person_id limit {n}
)
select person_id, 'condition' as domain, condition_concept_id as concept_id,
       condition_start_date as start_date, condition_end_date as end_date
  from {schema}.condition_occurrence join keep using (person_id)
union all
select person_id, 'drug', drug_concept_id,
       drug_exposure_start_date, drug_exposure_end_date
  from {schema}.drug_exposure join keep using (person_id)
union all
select person_id, 'visit', visit_concept_id,
       visit_start_date, visit_end_date
  from {schema}.visit_occurrence join keep using (person_id)
union all
select person_id, 'procedure', procedure_concept_id,
       procedure_date, cast(null as date)
  from {schema}.procedure_occurrence join keep using (person_id)
union all
select person_id, 'measurement', measurement_concept_id,
       measurement_date, cast(null as date)
  from {schema}.measurement join keep using (person_id)
union all
select person_id, 'observation', observation_concept_id,
       observation_date, cast(null as date)
  from {schema}.observation join keep using (person_id)
union all
select person_id, 'death', 0, death_date, cast(null as date)
  from {schema}.death join keep using (person_id)
"""

PERIODS_SQL = """
with keep as (
    select person_id from {schema}.person order by person_id limit {n}
)
select person_id, observation_period_start_date as start_date,
       observation_period_end_date as end_date
  from {schema}.observation_period join keep using (person_id)
"""


def _read(conn, sql):
    return pd.read_sql_query(sql, conn)


def test_extract_and_fit_live_cdm(cdm_conn, cdm_schema):
    events = _read(cdm_conn, EVENTS_SQL.format(schema=cdm_schema, n=SAMPLE))
    periods = _read(cdm_conn, PERIODS_SQL.format(schema=cdm_schema, n=SAMPLE))
    for df in (events, periods):
        for col in ("start_date", "end_date"):
            df[col] = pd.to_datetime(df[col])

    assert len(events) > 0
    n_patients = periods["person_id"].nunique()
    assert n_patients > 0
    # demo_cdm has several periods per patient; the fit must count people, not
    # period rows. This is the assertion that would have caught issue #1.
    assert len(periods) >= n_patients

    fit = pdm.fit_perdomain(events, periods, K=0)

    assert fit.meta["n_patients"] == n_patients
    assert len(fit.u) == n_patients
    assert len(fit.eta) == fit.n_concepts
    assert np.isfinite(fit.u).all()
    assert np.isfinite(fit.eta).all()


def test_clinical_incidence_live_cdm(cdm_conn, cdm_schema):
    events = _read(cdm_conn, EVENTS_SQL.format(schema=cdm_schema, n=SAMPLE))
    periods = _read(cdm_conn, PERIODS_SQL.format(schema=cdm_schema, n=SAMPLE))
    for df in (events, periods):
        for col in ("start_date", "end_date"):
            df[col] = pd.to_datetime(df[col])

    X, concept_ids, persons, T = pdm.clinical_incidence(events, periods)
    assert X.shape[0] == len(persons) == len(T)
    assert X.shape[1] == len(concept_ids)
    assert np.isfinite(T).all()
