# Testing

The previous tests never executed SQL against a database. The Databricks path
was checked with a SQL shim and a dry run, which is why the multiple-period bug
and the `MATERIALIZED` error reached you. The tests now have a live database
tier.

## Tiers

| tier | command | needs |
|---|---|---|
| unit | `pytest -q` | nothing; integration and Databricks tests skip |
| integration (Postgres) | `pytest -q -m integration` | `CDM_DSN`, `CDM_SCHEMA` |
| Databricks live | `pytest -q -m databricks` | connection card, circepy, connector |
| CI | push / PR | Postgres 16 service, fixture CDM |

## What each test does

- `tests/integration/test_cdm_fit.py` executes the **real extract SQL** against a
  live OMOP Postgres database, then runs `fit_perdomain` and
  `clinical_incidence`. `demo_cdm` has 2,694 patients and 5,343 observation
  periods, so it exercises the multiple-period case from issue #1.
- `tests/integration/test_databricks_live.py` runs
  `scripts/fit_databricks.py --n-patients 100 --k 2` against the configured
  warehouse. It is skipped unless the connection card, circepy and the
  Databricks connector are present. **This is the only test that proves the
  Databricks path**; run it before trusting a warehouse fit.
- `tests/test_databricks_config.py` checks the `cdms`/`cmds` loader and the SQL
  contract: the sample is a scratch table, the extract joins it, and there is
  no `WITH ... AS MATERIALIZED` / `WITH keep`.
- `tests/test_perdomain.py::test_multiple_observation_periods_per_person` is the
  unit regression for the period aggregation.

## Local runs

Against the OHDSI lab `demo_cdm` (the multi-period schema):

```bash
CDM_DSN=postgresql://postgres:mypass@localhost:5432/postgres \
CDM_SCHEMA=demo_cdm \
MPLCONFIGDIR=./.mplconfig .venv/bin/python -m pytest -q -m integration
# 2 passed
```

Against the CI fixture in a throwaway database:

```bash
docker exec ohdsi-lab-postgres createdb -U postgres omop_test
docker exec -i ohdsi-lab-postgres psql -U postgres -d omop_test \
  -v ON_ERROR_STOP=1 -f - < tests/fixtures/omop_test.sql
CDM_DSN=postgresql://postgres:mypass@localhost:5432/omop_test \
CDM_SCHEMA=cdm \
MPLCONFIGDIR=./.mplconfig .venv/bin/python -m pytest -q
# 35 passed, 1 skipped (the skipped one is the live Databricks test)
```

Against the real warehouse:

```bash
MPLCONFIGDIR=./.mplconfig .venv/bin/python -m pytest -q -m databricks
```

It reads `~/.config/ohdsi/databricks_connection.yml`, extracts 100 patients,
fits `K=2`, and asserts the summary is non-empty. It skips if the card or
`circe` is missing.

## CI

`.github/workflows/tests.yml` starts a Postgres 16 service, loads
`tests/fixtures/omop_test.sql`, and runs `pytest -q` on Python 3.10, 3.11 and
3.12. The fixture has five patients and patient 1 has two observation periods.
`ruff check .` runs before the tests (syntax and undefined names only; the
full style rules are not enabled yet).

## What is still not tested

There is no Databricks warehouse in CI; no credentials belong in the repo. The
live test is opt-in and must be run in the environment that has the connection
card. If it fails, the traceback is the warehouse path, not a shim.

## Standards taken from circepy

- pytest markers and `conftest.py` fixtures instead of ad hoc scripts
- a CI matrix on the supported Python versions
- a real database service in CI, with a checked-in fixture
- a separate `test` extra for database test dependencies
- an error-only lint gate (`ruff` with `E9`, `F63`, `F7`, `F82`)
- the live integration test is opt-in and skipped without the external service

Not yet adopted: `tox`, `mypy`, and coverage reporting.
