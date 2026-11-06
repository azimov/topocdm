# Testing

The old suite never executed SQL against a database. The Databricks path was
checked with a SQL shim and a dry run, which is why the multiple-period bug and
the `MATERIALIZED` error reached production use. The tests now run the same
`topocdm.db.run_fit` pipeline against Postgres and DuckDB before Databricks.

## Tiers

| tier | command | needs |
|---|---|---|
| unit | `pytest -q -m "not integration"` | nothing |
| unit + DuckDB | `pytest -q` | nothing; DuckDB is in-process |
| Postgres integration | `pytest -q -m integration` | `CDM_DSN`, `CDM_SCHEMA`, `CDM_RESULTS_SCHEMA` |
| Databricks live | `pytest -q -m databricks` | connection card, circepy, connector |
| CI | push / PR | Postgres 16 service, fixture CDM |

Every run enforces a coverage floor (`fail_under` in `pyproject.toml`, currently
65 %). Use `--no-cov` when running a single file so the floor does not apply.

## What each test does

- `tests/integration/test_duckdb_fit.py` loads
  `tests/fixtures/duckdb_test.sql` into an in-memory DuckDB CDM, including a
  patient with **two observation periods**, and runs `run_fit`: sample table,
  extraction, fit, drop. No server. The fixture is checked in as SQL so the
  data it tests against is reviewable.
- `tests/integration/test_cdm_fit.py` runs the same `run_fit` against a live
  OMOP Postgres database. `demo_cdm` has 2,694 patients and 5,343 periods, so
  it exercises the multiple-period case from issue #1.
- `tests/integration/test_databricks_live.py` runs
  `scripts/fit_databricks.py --n-patients 100 --k 2` against the configured
  warehouse. It is skipped unless the card, circepy and the connector exist.
  **This is the only test that proves the warehouse path.**
- `tests/test_databricks_config.py` checks the `cdms`/`cmds` loader, the
  scratch-table SQL, the absence of `WITH ... AS MATERIALIZED`, and the
  Postgres/DuckDB/Databricks sample syntax.
- `tests/test_sql.py` covers the portable SQL layer: per-dialect date
  arithmetic, the step plan and its dependencies, and checkpoint skip/execute/
  lineage behaviour.
- `tests/test_fast.py` compares the optional C kernel against the dense NumPy
  likelihood (single- and multi-group) and checks the no-library fallback.
- `tests/test_release.py` covers the release artefact: no per-patient terms, a
  normalised group prior, and the fixed-density invariant.
- `tests/test_omop.py` covers OMOP table generation: person/period keys, the
  domain-to-table mapping, and vocabulary validation.
- `tests/test_perdomain.py::test_multiple_observation_periods_per_person` is the
  unit regression for the period aggregation.

## The pipeline is shared

`topocdm.db.run_fit` is the only implementation. The backend is the only
dialect-specific part:

| backend | patient sample order |
|---|---|
| Postgres | `md5(person_id::text || ':seed')` |
| DuckDB | `hash(person_id::varchar || ':seed')` |
| Databricks | `hash(person_id, seed)` |

Databricks additionally translates the extraction with SQLRender
(`db.to_spark`). Postgres and DuckDB run the same templates unchanged.

## Local runs

No database server, only DuckDB:

```bash
MPLCONFIGDIR=./.mplconfig .venv/bin/python -m pytest -q tests/integration/test_duckdb_fit.py
# 1 passed
```

The OHDSI lab `demo_cdm`:

```bash
CDM_DSN=postgresql://postgres:mypass@localhost:5432/postgres \
CDM_SCHEMA=demo_cdm CDM_RESULTS_SCHEMA=public \
MPLCONFIGDIR=./.mplconfig .venv/bin/python -m pytest -q -m integration
# 3 passed (two Postgres, one DuckDB)
```

The CI fixture in a throwaway database:

```bash
docker exec ohdsi-lab-postgres createdb -U postgres omop_test
docker exec -i ohdsi-lab-postgres psql -U postgres -d omop_test \
  -v ON_ERROR_STOP=1 -f - < tests/fixtures/omop_test.sql
CDM_DSN=postgresql://postgres:mypass@localhost:5432/omop_test \
CDM_SCHEMA=cdm CDM_RESULTS_SCHEMA=public \
MPLCONFIGDIR=./.mplconfig .venv/bin/python -m pytest -q
# 37 passed, 1 skipped (the skipped one is the live Databricks test)
```

Without `CDM_DSN`:

```bash
MPLCONFIGDIR=./.mplconfig .venv/bin/python -m pytest -q
# 53 passed, 3 skipped (two Postgres, one Databricks)
```

The real warehouse:

```bash
MPLCONFIGDIR=./.mplconfig .venv/bin/python -m pytest -q -m databricks
```

It reads `~/.config/ohdsi/databricks_connection.yml`, extracts 100 patients,
fits `K=2`, and asserts the summary is non-empty.

## CI

Five workflows, mirroring the gates you can run locally:

| workflow | what it runs |
|---|---|
| `tests.yml` | `tox` on Python 3.10–3.14, Postgres 16 service + `omop_test.sql`, Codecov upload |
| `ruff.yml` | `ruff check .` and `ruff format --check .` (latest ruff) |
| `typecheck.yml` | `mypy topocdm` |
| `cross-platform.yml` | unit tests on Windows + macOS, and a wheel build + clean-venv smoke test |
| (local) | `tox`, `tox -e py311` reproduce the test matrix |

The CI fixture has five patients and patient 1 has two observation periods.

## What is still not tested

There is no Databricks warehouse in CI; no credentials belong in the repo. The
live test is opt-in and must be run in the environment that has the connection
card. If it fails, the traceback is the warehouse path, not a shim.

## Standards taken from circepy

- pytest markers and `conftest.py` fixtures instead of ad hoc scripts
- a CI matrix on the supported Python versions (3.10–3.14)
- a real database service in CI, with a checked-in fixture
- a separate `test` extra for database test dependencies
- the full ruff rule set (`E, F, I, UP, B, C4, SIM`) plus `ruff format`
- `tox` + `tox-gh-actions` so local and CI runs are identical
- `mypy` with a shipped `py.typed`
- coverage reporting with a floor and Codecov project/patch targets
- a wheel build and clean-venv smoke test
- the live integration test is opt-in and skipped without the external service

Run the gates locally with `CONTRIBUTING.md`'s four commands, or `tox` for the
full interpreter matrix.
