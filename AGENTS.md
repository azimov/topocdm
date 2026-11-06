# AGENTS.md

Instructions for coding agents working on this repository. Read this before
changing the model or the database path.

## What this is

`topocdm` is a research prototype: a fixed-density generative model of an OMOP
CDM, with a likelihood. It fits global parameters and generates synthetic CDMs.
The current state is documented in `docs/FIDELITY_REPORT.md` and
`docs/SUBJECT_SPECTRA.md`; the model fits the marginals and incidence, but
per-concept prevalence (especially the common bin) and the subject-graph
spectrum are not solved. Do not describe it as validated on a production CDM.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,test,databricks]"
.venv/bin/pip install "ohdsi-circepy @ git+https://github.com/OHDSI/Circepy.git@7c32a12485fc788be9882c9ea9c7cf1cd89255d6"

.venv/bin/pytest -q          # unit + DuckDB; Postgres/Databricks skip
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy topocdm
```

The C kernel is optional:

```bash
python setup.py build_ext --inplace      # or: bash scripts/build_fast.sh
```

The Python kernel must remain correct; tests must pass without `_fast*.so`.
`topocdm.fast` finds the library by platform suffix, so the build may land as
`_fast.cpython-311-darwin.so`, `_fast.so`, or `_fast.pyd`.

## Repository map

| path | purpose |
|---|---|
| `topocdm/perdomain.py` | the per-domain temporal model: `fit_perdomain`, `generate_cohort`, durations, visits, eras, `clinical_incidence` |
| `topocdm/db.py` | the one CDM pipeline: `run_fit`, `extract_sample`, `PostgresBackend`, `DuckDBBackend`, `DatabricksBackend`, `to_spark` |
| `scripts/fit_databricks.py` | thin CLI over `db.run_fit`; `--backend databricks|postgres|duckdb` |
| `topocdm/scalable.py`, `topocdm/distributed.py` | the older streaming/map-reduce fit; not the CDM path |
| `topocdm/release.py`, `topocdm/omop.py`, `topocdm/model.py` | release artefact, OMOP table generation, the original mixture model |
| `tests/` | unit tests, `tests/integration/` for live DuckDB/Postgres/Databricks |
| `docs/` | model spec, fidelity report, Databricks guide, testing guide |

## Backends

`db.run_fit` is the only extraction/fit path. The backend is the only
dialect-specific part:

| backend | sample order |
|---|---|
| Postgres | `md5(person_id::text || ':seed')` |
| DuckDB | `hash(person_id::varchar || ':seed')` |
| Databricks | `hash(person_id, seed)` + SQLRender translation |

The sample is a real table in `results_schema`, not a CTE. The warehouse
rejected `WITH ... AS MATERIALIZED`; do not reintroduce it.

## Invariants and traps

- **Aggregate in the database, not in the client.** Never stream a large object
  to Python if a `group by` will do; the engine is the only component that can
  scan a fact table without materialising it. Ask of every new CDM touch point:
  *could this be an aggregate?* Rows are a last resort and must be bounded
  (the coupling fit takes at most 5,000 patients). The client-side and
  SQL-side paths must agree bit-for-bit — count integers and divide once, or
  they differ by ~1e-16 and coverage flips on interval bounds. See
  `docs/DESIGN.md` §9.1 and `db.COUNT_EVENTS`.
- **`K` vs `k`.** `--k` / `K` is the coupling rank. `fit.k` is the Gamma shape
  of the patient propensity. They are different parameters.
- **The sample is patients, not period rows.** `--n-patients` is unique people.
  A patient can have several observation periods, so `period_rows` can exceed
  the sample. `db.extract_sample` raises if unique patients exceed the request.
- **The connection card key is `cdms`** (plural of CDM), with `cmds` accepted
  as a legacy fallback.
- **No per-patient parameters in the release.** `u_i` and the latent `z_i` are
  fit diagnostics, not released fields. Do not add a per-patient artefact.
- **Do not tune a hyperparameter on the metric you report.** Held-out data, or a
  fixed prior choice, not the fidelity gate.
- **No patient data in the repository.** `data/`, `.secrets/`, PDFs and pickles
  are gitignored. The fixtures are Synthea/MIMIC demo; never commit a real CDM
  extract.
- **The C kernel must not be required.** The pure-Python fallback is part of
  the supported path; run the tests once without `_fast.so` if you touch
  likelihood code.

## Tests

```bash
# unit + in-process DuckDB
.venv/bin/pytest -q

# live Postgres; demo_cdm has multiple periods per patient
CDM_DSN=postgresql://postgres:mypass@localhost:5432/postgres \
CDM_SCHEMA=demo_cdm CDM_RESULTS_SCHEMA=public \
.venv/bin/pytest -q -m integration

# full CI fixture
CDM_DSN=postgresql://postgres:mypass@localhost:5432/omop_test \
CDM_SCHEMA=cdm CDM_RESULTS_SCHEMA=public \
.venv/bin/pytest -q

# live warehouse (needs the connection card, circepy, connector)
.venv/bin/pytest -q -m databricks
```

`docs/TESTING.md` has the full matrix. If you change the SQL or the sample,
add a regression to `tests/test_databricks_config.py` and run the DuckDB and
Postgres integration tests.

## Quality gates

CI runs these four; run them locally before pushing:

```bash
.venv/bin/ruff check .             # E, F, I, UP, B, C4, SIM
.venv/bin/ruff format --check .    # formatting
.venv/bin/mypy topocdm             # types (py.typed shipped)
.venv/bin/pytest -q                # unit + in-process DuckDB
```

`pytest` enforces a coverage floor from `pyproject.toml`; use `--no-cov` when
running a single test file. `tox` reproduces the full 3.10–3.14 matrix.

## Working agreement

- Keep the pipeline in `db.py`; do not add a second extraction or fit path.
- Add a test for every bug fix. The multiple-period bug (#1) and the sample
  count display bug both had tests added after the fact.
- Run `ruff check .`, `ruff format --check .`, `mypy topocdm` and `pytest -q`
  before committing.
- Keep mechanical lint/format changes in a separate commit from behaviour.
- Update `docs/FIDELITY_REPORT.md` if the model or the sample changes.
- Do not publish credentials, drafts or data. The public repository must not
  contain a connection card or a `.secrets/` directory.

## Current known issues

- Per-concept prevalence fails the gate, especially the common bin
  (`docs/FIDELITY_REPORT.md`).
- The subject-graph spectrum is not reproduced; see `docs/SUBJECT_SPECTRA.md`.
- The model has not been fitted to a production CDM.
- The live Databricks path is exercised only by the opt-in `databricks` test.

## Docs

`docs/MODEL_V2.md` (spec), `docs/FIDELITY_REPORT.md` (current numbers),
`docs/DATABRICKS.md` (connection card, backends, sample), `docs/TESTING.md`
(test tiers), `docs/SUBJECT_SPECTRA.md` (subject-graph spectra).
