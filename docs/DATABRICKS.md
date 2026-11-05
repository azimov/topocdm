# Fitting the model from a Databricks-hosted CDM

`scripts/fit_databricks.py` pulls the per-domain extract from an OMOP CDM in
Databricks, fits `topocdm.perdomain.fit_perdomain`, and writes a pickled fit
plus a JSON summary.

## What it deliberately is not

The circepy execution layer is ibis-centric (`circe/execution/`, including the
Databricks backend). This script does not touch it. It uses the two pieces that
are not: **circepy's SQLRender port** for dialect translation, and the
**official Databricks SQL connector** for execution.

    OHDSI SQL (T-SQL)  ->  circe.sqlrender.translate(..., "spark")  ->  databricks.sql.connect

That is the whole connection. Everything below it is the model package.

## Install

```bash
pip install "ohdsi-circepy @ git+https://github.com/OHDSI/circepy.git@feature/decomposable-cohorts"
pip install "databricks-sql-connector[pyarrow]"
```

`[pyarrow]` lets the connector return Arrow tables (`fetchall_arrow`). Without
it the script falls back to `fetchall`, which is slower on large extracts.

## Credentials

Standard Databricks environment variables:

| variable | value |
|---|---|
| `DATABRICKS_SERVER_HOSTNAME` | `adb-….azuredatabricks.net` |
| `DATABRICKS_HTTP_PATH` | `/sql/1.0/warehouses/…` |
| `DATABRICKS_TOKEN` | personal access token |

The CDM location is `--cdm catalog.schema` (default `hive_metastore.omop`, or
`$CDM_SCHEMA`).

## Usage

```bash
# translate and print the Spark SQL; no connection, no credentials
python scripts/fit_databricks.py --cdm main.omop --dry-run

# full fit
python scripts/fit_databricks.py --cdm main.omop --k 20

# sample 20% of patients in SQL, then cap the local fit at 20k
python scripts/fit_databricks.py --cdm main.omop --sample-pct 20 --max-patients 20000
```

Outputs `results/databricks_fit.pkl` (the `PerDomainFit`) and
`results/databricks_fit.json` (summary).

## Notes

- **`--sample-pct` is a SQL filter on the translated query**:
  `pmod(hash(person_id), 100) < pct`. It is deterministic, so the same patients
  appear in every domain table. It is Spark SQL by construction, applied after
  translation.
- **`--max-patients`** is the second, local cap passed to `fit_perdomain`; it
  applies after the data have been pulled, so use `--sample-pct` for the
  network cost and `--max-patients` for the fit cost.
- **The fit runs locally** on the pulled data, not on the cluster. For a
  production database, the extraction is the expensive step; the fit is
  seconds to minutes (see `docs/RESULTS_PERDOMAIN.md` §5).
- **Date casts.** SQLRender turns `cast(x as date)` into the Databricks-safe
  `IF(try_cast(x AS DATE) IS NULL, to_date(cast(x AS STRING), 'yyyyMMdd'),
  try_cast(x AS DATE))`, which handles both date columns and `yyyyMMdd`
  integers. This is the part that would be easy to get wrong by hand-writing
  Spark SQL, and the reason to use the translator rather than a string rewrite.
