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

## Sampling a large CDM

For a large CDM, do not pull the event tables to the machine. `--n-patients`
selects a fixed-size patient sample **on the cluster**, joins every domain
table to it there, and downloads only the sampled rows.

```bash
python scripts/fit_databricks.py --cdm main.omop --n-patients 10000 --k 20
```

The sample is `order by hash(person_id, seed) limit n`: deterministic for a
given `--seed` and **nested**, so the 10k patients are a subset of the 20k.
Start at 10k, then 20k, 50k with the same seed; the smaller runs are always
subsets of the larger, which makes the increase comparable and auditable.

The cost is one sort of the person table. The event tables are filtered by the
join and are not downloaded whole. `--dry-run --n-patients 10000` prints the
exact Spark SQL — the `with keep as materialized (...)` CTE and the
`join keep using (person_id)` in every union branch — without connecting.

Programmatic use:

```python
from scripts.fit_databricks import connect, sample_cdm
conn = connect(args)
events, periods = sample_cdm(conn, n_patients=10000, cdm="main.omop", seed=0)
```

`sample_cdm` returns the same `(events, periods)` frames the fit expects.

## Notes

- **`--n-patients`** is the preferred sample for a large CDM: exact, nested,
  and executed entirely on the cluster. `--sample-pct` remains for a percentage
  filter (`pmod(hash(person_id), 100) < pct`); it is approximate and not nested
  across percentages. Do not combine the two.
- **`--max-patients`** is a second, local cap passed to `fit_perdomain`; it
  applies after the sample has been pulled, so it does not reduce the download.
  With `--n-patients` it is redundant unless set lower.
- **The fit runs locally** on the pulled data, not on the cluster. For a
  production database, the extraction is the expensive step; the fit is
  seconds to minutes (see `docs/RESULTS_PERDOMAIN.md` §5).
- **Date casts.** SQLRender turns `cast(x as date)` into the Databricks-safe
  `IF(try_cast(x AS DATE) IS NULL, to_date(cast(x AS STRING), 'yyyyMMdd'),
  try_cast(x AS DATE))`, which handles both date columns and `yyyyMMdd`
  integers. This is the part that would be easy to get wrong by hand-writing
  Spark SQL, and the reason to use the translator rather than a string rewrite.
