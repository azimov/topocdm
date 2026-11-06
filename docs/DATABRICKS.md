# Fitting the model from a Databricks-hosted CDM

`scripts/fit_databricks.py` reads a connection card, writes a patient sample to
a scratch schema on the cluster, pulls only that sample, fits
`topocdm.perdomain.fit_perdomain`, and writes a pickled fit plus a JSON summary.

## What it deliberately is not

The circepy execution layer is ibis-centric (`circe/execution/`, including the
Databricks backend). This script does not touch it. It uses the two pieces that
are not: **circepy's SQLRender port** for dialect translation, and the
**official Databricks SQL connector** for execution.

```
OHDSI SQL (T-SQL)  ->  circe.sqlrender.translate(..., "spark")  ->  databricks.sql.connect
```

That is the whole connection. Everything below it is the model package.

## Install

```bash
pip install -e ".[dev,databricks]"
pip install "ohdsi-circepy @ git+https://github.com/OHDSI/Circepy.git@7c32a12485fc788be9882c9ea9c7cf1cd89255d6"
```

The `databricks` extra brings `PyYAML` and `databricks-sql-connector[pyarrow]`.
The commit is the head of PR 60 (`feature/decomposable-cohorts`); the branch
name is an equivalent fallback.

## Connection card

The script reads `~/.config/ohdsi/databricks_connection.yml`:

```yaml
databricks:
  driver: "databricks"
  connection:
    server_hostname: "adb-....azuredatabricks.net"
    http_path: "/sql/1.0/warehouses/...."
    personal_access_token: "dapi...."

cdms:
  mycdm_version:
    cdm_schema: "catalog.schema"
    vocabulary_schema: "catalog.schema"
    results_schema: "scratch.schema"   # writable; the sample table goes here
```

- `--cmd mycdm_version` selects the entry. If omitted, the first entry is used.
- `cdm_schema` is the catalog and schema holding the OMOP tables.
- `results_schema` is a writable scratch location. It is required for
  `--n-patients`: the sample table is created there. The token needs
  `CREATE TABLE` and `DROP TABLE` on it.
- `vocabulary_schema` is read and recorded but not used by the extractor; it is
  kept for vocabulary joins that may be added later.
- `--cdm` and `--results-schema` override the card.
- `--host`, `--http-path` and `--token` override the connection fields; the
  environment variables `DATABRICKS_SERVER_HOSTNAME`, `DATABRICKS_HTTP_PATH`
  and `DATABRICKS_TOKEN` are also honoured.
- The token is never printed or written to the summary.

## Usage

```bash
# translate and print the SQL; no connection, no credentials
python scripts/fit_databricks.py --cmd mycdm_version --dry-run

# first fit: 10k patients, selected on the cluster
python scripts/fit_databricks.py --cmd mycdm_version --n-patients 10000 --k 20

# increase the sample; same seed, so 20k contains the 10k
python scripts/fit_databricks.py --cmd mycdm_version --n-patients 20000 --k 20

# keep the scratch table for inspection
python scripts/fit_databricks.py --cmd mycdm_version --n-patients 10000 --keep-scratch
```

Outputs `results/databricks_fit.pkl` (the `PerDomainFit`) and
`results/databricks_fit.json` (summary). The summary records the `cmd`, the
schemas, the sample table name, the requested sample size and seed, the patient
and record counts, and the fitted parameters.

## How the sample is taken

The sample is a real table in `results_schema`, not a CTE:

```sql
create or replace table scratch.schema.topocdm_sample_0_10000
as select person_id from catalog.schema.person
order by hash(person_id, 0)
limit 10000
```

Every domain query then joins to it on the cluster:

```sql
select ... from catalog.schema.condition_occurrence
  join scratch.schema.topocdm_sample_0_10000 using (person_id)
union all ...
```

This is why there is no `MATERIALIZED` keyword: the sample is materialised by
being a table, which also avoids the CTE syntax error on warehouses that reject
`AS MATERIALIZED`. The table is dropped after the fetch unless `--keep-scratch`
is passed.

The sample is deterministic for a given `--seed` and **nested**: the 10k
patients are a subset of the 20k. Only the sampled rows are downloaded; the
event tables are filtered by the join, not transferred whole.

## Notes

- **`--n-patients` is exact and nested**; there is no percentage sampling. The
  one cost is a sort of the person table, plus writing and dropping the scratch
  table.
- **`--max-patients`** is a second, local cap passed to `fit_perdomain`; it
  applies after the sample has been pulled and does not reduce the download.
- **The fit runs locally** on the pulled sample, not on the cluster. The
  extraction is the expensive step; the fit is seconds to minutes on 10k–100k
  patients.
- **Date casts.** SQLRender turns `cast(x as date)` into the Databricks-safe
  `IF(try_cast(x AS DATE) IS NULL, to_date(cast(x AS STRING), 'yyyyMMdd'),
  try_cast(x AS DATE))`, which handles both date columns and `yyyyMMdd`
  integers. This is the part that would be easy to get wrong by hand-writing
  Spark SQL, and the reason to use the translator rather than a string rewrite.
- **Dry run without credentials.** `--dry-run --cdm main.omop
  --results-schema scratch.scratch --n-patients 10000` prints placeholder SQL
  without reading the card, which is what the setup script uses.
