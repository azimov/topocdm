"""
Database backends and the shared CDM fit pipeline.

The Databricks path used to be the only executable path and was tested with a
SQL shim. The pipeline is now backend-agnostic: the same Python code runs
against Postgres and DuckDB first, and Databricks is one backend among three.

    backend = PostgresBackend(dsn)          # or DuckDBBackend(path)
    fit, summary = run_fit(backend, cdm, results, n_patients=100)

The only dialect-specific pieces are the patient-sample SQL and the SQLRender
translation for Databricks. Everything else -- the extract, the fit, the
summary, the drop -- is shared.
"""

from __future__ import annotations

import json
import os
import pickle

import pandas as pd

from . import perdomain as pdm


EXTRACT_EVENTS = """
select person_id, 'condition' as domain, condition_concept_id as concept_id,
       cast(condition_start_date as date) as start_date,
       cast(condition_end_date as date) as end_date
  from {cdm}.condition_occurrence {join}
union all
select person_id, 'drug', drug_concept_id,
       cast(drug_exposure_start_date as date),
       cast(drug_exposure_end_date as date)
  from {cdm}.drug_exposure {join}
union all
select person_id, 'visit', visit_concept_id,
       cast(visit_start_date as date),
       cast(visit_end_date as date)
  from {cdm}.visit_occurrence {join}
union all
select person_id, 'procedure', procedure_concept_id,
       cast(procedure_date as date),
       cast(null as date)
  from {cdm}.procedure_occurrence {join}
union all
select person_id, 'measurement', measurement_concept_id,
       cast(measurement_date as date),
       cast(null as date)
  from {cdm}.measurement {join}
union all
select person_id, 'observation', observation_concept_id,
       cast(observation_date as date),
       cast(null as date)
  from {cdm}.observation {join}
union all
select person_id, 'death', 0,
       cast(death_date as date),
       cast(null as date)
  from {cdm}.death {join}
"""

EXTRACT_PERIODS = """
select person_id,
       cast(observation_period_start_date as date) as start_date,
       cast(observation_period_end_date as date) as end_date
  from {cdm}.observation_period {join}
"""


def build_extract_sql(cdm: str, join: str = ""):
    """Return (events_sql, periods_sql) in portable SQL.

    The templates are standard enough for Postgres and DuckDB. The Databricks
    backend translates them with SQLRender before execution.
    """
    return (EXTRACT_EVENTS.format(cdm=cdm, join=join),
            EXTRACT_PERIODS.format(cdm=cdm, join=join))


def sample_table_name(results_schema: str, n_patients: int, seed: int) -> str:
    return f"{results_schema.rstrip('.')}.topocdm_sample_{int(seed)}_{int(n_patients)}"


def sample_select_sql(dialect: str, cdm: str, n_patients: int, seed: int) -> str:
    """The deterministic, nested patient sample for one dialect."""
    n = int(n_patients)
    s = int(seed)
    if dialect == "postgres":
        order = f"md5(person_id::text || ':{s}')"
    elif dialect == "duckdb":
        order = f"hash(person_id::varchar || ':{s}')"
    elif dialect == "spark":
        order = f"hash(person_id, {s})"
    else:
        raise ValueError(f"unknown dialect: {dialect}")
    return f"select person_id from {cdm}.person order by {order} limit {n}"


def create_sample_sql(dialect: str, table: str, cdm: str,
                      n_patients: int, seed: int) -> str:
    select = sample_select_sql(dialect, cdm, n_patients, seed)
    if dialect == "spark":
        return f"create or replace table {table} as {select}"
    return f"create table {table} as {select}"


def to_spark(tsql: str) -> str:
    """Translate OHDSI SQL to Spark SQL with circepy's SQLRender port."""
    try:
        from circe.sqlrender import translate
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "circepy is not installed. Install it with:\n"
            '  pip install "ohdsi-circepy @ git+https://github.com/OHDSI/'
            'Circepy.git@feature/decomposable-cohorts"'
        ) from exc
    return translate(tsql, "spark").strip().rstrip(";")


class Backend:
    """One connection; the dialect is the only subclass-specific behaviour."""

    dialect = "generic"

    def prepare(self, sql: str) -> str:
        return sql

    def execute(self, sql: str):  # pragma: no cover - abstract
        raise NotImplementedError

    def fetch(self, sql: str) -> pd.DataFrame:  # pragma: no cover - abstract
        raise NotImplementedError

    def create_sample(self, table: str, cdm: str, n_patients: int, seed: int):
        self.execute(f"drop table if exists {table}")
        self.execute(create_sample_sql(self.dialect, table, cdm, n_patients, seed))

    def drop(self, table: str):
        self.execute(f"drop table if exists {table}")

    def close(self):
        pass


class PostgresBackend(Backend):
    dialect = "postgres"

    def __init__(self, dsn: str):
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - environment guard
            raise SystemExit("psycopg is required: pip install 'psycopg[binary]'") from exc
        self.conn = psycopg.connect(dsn)

    def execute(self, sql: str):
        with self.conn.cursor() as cur:
            cur.execute(sql)
        self.conn.commit()

    def fetch(self, sql: str) -> pd.DataFrame:
        with self.conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)

    def close(self):
        self.conn.close()


class DuckDBBackend(Backend):
    dialect = "duckdb"

    def __init__(self, path: str = ":memory:"):
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - environment guard
            raise SystemExit("duckdb is required: pip install duckdb") from exc
        self.conn = duckdb.connect(path)

    def execute(self, sql: str):
        self.conn.execute(sql)

    def fetch(self, sql: str) -> pd.DataFrame:
        return self.conn.execute(sql).df()

    def close(self):
        self.conn.close()


class DatabricksBackend(Backend):
    dialect = "spark"

    def __init__(self, conn_cfg: dict | None = None, host: str | None = None,
                 http_path: str | None = None, token: str | None = None):
        try:
            from databricks import sql as dbsql
        except ImportError as exc:  # pragma: no cover - environment guard
            raise SystemExit(
                "databricks-sql-connector is not installed. Install it with:\n"
                "  pip install databricks-sql-connector"
            ) from exc
        cfg = conn_cfg or {}
        host = host or os.environ.get("DATABRICKS_SERVER_HOSTNAME") or cfg.get("server_hostname")
        http_path = http_path or os.environ.get("DATABRICKS_HTTP_PATH") or cfg.get("http_path")
        token = token or os.environ.get("DATABRICKS_TOKEN") or cfg.get("personal_access_token")
        if not (host and http_path and token):
            raise SystemExit(
                "Databricks connection is incomplete: set server_hostname, "
                "http_path and personal_access_token in the connection card, "
                "or pass --host/--http-path/--token")
        self.conn = dbsql.connect(server_hostname=host, http_path=http_path,
                                  access_token=token)

    def prepare(self, sql: str) -> str:
        return to_spark(sql)

    def execute(self, sql: str):
        with self.conn.cursor() as cur:
            cur.execute(sql)

    def fetch(self, sql: str) -> pd.DataFrame:
        with self.conn.cursor() as cur:
            cur.execute(sql)
            try:
                return cur.fetchall_arrow().to_pandas()
            except Exception:
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                return pd.DataFrame(rows, columns=cols)

    def create_sample(self, table: str, cdm: str, n_patients: int, seed: int):
        self.execute(create_sample_sql("spark", table, cdm, n_patients, seed))

    def close(self):
        self.conn.close()


def extract_sample(backend: Backend, cdm: str, results: str,
                   n_patients: int | None, seed: int = 0,
                   keep_scratch: bool = False):
    """Create the sample table, fetch events and periods, drop the table.

    Returns (events, periods, table_name). `table_name` is None for a full
    extract. The drop is in a finally block, so a failed fetch or Ctrl-C does
    not leave the scratch table behind unless keep_scratch is set.
    """
    table = None
    if n_patients is not None:
        table = sample_table_name(results, n_patients, seed)
        backend.create_sample(table, cdm, n_patients, seed)
        join = f" join {table} using (person_id)"
    else:
        join = ""

    events_sql, periods_sql = build_extract_sql(cdm, join)
    try:
        events = backend.fetch(backend.prepare(events_sql))
        periods = backend.fetch(backend.prepare(periods_sql))
    finally:
        if table is not None and not keep_scratch:
            backend.drop(table)

    for df in (events, periods):
        for col in ("start_date", "end_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

    n_unique = int(periods["person_id"].nunique())
    if n_patients is not None and n_unique > int(n_patients):
        raise RuntimeError(
            f"sample returned {n_unique} patients for n_patients={n_patients}; "
            "the sample table or extraction is not constrained to the sample")
    return events, periods, table


def summarize(fit, events: pd.DataFrame, periods: pd.DataFrame, *,
              cmd: str = "", cdm: str = "", results: str = "",
              table: str | None = None, n_patients: int | None = None,
              seed: int = 0) -> dict:
    return dict(
        cmd=cmd,
        cdm_schema=cdm,
        results_schema=results,
        sample_table=table,
        n_patients_requested=n_patients,
        seed=seed,
        patients=int(periods["person_id"].nunique()),
        period_rows=int(len(periods)),
        records=int(len(events)),
        per_domain={d: int(v) for d, v in events.groupby("domain").size().items()},
        k=float(fit.k),
        propensity_tmean=float(fit.propensity_tmean),
        death_frac=float(fit.death["frac"]),
        n_parameters=int(fit.n_parameters),
        K=int(fit.meta["K"]),
        n_atoms=int(fit.n_concepts),
    )


def run_fit(backend: Backend, cdm: str, results: str,
            n_patients: int | None, seed: int = 0, k: int = 20,
            max_patients: int | None = None, keep_scratch: bool = False,
            cmd: str = "", out: str | None = None,
            summary_path: str | None = None):
    """The one pipeline: sample -> extract -> fit -> save.

    This is the function the Databricks CLI calls, and the function the
    Postgres and DuckDB integration tests call. There is no second copy of the
    extraction or fit logic.
    """
    events, periods, table = extract_sample(
        backend, cdm, results, n_patients, seed, keep_scratch)
    fit = pdm.fit_perdomain(events, periods, K=k, max_patients=max_patients)
    summary = summarize(fit, events, periods, cmd=cmd, cdm=cdm,
                        results=results, table=table,
                        n_patients=n_patients, seed=seed)
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(fit, fh)
    if summary_path:
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=1)
    return fit, summary
