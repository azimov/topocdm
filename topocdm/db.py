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

import numpy as np
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

# Aggregate counts, computed in the database in the style of OHDSI's
# CohortIncidence: one row per (domain, concept), carrying the number of
# distinct patients with at least one record and the total number of records.
# The Python side never needs the event rows themselves to score fidelity, and
# at 100k patients those rows are 24M of them (~1.2 GB in a DataFrame), so
# counting here is what makes the report scale.
COUNT_EVENTS = """
select domain, concept_id,
       count(distinct person_id) as n_patients,
       count(*) as n_events
  from (
        select 'condition' as domain, condition_concept_id as concept_id, person_id
          from {cdm}.condition_occurrence {join}
        union all
        select 'drug', drug_concept_id, person_id
          from {cdm}.drug_exposure {join}
        union all
        select 'procedure', procedure_concept_id, person_id
          from {cdm}.procedure_occurrence {join}
        union all
        select 'measurement', measurement_concept_id, person_id
          from {cdm}.measurement {join}
        union all
        select 'observation', observation_concept_id, person_id
          from {cdm}.observation {join}
       ) ev
 group by domain, concept_id
"""

# Sufficient statistics for the fit, aggregated by the database.
#
# The fit never needs event *rows*; it needs these summaries, which are all
# O(C) or O(C x years) regardless of how many records the CDM holds:
#
#   EDGE_COUNTS         per (domain, concept) once per patient: the atoms, the
#                       per-patient presence that the coupling fit uses
#   DOMAIN_YEAR_COUNTS  events per domain per calendar year -> lam_t
#   DURATION_QUANTILES  the positive durations, sampled in SQL -> duration model
#
# Sampling the durations in SQL rather than shipping every interval is what
# keeps this bounded: 101 quantiles per domain instead of millions of rows.
DOMAIN_YEAR_COUNTS = """
select domain, year, count(*) as n_events
  from (
        select 'condition' as domain, cast(extract(year from condition_start_date) as integer) as year
          from {cdm}.condition_occurrence {join}
        union all
        select 'drug', cast(extract(year from drug_exposure_start_date) as integer) from {cdm}.drug_exposure {join}
        union all
        select 'procedure', cast(extract(year from procedure_date) as integer) from {cdm}.procedure_occurrence {join}
        union all
        select 'measurement', cast(extract(year from measurement_date) as integer) from {cdm}.measurement {join}
        union all
        select 'observation', cast(extract(year from observation_date) as integer) from {cdm}.observation {join}
        union all
        select 'visit', cast(extract(year from visit_start_date) as integer) from {cdm}.visit_occurrence {join}
       ) ev
 where year is not null
 group by domain, year
"""

# Zero-inflation and positive durations for the interval domains.
#
# `fit_durations` needs a zero fraction plus 101 quantiles of the positive
# durations. Shipping every interval row is the thing to avoid, so the database
# reports the zero counts and a *bounded* sample of positive durations; 101
# numbers per domain is the whole requirement. Plain date subtraction is used
# because it is valid in both Postgres and DuckDB (SQLRender translates it for
# Spark).
_DURATION_DAYS = """
        select '{domain}' as domain,
               cast({end_col} as date) - cast({start_col} as date) as days
          from {cdm}.{table} {join}
"""


def _durations_union() -> str:
    """The two interval domains as one SQL body, with {cdm}/{join} left intact.

    Substitution is done with a targeted replace rather than str.format so the
    outer placeholders survive for `_aggregate` to fill.
    """
    parts = []
    for domain, start_col, end_col, table in (
        ("condition", "condition_start_date", "condition_end_date", "condition_occurrence"),
        ("drug", "drug_exposure_start_date", "drug_exposure_end_date", "drug_exposure"),
    ):
        parts.append(
            _DURATION_DAYS.replace("{domain}", domain)
            .replace("{start_col}", start_col)
            .replace("{end_col}", end_col)
            .replace("{table}", table)
        )
    return " union all ".join(parts)


DURATION_STATS = (
    "select domain,"
    " sum(case when days is null or days <= 0 then 1 else 0 end) as n_zero,"
    " count(*) as n_total"
    " from (" + _durations_union() + ") ev group by domain"
)

DURATION_SAMPLE = (
    "select domain, days from (" + _durations_union() + ") ev where days > 0 limit {limit}"
)


def build_extract_sql(cdm: str, join: str = ""):
    """Return (events_sql, periods_sql) in portable SQL.

    The templates are standard enough for Postgres and DuckDB. The Databricks
    backend translates them with SQLRender before execution.
    """
    return (EXTRACT_EVENTS.format(cdm=cdm, join=join), EXTRACT_PERIODS.format(cdm=cdm, join=join))


def sample_table_name(results_schema: str, n_patients: int, seed: int) -> str:
    return f"{results_schema.rstrip('.')}.topocdm_sample_{int(seed)}_{int(n_patients)}"


def sample_select_sql(
    dialect: str, source: str, n_patients: int, seed: int, *, is_table: bool = False
) -> str:
    """The deterministic patient sample for one dialect.

    `source` is a CDM schema (sampling `source.person`) or, with `is_table`, an
    existing table of person_ids (used to take a bounded fit sample inside an
    already-sampled cohort, without returning to the client).
    """
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
    table = source if is_table else f"{source}.person"
    return f"select person_id from {table} order by {order} limit {n}"


def create_sample_sql(
    dialect: str,
    table: str,
    cdm: str,
    n_patients: int,
    seed: int,
    *,
    is_table: bool = False,
) -> str:
    select = sample_select_sql(dialect, cdm, n_patients, seed, is_table=is_table)
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

    def create_sample(
        self, table: str, cdm: str, n_patients: int, seed: int, *, is_table: bool = False
    ):
        self.execute(f"drop table if exists {table}")
        self.execute(
            create_sample_sql(self.dialect, table, cdm, n_patients, seed, is_table=is_table)
        )

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
            cols = [d[0] for d in (cur.description or [])]
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

    def __init__(
        self,
        conn_cfg: dict | None = None,
        host: str | None = None,
        http_path: str | None = None,
        token: str | None = None,
    ):
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
                "or pass --host/--http-path/--token"
            )
        self.conn = dbsql.connect(server_hostname=host, http_path=http_path, access_token=token)

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
                cols = [d[0] for d in (cur.description or [])]
                return pd.DataFrame(rows, columns=cols)

    def create_sample(
        self, table: str, cdm: str, n_patients: int, seed: int, *, is_table: bool = False
    ):
        self.execute(create_sample_sql("spark", table, cdm, n_patients, seed, is_table=is_table))

    def close(self):
        self.conn.close()


def extract_sample(
    backend: Backend,
    cdm: str,
    results: str,
    n_patients: int | None,
    seed: int = 0,
    keep_scratch: bool = False,
    fit_patients: int | None = None,
):
    """Create the sample table, fetch events and periods, drop the table.

    Returns (events, periods, table_name). `table_name` is None for a full
    extract. The drop is in a finally block, so a failed fetch or Ctrl-C does
    not leave the scratch table behind unless keep_scratch is set.

    `fit_patients` caps the rows pulled to the client. It is applied as a second,
    nested sample in SQL rather than a pandas filter, so a fit on 5,000 patients
    out of a 100,000-patient extract never materialises the 100,000 on this
    machine. The released parameters depend on no per-patient term, so a
    bounded sample estimates exactly the quantities that are published
    (`docs/DESIGN.md` §9, §9.1). Aggregates over the *full* sample are
    available separately via `count_events` and friends.
    """
    table = None
    if n_patients is not None:
        table = sample_table_name(results, n_patients, seed)
        backend.create_sample(table, cdm, n_patients, seed)
        if fit_patients is not None and int(fit_patients) < int(n_patients):
            # nested deterministic sample, still executed in the engine
            fit_table = sample_table_name(results, int(fit_patients), seed + 1)
            backend.create_sample(fit_table, table, int(fit_patients), seed + 1, is_table=True)
            join = f" join {fit_table} using (person_id)"
        else:
            fit_table = None
            join = f" join {table} using (person_id)"
    else:
        fit_table = None
        join = ""

    events_sql, periods_sql = build_extract_sql(cdm, join)
    try:
        events = backend.fetch(backend.prepare(events_sql))
        periods = backend.fetch(backend.prepare(periods_sql))
    finally:
        if not keep_scratch:
            for t in (fit_table, table):
                if t is not None:
                    backend.drop(t)

    for df in (events, periods):
        for col in ("start_date", "end_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

    n_unique = int(periods["person_id"].nunique())
    if n_patients is not None and n_unique > int(n_patients):
        raise RuntimeError(
            f"sample returned {n_unique} patients for n_patients={n_patients}; "
            "the sample table or extraction is not constrained to the sample"
        )
    return events, periods, table


def count_events(
    backend: Backend,
    cdm: str,
    results: str,
    n_patients: int | None,
    seed: int = 0,
    keep_scratch: bool = False,
) -> pd.DataFrame:
    """Per-concept patient counts and record totals, aggregated in SQL.

    Mirrors the shape CohortIncidence works in: the database does the
    counting and returns one row per (domain, concept), so scoring a
    100k-patient extract costs a few thousand rows in Python rather than 24M.
    The person sample is the same one `extract_sample` uses, so the counts
    describe exactly the patients the fit saw.

    Returns `domain, concept_id, n_patients, n_events`.
    """
    return _aggregate(backend, COUNT_EVENTS, cdm, results, n_patients, seed, keep_scratch)


def count_domain_years(
    backend: Backend,
    cdm: str,
    results: str,
    n_patients: int | None,
    seed: int = 0,
    keep_scratch: bool = False,
) -> pd.DataFrame:
    """Events per domain per calendar year, aggregated in SQL (feeds `lam_t`)."""
    return _aggregate(backend, DOMAIN_YEAR_COUNTS, cdm, results, n_patients, seed, keep_scratch)


def duration_sample(
    backend: Backend,
    cdm: str,
    results: str,
    n_patients: int | None,
    seed: int = 0,
    limit: int = 20000,
    keep_scratch: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zero counts and a bounded sample of positive durations, computed in SQL.

    Returns `(stats, sample)`. `stats` is one row per interval domain with
    `n_zero` and `n_total`; `sample` is at most `limit` positive durations.
    Together they give `fit_durations` its zero fraction and 101-point
    quantile function without shipping every interval in the CDM.
    """
    stats = _aggregate(backend, DURATION_STATS, cdm, results, n_patients, seed, keep_scratch)
    sample = _aggregate(
        backend,
        DURATION_SAMPLE.replace("{limit}", str(int(limit))),
        cdm,
        results,
        n_patients,
        seed,
        keep_scratch,
    )
    return stats, sample


def _cohort_table(
    backend: Backend,
    results: str,
    persons: np.ndarray,
    seed: int,
) -> str:
    """Materialise an explicit patient set as a scratch table.

    Used when the cohort was already fixed by the extract (a capped fit
    sample, a held-out set): the aggregate must describe *those* patients, not
    a fresh sample drawn with the same RNG. O(N) integers cross the wire, not
    event rows.
    """
    table = f"{results.rstrip('.')}.topocdm_cohort_{int(seed)}"
    backend.execute(f"drop table if exists {table}")
    backend.execute(f"create table {table} (person_id bigint)")
    ids = [int(p) for p in np.asarray(persons)]
    # insert in chunks so a 100k-patient cohort stays within statement limits
    for i in range(0, len(ids), 5000):
        chunk = ",".join(f"({v})" for v in ids[i : i + 5000])
        backend.execute(f"insert into {table} (person_id) values {chunk}")
    return table


def _aggregate(
    backend: Backend,
    template: str,
    cdm: str,
    results: str,
    n_patients: int | None,
    seed: int,
    keep_scratch: bool,
    persons: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run an aggregate over the fitted cohort.

    If `persons` is given the aggregate is restricted to exactly those
    patients, which is what keeps the scored cohort identical to the fitted
    one when the extract was capped.
    """
    table = None
    if persons is not None:
        table = _cohort_table(backend, results, persons, seed)
        join = f" join {table} using (person_id)"
    elif n_patients is not None:
        table = sample_table_name(results, n_patients, seed)
        backend.create_sample(table, cdm, n_patients, seed)
        join = f" join {table} using (person_id)"
    else:
        join = ""
    try:
        return backend.fetch(backend.prepare(template.format(cdm=cdm, join=join)))
    finally:
        if table is not None and not keep_scratch:
            backend.drop(table)


def summarize(
    fit,
    events: pd.DataFrame,
    periods: pd.DataFrame,
    *,
    cmd: str = "",
    cdm: str = "",
    results: str = "",
    table: str | None = None,
    n_patients: int | None = None,
    seed: int = 0,
) -> dict:
    return {
        "cmd": cmd,
        "cdm_schema": cdm,
        "results_schema": results,
        "sample_table": table,
        "n_patients_requested": n_patients,
        "seed": seed,
        "patients": int(periods["person_id"].nunique()),
        "period_rows": int(len(periods)),
        "records": int(len(events)),
        "per_domain": {d: int(v) for d, v in events.groupby("domain").size().items()},
        "k": float(fit.k),
        "propensity_tmean": float(fit.propensity_tmean),
        "death_frac": float(fit.death["frac"]),
        "n_parameters": int(fit.n_parameters),
        "K": int(fit.meta["K"]),
        "n_atoms": int(fit.n_concepts),
    }


def run_fit(
    backend: Backend,
    cdm: str,
    results: str,
    n_patients: int | None,
    seed: int = 0,
    k: int = 20,
    max_patients: int | None = None,
    keep_scratch: bool = False,
    cmd: str = "",
    out: str | None = None,
    summary_path: str | None = None,
    return_data: bool = False,
    return_counts: bool = False,
):
    """The one pipeline: sample -> extract -> fit -> save.

    This is the function the Databricks CLI calls, and the function the
    Postgres and DuckDB integration tests call. There is no second copy of the
    extraction or fit logic.

    With `return_data=True` the extracted `(events, periods)` are returned
    alongside the fit, so a caller can score fidelity without a second
    warehouse round-trip. That is the only way to compare against the *same*
    data the fit saw.

    With `return_counts=True` the per-concept aggregates are returned instead
    of the events. Prevalence and incidence need only those counts, and at
    100k patients the event frame is ~24M rows, so this is the path that keeps
    scoring flat in extract size.
    """
    # `max_patients` caps the *extract*, not a pandas filter: the fit is
    # bounded in SQL so a large cohort never lands on this machine. Aggregates
    # for the fidelity report are taken over the full sample separately.
    fit_cap = max_patients
    events, periods, table = extract_sample(
        backend, cdm, results, n_patients, seed, keep_scratch, fit_patients=fit_cap
    )
    counts = None
    if return_counts:
        # The counts must describe exactly the patients `periods` describes.
        # When the extract is capped by `fit_patients`, the scored cohort is the
        # capped one; counting the uncapped sample would put one cohort's
        # records over another cohort's person-time (it doubled incidence).
        # Passing the cohort explicitly avoids re-deriving the nested sample.
        counts = _aggregate(
            backend,
            COUNT_EVENTS,
            cdm,
            results,
            n_patients,
            seed,
            keep_scratch,
            persons=periods["person_id"].unique(),
        )
    fit = pdm.fit_perdomain(events, periods, K=k)
    summary = summarize(
        fit,
        events,
        periods,
        cmd=cmd,
        cdm=cdm,
        results=results,
        table=table,
        n_patients=n_patients,
        seed=seed,
    )
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(fit, fh)
    if summary_path:
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=1)
    if return_data:
        return fit, summary, events, periods
    if return_counts:
        # periods are small (one row per patient) and carry the person-time
        # incidence needs; the events frame is what has to stay in the warehouse
        return fit, summary, counts, periods
    return fit, summary
