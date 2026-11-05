#!/usr/bin/env python
"""
Fit the per-domain model against an OMOP CDM hosted on Databricks.

Connection and schemas come from the OHDSI connection card:

    ~/.config/ohdsi/databricks_connection.yml

    databricks:
      driver: "databricks"
      connection:
        server_hostname: "adb-....azuredatabricks.net"
        http_path: "/sql/1.0/warehouses/...."
        personal_access_token: "dapi...."

    cmds:
      mycdm_version:
        cdm_schema: "catalog.schema"
        vocabulary_schema: "catalog.schema"
        results_schema: "scratch.schema"   # writable; the sample table goes here

`results_schema` is required for sampling: the patient sample is written as a
real scratch table there, and the event tables are joined to it on the cluster.
Only the sampled rows are downloaded. The scratch table is dropped after the
fetch unless `--keep-scratch` is passed.

The connection is thin: OHDSI SQL is written in T-SQL, translated to Spark SQL
with circepy's SQLRender port, and sent through the official Databricks SQL
connector. There is no ibis in this path.

    pip install "ohdsi-circepy @ git+https://github.com/OHDSI/Circepy.git@feature/decomposable-cohorts"
    pip install "databricks-sql-connector[pyarrow]"

Usage:

    python scripts/fit_databricks.py --cmd mycdm_version --dry-run
    python scripts/fit_databricks.py --cmd mycdm_version --n-patients 10000 --k 20
    python scripts/fit_databricks.py --cmd mycdm_version --n-patients 20000 --k 20

The sample is ordered by a hash of `person_id`, so it is deterministic for a
given `--seed` and nested: 10k is a subset of 20k. `--cdm` overrides the
cdm_schema; `--results-schema` overrides the scratch schema.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topocdm import perdomain as pdm

DEFAULT_CONFIG = "~/.config/ohdsi/databricks_connection.yml"


# --------------------------------------------------------------------------
# OHDSI SQL in T-SQL. {join} is the semi-join against the scratch sample table
# (or empty for a full extract). No CTE and no MATERIALIZED: the sample is a
# real table, which is also what the cluster's scratch schema is for.
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def load_config(path: str = DEFAULT_CONFIG, cmd: str | None = None):
    """Read the OHDSI connection card.

    Returns (connection_dict, cdm_schema, results_schema, vocab_schema, cmd_name).
    Only the connection fields and the named command are returned; the token is
    never printed or written to the summary.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - setup installs it
        raise SystemExit("PyYAML is required: pip install PyYAML") from exc

    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise SystemExit(f"connection card not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    conn = (cfg.get("databricks") or {}).get("connection") or {}
    cmds = cfg.get("cmds") or {}
    if not cmds:
        raise SystemExit(f"no 'cmds' entries in {path}")

    name = cmd or next(iter(cmds))
    if name not in cmds:
        raise SystemExit(
            f"cmd '{name}' not in {path}; available: {', '.join(cmds)}")
    prof = cmds[name] or {}
    cdm = prof.get("cdm_schema")
    scratch = prof.get("results_schema")
    vocab = prof.get("vocabulary_schema") or cdm
    if not cdm:
        raise SystemExit(f"cmd '{name}' has no cdm_schema")
    return conn, cdm, scratch, vocab, name


def sample_table_name(results_schema: str, n_patients: int, seed: int) -> str:
    return f"{results_schema.rstrip('.')}.topocdm_sample_{int(seed)}_{int(n_patients)}"


def create_sample_sql(cdm: str, table: str, n_patients: int, seed: int) -> str:
    """Spark SQL that writes the patient sample to the scratch schema."""
    return (f"create or replace table {table}\n"
            f"as select person_id from {cdm}.person\n"
            f"order by hash(person_id, {int(seed)})\n"
            f"limit {int(n_patients)}")


def drop_table_sql(table: str) -> str:
    return f"drop table if exists {table}"


def build_extract_sql(cdm: str, join: str = ""):
    """Return (events_sql, periods_sql) in T-SQL, before translation."""
    return (EXTRACT_EVENTS.format(cdm=cdm, join=join),
            EXTRACT_PERIODS.format(cdm=cdm, join=join))


# --------------------------------------------------------------------------
# Databricks connection
# --------------------------------------------------------------------------

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


def connect(conn_cfg: dict, args):
    try:
        from databricks import sql as dbsql
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "databricks-sql-connector is not installed. Install it with:\n"
            "  pip install databricks-sql-connector"
        ) from exc

    host = (args.host or os.environ.get("DATABRICKS_SERVER_HOSTNAME")
            or conn_cfg.get("server_hostname"))
    http_path = (args.http_path or os.environ.get("DATABRICKS_HTTP_PATH")
                 or conn_cfg.get("http_path"))
    token = (args.token or os.environ.get("DATABRICKS_TOKEN")
             or conn_cfg.get("personal_access_token"))
    if not (host and http_path and token):
        raise SystemExit(
            "Databricks connection is incomplete: set server_hostname, "
            "http_path and personal_access_token in the connection card, "
            "or pass --host/--http-path/--token")
    return dbsql.connect(server_hostname=host, http_path=http_path,
                         access_token=token)


def execute(conn, spark_sql: str):
    """Run a statement that returns nothing (DDL/DML)."""
    with conn.cursor() as cur:
        cur.execute(spark_sql)


def fetch(conn, spark_sql: str) -> pd.DataFrame:
    """Run one translated SELECT and return a DataFrame."""
    with conn.cursor() as cur:
        cur.execute(spark_sql)
        try:
            return cur.fetchall_arrow().to_pandas()
        except Exception:
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            return pd.DataFrame(rows, columns=cols)


def _summary(fit, events, periods, *, cmd_name, cdm, scratch, table, n, seed):
    return dict(
        cmd=cmd_name,
        cdm_schema=cdm,
        results_schema=scratch,
        sample_table=table,
        n_patients_requested=n,
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="connection card path")
    ap.add_argument("--cmd", default=None,
                    help="which cmds entry to use (default: the first)")
    ap.add_argument("--n-patients", type=int, default=None,
                    help="patient sample; written to results_schema")
    ap.add_argument("--seed", type=int, default=0,
                    help="hash seed for the sample (deterministic, nested)")
    ap.add_argument("--k", type=int, default=20,
                    help="coupling rank (K); lower it for small cohorts")
    ap.add_argument("--max-patients", type=int, default=None,
                    help="local cap passed to fit_perdomain after download")
    ap.add_argument("--cdm", default=None, help="override cdm_schema")
    ap.add_argument("--results-schema", default=None,
                    help="override results_schema (writable scratch)")
    ap.add_argument("--host", default=None)
    ap.add_argument("--http-path", default=None)
    ap.add_argument("--token", default=None)
    ap.add_argument("--out", default="results/databricks_fit.pkl")
    ap.add_argument("--summary", default="results/databricks_fit.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the Spark SQL and exit; no connection")
    ap.add_argument("--keep-scratch", action="store_true",
                    help="keep the sample table after the fetch")
    args = ap.parse_args(argv)

    if args.n_patients is not None and args.n_patients <= 0:
        ap.error("--n-patients must be positive")

    # A dry run can work without a connection card, so the setup script can
    # prove the SQLRender path before credentials exist.
    if args.dry_run and (args.cdm or args.results_schema) and \
            not os.path.exists(os.path.expanduser(args.config)):
        conn_cfg, cmd_name = {}, "(dry-run)"
        cdm = args.cdm or "main.omop"
        scratch = args.results_schema or "scratch.scratch"
        vocab = cdm
    else:
        conn_cfg, cdm, scratch, vocab, cmd_name = load_config(
            args.config, args.cmd)
        cdm = args.cdm or cdm
        scratch = args.results_schema or scratch

    table = None
    join = ""
    if args.n_patients is not None:
        if not scratch:
            ap.error("--n-patients needs a writable results_schema in the "
                     "connection card (or --results-schema)")
        table = sample_table_name(scratch, args.n_patients, args.seed)
        join = f" join {table} using (person_id)"

    events_sql, periods_sql = build_extract_sql(cdm, join)
    events_spark = to_spark(events_sql)
    periods_spark = to_spark(periods_sql)
    sample_spark = (create_sample_sql(cdm, table, args.n_patients, args.seed)
                    if table else None)

    if args.dry_run:
        if sample_spark:
            print("--- sample table (Spark SQL, scratch) ---")
            print(sample_spark)
            print()
        print("--- events (Spark SQL) ---")
        print(events_spark)
        print("\n--- periods (Spark SQL) ---")
        print(periods_spark)
        if table and not args.keep_scratch:
            print("\n--- cleanup after fetch ---")
            print(drop_table_sql(table))
        return 0

    t0 = time.time()
    conn = connect(conn_cfg, args)
    try:
        if table:
            print(f"writing sample table {table} ...")
            execute(conn, sample_spark)
        print(f"extracting from {cdm} ...")
        events = fetch(conn, events_spark)
        periods = fetch(conn, periods_spark)
        if table and not args.keep_scratch:
            execute(conn, drop_table_sql(table))
    finally:
        conn.close()

    for df in (events, periods):
        for c in ("start_date", "end_date"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])

    print(f"pulled {len(events):,} records for {len(periods):,} patients "
          f"in {time.time() - t0:.1f}s")

    fit = pdm.fit_perdomain(events, periods, K=args.k,
                            max_patients=args.max_patients)
    summary = _summary(fit, events, periods, cmd_name=cmd_name, cdm=cdm,
                       scratch=scratch, table=table, n=args.n_patients,
                       seed=args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(fit, fh)
    with open(args.summary, "w") as fh:
        json.dump(summary, fh, indent=1)

    print(f"fitted: k={summary['k']:.2f}, tmean={summary['propensity_tmean']:.2f}, "
          f"death={summary['death_frac']:.3f}, n_params={summary['n_parameters']:,}, "
          f"K={summary['K']}")
    print(f"wrote {args.out} and {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
