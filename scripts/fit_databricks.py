#!/usr/bin/env python
"""
Fit the per-domain model against an OMOP CDM hosted on Databricks.

The connection is deliberately thin. OHDSI SQL is written in T-SQL, translated
to Spark SQL with circepy's SQLRender port, and sent through the official
Databricks SQL connector. There is no ibis in this path.

    pip install "ohdsi-circepy @ git+https://github.com/OHDSI/circepy.git@feature/decomposable-cohorts"
    pip install "databricks-sql-connector[pyarrow]"

Credentials come from the environment (the standard Databricks names):

    DATABRICKS_SERVER_HOSTNAME   e.g. adb-1234567890.azuredatabricks.net
    DATABRICKS_HTTP_PATH         e.g. /sql/1.0/warehouses/abcdef123456
    DATABRICKS_TOKEN             a personal access token

Sampling. For a large CDM the data must stay on the cluster. `--n-patients`
selects a fixed-size patient sample **on the cluster**, joins the event tables
to it there, and downloads only those patients' rows:

    python scripts/fit_databricks.py --cdm main.omop --n-patients 10000 --k 20

The sample is ordered by a hash of `person_id`, so it is deterministic for a
given `--seed` and **nested**: the 10k patients are a subset of the 20k. That
makes a gradual increase (10k -> 20k -> 50k) comparable and cheap to audit.
The one cost is a sort of the person table; the event tables are filtered by
the join, not downloaded whole.

`--sample-pct` remains available for a percentage sample; do not combine it
with `--n-patients`. `--max-patients` is a local cap passed to the fit after
download.

Usage:

    python scripts/fit_databricks.py --cdm main.omop --dry-run
    python scripts/fit_databricks.py --cdm main.omop --n-patients 10000 --k 20
    python scripts/fit_databricks.py --cdm main.omop --n-patients 20000 --k 20

The output is a pickled `PerDomainFit` plus a JSON summary. The fit itself is
`topocdm.perdomain.fit_perdomain`; nothing about the model is reimplemented
here.
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


# --------------------------------------------------------------------------
# OHDSI SQL in T-SQL. {keep} is the optional sampled-patient CTE and {join}
# is the semi-join against it. For a full extract both are empty strings.
# --------------------------------------------------------------------------

EXTRACT_EVENTS = """
{keep}
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
{keep}
select person_id,
       cast(observation_period_start_date as date) as start_date,
       cast(observation_period_end_date as date) as end_date
  from {cdm}.observation_period {join}
"""


def keep_clause(cdm: str, n_patients: int, seed: int = 0):
    """CTE and semi-join for a deterministic, nested patient sample."""
    keep = (f"with keep as materialized (\n"
            f"  select person_id from {cdm}.person\n"
            f"  order by hash(person_id, {int(seed)})\n"
            f"  limit {int(n_patients)}\n"
            f")")
    join = "join keep using (person_id)"
    return keep, join


def build_extract_sql(cdm: str, n_patients: int | None = None,
                      seed: int = 0):
    """Return (events_sql, periods_sql) in T-SQL, before translation."""
    if n_patients is not None:
        if n_patients <= 0:
            raise ValueError("n_patients must be positive")
        keep, join = keep_clause(cdm, n_patients, seed)
    else:
        keep, join = "", ""
    events = EXTRACT_EVENTS.format(cdm=cdm, keep=keep, join=join)
    periods = EXTRACT_PERIODS.format(cdm=cdm, keep=keep, join=join)
    return events, periods


def to_spark(tsql: str, sample_pct: int | None = None) -> str:
    """Translate T-SQL to Spark SQL and optionally add the percentage sample."""
    try:
        from circe.sqlrender import translate
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "circepy is not installed. Install it with:\n"
            '  pip install "ohdsi-circepy @ git+https://github.com/OHDSI/'
            'circepy.git@feature/decomposable-cohorts"'
        ) from exc

    spark = translate(tsql, "spark").strip().rstrip(";")
    if sample_pct is not None:
        spark = (f"select * from (\n{spark}\n) s "
                 f"where pmod(hash(person_id), 100) < {int(sample_pct)}")
    return spark


def connect(args):
    try:
        from databricks import sql as dbsql
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "databricks-sql-connector is not installed. Install it with:\n"
            "  pip install databricks-sql-connector"
        ) from exc

    host = args.host or os.environ.get("DATABRICKS_SERVER_HOSTNAME")
    http_path = args.http_path or os.environ.get("DATABRICKS_HTTP_PATH")
    token = args.token or os.environ.get("DATABRICKS_TOKEN")
    if not (host and http_path and token):
        raise SystemExit(
            "set DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH and "
            "DATABRICKS_TOKEN (or pass --host/--http-path/--token)")

    return dbsql.connect(server_hostname=host, http_path=http_path,
                         access_token=token)


def fetch(conn, spark_sql: str) -> pd.DataFrame:
    """Execute one translated statement and return a DataFrame."""
    with conn.cursor() as cur:
        cur.execute(spark_sql)
        try:
            return cur.fetchall_arrow().to_pandas()
        except Exception:
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            return pd.DataFrame(rows, columns=cols)


def sample_cdm(conn, n_patients: int, cdm: str, seed: int = 0):
    """Download a fixed-size patient sample from a Databricks-hosted CDM.

    The sample and the event joins are executed on the cluster; only the
    sampled patients' rows are transferred. The sample is deterministic for a
    given seed and nested, so 10k is a subset of 20k.

    Returns (events, periods) as pandas DataFrames.
    """
    events_sql, periods_sql = build_extract_sql(cdm, n_patients, seed)
    events = fetch(conn, to_spark(events_sql))
    periods = fetch(conn, to_spark(periods_sql))
    return events, periods


def _summary(fit, events: pd.DataFrame, periods: pd.DataFrame) -> dict:
    return dict(
        patients=int(len(periods)),
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
    ap.add_argument("--cdm", default=os.environ.get("CDM_SCHEMA",
                                                    "hive_metastore.omop"),
                    help="catalog.schema containing the OMOP tables")
    ap.add_argument("--host", default=None)
    ap.add_argument("--http-path", default=None)
    ap.add_argument("--token", default=None)
    ap.add_argument("--k", type=int, default=20,
                    help="coupling rank (K); lower it for small cohorts")
    ap.add_argument("--n-patients", type=int, default=None,
                    help="fixed patient sample, selected on the cluster")
    ap.add_argument("--seed", type=int, default=0,
                    help="hash seed for the patient sample (deterministic)")
    ap.add_argument("--max-patients", type=int, default=None,
                    help="local cap passed to fit_perdomain after download")
    ap.add_argument("--sample-pct", type=int, default=None,
                    help="percentage SQL sample; do not combine with --n-patients")
    ap.add_argument("--out", default="results/databricks_fit.pkl")
    ap.add_argument("--summary", default="results/databricks_fit.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the translated SQL and exit")
    args = ap.parse_args(argv)

    if args.n_patients is not None and args.sample_pct is not None:
        ap.error("use --n-patients or --sample-pct, not both")
    if args.sample_pct is not None and not 1 <= args.sample_pct <= 100:
        ap.error("--sample-pct must be between 1 and 100")

    events_sql, periods_sql = build_extract_sql(args.cdm, args.n_patients,
                                                args.seed)
    events_spark = to_spark(events_sql, args.sample_pct)
    periods_spark = to_spark(periods_sql, args.sample_pct)

    if args.dry_run:
        print("--- events (Spark SQL) ---")
        print(events_spark)
        print("\n--- periods (Spark SQL) ---")
        print(periods_spark)
        return 0

    t0 = time.time()
    conn = connect(args)
    try:
        if args.n_patients is not None:
            print(f"sampling {args.n_patients:,} patients on the cluster "
                  f"(seed {args.seed}) ...")
            events = fetch(conn, events_spark)
            periods = fetch(conn, periods_spark)
        else:
            print(f"extracting from {args.cdm} ...")
            events = fetch(conn, events_spark)
            periods = fetch(conn, periods_spark)
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
    summary = _summary(fit, events, periods)

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
