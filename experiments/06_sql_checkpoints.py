"""
06 -- checkpointed SQL collection on the local CDM, validated against the Python path.

The claim: pushing the aggregation into SQL, decomposed into independently
materialised steps, produces exactly what the Python extractor produces -- and
re-running is free because the checkpoints already exist.

Run: MPLCONFIGDIR=./.mplconfig .venv/bin/python experiments/06_sql_checkpoints.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topocdm.sql import plan, CheckpointExecutor

CONTAINER = "ohdsi-lab-postgres"
OUT_SCHEMA = "topocdm"


def psql(sql: str, rows: bool = False):
    """Run SQL through the container's psql. Returns rows when asked."""
    args = ["docker", "exec", "-i", CONTAINER, "psql", "-U", "postgres",
            "-d", "postgres", "-v", "ON_ERROR_STOP=1"]
    args += ["-t", "-A", "-F", ","] if rows else []
    p = subprocess.run(args, input=sql, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip()[:500])
    if not rows:
        return None
    out = [ln for ln in p.stdout.splitlines() if ln and not ln.startswith("(")]
    return out


def main():
    steps = plan(cdm="cdm", vocab="vocab", out_schema=OUT_SCHEMA,
                 dialect="postgres")
    ex = CheckpointExecutor(run_sql=psql, query_sql=lambda q: psql(q, rows=True),
                            out_schema=OUT_SCHEMA, dialect="postgres")

    psql(f"drop schema if exists {OUT_SCHEMA} cascade")
    print("cold start -- materialising every step from the raw CDM")
    t0 = time.time()
    ex.run(steps)
    first = time.time() - t0
    for s in ex.executed:
        print(f"  {s.name:16s} {s.content_hash}")
    print(f"  {len(ex.executed)} executed, {len(ex.skipped)} skipped, {first:.1f}s")

    print("\nsecond run -- unchanged plan should be a no-op")
    ex2 = CheckpointExecutor(run_sql=psql, query_sql=lambda q: psql(q, rows=True),
                             out_schema=OUT_SCHEMA, dialect="postgres")
    t0 = time.time()
    ex2.run(steps)
    second = time.time() - t0
    print(f"  {len(ex2.executed)} executed, {len(ex2.skipped)} skipped, {second:.1f}s")

    print("\nlineage")
    for row in ex.lineage():
        print(f"  {row['name']:16s} {row['content_hash']}  <- {row['depends_on']}")

    # ---- validate against the Python extraction
    rows = psql(f"select person_id, concept_id, n from {OUT_SCHEMA}.incidence",
                rows=True)
    sql_inc = pd.DataFrame(
        [(int(a), int(b), int(c)) for a, b, c in
         (r.split(",") for r in rows)],
        columns=["person_id", "concept_id", "n"])
    py_inc = pd.read_csv("data/events.csv")
    merged = sql_inc.merge(py_inc, on=["person_id", "concept_id"],
                           suffixes=("_sql", "_py"), how="outer",
                           indicator=True)
    both = merged[merged._merge == "both"]
    print(f"\nvalidation against the Python extractor")
    print(f"  rows: sql {len(sql_inc):,}  python {len(py_inc):,}  "
          f"matched {len(both):,}")
    print(f"  only in sql {int((merged._merge=='left_only').sum()):,}  "
          f"only in python {int((merged._merge=='right_only').sum()):,}")
    if len(both):
        print(f"  counts identical: "
              f"{bool((both.n_sql == both.n_py).all())}")
        print(f"  total events: sql {int(sql_inc.n.sum()):,}  "
              f"python {int(py_inc.n.sum()):,}")

    # ---- person-time, for the fit
    pt = psql(f"select count(*), sum(days_observed) from {OUT_SCHEMA}.person_time",
              rows=True)
    print(f"  person_time: {pt[0]}  (persons, total days)")


if __name__ == "__main__":
    main()
