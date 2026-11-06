#!/usr/bin/env python
"""
Fit the per-domain model against a CDM on Databricks, Postgres or DuckDB.

The pipeline is shared: `topocdm.db.run_fit` creates the patient sample,
extracts the events, fits, and drops the sample. The backend is the only
dialect-specific part.

    python scripts/fit_databricks.py --backend postgres --dsn ... --n-patients 100
    python scripts/fit_databricks.py --backend duckdb --duckdb /tmp/cdm.duckdb --n-patients 100
    python scripts/fit_databricks.py --backend databricks --n-patients 100

The connection card is the OHDSI one:

    ~/.config/ohdsi/databricks_connection.yml

    databricks:
      connection:
        server_hostname: "adb-....azuredatabricks.net"
        http_path: "/sql/1.0/warehouses/...."
        personal_access_token: "dapi...."

    cdms:
      mycdm_version:
        cdm_schema: "catalog.schema"
        vocabulary_schema: "catalog.schema"
        results_schema: "scratch.schema"   # writable; the sample table goes here

`--cdm` and `--results-schema` override the card, and are enough for Postgres
and DuckDB when no card is present. The `results_schema` must be writable: the
patient sample is written there as a real table, not a CTE.

`--dry-run` prints the sample and extract SQL for the selected backend without
connecting.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topocdm import db

DEFAULT_CONFIG = "~/.config/ohdsi/databricks_connection.yml"


def load_config(path: str = DEFAULT_CONFIG, cmd: str | None = None):
    """Read the OHDSI connection card.

    Returns (connection_dict, cdm_schema, results_schema, vocab_schema, cmd_name).
    The canonical profile map is 'cdms'; 'cmds' is accepted for older cards.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - setup installs it
        raise SystemExit("PyYAML is required: pip install PyYAML") from exc

    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise SystemExit(f"connection card not found: {path}")
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    conn = (cfg.get("databricks") or {}).get("connection") or {}
    cdms = cfg.get("cdms") or cfg.get("cmds") or {}
    if not cdms:
        raise SystemExit(f"no 'cdms' entries in {path}")

    name = cmd or next(iter(cdms))
    if name not in cdms:
        raise SystemExit(f"cmd '{name}' not in {path}; available: {', '.join(cdms)}")
    prof = cdms[name] or {}
    cdm = prof.get("cdm_schema")
    scratch = prof.get("results_schema")
    vocab = prof.get("vocabulary_schema") or cdm
    if not cdm:
        raise SystemExit(f"cmd '{name}' has no cdm_schema")
    return conn, cdm, scratch, vocab, name


def _resolve_schemas(args):
    """Return (conn_cfg, cdm, results, cmd_name) from the card and overrides."""
    try:
        conn, cdm, results, _vocab, name = load_config(args.config, args.cmd)
    except SystemExit:
        if args.cdm and args.results_schema:
            conn, cdm, results, name = {}, args.cdm, args.results_schema, "(cli)"
        else:
            raise
    return conn, (args.cdm or cdm), (args.results_schema or results), name


def _dry_run(
    backend: str, cdm: str, results: str, n_patients: int | None, seed: int, keep_scratch: bool
):
    dialect = {"databricks": "spark", "postgres": "postgres", "duckdb": "duckdb"}[backend]
    table = None
    join = ""
    if n_patients is not None:
        table = db.sample_table_name(results, n_patients, seed)
        join = f" join {table} using (person_id)"
        print("--- sample table ---")
        print(db.create_sample_sql(dialect, table, cdm, n_patients, seed))
        print()
    events, periods = db.build_extract_sql(cdm, join)
    if dialect == "spark":
        events, periods = db.to_spark(events), db.to_spark(periods)
    print("--- events ---")
    print(events)
    print("\n--- periods ---")
    print(periods)
    if table is not None and not keep_scratch:
        print("\n--- cleanup ---")
        print(f"drop table if exists {table}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["databricks", "postgres", "duckdb"], default="databricks")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--cmd", default=None, help="which cdms entry to use (default: the first)")
    ap.add_argument("--cdm", default=None, help="override cdm_schema")
    ap.add_argument(
        "--results-schema", default=None, help="override results_schema (writable scratch)"
    )
    ap.add_argument("--dsn", default=None, help="Postgres DSN (default: CDM_DSN)")
    ap.add_argument("--duckdb", default=":memory:", help="DuckDB path for --backend duckdb")
    ap.add_argument(
        "--n-patients", type=int, default=None, help="patient sample; written to results_schema"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--k", type=int, default=20, help="coupling rank (K); lower it for small cohorts"
    )
    ap.add_argument(
        "--max-patients", type=int, default=None, help="local cap passed to fit_perdomain"
    )
    ap.add_argument(
        "--keep-scratch", action="store_true", help="keep the sample table after the fetch"
    )
    ap.add_argument("--host", default=None)
    ap.add_argument("--http-path", default=None)
    ap.add_argument("--token", default=None)
    ap.add_argument("--out", default="results/databricks_fit.pkl")
    ap.add_argument("--summary", default="results/databricks_fit.json")
    ap.add_argument(
        "--fidelity",
        default=None,
        help="write a prevalence/incidence fidelity report here (JSON); "
        "default: alongside --summary, or 'none' to skip",
    )
    ap.add_argument(
        "--fidelity-draws", type=int, default=20, help="synthetic cohorts in the report"
    )
    ap.add_argument(
        "--fidelity-figure",
        default=None,
        help="also write a two-panel PNG of prevalence and incidence",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the SQL and exit; no connection")
    args = ap.parse_args(argv)

    if args.n_patients is not None and args.n_patients <= 0:
        ap.error("--n-patients must be positive")

    conn_cfg, cdm, results, cmd_name = _resolve_schemas(args)
    if args.n_patients is not None and not results:
        ap.error("--n-patients needs a writable results_schema")

    if args.dry_run:
        _dry_run(args.backend, cdm, results, args.n_patients, args.seed, args.keep_scratch)
        return 0

    t0 = time.time()
    if args.backend == "databricks":
        backend = db.DatabricksBackend(conn_cfg, args.host, args.http_path, args.token)
    elif args.backend == "postgres":
        dsn = args.dsn or os.environ.get("CDM_DSN")
        if not dsn:
            ap.error("--backend postgres needs --dsn or CDM_DSN")
        backend = db.PostgresBackend(dsn)
    else:
        backend = db.DuckDBBackend(args.duckdb)

    try:
        fit, summary, events, periods = db.run_fit(
            backend,
            cdm,
            results,
            args.n_patients,
            seed=args.seed,
            k=args.k,
            max_patients=args.max_patients,
            keep_scratch=args.keep_scratch,
            cmd=cmd_name,
            out=args.out,
            summary_path=args.summary,
            return_data=True,
        )
    finally:
        backend.close()

    print(
        f"pulled {summary['records']:,} records for "
        f"{summary['patients']:,} patients "
        f"({summary['period_rows']:,} periods) in {time.time() - t0:.1f}s"
    )
    print(
        f"fitted: k={summary['k']:.2f}, tmean={summary['propensity_tmean']:.2f}, "
        f"death={summary['death_frac']:.3f}, n_params={summary['n_parameters']:,}, "
        f"K={summary['K']}"
    )
    print(f"wrote {args.out} and {args.summary}")

    if args.fidelity is not None and args.fidelity.lower() != "none":
        _write_fidelity(args, fit, events, periods)
    return 0


def _write_fidelity(args, fit, events, periods):
    """Score prevalence/incidence against the data the fit was produced from."""
    import json

    from topocdm import fidelity

    path = args.fidelity
    if path is None:
        base = os.path.splitext(args.summary)[0]
        path = f"{base}_fidelity.json"

    print()
    print(f"fidelity: generating {args.fidelity_draws} synthetic cohorts ...")
    t0 = time.time()
    rep = fidelity.fidelity_report(
        fit,
        events,
        periods,
        n_draws=args.fidelity_draws,
        seed=args.seed,
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)

    cov = rep["coverage"]
    print(
        f"  scored {rep['n_concepts_scored']:,} of {rep['n_concepts']:,} concepts "
        f"on {rep['n_patients_real']:,} patients ({time.time() - t0:.1f}s)"
    )
    print(
        f"  prevalence coverage {cov['prevalence']:.0%}   incidence coverage {cov['incidence']:.0%}"
    )
    for name, b in rep["bins"].items():
        if b.get("n"):
            print(
                f"    {name:14s} n={b['n']:4d}  "
                f"P cov {b['prevalence_coverage']:5.0%}  "
                f"P ratio {b['prevalence_ratio_median']:5.2f}  "
                f"(over {b['prevalence_over']}, under {b['prevalence_under']})"
            )
    g = rep["gates"]
    verdict = "PASS" if g["passes_pooled"] and g["passes_common_bin"] else "FAIL"
    print(f"  gate: {verdict} (floor {g['coverage_floor']:.0%})")
    print(f"wrote {path}")

    if args.fidelity_figure:
        _plot_fidelity(rep, args.fidelity_figure)


def _plot_fidelity(rep, path):
    """Two panels: prevalence and incidence, real against the 5-95% band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    c = rep["concepts"]
    sup = np.array(c["supported"])
    order = np.argsort(-np.array(c["prevalence_real"]))
    order = order[sup[order]]
    idx = np.arange(len(order))
    pv = np.array(c["prevalence_real"])[order]
    pm, plo, phi = (
        np.array(c[k])[order] for k in ("prevalence_median", "prevalence_lo", "prevalence_hi")
    )
    iv = np.array(c["incidence_real"])[order]
    im, ilo, ihi = (
        np.array(c[k])[order] for k in ("incidence_median", "incidence_lo", "incidence_hi")
    )

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, real, med, lo, hi, label, scale in [
        (axes[0], pv, pm, plo, phi, "prevalence", 1.0),
        (axes[1], iv, im, ilo, ihi, "incidence per 1000 PY", 1.0),
    ]:
        ax.fill_between(idx, lo * scale, hi * scale, color="#2C6E9B", alpha=0.25, label="5-95% syn")
        ax.plot(idx, med * scale, color="#2C6E9B", lw=1.2, label="synthetic median")
        ax.plot(idx, real, color="#1f2933", lw=1.0, label="real")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_title(
        f"Fidelity: {rep['n_concepts_scored']:,} concepts, "
        f"{rep['n_patients_real']:,} patients, {rep['n_draws']} draws"
    )
    axes[1].set_xlabel("concept rank (by real prevalence)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
