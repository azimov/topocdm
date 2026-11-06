"""The connection card and the portable sample/extract SQL."""

from __future__ import annotations

import importlib.util
import pathlib
import textwrap

from topocdm import db

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fit_databricks", ROOT / "scripts" / "fit_databricks.py")
fit_databricks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fit_databricks)


def _card(tmp_path, key):
    text = textwrap.dedent(f"""
    databricks:
      connection:
        server_hostname: "adb-test"
        http_path: "/sql/1.0/warehouses/test"
        personal_access_token: "dapi-test"
    {key}:
      test:
        cdm_schema: "main.omop"
        vocabulary_schema: "main.omop"
        results_schema: "scratch.me"
    """).lstrip()
    path = tmp_path / "card.yml"
    path.write_text(text)
    return path


def test_cdms_key(tmp_path):
    conn, cdm, scratch, vocab, name = fit_databricks.load_config(
        str(_card(tmp_path, "cdms")))
    assert name == "test"
    assert cdm == "main.omop"
    assert scratch == "scratch.me"
    assert vocab == "main.omop"
    assert conn["server_hostname"] == "adb-test"


def test_legacy_cmds_key_still_loads(tmp_path):
    _, cdm, _, _, name = fit_databricks.load_config(
        str(_card(tmp_path, "cmds")))
    assert name == "test"
    assert cdm == "main.omop"


def test_sampled_sql_uses_a_scratch_table_not_a_cte():
    table = db.sample_table_name("scratch.me", 1000, 0)
    assert table == "scratch.me.topocdm_sample_0_1000"

    sample = db.create_sample_sql("spark", table, "main.omop", 1000, 0)
    assert "create or replace table scratch.me.topocdm_sample_0_1000" in sample
    assert "order by hash(person_id, 0)" in sample
    assert "limit 1000" in sample

    events, periods = db.build_extract_sql(
        "main.omop", f" join {table} using (person_id)")
    join = "join scratch.me.topocdm_sample_0_1000 using (person_id)"
    assert join in events
    assert join in periods
    # the warehouse rejected WITH ... AS MATERIALIZED; the sample is a table
    assert "materialized" not in events.lower()
    assert "with keep" not in events.lower()


def test_full_extract_has_no_scratch_join():
    events, periods = db.build_extract_sql("main.omop")
    assert "join scratch" not in events
    assert "join scratch" not in periods


def test_each_dialect_gets_a_deterministic_sample():
    pg = db.sample_select_sql("postgres", "main.omop", 100, 7)
    duck = db.sample_select_sql("duckdb", "main.omop", 100, 7)
    spark = db.sample_select_sql("spark", "main.omop", 100, 7)

    assert "md5(" in pg and "limit 100" in pg
    assert "hash(" in duck and "limit 100" in duck
    assert "hash(person_id, 7)" in spark and "limit 100" in spark

    assert db.create_sample_sql("spark", "s.t", "main.omop", 10, 0).startswith(
        "create or replace table")
    assert db.create_sample_sql("postgres", "s.t", "main.omop", 10, 0).startswith(
        "create table")
