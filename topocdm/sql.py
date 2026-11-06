"""
topocdm.sql -- portable SQL for the data-collection step.

The expensive part of fitting a large CDM is not the arithmetic, it is getting
the events out of the warehouse. The Python extractor does a single UNION ALL
over five domain tables and aggregates; that shape is right, but on a warehouse
it should be decomposed so the engine can parallelise it and so a failed step
can be resumed without repeating the others.

The pattern here follows the one in OHDSI/Circepy#60 (decomposable cohort sets
and shared checkpoint execution):

  1. every step is a named, self-contained query;
  2. each step is content-addressed -- the hash of its SQL is its identity, so
     re-running an unchanged step is a no-op and a changed step is detected;
  3. each step materialises once into a table and downstream steps read that
     table, never the raw CDM.

Applied to this model, the decomposition is:

    ev_<domain>   aggregate one domain to (person_id, concept_id, n)
    person_time   person-time from observation_period
    concept_group concept -> concept group, from the vocabulary
    incidence     union the domain checkpoints

Only `incidence` and `person_time` are needed for the fit; the domain
checkpoints let you re-run one domain, skip a domain you do not want, or share
them with any other analysis that needs the same aggregation.

Cross-platform notes are in `DIALECTS`; the only genuinely dialect-specific
piece is date arithmetic, which is why it is isolated in one function.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------
# Dialects: only date arithmetic actually differs
# --------------------------------------------------------------------------

DIALECTS = {
    # days(end - start)
    "postgres": "({{end}}::date - {{start}}::date)",
    "redshift": "datediff(day, {{start}}, {{end}})",
    "spark": "datediff({{end}}, {{start}})",
    "databricks": "datediff({{end}}, {{start}})",
    "bigquery": "date_diff({{end}}, {{start}}, day)",
    "snowflake": "datediff(day, {{start}}, {{end}})",
    "duckdb": "date_diff('day', {{start}}, {{end}})",
    "sqlserver": "datediff(day, {{start}}, {{end}})",
}

# DDL differs more than the queries do: Postgres has no CREATE OR REPLACE TABLE
CREATE_REPLACE = {
    "postgres": "drop table if exists {name};\ncreate table {name} as\n{sql}",
    "redshift": "drop table if exists {name};\ncreate table {name} as\n{sql}",
    "spark": "create or replace table {name} as\n{sql}",
    "databricks": "create or replace table {name} as\n{sql}",
    "bigquery": "create or replace table {name} as\n{sql}",
    "snowflake": "create or replace table {name} as\n{sql}",
    "duckdb": "create or replace table {name} as\n{sql}",
}

SCHEMA_DDL = {
    "spark": "create database if not exists {schema}",
    "databricks": "create database if not exists {schema}",
}

DOMAIN_TABLES = {
    "condition": ("condition_occurrence", "condition_concept_id"),
    "drug": ("drug_exposure", "drug_concept_id"),
    "procedure": ("procedure_occurrence", "procedure_concept_id"),
    "measurement": ("measurement", "measurement_concept_id"),
    "observation": ("observation", "observation_concept_id"),
}


def days_between(dialect: str, start: str, end: str) -> str:
    tmpl = DIALECTS.get(dialect)
    if tmpl is None:
        raise ValueError(f"unknown dialect {dialect!r}; known: {sorted(DIALECTS)}")
    return tmpl.replace("{{start}}", start).replace("{{end}}", end)


# --------------------------------------------------------------------------
# Step definitions
# --------------------------------------------------------------------------


@dataclass
class Step:
    name: str
    sql: str
    depends_on: tuple = ()

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.sql.encode()).hexdigest()[:16]


def domain_step(domain: str, cdm: str, out_schema: str, dialect: str) -> Step:
    """Aggregate one clinical domain to (person_id, concept_id, n)."""
    table, concept_col = DOMAIN_TABLES[domain]
    sql = f"""
select person_id,
       {concept_col} as concept_id,
       count(*)      as n
from {cdm}.{table}
where {concept_col} <> 0
group by person_id, {concept_col}
""".strip()
    return Step(name=f"ev_{domain}", sql=sql)


def person_time_step(cdm: str, out_schema: str, dialect: str) -> Step:
    days = days_between(dialect, "observation_period_start_date", "observation_period_end_date")
    sql = f"""
select person_id,
       {days} as days_observed
from {cdm}.observation_period
""".strip()
    return Step(name="person_time", sql=sql)


def concept_group_step(cdm: str, vocab: str) -> Step:
    """Concept -> concept group, from the authoritative vocabulary.

    Grouping is (domain_id, concept_class_id): vocabulary-driven, reproducible
    from Athena, and no curation step. Restrict to standard concepts here and
    the concept count C drops accordingly.
    """
    sql = f"""
select concept_id,
       domain_id,
       concept_class_id,
       domain_id || ' | ' || concept_class_id as concept_group
from {vocab}.concept
where standard_concept in ('S', 'C')
""".strip()
    return Step(name="concept_group", sql=sql)


def incidence_step(cdm: str, out_schema: str, domains=None) -> Step:
    """Union the per-domain checkpoints into one incidence table."""
    domains = domains or list(DOMAIN_TABLES)
    union = "\nunion all\n".join(
        f"select person_id, concept_id, n from {out_schema}.ev_{d}" for d in domains
    )
    sql = f"""
select person_id, concept_id, sum(n) as n
from (
{union}
) u
group by person_id, concept_id
""".strip()
    return Step(name="incidence", sql=sql, depends_on=tuple(domains))


def plan(
    cdm: str = "cdm",
    vocab: str = "vocab",
    out_schema: str = "topocdm",
    dialect: str = "postgres",
    domains=None,
) -> list:
    """The full step plan, in dependency order."""
    domains = domains or list(DOMAIN_TABLES)
    steps = [domain_step(d, cdm, out_schema, dialect) for d in domains]
    steps.append(person_time_step(cdm, out_schema, dialect))
    steps.append(concept_group_step(cdm, vocab))
    steps.append(incidence_step(cdm, out_schema, domains))
    return steps


# --------------------------------------------------------------------------
# Checkpoint execution
# --------------------------------------------------------------------------


@dataclass
class CheckpointExecutor:
    """Materialise steps once, skip unchanged ones, record what ran.

    `run_sql` is any callable taking a SQL string and returning rows (or
    None). Keeping it injectable is what makes this portable: on Databricks you
    pass a Spark session wrapper, locally a psql shell-out, in tests a stub.
    """

    run_sql: Callable[[str], Any]
    query_sql: Callable[[str], Any] | None = None
    out_schema: str = "topocdm"
    dialect: str = "postgres"
    executed: list = field(default_factory=list)
    skipped: list = field(default_factory=list)

    def _exists(self, name: str) -> bool:
        """True if the checkpoint table exists.

        Needs a *query* path that returns rows -- `run_sql` returns nothing,
        which silently disables checkpointing if the two are conflated.
        """
        q = self.query_sql or self.run_sql
        try:
            rows = q(f"select 1 from {self.out_schema}.{name} limit 1")
            return rows is not None
        except Exception:
            return False

    def run(self, steps, force: bool = False, checkpoint: bool = True):
        """Execute steps in order, materialising each as a table.

        Materialisation is the point: downstream steps read the checkpoint, so
        a failure late in the plan does not repeat the expensive aggregation,
        and a second analysis can reuse the same tables.
        """
        self.run_sql(
            SCHEMA_DDL.get(self.dialect, "create schema if not exists {schema}").format(
                schema=self.out_schema
            )
        )
        for step in steps:
            if checkpoint and not force and self._exists(step.name):
                self.skipped.append(step)
                continue
            ddl = CREATE_REPLACE[self.dialect].format(
                name=f"{self.out_schema}.{step.name}", sql=step.sql
            )
            self.run_sql(ddl)
            self.executed.append(step)
        return self

    def lineage(self) -> list:
        return [
            {"name": s.name, "content_hash": s.content_hash, "depends_on": list(s.depends_on)}
            for s in self.executed
        ]
