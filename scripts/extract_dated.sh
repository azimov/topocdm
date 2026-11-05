#!/usr/bin/env bash
# Extract dated events WITH end dates, per domain.
#
# The earlier extract (extract_temporal.sh -> events_dated.csv) carried start
# dates only. Condition and drug are interval processes (median duration 364 and
# 365 days) and that structure is lost without end dates. This extract keeps
# the interval/point/absorbing distinction explicit.
#
# Output (data/, gitignored): events_full.csv
#   person_id, domain, concept_id, start_date, end_date
#
#   domain     type        end_date
#   condition  interval    condition_end_date
#   drug       interval    drug_exposure_end_date
#   visit      interval    visit_end_date (degenerate: median 0 days)
#   procedure  point       NULL
#   measurement point      NULL
#   observation point      NULL
#   death      absorbing   NULL
#
# Unmapped records (concept_id = 0) are KEPT: they are a domain property
# (44% of procedures, 2.4% of conditions, 2.3% of measurements) and the model
# must reproduce them as a first-class outcome, not discard them.
#
# Large databases: set SAMPLE_PCT (e.g. SAMPLE_PCT=0.05 for 5%) to draw a
# consistent random patient subset in one pass. The CTE is MATERIALIZED so the
# same patients appear in every domain table; without it the unions would
# sample independently.
set -euo pipefail
OUT=${OUT:-data}
SAMPLE_PCT=${SAMPLE_PCT:-}
mkdir -p "$OUT"
PSQL_CMD=${PSQL:-}
if [ -n "$SAMPLE_PCT" ]; then
  KEEP_CTE="with keep as materialized (select person_id from cdm.person where random() < ${SAMPLE_PCT})"
  JOIN_COND="join keep using (person_id)"
else
  KEEP_CTE=""
  JOIN_COND=""
fi
run() {
  if [ -n "$PSQL_CMD" ]; then
    eval "$PSQL_CMD" -c "\\copy ($2) TO STDOUT WITH CSV HEADER" > "$1"
  else
    docker exec ohdsi-lab-postgres psql -U postgres -d postgres \
      -c "\\copy ($2) TO STDOUT WITH CSV HEADER" > "$1"
  fi
}
echo "dated events with end dates -> $OUT/events_full.csv (sample: ${SAMPLE_PCT:-all})"
run "$OUT/events_full.csv" "
${KEEP_CTE}
select c.person_id, 'condition' as domain, c.condition_concept_id as concept_id,
       c.condition_start_date::date as start_date, c.condition_end_date::date as end_date
  from cdm.condition_occurrence c ${JOIN_COND}
union all select c.person_id, 'drug', c.drug_concept_id,
       c.drug_exposure_start_date::date, c.drug_exposure_end_date::date
  from cdm.drug_exposure c ${JOIN_COND}
union all select c.person_id, 'visit', c.visit_concept_id,
       c.visit_start_date::date, c.visit_end_date::date
  from cdm.visit_occurrence c ${JOIN_COND}
union all select c.person_id, 'procedure', c.procedure_concept_id,
       c.procedure_date::date, NULL::date
  from cdm.procedure_occurrence c ${JOIN_COND}
union all select c.person_id, 'measurement', c.measurement_concept_id,
       c.measurement_date::date, NULL::date
  from cdm.measurement c ${JOIN_COND}
union all select c.person_id, 'observation', c.observation_concept_id,
       c.observation_date::date, NULL::date
  from cdm.observation c ${JOIN_COND}
union all select c.person_id, 'death', 0,
       c.death_date::date, NULL::date
  from cdm.death c ${JOIN_COND}"
wc -l "$OUT/events_full.csv"
