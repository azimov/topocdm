#!/usr/bin/env bash
# Extract dated events and visits, for the temporal model and for
# CohortIncidence-style comparisons.
set -euo pipefail
OUT=${OUT:-data}
mkdir -p "$OUT"
PSQL_CMD=${PSQL:-}
run() {
  if [ -n "$PSQL_CMD" ]; then
    eval "$PSQL_CMD" -c "\\copy ($2) TO STDOUT WITH CSV HEADER" > "$1"
  else
    docker exec ohdsi-lab-postgres psql -U postgres -d postgres \
      -c "\\copy ($2) TO STDOUT WITH CSV HEADER" > "$1"
  fi
}
echo "dated events -> $OUT/events_dated.csv"
run "$OUT/events_dated.csv" "
select person_id, condition_concept_id as concept_id, condition_start_date::date as event_date
  from cdm.condition_occurrence where condition_concept_id <> 0
union all select person_id, drug_concept_id, drug_exposure_start_date::date
  from cdm.drug_exposure where drug_concept_id <> 0
union all select person_id, procedure_concept_id, procedure_date::date
  from cdm.procedure_occurrence where procedure_concept_id <> 0
union all select person_id, measurement_concept_id, measurement_date::date
  from cdm.measurement where measurement_concept_id <> 0
union all select person_id, observation_concept_id, observation_date::date
  from cdm.observation where observation_concept_id <> 0"
echo "visits -> $OUT/visits.csv"
run "$OUT/visits.csv" "select person_id, visit_start_date::date as visit_date from cdm.visit_occurrence"
wc -l "$OUT"/events_dated.csv "$OUT"/visits.csv
