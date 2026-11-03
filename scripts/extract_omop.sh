#!/usr/bin/env bash
# Extract a patient x concept incidence table from an OMOP CDM.
#
# Uses `docker exec` against the local OHDSI lab container, so it needs no
# credentials on the host. Point it at another server by setting
#   PSQL="psql \"host=... dbname=... user=...\""
# and the script will use that instead.
#
# Outputs (data/, gitignored):
#   events.csv   person_id, concept_id, n
#   concepts.csv concept_id, domain_id, vocabulary_id, concept_class_id, concept_name
#   periods.csv  person_id, start, end
set -euo pipefail

OUT=${OUT:-data}
mkdir -p "$OUT"
PSQL_CMD=${PSQL:-}

run() {  # run <file> <sql>
  if [ -n "$PSQL_CMD" ]; then
    eval "$PSQL_CMD" -c "\\copy ($2) TO STDOUT WITH CSV HEADER" > "$1"
  else
    docker exec ohdsi-lab-postgres psql -U postgres -d postgres \
      -c "\\copy ($2) TO STDOUT WITH CSV HEADER" > "$1"
  fi
}

echo "extracting events -> $OUT/events.csv"
run "$OUT/events.csv" "
with ev as (
  select person_id, condition_concept_id as concept_id from cdm.condition_occurrence where condition_concept_id <> 0
  union all select person_id, drug_concept_id        from cdm.drug_exposure        where drug_concept_id <> 0
  union all select person_id, procedure_concept_id   from cdm.procedure_occurrence  where procedure_concept_id <> 0
  union all select person_id, measurement_concept_id from cdm.measurement          where measurement_concept_id <> 0
  union all select person_id, observation_concept_id from cdm.observation          where observation_concept_id <> 0
)
select person_id, concept_id, count(*)::int as n from ev group by 1,2"

echo "extracting concepts -> $OUT/concepts.csv"
run "$OUT/concepts.csv" "
select concept_id, domain_id, vocabulary_id, concept_class_id, concept_name
from vocab.concept"

echo "extracting observation periods -> $OUT/periods.csv"
run "$OUT/periods.csv" "
select person_id,
       observation_period_start_date as start_date,
       observation_period_end_date   as end_date
from cdm.observation_period"

wc -l "$OUT"/*.csv
