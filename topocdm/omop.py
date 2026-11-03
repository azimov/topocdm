"""
topocdm.omop -- write a generated cohort into OMOP CDM domain tables.

The model produces a patient x concept count matrix. To be usable, that has to
become rows in the domain tables, with real concept_ids from a standard
vocabulary and dates attached.

Two things are deliberately explicit here:

1. **Domain routing comes from the vocabulary.** A concept belongs to exactly
   one domain (`CONCEPT.domain_id`), and that decides which table its events go
   into. We never guess: if a concept's domain is unknown, it is refused rather
   than silently dropped into OBSERVATION.

2. **Dates are generated, not modelled.** The current fit is cross-sectional --
   it estimates rates over the observation period, not trajectories. Visit
   dates are therefore drawn uniformly within each person's observation period
   and events are attached to visits. This produces plausible clustering
   (events arriving in batches, as they do) but it is a placeholder, and the
   channel/time extension replaces it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

# OMOP CDM v5.4/5.5: domain_id -> table name
DOMAIN_TO_TABLE = {
    "Condition": "condition_occurrence",
    "Drug": "drug_exposure",
    "Procedure": "procedure_occurrence",
    "Measurement": "measurement",
    "Observation": "observation",
    "Device": "device_exposure",
    "Specimen": "specimen",
    "Visit": "visit_occurrence",
}

# standard OMOP type concepts
TYPE_CONCEPT = {
    "EHR": 32817,
    "EHR encounter diagnosis": 32817,
}

GENDER_MALE = 8507
GENDER_FEMALE = 8532
RACE_UNKNOWN = 0
ETHNICITY_UNKNOWN = 0


@dataclass
class ConceptVocab:
    """The slice of the vocabulary the generator needs.

    concept_id      OMOP concept identifier
    domain_id       'Condition', 'Drug', 'Measurement', ...
    vocabulary_id   'SNOMED', 'RxNorm', 'LOINC', ...
    concept_name    optional label
    """

    concept_id: np.ndarray
    domain_id: np.ndarray
    vocabulary_id: np.ndarray | None = None
    concept_name: np.ndarray | None = None

    def __post_init__(self):
        self.concept_id = np.asarray(self.concept_id, dtype=np.int64)
        self.domain_id = np.asarray(self.domain_id)
        if self.vocabulary_id is None:
            self.vocabulary_id = np.full(len(self.concept_id), "Unknown", dtype=object)
        if self.concept_name is None:
            self.concept_name = np.array([f"concept {c}" for c in self.concept_id],
                                         dtype=object)
        bad = ~np.isin(self.domain_id, list(DOMAIN_TO_TABLE))
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} concepts have a domain with no CDM table: "
                f"{sorted(set(self.domain_id[bad]))[:5]}")

    def __len__(self):
        return len(self.concept_id)

    def table_for(self, concept_ids) -> np.ndarray:
        idx = {int(c): t for c, t in zip(self.concept_id,
                                         [DOMAIN_TO_TABLE[d] for d in self.domain_id])}
        try:
            return np.array([idx[int(c)] for c in concept_ids], dtype=object)
        except KeyError as exc:
            raise ValueError(f"concept {exc} is not in the supplied vocabulary") from None


def _dates(start: date, days: np.ndarray) -> np.ndarray:
    return np.array([start + timedelta(days=int(d)) for d in days], dtype=object)


def visits_per_patient(events: np.ndarray, rng) -> np.ndarray:
    """Batch events into visits.

    Real CDMs record several things at one encounter. A patient with k events
    gets roughly k / E[events per visit] visits, with a minimum of one for
    anyone who appears at all.
    """
    per_visit = 1.0 + rng.poisson(1.5, size=len(events))
    v = np.ceil(events / per_visit).astype(int)
    v[events == 0] = 0
    v[events > 0] = np.maximum(v[events > 0], 1)
    return v


def generate_omop_tables(X, T, vocab: ConceptVocab, start_date: str = "2015-01-01",
                         person_id_offset: int = 0, id_offset: int = 0,
                         rng=None, gender_ratio: float = 0.5,
                         age_range=(18, 90), type_concept_id: int = 32817,
                         emit_person_columns: bool = True) -> dict:
    """Turn a count matrix into OMOP domain tables.

    `X` is (patients x concepts) counts; `T` is person-time in years and sets
    each observation period. Returns a dict of DataFrames keyed by CDM table
    name, including PERSON and OBSERVATION_PERIOD.
    """
    rng = rng or np.random.default_rng(0)
    X = np.asarray(X)
    n, c = X.shape
    T = np.asarray(T, dtype=float)
    if len(vocab) != c:
        raise ValueError(f"vocabulary has {len(vocab)} concepts but X has {c} columns")

    start = date.fromisoformat(start_date)
    person_ids = np.arange(1, n + 1) + person_id_offset

    op_days = np.round(T * 365.25).astype(int)
    op_start = start + timedelta(days=0)
    op_end_days = op_days

    # ---------------- PERSON ----------------
    year_of_birth = rng.integers(age_range[0], age_range[1], size=n)
    year_of_birth = (op_start.year - year_of_birth).astype(int)
    gender = np.where(rng.random(n) < gender_ratio, GENDER_FEMALE, GENDER_MALE)
    person = pd.DataFrame({
        "person_id": person_ids,
        "gender_concept_id": gender,
        "year_of_birth": year_of_birth,
        "month_of_birth": rng.integers(1, 13, size=n),
        "day_of_birth": rng.integers(1, 29, size=n),
        "race_concept_id": RACE_UNKNOWN,
        "ethnicity_concept_id": ETHNICITY_UNKNOWN,
        "person_source_value": [f"synth-{i}" for i in person_ids],
        "gender_source_value": np.where(gender == GENDER_FEMALE, "F", "M"),
        "gender_source_concept_id": gender,
        "race_source_value": "unknown",
        "race_source_concept_id": RACE_UNKNOWN,
        "ethnicity_source_value": "unknown",
        "ethnicity_source_concept_id": ETHNICITY_UNKNOWN,
    }) if emit_person_columns else None

    obs_period = pd.DataFrame({
        "observation_period_id": np.arange(1, n + 1) + id_offset,
        "person_id": person_ids,
        "observation_period_start_date": _dates(op_start, np.zeros(n)),
        "observation_period_end_date": _dates(op_start, op_end_days),
        "period_type_concept_id": type_concept_id,
    })

    # ---------------- visits ----------------
    events_total = X.sum(axis=1).astype(int)
    n_visits = visits_per_patient(events_total, rng)
    visit_person = np.repeat(person_ids, n_visits)
    visit_day = np.concatenate([
        np.sort(rng.integers(0, max(int(d), 1), size=v)) if v else np.empty(0, dtype=int)
        for v, d in zip(n_visits, op_days)
    ]).astype(int)
    visit_ids = np.arange(1, len(visit_person) + 1) + id_offset
    visit = pd.DataFrame({
        "visit_occurrence_id": visit_ids,
        "person_id": visit_person,
        "visit_concept_id": 9202,          # 'Outpatient Visit'
        "visit_start_date": _dates(op_start, visit_day),
        "visit_end_date": _dates(op_start, visit_day),
        "visit_type_concept_id": type_concept_id,
        "visit_source_value": "synthetic outpatient",
        "visit_source_concept_id": 9202,
        "discharge_to_concept_id": 0,
        "admitted_from_concept_id": 0,
    })

    # map (person, visit index) -> visit_occurrence_id
    visit_ptr = {}
    pos = 0
    for pid, v in zip(person_ids, n_visits):
        visit_ptr[int(pid)] = visit_ids[pos:pos + v]
        pos += v

    # ---------------- events ----------------
    tables = {}
    for j in range(c):
        table = DOMAIN_TO_TABLE[vocab.domain_id[j]]
        counts = X[:, j].astype(int)
        rows = np.nonzero(counts)[0]
        if rows.size == 0:
            continue
        pid = np.repeat(person_ids[rows], counts[rows])
        # allocate each event to one of that patient's visits
        vid = np.empty(pid.size, dtype=np.int64)
        k = 0
        for r in rows:
            m = counts[r]
            vid[k:k + m] = rng.choice(visit_ptr[int(person_ids[r])], size=m)
            k += m
        vmap = dict(zip(visit_ids, visit_day))
        day = np.array([vmap[int(v)] for v in vid], dtype=int)

        ev = pd.DataFrame({
            "person_id": pid,
            "concept_id": int(vocab.concept_id[j]),
            "date": _dates(op_start, day),
            "visit_occurrence_id": vid,
            "source_value": str(vocab.concept_name[j]),
            "source_concept_id": int(vocab.concept_id[j]),
            "type_concept_id": type_concept_id,
        })
        tables.setdefault(table, []).append(ev)

    out = {"person": person, "observation_period": obs_period,
           "visit_occurrence": visit}

    field_names = {
        "condition_occurrence": ("condition_occurrence_id", "condition_concept_id",
                                 "condition_start_date", "condition_source_value",
                                 "condition_source_concept_id"),
        "drug_exposure": ("drug_exposure_id", "drug_concept_id",
                          "drug_exposure_start_date", "drug_source_value",
                          "drug_source_concept_id"),
        "procedure_occurrence": ("procedure_occurrence_id", "procedure_concept_id",
                                 "procedure_date", "procedure_source_value",
                                 "procedure_source_concept_id"),
        "measurement": ("measurement_id", "measurement_concept_id",
                        "measurement_date", "measurement_source_value",
                        "measurement_source_concept_id"),
        "observation": ("observation_id", "observation_concept_id",
                        "observation_date", "observation_source_value",
                        "observation_source_concept_id"),
        "device_exposure": ("device_exposure_id", "device_concept_id",
                            "device_exposure_start_date", "device_source_value",
                            "device_source_concept_id"),
    }

    for table, frames in tables.items():
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values(["person_id", "date"]).reset_index(drop=True)
        id_col, concept_col, date_col, src_col, src_concept_col = field_names[table]
        out[table] = pd.DataFrame({
            id_col: np.arange(1, len(df) + 1) + id_offset,
            "person_id": df["person_id"],
            concept_col: df["concept_id"],
            date_col: df["date"],
            "type_concept_id": df["type_concept_id"],
            "visit_occurrence_id": df["visit_occurrence_id"],
            src_col: df["source_value"],
            src_concept_col: df["source_concept_id"],
        })
    return out


def write_omop_tables(tables: dict, outdir: str) -> list:
    """Write each table to CSV. Returns the list of paths written."""
    import os
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for name, df in tables.items():
        if df is None:
            continue
        p = os.path.join(outdir, f"{name}.csv")
        df.to_csv(p, index=False)
        paths.append(p)
    return paths
