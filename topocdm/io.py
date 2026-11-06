"""
topocdm.io -- load an OMOP CDM extract into a SparseCDM.

The extract is three CSVs produced by scripts/extract_omop.sh:

    events.csv    person_id, concept_id, n
    concepts.csv  concept_id, domain_id, vocabulary_id, concept_class_id, ...
    periods.csv   person_id, start_date, end_date

Concept grouping defaults to (domain_id, concept_class_id), which is a
vocabulary-driven partition: it does not require a curation step and it is
reproducible from Athena alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from topocdm.scalable import SparseCDM


def load_incidence(
    events="data/events.csv",
    concepts="data/concepts.csv",
    periods="data/periods.csv",
    group_by=("domain_id", "concept_class_id"),
    min_events=1,
) -> tuple[SparseCDM, list, pd.DataFrame]:
    ev = pd.read_csv(events)
    pe = pd.read_csv(periods, parse_dates=["start_date", "end_date"])
    cv = pd.read_csv(concepts, usecols=["concept_id", *group_by])

    # restrict to concepts that actually occur
    used = np.sort(ev["concept_id"].unique())
    ev = ev[ev["concept_id"].isin(used)]
    cv = cv[cv["concept_id"].isin(used)].set_index("concept_id").loc[used]

    persons = np.sort(pe["person_id"].unique())
    prow = {p: i for i, p in enumerate(persons)}
    crow = {c: j for j, c in enumerate(used)}

    rows = ev["person_id"].map(prow).to_numpy()
    cols = ev["concept_id"].map(crow).to_numpy()
    vals = ev["n"].to_numpy(dtype=float)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(persons), len(used)))

    # drop concepts below the minimum event count (small-cell discipline)
    if min_events > 1:
        keep = np.asarray(X.sum(axis=0)).ravel() >= min_events
        X = X[:, keep]
        used = used[keep]
        cv = cv.loc[used]

    days = (
        pe.set_index("person_id").loc[persons, "end_date"]
        - pe.set_index("person_id").loc[persons, "start_date"]
    ).dt.days
    T = np.maximum(days.to_numpy(dtype=float) / 365.25, 1.0 / 365.25)

    labels = cv[list(group_by)].astype(str).agg(" | ".join, axis=1)
    codes, uniques = pd.factorize(labels)
    return (
        SparseCDM(
            X=X,
            T=T,
            concept_group=codes.astype(np.int64),
            concept_ids=used.astype(np.int64),
            person_ids=persons.astype(np.int64),
        ),
        list(uniques),
        cv,
    )
