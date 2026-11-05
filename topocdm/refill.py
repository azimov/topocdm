"""
topocdm.refill -- repeat-prescription continuity for generated drug records.

The generators draw each drug record as an independent interval. Real data has
courses: a patient on a chronic drug gets a fill, then another when it runs
out, so several records sit within a few weeks of each other and the era
builder merges them into one spell. Without that, the generated database has
too many, too short drug eras (3.6x on Synthea).

This module estimates the observed distribution of

    fills per era        how many records merge into one continuous spell
    gap within an era    days between consecutive fills of the same course

and re-clusters a generated drug table into the same kind of courses. It
preserves the number of records and each record's duration; only the dates
move. That is deliberate: prevalence and incidence are count-based, so they are
unchanged, while the era count (which is date-based) is repaired.

It is a re-clustering of generated records, not a fully generative renewal
process. It is labelled as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_refill(drug_records: pd.DataFrame, gap_days: int = 30) -> dict:
    """Estimate fills-per-era and within-era gap distributions.

    drug_records: person_id, concept_id, start_date, end_date
    """
    df = drug_records[["person_id", "concept_id", "start_date", "end_date"]].copy()
    df = df.sort_values(["person_id", "concept_id", "start_date"])
    fills, gaps = [], []
    for _, g in df.groupby(["person_id", "concept_id"], sort=False):
        starts = pd.to_datetime(g["start_date"]).to_numpy(dtype="datetime64[D]")
        ends = pd.to_datetime(g["end_date"]).to_numpy(dtype="datetime64[D]")
        era_start = 0
        for i in range(1, len(g)):
            gap = float((starts[i] - ends[i - 1]) / np.timedelta64(1, "D"))
            if gap <= gap_days:
                gaps.append(max(gap, 0.0))
            else:
                fills.append(i - era_start)
                era_start = i
        fills.append(len(g) - era_start)
    return dict(fills=np.asarray(fills, dtype=np.int64),
                gaps=np.asarray(gaps, dtype=float))


def apply_refill(drug_table: pd.DataFrame, refill: dict,
                 periods: pd.DataFrame, rng, gap_days: int = 30) -> pd.DataFrame:
    """Re-date generated drug records into courses.

    drug_table: person_id, drug_concept_id, drug_exposure_start_date,
                drug_exposure_end_date
    The number of records and each record's duration are preserved.
    """
    if drug_table is None or len(drug_table) == 0:
        return drug_table
    fills = np.asarray(refill["fills"], dtype=np.int64)
    gaps = np.asarray(refill["gaps"], dtype=float)
    if len(fills) == 0:
        return drug_table
    fills = np.maximum(fills, 1)
    if len(gaps) == 0:
        gaps = np.array([0.0])

    end_map = periods.set_index("person_id")["end_date"]
    df = drug_table.copy()
    df["drug_exposure_start_date"] = pd.to_datetime(df["drug_exposure_start_date"])
    df["drug_exposure_end_date"] = pd.to_datetime(df["drug_exposure_end_date"])
    rows = []
    for (pid, cid), g in df.groupby(["person_id", "drug_concept_id"], sort=False):
        g = g.sort_values("drug_exposure_start_date").reset_index(drop=True)
        n = len(g)
        if n < 2:
            rows.extend((pid, cid, r.drug_exposure_start_date,
                         r.drug_exposure_end_date) for r in g.itertuples())
            continue
        starts = g["drug_exposure_start_date"].to_numpy(dtype="datetime64[D]")
        dur = (g["drug_exposure_end_date"]
               - g["drug_exposure_start_date"]).dt.days.to_numpy(dtype=np.int64)
        ob_end = np.datetime64(pd.to_datetime(end_map.get(pid)), "D")

        # partition n records into courses with the observed fill distribution
        sizes, total = [], 0
        while total < n:
            f = int(fills[rng.integers(0, len(fills))])
            f = max(1, min(f, n - total))
            sizes.append(f)
            total += f

        idx = 0
        for s in sizes:
            base = starts[idx]                       # keep one original anchor
            offset = 0.0
            for j in range(s):
                if j > 0:
                    offset += float(gaps[rng.integers(0, len(gaps))])
                start = base + np.timedelta64(int(round(offset)), "D")
                start = min(start, ob_end)
                end = min(start + np.timedelta64(int(dur[idx]), "D"), ob_end)
                rows.append((pid, cid, pd.to_datetime(start), pd.to_datetime(end)))
                idx += 1
    return pd.DataFrame(rows, columns=[
        "person_id", "drug_concept_id", "drug_exposure_start_date",
        "drug_exposure_end_date"]) if rows else drug_table
