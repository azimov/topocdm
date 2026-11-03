"""
topocdm.visits -- the visit layer: how records are grouped and ordered in time.

The count model says how many records of each concept a patient has. It says
nothing about how those records are arranged: whether they arrive in one
encounter or twenty, or in what order. On a real CDM that is a large error --
91% of the gaps between consecutive events in this dataset are under a day.

This module adds the missing layer without refitting the counts:

  1. visits are generated from the fitted visit rate (as before);
  2. a sequence of concept groups is drawn for those visits from a first-order
     Markov chain, fitted by counting transitions between consecutive visits;
  3. each of the patient's records is assigned to a visit whose group matches,
     so records of related concepts land in the same encounter and the order
     across visits follows the chain.

The marginal counts are untouched -- prevalence and incidence are exactly what
the count model produced. Only their arrangement in time changes.

The chain is fitted by counting, which is one pass and H*H numbers. Generating
is where the cost is: it becomes a sequential simulation per patient rather
than one multinomial draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VisitModel:
    trans: np.ndarray          # (H, H) P(next group | current group)
    start: np.ndarray          # (H,) distribution of the group at a first visit
    visits_per_patient: float  # mean
    records_per_visit: np.ndarray   # empirical distribution, sampled from
    concept_group: np.ndarray = None   # (C,) group per concept column

    def summary(self) -> dict:
        H = self.trans.shape[0]
        marg = self.trans.mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.log2(np.maximum(self.trans, 1e-12) / np.maximum(marg, 1e-12))
        mi = float(np.nansum(self.trans * np.log2(
            np.maximum(self.trans, 1e-12) / np.maximum(marg, 1e-12))) / H)
        return dict(n_groups=int(H), mean_visits=float(self.visits_per_patient),
                    mean_records_per_visit=float(np.mean(self.records_per_visit)),
                    persistence=float(np.mean(np.diag(self.trans))),
                    lift_range=(float(np.nanmin(lift)), float(np.nanmax(lift))))


def fit_visit_model(visits: pd.DataFrame, events: pd.DataFrame,
                    concept_group_of: dict, n_groups: int,
                    concept_group: np.ndarray | None = None) -> VisitModel:
    """Count transitions between consecutive visits' concept groups.

    A visit's "group" is taken to be the modal concept group of the records on
    that date. That is a coarsening, and it is deliberate: ordering *within* a
    day is not identifiable (everything carries the same date), while ordering
    *between* visits is.
    """
    ev = events.copy()
    ev["grp"] = ev["concept_id"].map(concept_group_of)
    ev = ev.dropna(subset=["grp"])
    ev["grp"] = ev["grp"].astype(int)

    # modal group per (patient, date)
    per_visit = (ev.groupby(["person_id", "event_date", "grp"]).size()
                   .rename("n").reset_index())
    idx = per_visit.groupby(["person_id", "event_date"])["n"].idxmax()
    g = per_visit.loc[idx].sort_values(["person_id", "event_date"])
    g["next_grp"] = g.groupby("person_id")["grp"].shift(-1)

    tr = g.dropna(subset=["next_grp"])
    M = np.zeros((n_groups, n_groups))
    np.add.at(M, (tr["grp"].to_numpy(), tr["next_grp"].to_numpy().astype(int)), 1.0)

    # rows -> probabilities, with a small prior so unseen transitions are possible
    M = M + 0.5
    trans = M / M.sum(axis=1, keepdims=True)
    start = np.bincount(g["grp"].to_numpy(), minlength=n_groups).astype(float) + 0.5
    start /= start.sum()

    vpp = g.groupby("person_id")["event_date"].nunique()
    rpv = per_visit.groupby(["person_id", "event_date"])["n"].sum().to_numpy()

    return VisitModel(trans=trans, start=start,
                      visits_per_patient=float(vpp.mean()),
                      records_per_visit=rpv,
                      concept_group=concept_group)


def assign_to_visits(counts: np.ndarray, n_visits: int, model: VisitModel,
                     rng) -> list:
    """Place a patient's records into an ordered sequence of visits.

    Returns a list of (visit_index, concept_index) pairs. The counts per
    concept are preserved exactly; only their placement changes.
    """
    if n_visits <= 0:
        return []
    H = model.trans.shape[0]
    # a group sequence for this patient's visits
    seq = np.empty(n_visits, dtype=int)
    seq[0] = rng.choice(H, p=model.start)
    for v in range(1, n_visits):
        seq[v] = rng.choice(H, p=model.trans[seq[v - 1]])

    # visits available for each group
    by_group = {h: np.nonzero(seq == h)[0] for h in range(H)}

    out = []
    for j in np.nonzero(counts > 0)[0]:
        h = int(model.concept_group[j])
        slots = by_group.get(h)
        if slots is None or slots.size == 0:
            slots = np.arange(n_visits)          # fall back: any visit
        pick = rng.choice(slots, size=int(counts[j]), replace=True)
        out.extend((int(v), int(j)) for v in pick)
    return out
