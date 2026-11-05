"""
topocdm.hurdle -- per-concept prevalence calibration.

The generators reproduce the total number of records per concept (incidence)
but not how many patients have the concept at all (prevalence). The diagnosis
in docs/FIDELITY_REPORT.md separates two mechanisms: rare concepts are spread
across too many patients, and common concepts are assigned to the wrong number
of patients in opposite directions on the two fixtures. Both are patient-level
presence errors, not count errors.

This module adds a per-concept presence probability:

    pi_j = share of training patients with at least one record of concept j

and reallocates a generated patient-by-concept count matrix so that the number
of positive patients for each concept matches pi_j, while the column total
(the incidence) is preserved. It is a hurdle/zero-inflation layer with one
parameter per concept and no per-patient parameter, so it does not change the
release's privacy story. It is a post-generation calibration, not a change to
the event process: co-occurrence and refill are untouched.

Fit on training patients; apply to synthetic patients. Applying it in-sample
is circular and is not a quality claim.
"""

from __future__ import annotations

import numpy as np


def fit_prevalence(X_train) -> np.ndarray:
    """Per-concept prevalence (share of patients with >= 1 record)."""
    X = np.asarray(X_train, dtype=float)
    return (X > 0).mean(axis=0)


def sample_prevalence(X_train, rng, alpha: float = 0.5) -> np.ndarray:
    """Draw per-concept prevalence from its Beta posterior (Jeffreys prior).

    A point estimate pi_j makes the generated prevalence nearly deterministic,
    so the simulation interval collapses and the held-out data falls outside it.
    Sampling pi_j per cohort propagates the uncertainty in the training
    prevalence and restores the interval.
    """
    X = np.asarray(X_train, dtype=bool)
    k = X.sum(axis=0).astype(float)
    n = float(X.shape[0])
    return rng.beta(k + alpha, n - k + alpha)


def apply_prevalence(X, pi, rng=None, max_add_frac: float = 1.0) -> np.ndarray:
    """Reallocate counts so each concept's positive-patient share matches pi.

    Column totals are preserved exactly, so incidence is unchanged. When a
    concept must gain patients, one record is taken from a patient with at
    least two records of that concept; if there is no such donor the concept is
    left unchanged (and the shortfall is reported by the caller through the
    prevalence coverage).

    X: (n, C) count matrix. pi: (C,) target prevalence.
    """
    X = np.array(X, dtype=float)
    n, c = X.shape
    pi = np.asarray(pi, dtype=float)
    for j in range(c):
        col = X[:, j]
        total = float(col.sum())
        if total <= 0:
            continue
        pos = np.nonzero(col)[0]
        k = len(pos)
        target = int(round(float(pi[j]) * n))
        # a positive patient needs at least one record; never exceed the total
        target = max(1, min(target, int(total)))
        if target == k:
            continue

        if target < k:
            # concentrate: drop the positives with the fewest records, move
            # their records to the largest remaining positive
            order = pos[np.argsort(col[pos], kind="stable")]
            drop = order[: k - target]
            keep = np.setdiff1d(pos, drop)
            moved = float(col[drop].sum())
            col[drop] = 0.0
            if len(keep):
                col[keep[np.argmax(col[keep])]] += moved
            X[:, j] = col
        else:
            # spread: add negative patients, each taking one record from a
            # positive with at least two records
            neg = np.setdiff1d(np.arange(n), pos)
            if len(neg) == 0:
                continue
            # prefer patients with a higher overall record count (propensity)
            rowsum = X.sum(axis=1)
            neg = neg[np.argsort(-rowsum[neg], kind="stable")]
            need = target - k
            for i in neg[:need]:
                donors = pos[col[pos] >= 2]
                if len(donors) == 0:
                    break
                d = donors[np.argmax(col[donors])]
                col[d] -= 1.0
                col[i] = 1.0
                pos = np.append(pos, i)
            X[:, j] = col
    return X
