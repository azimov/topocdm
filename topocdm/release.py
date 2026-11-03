"""
topocdm.release -- the artefact that leaves the institution.

A `Release` is everything that is safe and sufficient to share: concept
intensities, the block affinity matrix, mixing weights, a propensity shape, and
a target density. It contains **no per-patient parameters**. That is the whole
argument for the privacy posture, and it is also why fitting on a patient
subsample costs nothing that the release would have used.

    Release  =  lambda (C) + B (G x H) + pi (G) + k (1) + a concept list

Nothing in it indexes a person. Generation samples *new* patients from the
propensity distribution, so a synthetic cohort is a draw from a population
model, not a perturbation of the observed one.

Density is a parameter of the generator, not of the fit. Fitting on real data
estimates `density_observed`; generating takes whatever density you ask for, so
the same release can produce a sparse cohort or a dense one without refitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from topocdm.scalable import SparseCDM


@dataclass
class Release:
    lam: np.ndarray
    concept_group: np.ndarray
    concept_ids: np.ndarray
    B: np.ndarray | None = None
    pi: np.ndarray | None = None
    propensity_shape: float = 1.0
    density: float = 1.0            # events per person-year, as observed
    n_patients_fitted: int = 0
    n_events_fitted: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def n_concepts(self) -> int:
        return len(self.lam)

    @property
    def n_parameters(self) -> int:
        """How many numbers this release actually contains."""
        n = len(self.lam) + 1 + len(self.concept_ids)
        if self.B is not None:
            n += self.B.size + len(self.pi)
        return int(n)

    def summary(self) -> dict:
        return dict(
            n_concepts=self.n_concepts,
            n_concept_groups=int(self.concept_group.max()) + 1,
            n_patient_groups=0 if self.B is None else int(self.B.shape[0]),
            n_parameters=self.n_parameters,
            density=self.density,
            propensity_shape=self.propensity_shape,
            contains_patient_level_parameters=False,
        )


def make_release(params, concept_ids, concept_group, propensity_shape: float,
                 density: float, n_patients: int = 0, n_events: int = 0,
                 meta: dict | None = None, pi=None) -> Release:
    """Build a Release from fitted parameters, dropping every patient term.

    `pi` -- the group mixing weights -- is a global parameter and belongs in the
    release. It must not be replaced by a uniform vector: if the fitted groups
    are unbalanced, uniform mixing over-samples the rare high-intensity groups
    and inflates the generated event rate.
    """
    B = None if params.B is None else np.asarray(params.B, dtype=float)
    if B is not None:
        if pi is None:
            cnt = np.bincount(np.asarray(params.patient_group),
                              minlength=B.shape[0]).astype(float)
            pi = np.maximum(cnt, 1e-9) / max(cnt.sum(), 1.0)
        pi = np.asarray(pi, dtype=float)
        pi = pi / pi.sum()
    return Release(lam=np.asarray(params.lam, dtype=float),
                   concept_group=np.asarray(concept_group),
                   concept_ids=np.asarray(concept_ids),
                   B=B, pi=pi, propensity_shape=float(propensity_shape),
                   density=float(density), n_patients_fitted=int(n_patients),
                   n_events_fitted=int(n_events), meta=meta or {})


def sample_release(release: Release, n_patients: int, years: float = 5.0,
                   density: float | None = None, rng=None,
                   person_time: np.ndarray | None = None) -> SparseCDM:
    """Draw a synthetic cohort at a chosen density.

    Events are allocated multinomially to a *fixed total* M = density * sum(T),
    which is what the fixed-density model means. Increasing `density` makes a
    denser cohort at the same number of patients, without refitting anything.
    """
    rng = rng or np.random.default_rng(0)
    c = release.n_concepts
    if person_time is not None:
        T = np.asarray(person_time, dtype=float).copy()
        n_patients = T.size
    else:
        T = np.full(n_patients, float(years))
    target = (release.density if density is None else float(density)) * T.sum()

    k = max(release.propensity_shape, 1e-6)
    u = rng.gamma(k, 1.0 / k, size=n_patients)
    u /= u.mean()

    H = int(release.concept_group.max()) + 1
    Lam_h = np.bincount(release.concept_group, weights=release.lam, minlength=H)
    if release.B is None:
        Lam = np.full(n_patients, Lam_h.sum())
        g = np.zeros(n_patients, dtype=int)
    else:
        g = rng.choice(release.B.shape[0], size=n_patients, p=release.pi)
        Lam = (release.B @ Lam_h)[g]

    w = T * u * Lam
    M = int(round(target))
    counts = rng.multinomial(M, w / w.sum())

    rows, cols, vals = [], [], []
    for i in np.nonzero(counts > 0)[0]:
        q = release.lam if release.B is None else release.lam * release.B[g[i], release.concept_group]
        draw = rng.multinomial(int(counts[i]), q / q.sum())
        nz = np.nonzero(draw)[0]
        if nz.size:
            rows.append(np.full(nz.size, i, dtype=np.int64))
            cols.append(nz.astype(np.int64))
            vals.append(draw[nz].astype(float))

    X = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_patients, c))
    return SparseCDM(X=X, T=T, concept_group=release.concept_group,
                     concept_ids=release.concept_ids,
                     person_ids=np.arange(1, n_patients + 1))
