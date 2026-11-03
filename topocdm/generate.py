"""
topocdm.generate -- sample a synthetic CDM from a fitted topology model.

Generation respects the fixed-density constraint by construction: the number
of events is drawn as a fixed total M and allocated multinomially over cells.
So a synthetic database has exactly as many records as the target, and the
question of whether it has the right *arrangement* of them is a real question
rather than an artefact of getting the total wrong.

Two sampling modes:

  replay   keep the fitted patient propensities and groups, redraw only the
           counts. This is a posterior predictive check: it asks whether the
           fitted model reproduces the observed data.
  fresh    draw new patients from the fitted propensity distribution. This is
           the synthetic-CDM generator, and it is what a data-sharing use case
           would actually ship.
"""

from __future__ import annotations

import numpy as np

from topocdm.fit import PropensityDist
from topocdm.model import CDMIncidence, Params


def sample_cdm(params: Params, template: CDMIncidence, M: int | None = None,
               patient_dist: PropensityDist | None = None,
               rng: np.random.Generator | None = None,
               mode: str = "replay") -> CDMIncidence:
    """Draw a synthetic incidence matrix with a fixed total event count."""
    rng = rng or np.random.default_rng(0)
    n, c = template.n, template.c
    M = int(template.total if M is None else M)

    if mode == "replay":
        u = params.u.copy()
        g = params.patient_group.copy()
    elif mode == "fresh":
        if patient_dist is None:
            raise ValueError("fresh mode needs a PropensityDist")
        u = patient_dist.sample(n, rng)
        if params.B is not None:
            freq = np.bincount(params.patient_group,
                               minlength=params.n_groups).astype(float)
            g = rng.choice(params.n_groups, size=n, p=freq / freq.sum())
        else:
            g = np.zeros(n, dtype=int)
    else:
        raise ValueError(mode)

    T = template.T

    # per-patient expected totals, then allocate M events over patients
    if params.B is None:
        w = T * u
    else:
        Lam_h = np.bincount(template.concept_group, weights=params.lam,
                            minlength=params.B.shape[1])
        w = T * u * (params.B @ Lam_h)[g]

    p_i = w / w.sum()
    counts = rng.multinomial(M, p_i)

    X = np.zeros((n, c), dtype=np.int64)
    B = params.B
    for i in np.nonzero(counts > 0)[0]:
        q = params.lam if B is None else params.lam * B[g[i], template.concept_group]
        q = q / q.sum()
        X[i] = rng.multinomial(int(counts[i]), q)

    return CDMIncidence(X=X, T=T.copy(), concept_group=template.concept_group.copy(),
                        patient_group=g, concept_names=template.concept_names,
                        meta=dict(source="topocdm", mode=mode, M=M))


def sample_replicates(params: Params, template: CDMIncidence, n_rep: int = 20,
                      **kwargs) -> list:
    """Draw several synthetic CDMs; useful for Monte Carlo null distributions."""
    rng = kwargs.pop("rng", None) or np.random.default_rng(0)
    return [sample_cdm(params, template, rng=rng, **kwargs) for _ in range(n_rep)]
