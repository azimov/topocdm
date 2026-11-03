"""
topocdm.simulate -- ground-truth CDMs with known topology, for validation.

The point of a simulator here is not realism; it is that you know the answer.
Every quantity the fitting procedure is supposed to recover is available, so
"the model fits" can be checked against "the model recovered the truth", which
are different claims.

Structure follows the model exactly, so parameter recovery is meaningful:

    u_i      ~ LogNormal(0, sigma_u), normalised to mean 1
    lambda_j ~ LogNormal(0, sigma_lam), scaled to a target events-per-year
    B[g,h]   = 1 + amplification on the diagonal, plus lognormal noise
    g_i      ~ Categorical(equal)
    X_ij     ~ Poisson(T_i u_i lambda_j B[g_i, h_j])

The block matrix is the part that carries co-occurrence structure beyond the
marginals. Setting `block_amplification = 0` gives a null world in which the
only topology is patient-level propensity -- which is exactly the null that
the epidemiological argument needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topocdm.model import CDMIncidence, Params


@dataclass
class SimConfig:
    n: int = 2000
    c: int = 60
    n_concept_groups: int = 6
    n_patient_groups: int = 4
    years: float = 5.0
    sigma_u: float = 0.9
    sigma_lam: float = 1.0
    events_per_person_year: float = 8.0
    block_amplification: float = 3.0
    block_noise: float = 0.25
    seed: int = 0


def simulate(cfg: SimConfig | None = None, seed: int | None = None):
    """Return (data, true_params)."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    n, c = cfg.n, cfg.c
    H = cfg.n_concept_groups
    G = cfg.n_patient_groups

    T = np.full(n, float(cfg.years))

    u = rng.lognormal(0.0, cfg.sigma_u, size=n)
    u = u / u.mean()

    lam = rng.lognormal(0.0, cfg.sigma_lam, size=c)
    concept_group = np.arange(c) % H
    # spread the concept groups so that each has several members
    concept_group = rng.permutation(concept_group)

    B = np.exp(rng.normal(0.0, cfg.block_noise, size=(G, H)))
    for g in range(min(G, H)):
        B[g, g] *= 1.0 + cfg.block_amplification

    pg = rng.integers(0, G, size=n)

    # calibrate lambda to the requested overall event rate
    mu = T[:, None] * u[:, None] * lam[None, :] * B[pg][:, concept_group]
    lam *= (cfg.events_per_person_year * n) / max(mu.sum(), 1e-9)
    mu = T[:, None] * u[:, None] * lam[None, :] * B[pg][:, concept_group]

    X = rng.poisson(mu)

    true = Params(u=u, lam=lam, B=B, patient_group=pg,
                  concept_group=concept_group.copy(), n_groups=G)
    data = CDMIncidence(X=X, T=T, concept_group=concept_group.copy(),
                        patient_group=pg,
                        concept_names=[f"concept_{j:03d}" for j in range(c)],
                        meta=dict(sim=cfg.__dict__, true_events=float(mu.sum())))
    return data, true


def simulate_null(cfg: SimConfig | None = None, seed: int | None = None):
    """Same marginals, no block structure: an 'ingrown toenail world'.

    Every patient still has a propensity and every concept still has an
    intensity, so the degree distribution and the prevalence profile are
    unchanged. Only the co-occurrence structure is removed. This is the null
    against which 'is there real signal in the topology' is tested.
    """
    cfg = cfg or SimConfig()
    cfg = SimConfig(**{**cfg.__dict__, "block_amplification": 0.0,
                       "block_noise": 1e-9})
    return simulate(cfg, seed)
