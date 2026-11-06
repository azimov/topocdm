"""
topocdm.model -- the fixed-density topology model and its likelihood.

The question this module answers: can a fixed-density generative graph model
have a tractable likelihood?

It can, and the resolution is simple. "Fixed density" means the total number of
events is fixed. Conditioning a Poisson model on its total gives a multinomial,
and the multinomial likelihood over the cells of a patient x concept incidence
matrix is closed-form. The fixed-density constraint is therefore not an
obstacle to inference; it is a conditioning event.

    X_ij  ~  Poisson( mu_ij )                 mu_ij = T_i u_i lambda_j B[g_i, h_j]
    X | M ~  Multinomial( M, mu / sum(mu) )   M = sum_ij X_ij

Two models that differ only in their scale parameter have the same conditional
MLE, so the Poisson MLE *is* the fixed-density MLE. Fit Poisson; generate by
conditioning on M.

The other half of the resolution is that a CDM is a multigraph. A patient can
be coded with the same concept many times, so there is no sampling without
replacement. That is exactly what makes the Poisson route available, and it is
why CiGRAM's intractable weighted sampling can be dropped without losing the
fixed-density structure: the edge budget is still fixed, but edges are
exchangeable within cells.

Parameters, and what each one means epidemiologically:

    u_i      patient care-seeking propensity (how many codes they accumulate)
    lambda_j concept intensity (how often it occurs per unit of contact)
    B[g,h]   affinity between patient group g and concept group h, above and
             beyond u_i and lambda_j -- this is the multimorbidity structure,
             i.e. the part of the topology that marginals do not contain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaln


@dataclass
class Params:
    """Fitted parameters of the topology model."""

    u: np.ndarray  # (N,) patient propensity, mean 1
    lam: np.ndarray  # (C,) concept intensity
    B: np.ndarray | None = None  # (G, H) block affinity, or None
    patient_group: np.ndarray | None = None  # (N,)
    concept_group: np.ndarray | None = None  # (C,)
    n_groups: int = 1

    def copy(self) -> Params:
        return Params(
            u=self.u.copy(),
            lam=self.lam.copy(),
            B=None if self.B is None else self.B.copy(),
            patient_group=None if self.patient_group is None else self.patient_group.copy(),
            concept_group=None if self.concept_group is None else self.concept_group.copy(),
            n_groups=self.n_groups,
        )


@dataclass
class CDMIncidence:
    """A CDM reduced to an incidence matrix.

    X         (N, C) integer counts of concept j for patient i
    T         (N,)   person-time at risk, in years
    concept_group (C,) integer group index per concept
    patient_group (N,) optional observed stratum (age band, care site, ...)
    """

    X: np.ndarray
    T: np.ndarray
    concept_group: np.ndarray
    patient_group: np.ndarray | None = None
    concept_names: list | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def c(self) -> int:
        return self.X.shape[1]

    @property
    def total(self) -> int:
        return int(self.X.sum())

    def subset_patients(self, idx) -> CDMIncidence:
        return CDMIncidence(
            X=self.X[idx],
            T=self.T[idx],
            concept_group=self.concept_group,
            patient_group=None if self.patient_group is None else self.patient_group[idx],
            concept_names=self.concept_names,
            meta=dict(self.meta),
        )


# --------------------------------------------------------------------------
# Mean structure
# --------------------------------------------------------------------------


def mean_matrix(p: Params, data: CDMIncidence) -> np.ndarray:
    """Expected counts, mu_ij = T_i u_i lambda_j B[g_i, h_j]."""
    base = data.T[:, None] * p.u[:, None] * p.lam[None, :]
    if p.B is None:
        return base
    if p.patient_group is None or p.concept_group is None:
        raise ValueError("group labels are required when B is set")
    return base * p.B[np.ix_(p.patient_group, p.concept_group)]


def concept_multipliers(p: Params) -> np.ndarray:
    """Per-concept multiplier for a patient in group g: lam_j * B[g, h_j]."""
    if p.B is None:
        return p.lam[None, :]
    return p.lam[None, :] * p.B[:, p.concept_group]


# --------------------------------------------------------------------------
# Likelihoods
# --------------------------------------------------------------------------


def loglik_poisson(p: Params, data: CDMIncidence) -> float:
    """Unconditional Poisson log-likelihood."""
    mu = mean_matrix(p, data)
    return float(np.sum(data.X * np.log(np.maximum(mu, 1e-300)) - mu - gammaln(data.X + 1)))


def loglik_conditional(p: Params, data: CDMIncidence) -> float:
    """Log-likelihood *given* the total count M -- the fixed-density likelihood.

    This is the multinomial log-likelihood of the cell allocation. It is what
    should be compared between models and reported as the fit of the topology,
    because it does not reward a model simply for getting the overall event
    rate right.
    """
    mu = mean_matrix(p, data)
    tot = mu.sum()
    if tot <= 0:
        return -np.inf
    logp = np.log(np.maximum(mu, 1e-300)) - np.log(tot)
    return float(np.sum(data.X * logp))


def deviance_conditional(p: Params, data: CDMIncidence) -> float:
    """Conditional deviance against the saturated model: -2 log LR.

    Saturated log-likelihood for a multinomial with M draws over observed
    cells is sum X log(X/M).
    """
    M = data.total
    nz = data.X > 0
    ll_sat = float(np.sum(data.X[nz] * np.log(data.X[nz] / M)))
    return float(-2.0 * (loglik_conditional(p, data) - ll_sat))


def n_params(p: Params, data: CDMIncidence) -> int:
    """Free parameters, net of the identifying constraints.

    u is constrained to mean 1 (-1); lambda and B have a shared scale within
    each concept group, removed by the B-normalisation (-H).
    """
    n = data.n + data.c
    if p.B is not None:
        n += p.B.size - p.B.shape[1]
    return n - 1


def aic(p: Params, data: CDMIncidence) -> float:
    return -2.0 * loglik_conditional(p, data) + 2.0 * n_params(p, data)


def bic(p: Params, data: CDMIncidence) -> float:
    return -2.0 * loglik_conditional(p, data) + np.log(data.total) * n_params(p, data)


# --------------------------------------------------------------------------
# Sampled quantities used by the generator and the null model
# --------------------------------------------------------------------------


def expected_prevalence(p: Params, data: CDMIncidence) -> np.ndarray:
    """P(a patient has at least one record of concept j), under the model."""
    mu = mean_matrix(p, data)
    return 1.0 - np.exp(-mu).mean(axis=0)


def expected_incidence(p: Params, data: CDMIncidence) -> np.ndarray:
    """Expected events per person-year for each concept."""
    mu = mean_matrix(p, data)
    return mu.sum(axis=0) / data.T.sum()


def loglik_mixture(p: Params, data: CDMIncidence, pis: np.ndarray | None = None) -> float:
    """Marginal (mixture) log-likelihood, integrating out the group label.

        L = sum_i log sum_g pi_g prod_j Poisson(x_ij | mu_ijg)

    This is the likelihood the model actually implies. The classification
    likelihood used for fitting (which conditions on a hard assignment)
    over-states the fit, because every patient's group is then an estimated
    parameter that never appears in the parameter count. Any AIC, BIC or
    likelihood ratio reported for this model must use the mixture form, or
    model selection will march off towards infinitely many groups.
    """
    H = int(data.concept_group.max()) + 1
    if p.B is None:
        G = 1
        B = np.ones((1, H))
        pis = np.ones(1)
    else:
        G = p.B.shape[0]
        B = p.B
        if pis is None:
            if p.patient_group is None:
                raise ValueError("patient_group is required when B is set")
            cnt = np.bincount(p.patient_group, minlength=G).astype(float)
            pis = np.maximum(cnt, 0.5) / max(cnt.sum(), 1e-9)

    base = data.T[:, None] * p.u[:, None] * p.lam[None, :]
    const = gammaln(data.X + 1)
    per_group = np.empty((G, data.n))
    for g in range(G):
        mu = np.maximum(base * B[g, data.concept_group][None, :], 1e-300)
        per_group[g] = np.sum(data.X * np.log(mu) - mu - const, axis=1)
    from scipy.special import logsumexp

    return float(logsumexp(per_group + np.log(pis)[:, None], axis=0).sum())


def n_params_mixture(p: Params, data: CDMIncidence) -> int:
    """Free parameters of the mixture model.

    u (N, minus one for the mean-1 constraint), lambda and B together
    (C + G*(H-1), after the within-concept-group normalisation), and G-1
    mixing weights. Crucially the N group assignments are *not* parameters:
    they are integrated out by `loglik_mixture`.
    """
    H = int(data.concept_group.max()) + 1
    n = data.n - 1 + data.c
    if p.B is not None:
        G = p.B.shape[0]
        n += G * (H - 1) + (G - 1)
    return n


def aic_mixture(p: Params, data: CDMIncidence) -> float:
    return -2.0 * loglik_mixture(p, data) + 2.0 * n_params_mixture(p, data)


def bic_mixture(p: Params, data: CDMIncidence) -> float:
    return -2.0 * loglik_mixture(p, data) + np.log(max(data.total, 2)) * n_params_mixture(p, data)
