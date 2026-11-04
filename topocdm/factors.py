"""
topocdm.factors -- a low-rank interaction term on top of the block model.

The block term B[g, h] is nearly rank one in practice: whatever the number of
patient groups or concept groups, the fitted B is close to uniform, so the mean
matrix has an effective rank of a handful. Measured on the Synthea slice, the
real incidence matrix has 81 singular values above 0.1; the block model
produces 2-6. That is why the whole-CDM spectrum does not match, and no amount
of re-partitioning fixes it, because the rank is a property of B and not of how
finely it is cut.

This adds a rank-K bilinear term on the log scale:

    mu_ij = mu0_ij * exp( sum_k a_ik b_jk )

where mu0 is the block model's mean and (a, b) are patient and concept
loadings. The rank of the *interaction* is now K, and K is a free parameter
rather than a consequence of the partition.

Fitting is alternating Poisson regression with a diagonal Newton step. For
patient i, given b, the log-likelihood is

    sum_j x_ij (a_i . b_j) - sum_j mu_ij

so the gradient is (x_i - mu_i) . b and the diagonal Hessian is
sum_j mu_ij b_jk^2. That gives a multiplicative-free update that needs only the
N x C matrix of means, which is a few tens of MB at this scale.

Identifiability: a and b are only determined up to a rotation. After each pass
b is replaced by its QR factor and a is rotated to match, so a b^T is unchanged
and b has orthonormal columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FactorFit:
    a: np.ndarray          # (N, K) patient loadings
    b: np.ndarray          # (C, K) concept loadings (orthonormal columns)
    history: list


def _mu(mu0, a, b, clip: float = 8.0):
    """Mean matrix with the interaction exponent clipped.

    Without the clip, an over-long Newton step overflows `exp` and the fit is
    lost; the interaction is a correction, so a bounded range is enough.
    """
    return mu0 * np.exp(np.clip(a @ b.T, -clip, clip))


def fit_factors(mu0: np.ndarray, X: np.ndarray, K: int = 20, n_iter: int = 12,
                seed: int = 0, ridge: float = 1e-6, step: float = 0.25,
                verbose: bool = False) -> FactorFit:
    """Alternating diagonal-Newton fit of the rank-K log-scale interaction."""
    rng = np.random.default_rng(seed)
    N, C = mu0.shape
    # start at the block model and take damped steps: a full Newton step on two
    # factors that are only identified up to rotation diverges
    a = np.zeros((N, K))
    b = rng.normal(0, 0.01, size=(C, K))
    history = []

    for it in range(n_iter):
        # ---- Newton step for a given b
        mu = _mu(mu0, a, b)
        a = a + step * ((X - mu) @ b) / (mu @ (b ** 2) + ridge)

        # ---- Newton step for b given a
        mu = _mu(mu0, a, b)
        b = b + step * ((X - mu).T @ a) / (mu.T @ (a ** 2) + ridge)

        # a and b are identified only up to rotation; make b orthonormal and
        # rotate a to match, leaving a b^T unchanged
        Q, R = np.linalg.qr(b)
        b = Q
        a = a @ R.T

        mu = _mu(mu0, a, b)
        ll = float(np.sum(X * np.log(np.maximum(mu, 1e-12)) - mu))
        if history and ll < history[-1]:
            break                      # damped step overshot; keep the best
        history.append(ll)
        if verbose:
            print(f"  iter {it:2d}  ll {ll:,.1f}")
    return FactorFit(a=a, b=b, history=history)
