"""
topocdm.fit -- maximum likelihood for the fixed-density topology model.

Fitting is iterative proportional fitting. For the degree-corrected model
(no blocks) the M-step is exactly IPF: alternate matching row sums and column
sums. With blocks, add a block update and, if the patient groups are latent, a
classification-EM step between the assignment and the fitting.

All updates are closed form, so there is no numerical optimiser in the loop.
That matters for two reasons: it is fast enough to run over a real CDM, and
the convergence diagnostics are honest -- you can watch the log-likelihood
increase monotonically, which you cannot do with a stochastic optimiser.

Derivation of the M-step, for the record. With

    mu_ij = T_i u_i lambda_j B[g_i, h_j]

and writing A_g = sum_{i in g} T_i u_i and Lam_h = sum_{j in h} lambda_j:

    u_i         = rowsum_i  / ( T_i * (B @ Lam)[g_i] )
    lambda_j    = colsum_j  / ( A @ B )[h_j]
    B[g, h]     = num[g, h] / ( A_g * Lam_h )

where num[g,h] is the observed count in the (patient group, concept group)
cell. Each line is the exact maximiser given the others, so the loop is a
coordinate ascent and the log-likelihood cannot decrease.

Identifiability constraints, applied explicitly:

    mean(u) = 1                          fixes the u / lambda scale
    sum_g w_g B[g, h] / sum_g w_g = 1    fixes the lambda / B scale within each
                                         concept group, w_g = sum_{i in g} T_i u_i

Both are enforced by renormalisation after each update, with the compensating
factor pushed into lambda so that the mean matrix is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topocdm.model import (CDMIncidence, Params, loglik_conditional,
                           loglik_mixture)

EPS = 1e-12


@dataclass
class FitResult:
    params: Params
    loglik: float
    n_iter: int
    converged: bool
    history: list
    n_restarts: int = 1
    resp: object = None
    pi: object = None


def _normalise(p: Params, data: CDMIncidence) -> Params:
    """Apply both identifying constraints, keeping the mean matrix fixed."""
    m = p.u.mean()
    if m > 0:
        p.u = p.u / m
        p.lam = p.lam * m
    if p.B is not None:
        w = np.bincount(p.patient_group, weights=data.T * p.u,
                        minlength=p.n_groups).astype(float)
        tot = w.sum()
        if tot > 0:
            for h in range(p.B.shape[1]):
                c = float(np.sum(w * p.B[:, h]) / tot)
                if c > 0:
                    p.B[:, h] /= c
                    p.lam[p.concept_group == h] *= c
    return p


def fit_ipf(data: CDMIncidence, n_patient_groups: int = 1,
            tol: float = 1e-10, max_iter: int = 500,
            init: Params | None = None, verbose: bool = False,
            soft: bool = True) -> FitResult:
    """Fit by EM (soft responsibilities) or classification EM (hard).

    Soft EM maximises the mixture likelihood, which is the likelihood the model
    actually implies. Classification EM maximises a different objective and
    will happily add groups to fit noise, so it is available only for
    comparison; `soft=True` is the default.
    """
    n, c = data.n, data.c
    H = int(data.concept_group.max()) + 1
    G = max(1, int(n_patient_groups))

    X = data.X.astype(float)
    rowsum = X.sum(axis=1)
    colsum = X.sum(axis=0)
    total = max(X.sum(), EPS)

    col_H = np.zeros((c, H))
    col_H[np.arange(c), data.concept_group] = 1.0
    n_ih = X @ col_H

    if init is None:
        u = np.maximum(rowsum / max(data.T.sum(), EPS), EPS)
        u = u / max(u.mean(), EPS)
        lam = np.maximum(colsum / total, EPS)
        pg = np.zeros(n, dtype=int)
        B = np.ones((G, H)) if G > 1 else None
        p = Params(u=u, lam=lam, B=B, patient_group=pg,
                   concept_group=data.concept_group.copy(), n_groups=G)
        p = _normalise(p, data)
    else:
        p = init.copy()

    # responsibilities; one-hot for the single-group case
    if G == 1:
        R = np.ones((n, 1))
    else:
        R = np.zeros((n, G))
        R[np.arange(n), p.patient_group] = 1.0

    pi = R.mean(axis=0)

    history = []
    best_ll, best_p, best_R = -np.inf, p.copy(), R.copy()
    prev = -np.inf
    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        # ---------------- M-step ----------------
        if p.B is None:
            p.u = np.maximum(rowsum / np.maximum(data.T * p.lam.sum(), EPS), EPS)
            p.lam = np.maximum(colsum / max(float(np.sum(data.T * p.u)), EPS), EPS)
        else:
            Lam_h = col_H.T @ p.lam                        # (H,)
            pi = R.mean(axis=0)

            # u first, then A_g from the UPDATED u, so that lambda and B use
            # consistent quantities. Using a stale A_g breaks the EM
            # monotonicity guarantee.
            blam = p.B @ Lam_h                             # (G,)
            p.u = np.maximum(rowsum / np.maximum(data.T * (R @ blam), EPS), EPS)
            p = _normalise(p, data)
            Lam_h = col_H.T @ p.lam
            A_g = R.T @ (data.T * p.u)                     # (G,)

            denom_l = (A_g @ p.B)[data.concept_group]
            p.lam = np.maximum(colsum / np.maximum(denom_l, EPS), EPS)

            Lam_h = col_H.T @ p.lam
            num = R.T @ n_ih                               # (G, H)
            den = np.outer(A_g, Lam_h)
            p.B = np.maximum((num + 0.5) / np.maximum(den + 0.5, EPS), EPS)

        p = _normalise(p, data)

        # ---------------- E-step ----------------
        if p.B is not None and G > 1:
            Lam_h = col_H.T @ p.lam
            logB = np.log(p.B)
            score = n_ih @ logB.T - (data.T * p.u)[:, None] * (p.B @ Lam_h)[None, :]
            score = score + np.log(np.maximum(pi, 1e-300))[None, :]
            if soft:
                m = score.max(axis=1, keepdims=True)
                R = np.exp(score - m)
                R /= R.sum(axis=1, keepdims=True)
            else:
                R = np.zeros_like(score)
                R[np.arange(n), np.argmax(score, axis=1)] = 1.0
            p.patient_group = np.argmax(R, axis=1)

        ll = loglik_mixture(p, data, pis=pi)
        history.append(ll)
        if ll > best_ll:
            best_ll, best_p, best_R = ll, p.copy(), R.copy()
        if verbose and it % 25 == 0:
            print(f"  iter {it:4d}  mix-loglik {ll:.4f}")
        if abs(ll - prev) < tol and it > 2:
            converged = True
            break
        prev = ll

    best_p.patient_group = np.argmax(best_R, axis=1)
    return FitResult(params=best_p, loglik=best_ll, n_iter=it,
                     converged=converged, history=history, resp=best_R,
                     pi=best_R.mean(axis=0) if G > 1 else np.ones(1))


def _init_groups(data: CDMIncidence, G: int, rng) -> np.ndarray:
    """Initialise latent patient groups from their concept-group profile.

    A random init collapses under classification EM: the first M-step puts all
    patients in one group, the empty groups get B -> 0, and the E-step can
    never move anyone back. Clustering the normalised profile avoids that, and
    is close to how the blocks are actually defined.
    """
    H = int(data.concept_group.max()) + 1
    col_H = np.zeros((data.c, H))
    col_H[np.arange(data.c), data.concept_group] = 1.0
    prof = data.X.astype(float) @ col_H
    tot = prof.sum(axis=1, keepdims=True)
    prof = np.divide(prof, np.maximum(tot, EPS))
    prof = np.hstack([prof, (tot > 0).astype(float)])   # distinguish all-zero rows

    if G == 1:
        return np.zeros(data.n, dtype=int)
    try:
        from scipy.cluster.vq import kmeans2
        _, lab = kmeans2(prof, G, minit="++", seed=int(rng.integers(1, 1 << 30)))
        lab = lab.astype(int)
    except Exception:
        lab = rng.integers(0, G, size=data.n)
    # make sure every group is non-empty
    for g in range(G):
        if not np.any(lab == g):
            lab[rng.integers(0, data.n)] = g
    return lab


def fit(data: CDMIncidence, n_patient_groups: int = 1, n_restarts: int = 1,
        seed: int = 0, **kwargs) -> FitResult:
    """Fit, with k-means initialisation and random restarts for the block model."""
    G = max(1, int(n_patient_groups))
    if G == 1:
        return fit_ipf(data, 1, **kwargs)

    H = int(data.concept_group.max()) + 1
    n = data.n
    best = None
    for r in range(max(1, n_restarts)):
        rng = np.random.default_rng(seed + r)
        u0 = np.maximum(data.X.sum(axis=1) / max(data.T.sum(), EPS), EPS)
        u0 = u0 / u0.mean()
        lam0 = np.maximum(data.X.sum(axis=0) / max(data.total, EPS), EPS)
        lab = _init_groups(data, G, rng) if r == 0 else \
            rng.integers(0, G, size=n)
        init = Params(u=u0, lam=lam0, B=np.ones((G, H)),
                      patient_group=lab,
                      concept_group=data.concept_group.copy(), n_groups=G)
        res = fit_ipf(data, G, init=init, **kwargs)
        res.n_restarts = r + 1
        if best is None or res.loglik > best.loglik:
            best = res
    return best


# --------------------------------------------------------------------------
# Propensity distribution, for generating new patients rather than replaying
# the fitted ones.
# --------------------------------------------------------------------------

@dataclass
class PropensityDist:
    kind: str
    params: dict

    def sample(self, n: int, rng) -> np.ndarray:
        if self.kind == "lognormal":
            u = rng.lognormal(self.params["mu"], self.params["sigma"], size=n)
        elif self.kind == "gamma":
            u = rng.gamma(self.params["shape"], self.params["scale"], size=n)
        else:
            raise ValueError(self.kind)
        return u / u.mean()


def fit_propensity_nb(data: CDMIncidence, p: Params) -> PropensityDist:
    """Fit the propensity distribution by marginal likelihood.

    Plugging the fitted u_i into a log-normal over-states the spread, because
    u_i is itself estimated from a handful of events: measurement error adds
    variance. Integrating the propensity out instead gives a negative binomial
    for each patient's total event count,

        X_i. ~ NegBin(mean = T_i * Lambda_i, size = k)

    and k is estimated by maximising that likelihood over patients. This is the
    correct marginal, and it is what makes fresh-patient generation reproduce
    the observed concentration rather than exaggerating it.
    """
    from scipy import optimize
    from scipy.special import gammaln

    lam_h = np.bincount(data.concept_group, weights=p.lam,
                        minlength=int(data.concept_group.max()) + 1)
    if p.B is None:
        Lam = np.full(data.n, lam_h.sum())
    else:
        Lam = (p.B @ lam_h)[p.patient_group]
    mu = data.T * Lam
    x = data.X.sum(axis=1).astype(float)

    def nll(theta):
        k = np.exp(theta[0])
        ll = (gammaln(x + k) - gammaln(k) - gammaln(x + 1)
              + k * (np.log(k) - np.log(k + mu))
              + x * (np.log(np.maximum(mu, 1e-12)) - np.log(k + mu)))
        return -float(np.sum(ll))

    r = optimize.minimize_scalar(lambda t: nll(np.array([t])),
                                 bounds=(-6.0, 6.0), method="bounded")
    k = float(np.exp(r.x))
    return PropensityDist("gamma", dict(shape=k, scale=1.0 / k))


def fit_propensity(u: np.ndarray, floor: float = 1e-3) -> PropensityDist:
    """Fit a log-normal to the fitted propensities.

    A patient with zero events has an MLE of u_i = 0, so a floor is applied
    before fitting on the log scale. The floor is a modelling choice and is
    reported alongside the fit.
    """
    x = np.clip(u, floor, None)
    logx = np.log(x)
    return PropensityDist("lognormal", dict(mu=float(logx.mean()),
                                            sigma=float(logx.std(ddof=1))))
