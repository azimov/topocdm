"""
topocdm.evaluate -- does the synthetic CDM behave like the real one?

The comparison that matters is not "do the marginals match". A model with only
concept intensities matches every marginal by construction, and that is the
standard against which synthetic data is usually validated. The comparison
that separates a topology model from a marginal model is the *joint* one:

    marginal agreement   prevalence per concept, incidence per concept
    topology agreement   codes-per-patient distribution, pairwise co-occurrence,
                         concentration (what share of events sit on what share
                         of patients)

Both are reported here, and both are reported against a held-out set of
patients that played no part in fitting.
"""

from __future__ import annotations

import numpy as np

from topocdm.model import CDMIncidence, Params


def prevalence(data: CDMIncidence) -> np.ndarray:
    """Fraction of patients with at least one record of each concept."""
    return (data.X > 0).mean(axis=0)


def incidence(data: CDMIncidence) -> np.ndarray:
    """Events per person-year for each concept."""
    return data.X.sum(axis=0) / data.T.sum()


def degree(data: CDMIncidence) -> np.ndarray:
    """Distinct concepts recorded per patient."""
    return (data.X > 0).sum(axis=1)


def degree_hist(data: CDMIncidence, max_deg: int | None = None) -> np.ndarray:
    d = degree(data)
    m = int(d.max()) + 1 if max_deg is None else max_deg + 1
    return np.bincount(d, minlength=m).astype(float) / len(d)


def cooccurrence(data: CDMIncidence, concepts: np.ndarray | None = None) -> np.ndarray:
    """P(a patient has both i and j), upper triangle. Heavy tail makes this
    the honest test of joint structure."""
    A = (data.X > 0).astype(float)
    if concepts is not None:
        A = A[:, concepts]
    return A.T @ A / A.shape[0]


def gini(x: np.ndarray) -> float:
    """Concentration of a non-negative vector."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) / (n * x.sum())) - (n + 1) / n)


def _calib(y: np.ndarray, yhat: np.ndarray) -> dict:
    """Calibration of synthetic values against real ones."""
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    if len(y) < 3:
        return dict(r=np.nan, slope=np.nan, intercept=np.nan, med_ratio=np.nan)
    slope, intercept = np.polyfit(yhat, y, 1)
    ratio = y / np.maximum(yhat, 1e-12)
    return dict(r=float(np.corrcoef(y, yhat)[0, 1]),
                slope=float(slope), intercept=float(intercept),
                med_ratio=float(np.median(ratio)))


def compare(real: CDMIncidence, synth: CDMIncidence) -> dict:
    """Full comparison table between a real and a synthetic CDM."""
    pv, pvs = prevalence(real), prevalence(synth)
    inc, incs = incidence(real), incidence(synth)

    # degree distributions on a common support
    m = max(int(degree(real).max()), int(degree(synth).max()))
    dh, dhs = degree_hist(real, m), degree_hist(synth, m)

    # co-occurrence is only informative on concepts with intermediate
    # prevalence: pairs that are near-universal are ~1 by construction and
    # swamp the correlation.
    mid = np.where((pv > 0.05) & (pv < 0.6))[0]
    if mid.size < 5:
        mid = np.argsort(-pv)[: min(25, len(pv))]
    top = mid[: min(25, mid.size)]

    return dict(
        prevalence=_calib(pv, pvs),
        incidence=_calib(inc, incs),
        degree_l1=float(np.abs(dh - dhs).sum()),
        degree_ks=float(np.max(np.abs(np.cumsum(dh) - np.cumsum(dhs)))),
        degree_mean=(float(degree(real).mean()), float(degree(synth).mean())),
        degree_p99=(float(np.percentile(degree(real), 99)),
                    float(np.percentile(degree(synth), 99))),
        cooccurrence_r=float(np.corrcoef(
            cooccurrence(real, top)[np.triu_indices(len(top), 1)],
            cooccurrence(synth, top)[np.triu_indices(len(top), 1)])[0, 1]),
        gini_events=(gini(real.X.sum(axis=1)), gini(synth.X.sum(axis=1))),
        total_events=(real.total, synth.total),
        n=(real.n, synth.n),
    )


def predictive_loglik(p: Params, data: CDMIncidence) -> float:
    """Mean conditional log-likelihood per event on a held-out CDM."""
    from topocdm.model import loglik_conditional
    return loglik_conditional(p, data) / max(data.total, 1)


def predictive_loglik_nb(data: CDMIncidence, p: Params, k: float,
                         marginalise_groups: bool = True) -> float:
    """Held-out predictive log-likelihood per event, integrating out propensity.

    A held-out patient has no fitted u_i, so the propensity is integrated over
    its fitted Gamma distribution. That gives a negative binomial per cell,

        X_ij | g ~ NegBin(mean = T_i lambda_j B[g, h_j], size = k)

    and if the patient's group is unknown the likelihood is averaged over
    groups with their fitted frequencies. This is the honest held-out score:
    it is a prediction for a patient the model has never seen.
    """
    from scipy.special import logsumexp
    from scipy.stats import nbinom

    H = int(data.concept_group.max()) + 1
    if p.B is None:
        G, pis = 1, np.ones(1)
        B = np.ones((1, H))
    else:
        G = p.B.shape[0]
        cnt = np.bincount(p.patient_group, minlength=G).astype(float)
        pis = cnt / cnt.sum()
        B = p.B

    lam_h = np.bincount(data.concept_group, weights=p.lam, minlength=H)
    parts = []
    for g in range(G):
        mu = np.maximum(data.T[:, None] *
                        (p.lam * B[g, data.concept_group])[None, :], 1e-12)
        prob = k / (k + mu)
        lp = nbinom.logpmf(data.X, k, prob)
        if marginalise_groups and G > 1:
            lp = lp + np.log(max(pis[g], 1e-300))
        parts.append(lp)
    stacked = np.stack(parts, axis=0)                 # (G, N, C)
    total = logsumexp(stacked, axis=0)
    return float(total.sum() / max(data.total, 1))
