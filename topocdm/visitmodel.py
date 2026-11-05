"""
topocdm.visitmodel -- a visit-anchored, time-dynamic bipartite model.

The previous model treated a patient as a bag of concepts with one free
propensity each. That has two problems the brief names: it is not anchored to
visits, and its parameter count is dominated by per-patient quantities.

This model takes the visit as the unit of observation, which is what the CDM
actually is. A patient is a *sequence* of visits; a visit is a set of concepts.

    patient i  --(time)-->  visit v  --(mixture)-->  topic z_v  --(draw)-->  concepts

Generative story:

  1. patient i draws a starting topic  z_1 ~ rho
  2. each later visit                z_v | z_{v-1} ~ T[z_{v-1}, .]     (time dynamic)
  3. a visit of topic k draws its concepts from a catalogue theta_k over C
  4. visit times are drawn from a gap distribution per topic

Because the concepts at a visit are drawn from theta_k and a patient's record is
a sum over their visits,

    E[X_ij]  =  sum_k  c_ik * theta_kj

where c_ik is the number of visits of topic k the patient had. That is a
rank-K factorisation in which the per-patient quantity c_ik is a *visit count*,
not a free parameter, and the model parameters are

    theta   K x C    concept catalogue per visit type
    T       K x K    topic-to-topic transition
    rho     K        starting distribution
    gaps    K        inter-visit time per topic

so the count is K*C + K^2 + 2K and does not involve the number of patients at
all. The rank of the mean matrix is at most K, and K is a knob.

Fitting uses Poisson non-negative matrix factorisation (KL divergence) on the
visit x concept matrix, which is the maximum-likelihood fit for this
generative story given topic assignments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class VisitTopicModel:
    theta: np.ndarray          # (K, C) concept catalogue per visit type
    trans: np.ndarray          # (K, K) topic transition
    rho: np.ndarray            # (K,) starting distribution
    visits_per_patient: float
    gap_mean: float
    visit_size: float = 1.0
    visit_count_pool: np.ndarray = None   # empirical visits-per-patient, for sampling
    n_visits_fitted: int = 0

    @property
    def n_parameters(self) -> int:
        K, C = self.theta.shape
        return int(K * C + K * K + 2 * K)

    def summary(self) -> dict:
        K = self.theta.shape[0]
        return dict(K=int(K), C=int(self.theta.shape[1]),
                    visit_size=float(self.visit_size),
                    n_parameters=self.n_parameters,
                    visits_per_patient=float(self.visits_per_patient),
                    self_transition=float(np.mean(np.diag(self.trans))))


def fit_topics(V: sparse.csr_matrix, K: int = 30, n_iter: int = 40,
               seed: int = 0, verbose: bool = False):
    """Fit the visit-topic mixture by EM (the aspect / PLSA model).

    Each visit is a bag of concepts. A visit of topic k draws its concepts
    independently from theta_k, so conditioning on the visit's size the visit is
    multinomial with probabilities theta_k. That makes the E-step a sparse
    matrix product and the M-step a weighted sum -- no dense V x C matrix at any
    point.

        E: score_vk = log pi_k + sum_j x_vj log theta_kj ;  r_v. = softmax(score)
        M: theta_kj = sum_v r_vk x_vj / sum_v r_vk ;  pi_k = mean_v r_vk

    Minimising KL divergence is the same as maximising this likelihood.
    """
    rng = np.random.default_rng(seed)
    n, c = V.shape
    # start from a light random catalogue; EM sharpens quickly
    H = rng.random((K, c)) + 0.5
    H /= H.sum(1, keepdims=True)
    pi = np.full(K, 1.0 / K)
    prev = -np.inf

    for it in range(n_iter):
        score = np.asarray(V @ np.log(H).T) + np.log(np.maximum(pi, 1e-12))[None, :]
        m = score.max(axis=1, keepdims=True)
        R = np.exp(score - m)
        Z = R.sum(axis=1, keepdims=True)
        R /= Z
        ll = float(np.sum(np.log(Z.ravel() + 1e-300) + m.ravel()))

        H = np.asarray((R.T @ V)) + 1e-9
        H /= H.sum(1, keepdims=True)
        pi = np.maximum(R.mean(axis=0), 1e-9); pi /= pi.sum()

        if verbose and (it + 1) % 10 == 0:
            print(f"  iter {it+1:3d}  loglik {ll:,.0f}")
        if ll - prev < 1e-6 * max(1.0, abs(prev)):
            break
        prev = ll
    return R, H, pi, ll


def fit_transitions(topic_of_visit: np.ndarray, person_of_visit: np.ndarray,
                    order: np.ndarray | None = None):
    """Topic-to-topic transition counts from consecutive visits of a patient."""
    K = int(topic_of_visit.max()) + 1
    if order is None:
        order = np.lexsort((np.arange(len(person_of_visit)), person_of_visit))
    t = topic_of_visit[order]
    p = person_of_visit[order]
    same = p[1:] == p[:-1]
    T = np.zeros((K, K))
    np.add.at(T, (t[:-1][same], t[1:][same]), 1.0)
    T += 0.5
    T = T / T.sum(1, keepdims=True)
    rho = np.bincount(t[np.r_[True, p[1:] != p[:-1]]], minlength=K).astype(float) + 0.5
    rho /= rho.sum()
    return T, rho


def simulate_topic_counts(n_patients: int, visit_counts, model: VisitTopicModel,
                          rng) -> np.ndarray:
    """Draw (n_patients, K) counts of visits per topic.

    `visit_counts` may be a scalar or a per-patient array. Heterogeneity in the
    number of visits is the main source of patient-level variation, so it has to
    be drawn per patient -- drawing one value for the cohort makes every
    synthetic patient look the same and collapses the rank.
    """
    K = model.theta.shape[0]
    n = np.asarray(visit_counts, dtype=int)
    if n.ndim == 0:
        n = np.full(n_patients, int(n))
    c = np.zeros((n_patients, K))
    if n.max() <= 0:
        return c
    z = rng.choice(K, size=n_patients, p=model.rho)
    for v in range(int(n.max())):
        active = v < n
        np.add.at(c, (np.nonzero(active)[0], z[active]), 1.0)
        u = rng.random(n_patients)[:, None]
        z = (np.cumsum(model.trans[z], axis=1) < u).sum(axis=1) % K
    return c


def simulate_visits(model: VisitTopicModel, n_patients: int, rng,
                    visits_per_patient: int | None = None) -> np.ndarray:
    """Generate a synthetic patient x concept count matrix."""
    if visits_per_patient is None and model.visit_count_pool is not None:
        pool = np.asarray(model.visit_count_pool)
        V = rng.choice(pool, size=n_patients, replace=True).astype(int)
    else:
        V = np.full(n_patients, int(visits_per_patient or round(model.visits_per_patient)))
    c = simulate_topic_counts(n_patients, V, model, rng)
    mu = c @ model.theta * model.visit_size
    return rng.poisson(mu)
