"""
topocdm.distributed -- the fit as a map/reduce over patient partitions.

The streaming fit is already a loop over chunks. To scale it out, the loop only
has to be split into three pieces:

    map      each partition computes its own sufficient statistics
    reduce   sum them (small: O(C + G*H), kilobytes)
    update   one global M-step, identical on every worker

That works because the model's global parameters depend on the data only
through sums over patients -- events per concept, counts per (patient group,
concept group), propensity-weighted group mass, and the mixing counts. Nothing
being reduced is indexed by a patient, so the shuffle is kilobytes regardless
of whether there are 10^4 or 10^9 patients.

On Spark this is

    rdd.mapPartitions(partial_stats).treeAggregate(zero, merge, merge)

with the M-step on the driver between epochs. On Databricks the same functions
plug into a pandas-UDF or a plain `mapPartitions` closure; `partition_stats`
and `merge_stats` are the only things a worker needs to know about.

This module is dependency-free so it can be shipped to workers as-is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.special import logsumexp

from topocdm.model import Params

EPS = 1e-12


@dataclass
class Stats:
    colsum: np.ndarray        # (C,)   events per concept
    num_gh: np.ndarray        # (G,H)  events per patient group x concept group
    A_g: np.ndarray           # (G,)   propensity-weighted group mass
    pi_num: np.ndarray        # (G,)   responsibility mass
    ll: float = 0.0
    n: int = 0
    nnz: int = 0
    sum_T: float = 0.0
    sum_events: float = 0.0
    u_sum: float = 0.0

    @staticmethod
    def zeros(C, G, H) -> "Stats":
        return Stats(colsum=np.zeros(C), num_gh=np.zeros((G, H)),
                     A_g=np.zeros(G), pi_num=np.zeros(G))


def merge(a: Stats, b: Stats) -> Stats:
    return Stats(colsum=a.colsum + b.colsum,
                 num_gh=a.num_gh + b.num_gh,
                 A_g=a.A_g + b.A_g,
                 pi_num=a.pi_num + b.pi_num,
                 ll=a.ll + b.ll, n=a.n + b.n, nnz=a.nnz + b.nnz,
                 sum_T=a.sum_T + b.sum_T,
                 sum_events=a.sum_events + b.sum_events,
                 u_sum=a.u_sum + b.u_sum)


def _concept_group_matrix(concept_group, H):
    c = len(concept_group)
    return sparse.csr_matrix((np.ones(c), (np.arange(c), concept_group)),
                             shape=(c, H))


def partition_stats(X, T, lam, B, pi, concept_group, n_groups: int,
                    local_iters: int = 3, want_ll: bool = True) -> Stats:
    """Sufficient statistics for one partition. This is the map step."""
    X = sparse.csr_matrix(X)
    T = np.asarray(T, dtype=float)
    c = X.shape[1]
    G = int(n_groups)
    H = int(concept_group.max()) + 1
    col_H = _concept_group_matrix(concept_group, H)

    n_ih = (X @ col_H).toarray()
    rowsum = np.asarray(X.sum(axis=1)).ravel()
    u = rowsum / np.maximum(T * lam.sum(), EPS)

    if B is None or G == 1:
        R = np.ones((X.shape[0], 1))
    else:
        Lam_h = np.bincount(concept_group, weights=lam, minlength=H)
        logB = np.log(B)
        blam = B @ Lam_h
        for _ in range(local_iters):
            score = n_ih @ logB.T - (T * u)[:, None] * blam[None, :]
            score = score + np.log(np.maximum(pi, 1e-300))[None, :]
            m = score.max(axis=1, keepdims=True)
            R = np.exp(score - m)
            R /= R.sum(axis=1, keepdims=True)
            u = rowsum / np.maximum(T * (R @ blam), EPS)

    st = Stats(
        colsum=np.asarray(X.sum(axis=0)).ravel().astype(float),
        num_gh=(R.T @ n_ih) if G > 1 else np.zeros((1, H)),
        A_g=(R.T @ (T * u)) if G > 1 else np.array([float(np.sum(T * u))]),
        pi_num=R.sum(axis=0) if G > 1 else np.array([float(X.shape[0])]),
        n=X.shape[0], nnz=X.nnz, sum_T=float(T.sum()),
        sum_events=float(rowsum.sum()), u_sum=float(u.sum()),
    )
    if want_ll:
        st.ll = chunk_ll(X, T, u, lam, B, pi, concept_group, G)
    return st


def chunk_ll(X, T, u, lam, B, pi, concept_group, G) -> float:
    """Poisson mixture log-likelihood for a partition, via the C kernel."""
    try:
        from topocdm import fast
        if fast.AVAILABLE:
            v = fast.chunk_loglik(X, T, u, lam, B, pi, concept_group, G)
            if v is not None:
                return float(v)
    except Exception:
        pass
    Xd = X.toarray()
    base = np.maximum(T[:, None] * u[:, None] * lam[None, :], 1e-300)
    from scipy.special import gammaln
    const = gammaln(Xd + 1)
    if B is None or G == 1:
        return float(np.sum(Xd * np.log(base) - base - const))
    per = np.empty((G, Xd.shape[0]))
    for g in range(G):
        mu = np.maximum(base * B[g, concept_group][None, :], 1e-300)
        per[g] = np.sum(Xd * np.log(mu) - mu - const, axis=1)
    per += np.log(np.maximum(pi, 1e-300))[:, None]
    return float(logsumexp(per, axis=0).sum())


def m_step(stats: Stats, lam, B, pi, concept_group, n_groups: int):
    """The global update, applied once per epoch on the driver."""
    G = int(n_groups)
    cg = np.asarray(concept_group)
    H = int(cg.max()) + 1
    pi = np.maximum(stats.pi_num / max(stats.pi_num.sum(), EPS), 1e-9)
    pi /= pi.sum()
    if B is None:
        lam = np.maximum(stats.colsum / max(stats.A_g.sum(), EPS), EPS)
        return lam, B, pi
    lam = np.maximum(stats.colsum / np.maximum((stats.A_g @ B)[cg], EPS), EPS)
    Lam_h = np.bincount(cg, weights=lam, minlength=H)
    B = np.maximum((stats.num_gh + 0.5) /
                   (np.outer(stats.A_g, Lam_h) + 0.5), EPS)
    tot = stats.A_g.sum()
    if tot > 0:
        for h in range(H):
            cc = float(np.sum(stats.A_g * B[:, h]) / tot)
            if cc > 0:
                B[:, h] /= cc
                lam[cg == h] *= cc
    return lam, B, pi


def fit_partitioned(partitions, C, concept_group, n_groups, n_epochs=8,
                    local_iters=3, init=None, verbose=False):
    """Reference driver: same maths as fit_streaming, expressed as map/reduce.

    `partitions` is a list of (X_csr, T) pairs. In production these come from
    Spark partitions; here they are a list, which is enough to prove that the
    reduction reproduces the sequential fit exactly.
    """
    G = int(n_groups)
    H = int(np.asarray(concept_group).max()) + 1
    cg = np.asarray(concept_group)
    if init is None:
        lam = np.full(C, 1.0 / C)
        B = np.ones((G, H)) if G > 1 else None
        pi = np.full(G, 1.0 / G)
    else:
        lam, B = init.lam.copy(), (None if init.B is None else init.B.copy())
        pi = np.full(G, 1.0 / G)

    history = []
    for epoch in range(1, n_epochs + 1):
        parts = [partition_stats(X, T, lam, B, pi, cg, G,
                                 local_iters=local_iters)
                 for X, T in partitions]
        st = Stats.zeros(C, G, H)
        for p in parts:
            st = merge(st, p)
        lam, B, pi = m_step(st, lam, B, pi, cg, G)
        history.append(st.ll)
        if verbose:
            print(f"  epoch {epoch:3d}  ll {st.ll:,.1f}  n {st.n:,}")
        if len(history) > 2 and abs(history[-1] - history[-2]) < \
                1e-7 * max(1.0, abs(history[-2])):
            break
    return Params(u=np.ones(1), lam=lam, B=B, patient_group=np.zeros(1, int),
                  concept_group=cg.copy(), n_groups=G), history
