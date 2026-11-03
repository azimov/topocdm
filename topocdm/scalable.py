"""
topocdm.scalable -- fitting on CDMs too large to hold in memory.

The model is separable across patients once the global parameters are fixed:

    mu_ij = T_i u_i lambda_j B[g_i, h_j]

u_i is local. lambda, B and the mixing weights pi are global, and they depend on
the data only through three sums over patients:

    colsum_j  = sum_i X_ij                  events per concept
    n_ih      = sum_{j in h} X_ij           events per patient per concept group
    A_g       = sum_i R_ig T_i u_i          propensity-weighted group mass

Everything else -- the E-step, the u update, the likelihood contribution -- is
per patient and needs only n_ih, a vector of length H, not the full row. So no
N x C matrix is ever materialised, and the working set is one chunk.

Subsampling patients is not a compromise here, it is the point. The released
model contains no per-patient parameters, so fitting on a sample costs nothing
that the release would have used, and a fresh random sample each epoch keeps
the estimate honest while never putting the whole database in one process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
from scipy import sparse
from scipy.special import gammaln, logsumexp

from topocdm.model import Params

try:                                    # optional compiled kernel
    from topocdm import fast as _FAST
    if not _FAST.AVAILABLE:
        _FAST = None
except Exception:                       # pragma: no cover
    _FAST = None

EPS = 1e-12


@dataclass
class SparseCDM:
    """A CDM as a sparse patient x concept count matrix."""

    X: sparse.csr_matrix
    T: np.ndarray
    concept_group: np.ndarray
    concept_ids: np.ndarray | None = None
    person_ids: np.ndarray | None = None

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def c(self) -> int:
        return self.X.shape[1]

    @property
    def total(self) -> int:
        return int(self.X.sum())

    @property
    def density(self) -> float:
        """Events per person-year."""
        return self.total / float(self.T.sum())

    def take(self, idx) -> "SparseCDM":
        return SparseCDM(
            X=self.X[idx], T=self.T[idx], concept_group=self.concept_group,
            concept_ids=self.concept_ids,
            person_ids=None if self.person_ids is None else self.person_ids[idx],
        )


def to_sparse(data) -> SparseCDM:
    """Convert a dense CDMIncidence (or pass a SparseCDM through)."""
    if isinstance(data, SparseCDM):
        return data
    X = data.X if sparse.issparse(data.X) else sparse.csr_matrix(data.X)
    return SparseCDM(X=sparse.csr_matrix(X), T=np.asarray(data.T, dtype=float),
                     concept_group=np.asarray(data.concept_group),
                     concept_ids=getattr(data, "concept_ids", None))


def _concept_group_matrix(concept_group: np.ndarray, H: int) -> sparse.csr_matrix:
    c = len(concept_group)
    return sparse.csr_matrix((np.ones(c), (np.arange(c), concept_group)),
                             shape=(c, H))


def _profiles(X, col_H):
    """Normalised concept-group profile per patient, for initialisation."""
    n_ih = (X @ col_H).toarray()
    tot = n_ih.sum(axis=1, keepdims=True)
    return n_ih / np.maximum(tot, 1.0)


def _init_from_sample(make_chunks, col_H, G, sample_cap=20000, seed=0):
    """k-means on patient profiles, then a pass to build the initial B.

    A uniform B gives every group an identical score, so responsibilities stay
    uniform and the block structure never appears. Symmetry has to be broken
    deliberately, and clustering the profiles is the cheapest way to do it.
    """
    from scipy.cluster.vq import kmeans2

    pts, got = [], 0
    for chunk in make_chunks(0):
        X = chunk[1]
        pts.append(_profiles(sparse.csr_matrix(X), col_H))
        got += pts[-1].shape[0]
        if got >= sample_cap:
            break
    if not pts or got < 2 * G:
        return None
    P = np.vstack(pts)[:sample_cap]
    try:
        cent, _ = kmeans2(P, G, minit="++", seed=int(seed))
    except Exception:
        return None
    return cent


def _assign_by_centroids(prof, cent):
    d = ((prof[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d, axis=1)


@dataclass
class StreamingFit:
    params: Params
    loglik: float
    history: list
    n_patients_seen: int
    n_events_seen: int
    epochs: int
    density: float


def _init_from_sample_fit(make_chunks, c, cg_idx, G, sample_cap, seed,
                          dense_limit=5e7):
    """Fit properly on a bounded sample, then use that as the streaming start.

    This is the robust way to initialise: the sample is small enough to run the
    exact EM, and the stream then only refines lambda and B on the rest of the
    data. It also keeps the initialisation and the refinement using the same
    code path, so there is no second implementation to get subtly wrong.
    """
    from topocdm.fit import fit
    from topocdm.model import CDMIncidence

    Xs, Ts, got = [], [], 0
    for chunk in make_chunks(0):
        X, T = chunk[1], chunk[2]
        X = sparse.csr_matrix(X)
        Xs.append(X)
        Ts.append(np.asarray(T, dtype=float))
        got += X.shape[0]
        if got >= sample_cap:
            break
    if not Xs or got < 10 * G:
        return None
    X = sparse.vstack(Xs)
    T = np.concatenate(Ts)
    if X.shape[0] * c > dense_limit:
        return None                      # sample too wide to densify; fall back
    data = CDMIncidence(X=X.toarray(), T=T, concept_group=cg_idx.copy())
    try:
        res = fit(data, n_patient_groups=G, n_restarts=1, seed=seed,
                  max_iter=60, tol=1e-6)
    except Exception:
        return None
    return res.params


def _init_B(make_chunks, col_H, G, H, lam, cg_idx, n, sample_cap, seed):
    """Initial B from a clustering pass; falls back to a random perturbation."""
    cent = _init_from_sample(make_chunks, col_H, G, sample_cap=sample_cap, seed=seed)
    if cent is None:
        rng = np.random.default_rng(seed)
        B = np.exp(rng.normal(0.0, 0.3, size=(G, H)))
    else:
        num = np.zeros((G, H))
        A_g = np.zeros(G)
        Lam_h = np.full(H, lam.sum() / H)
        for chunk in make_chunks(0):
            X, T = chunk[1], chunk[2]
            X = sparse.csr_matrix(X)
            if X.shape[0] == 0:
                continue
            T = np.asarray(T, dtype=float)
            n_ih = (X @ col_H).toarray()
            lab = _assign_by_centroids(_profiles(X, col_H), cent)
            rowsum = np.asarray(X.sum(axis=1)).ravel()
            u = rowsum / np.maximum(T * lam.sum(), EPS)
            for g in range(G):
                m = lab == g
                if m.any():
                    num[g] += n_ih[m].sum(axis=0)
                    A_g[g] += float(np.sum(T[m] * u[m]))
        B = np.maximum((num + 0.5) / (np.outer(np.maximum(A_g, EPS), Lam_h) + 0.5), EPS)
    for h in range(H):
        cc = float(B[:, h].mean())
        if cc > 0:
            B[:, h] /= cc
    return B


def _dense_chunk_loglik(X, T, u, lam, B, pi, cg_idx, G) -> float:
    """Exact mixture log-likelihood for the rows of X (dense evaluation)."""
    Xd = X.toarray()
    base = np.maximum(T[:, None] * u[:, None] * lam[None, :], 1e-300)
    const = gammaln(Xd + 1)
    if B is None or G == 1:
        return float(np.sum(Xd * np.log(base) - base - const))
    per = np.empty((G, Xd.shape[0]))
    for g in range(G):
        mu = np.maximum(base * B[g, cg_idx][None, :], 1e-300)
        per[g] = np.sum(Xd * np.log(mu) - mu - const, axis=1)
    per += np.log(np.maximum(pi, 1e-300))[:, None]
    return float(logsumexp(per, axis=0).sum())


def _chunk_loglik(X, T, u, lam, B, pi, n_ih, rowsum, cg_idx, H, G,
                  rows: int = 512, rng=None) -> float:
    """Mixture log-likelihood for one chunk, estimated on a row subsample.

    The likelihood is a sum over patients, so evaluating it on `rows` randomly
    chosen patients and scaling by n/rows is unbiased. This exists because the
    exact evaluation needs `gammaln` over the dense chunk, which dominates the
    runtime at scale; the estimate is only used to monitor convergence, while
    any reported fit statistic should be computed by `loglik_mixture` on a
    held-out or sampled set.
    """
    if _FAST is not None:
        v = _FAST.chunk_loglik(X, T, u, lam, B, pi, cg_idx, G)
        if v is not None:
            return v

    n = X.shape[0]
    if n <= rows:
        return _dense_chunk_loglik(X, T, u, lam, B, pi, cg_idx, G)
    rng = rng or np.random.default_rng(0)
    sel = rng.choice(n, size=rows, replace=False)
    sub = _dense_chunk_loglik(X[sel], T[sel], u[sel], lam, B, pi, cg_idx, G)
    return float(sub * n / rows)


def fit_streaming(make_chunks: Callable[[int], Iterator], n, c, concept_group,
                  n_groups: int = 1, n_epochs: int = 25, local_iters: int = 3,
                  tol: float = 1e-7, init: Params | None = None,
                  init_sample: int = 5000, seed: int = 0,
                  ll_rows: int = 512, verbose: bool = False) -> StreamingFit:
    """Fit by epoch-wise EM over a stream of patient chunks.

    `make_chunks(epoch)` returns an iterable of (X_chunk, T_chunk) with X_chunk
    a CSR matrix of shape (n_chunk, c). Chunks should partition the sampled
    patients within an epoch; they may be redrawn between epochs.
    """
    H = int(concept_group.max()) + 1
    G = max(1, int(n_groups))
    concept_group = np.asarray(concept_group)
    col_H = _concept_group_matrix(concept_group, H)
    cg_idx = concept_group

    if init is not None:
        lam = init.lam.copy()
        B = None if init.B is None else init.B.copy()
        pi = np.full(G, 1.0 / G)
    else:
        lam = np.full(c, 1.0 / c)
        pi = np.full(G, 1.0 / G)
        B = None
        if G > 1:
            warm = _init_from_sample_fit(make_chunks, c, cg_idx, G,
                                         init_sample, seed)
            if warm is not None:
                lam = warm.lam.copy()
                B = warm.B.copy()
            else:
                B = _init_B(make_chunks, col_H, G, H, lam, cg_idx, n,
                            init_sample, seed)

    u_all = np.ones(n)
    lab_all = np.zeros(n, dtype=int)

    ll_rng = np.random.default_rng(seed + 999)
    history = []
    n_seen = 0
    n_events = 0
    prev = -np.inf

    for epoch in range(1, n_epochs + 1):
        colsum = np.zeros(c)
        num_gh = np.zeros((G, H))
        A_g = np.zeros(G)
        pi_num = np.zeros(G)
        ll = 0.0
        seen = 0

        for chunk in make_chunks(epoch):
            idx, X, T = chunk
            X = sparse.csr_matrix(X)
            T = np.asarray(T, dtype=float)
            if X.shape[0] == 0:
                continue

            n_ih = (X @ col_H).toarray()
            rowsum = np.asarray(X.sum(axis=1)).ravel()
            u = rowsum / np.maximum(T * lam.sum(), EPS)

            if G == 1 or B is None:
                R = np.ones((X.shape[0], 1))
            else:
                Lam_h = np.bincount(cg_idx, weights=lam, minlength=H)
                logB = np.log(B)
                blam = B @ Lam_h
                for _ in range(local_iters):
                    score = n_ih @ logB.T - (T * u)[:, None] * blam[None, :]
                    score = score + np.log(np.maximum(pi, 1e-300))[None, :]
                    m = score.max(axis=1, keepdims=True)
                    R = np.exp(score - m)
                    R /= R.sum(axis=1, keepdims=True)
                    u = rowsum / np.maximum(T * (R @ blam), EPS)

            u_all[idx] = u
            if R.shape[1] > 1:
                lab_all[idx] = np.argmax(R, axis=1)

            colsum += np.asarray(X.sum(axis=0)).ravel()
            num_gh += R.T @ n_ih
            A_g += R.T @ (T * u)
            pi_num += R.sum(axis=0)
            seen += X.shape[0]
            ll += _chunk_loglik(X, T, u, lam, B, pi, n_ih, rowsum, cg_idx,
                                H, G, rows=ll_rows, rng=ll_rng)

        pi = np.maximum(pi_num / max(pi_num.sum(), EPS), 1e-9)
        pi /= pi.sum()
        if B is None:
            lam = np.maximum(colsum / max(A_g.sum(), EPS), EPS)
        else:
            lam = np.maximum(colsum / np.maximum((A_g @ B)[cg_idx], EPS), EPS)
            Lam_h = np.bincount(cg_idx, weights=lam, minlength=H)
            B = np.maximum((num_gh + 0.5) / (np.outer(A_g, Lam_h) + 0.5), EPS)
            tot = A_g.sum()
            if tot > 0:
                for h in range(H):
                    cc = float(np.sum(A_g * B[:, h]) / tot)
                    if cc > 0:
                        B[:, h] /= cc
                        lam[cg_idx == h] *= cc

        n_seen = seen
        n_events += int(colsum.sum())
        history.append(ll)
        if verbose:
            print(f"  epoch {epoch:3d}  loglik {ll:,.1f}  patients {seen:,}")
        if abs(ll - prev) < tol * max(1.0, abs(prev)):
            break
        prev = ll

    params = Params(u=u_all, lam=lam, B=B,
                    patient_group=lab_all,
                    concept_group=cg_idx.copy(), n_groups=G)
    return StreamingFit(params=params,
                        loglik=float(history[-1]) if history else -np.inf,
                        history=history, n_patients_seen=n_seen,
                        n_events_seen=n_events, epochs=len(history),
                        density=n_events / max(n_seen, 1))


def array_chunks(data: SparseCDM, chunk_size: int, sample_frac: float = 1.0,
                 seed: int = 0, redraw: bool = True):
    """`make_chunks` over a SparseCDM, with optional per-epoch subsampling."""
    def make(epoch: int):
        rng = np.random.default_rng(seed + (epoch if redraw else 0))
        n = data.n
        if sample_frac < 1.0:
            k = max(1, int(round(sample_frac * n)))
            idx = rng.choice(n, size=k, replace=False)
        else:
            idx = np.arange(n)
        if redraw:
            idx = rng.permutation(idx)
        for s in range(0, len(idx), chunk_size):
            sel = idx[s:s + chunk_size]
            yield sel, data.X[sel], data.T[sel]
    return make


def fit_propensity_streaming(make_chunks, params: Params) -> float:
    """Negative-binomial shape k, accumulated over chunks (one pass)."""
    from scipy import optimize

    H = int(params.concept_group.max()) + 1
    lam_h = np.bincount(params.concept_group, weights=params.lam, minlength=H)
    col_H = _concept_group_matrix(params.concept_group, H)

    xs, mus = [], []
    for chunk in make_chunks(0):
        X, T = chunk[1], chunk[2]
        X = sparse.csr_matrix(X)
        if X.shape[0] == 0:
            continue
        x = np.asarray(X.sum(axis=1)).ravel().astype(float)
        if params.B is None:
            mu = np.asarray(T, dtype=float) * lam_h.sum()
        else:
            n_ih = (X @ col_H).toarray()
            g = np.argmax(n_ih @ np.log(np.maximum(params.B, 1e-300)).T, axis=1)
            mu = np.asarray(T, dtype=float) * (params.B @ lam_h)[g]
        xs.append(x)
        mus.append(mu)

    x = np.concatenate(xs)
    mu = np.concatenate(mus)

    def nll(theta):
        k = np.exp(theta[0])
        return -float(np.sum(gammaln(x + k) - gammaln(k) - gammaln(x + 1)
                             + k * (np.log(k) - np.log(k + mu))
                             + x * (np.log(np.maximum(mu, 1e-12)) - np.log(k + mu))))

    r = optimize.minimize_scalar(lambda t: nll(np.array([t])),
                                 bounds=(-6.0, 6.0), method="bounded")
    return float(np.exp(r.x))


def simulate_sparse(n: int = 100_000, c: int = 500, n_concept_groups: int = 10,
                    n_patient_groups: int = 5, years: float = 5.0,
                    sigma_u: float = 1.2, events_per_person_year: float = 8.0,
                    block_amplification: float = 4.0, block_noise: float = 0.1,
                    seed: int = 0) -> tuple:
    """Generate a large CDM directly in sparse form.

    Sampling total events per patient and then allocating them over concepts
    avoids ever building the dense N x C matrix, which is what makes a
    10^5-patient benchmark run on a laptop. The per-patient concept vector is
    built inside the loop, so peak memory is O(events), not O(N*C).
    """
    rng = np.random.default_rng(seed)
    T = np.full(n, float(years))
    u = rng.lognormal(0.0, sigma_u, size=n)
    u /= u.mean()

    lam0 = rng.lognormal(0.0, 1.0, size=c)
    lam0 /= lam0.sum()
    cg = rng.permutation(np.arange(c) % n_concept_groups)
    B = np.exp(rng.normal(0.0, block_noise, size=(n_patient_groups, n_concept_groups)))
    for g in range(min(n_patient_groups, n_concept_groups)):
        B[g, g] *= 1.0 + block_amplification
    pg = rng.integers(0, n_patient_groups, size=n)

    # calibrate the overall intensity so the total event count is on target
    Lam_h0 = np.bincount(cg, weights=lam0, minlength=n_concept_groups)
    per_patient0 = (B @ Lam_h0)[pg]
    target = events_per_person_year * float(T.sum())
    scale = target / max(float(np.sum(T * u * per_patient0)), 1e-9)
    lam = lam0 * scale

    Lam_h = np.bincount(cg, weights=lam, minlength=n_concept_groups)
    mu_total = T * u * (B @ Lam_h)[pg]
    counts = rng.poisson(mu_total)

    rows, cols, vals = [], [], []
    for i in np.nonzero(counts > 0)[0]:
        k = int(counts[i])
        q = lam * B[pg[i], cg]
        draw = rng.multinomial(k, q / q.sum())
        nz = np.nonzero(draw)[0]
        if nz.size:
            rows.append(np.full(nz.size, i, dtype=np.int64))
            cols.append(nz.astype(np.int64))
            vals.append(draw[nz].astype(np.float64))

    X = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, c))
    data = SparseCDM(X=X, T=T, concept_group=cg,
                     person_ids=np.arange(1, n + 1),
                     concept_ids=np.arange(1, c + 1))
    truth = Params(u=u, lam=lam, B=B, patient_group=pg,
                   concept_group=cg.copy(), n_groups=n_patient_groups)
    return data, truth
