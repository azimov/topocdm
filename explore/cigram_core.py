"""
cigram_core.py -- a compact, dependency-light reimplementation of CiGRAM.

Reference implementation: Gilbert, J.P. (2015) "A probabilistic model for the
analysis of modular structure in complex biological networks", PhD thesis,
University of Nottingham, Chapter 5 (Algorithms 2 and 3).

This is a *probe*, not a port. It reproduces the generative logic closely
enough to test the model's stated properties (degree heterogeneity, the
density-assortativity limit, the K-dependence of the tail), and it adds a
multiplex extension (CiGRAM-M) that the thesis does not contain.

The original is fixed-density, unweighted, undirected. So is this.

Key pieces, with thesis equation numbers:

    theta_i  ~ WrappedNormal(0, sigma_f)              folded to (-pi, pi]
    alpha_i   = g(theta_i; mu=pi, sigma_s)            Eq 5.3
    beta_i    = alpha_i / sum(alpha)                  Eq 5.4
    P(i->j)  ~ beta_j * exp(-a * delta(theta_i,theta_j))
                                                      Eq 5.13
    delta     = | |theta_i| - |theta_j| |             (radial, semi-circle)

Sampling without replacement uses the Gumbel-top-k trick, which is the modern
equivalent of the Efraimidis-Spirakis reservoir sampler used in the thesis
(Algorithm 1). Same distribution, vectorised.
"""

from __future__ import annotations

import numpy as np

TAU = 2.0 * np.pi


# --------------------------------------------------------------------------
# Wrapped Gaussian machinery (thesis Section 5.1.1)
# --------------------------------------------------------------------------

def wrapped_gauss(theta, mu, sigma, kmax=6):
    """Wrapped normal density, Eq 5.1: sum over the integer lattice."""
    theta = np.asarray(theta, dtype=float)
    out = np.zeros_like(theta)
    for k in range(-kmax, kmax + 1):
        out += np.exp(-((theta - mu + TAU * k) ** 2) / (2.0 * sigma ** 2))
    return out / (sigma * np.sqrt(TAU))


def wrap(theta):
    """Fold an angle into (-pi, pi]."""
    return (np.asarray(theta, dtype=float) + np.pi) % TAU - np.pi


def node_positions(n, sigma_f, rng):
    """Draw node positions: Eq 5.2."""
    return wrap(rng.normal(0.0, sigma_f, size=n))


def node_scores(theta, sigma_s):
    """Score function s(theta) = g(theta; mu=pi, sigma_s): Eq 5.3."""
    return wrapped_gauss(theta, mu=np.pi, sigma=sigma_s)


# --------------------------------------------------------------------------
# Weighted sampling without replacement
# --------------------------------------------------------------------------

def _gumbel_topk(weights, k, rng):
    """Sample k distinct indices with probability proportional to weights.

    Uniformisation trick: arg-top-k of (log w + Gumbel) is a without-replacement
    sample weighted by w. O(n) per draw, fully vectorised.
    """
    n = weights.shape[0]
    k = int(min(k, np.count_nonzero(weights > 0)))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    with np.errstate(divide="ignore"):
        logw = np.where(weights > 0, np.log(weights), -np.inf)
    keys = logw + rng.gumbel(size=n)
    if k >= n:
        return np.argsort(-keys)
    return np.argpartition(-keys, k - 1)[:k]


# --------------------------------------------------------------------------
# Core generator (thesis Algorithm 2)
# --------------------------------------------------------------------------

class CigramGraph:
    """Container for a generated graph and its ground-truth structure."""

    def __init__(self, adj, theta, alpha, communities=None, community_theta=None):
        self.adj = adj
        self.theta = theta
        self.alpha = alpha
        self.communities = communities if communities is not None else np.zeros(
            adj.shape[0], dtype=int)
        self.community_theta = community_theta

    @property
    def n(self):
        return self.adj.shape[0]

    @property
    def degrees(self):
        return self.adj.sum(axis=1)

    @property
    def m(self):
        return int(self.adj.sum() // 2)


def _fill_subgraph(nodes, theta, beta, m, a, rng, min_degree=1, adj=None,
                   global_index=None):
    """Fill a block (thesis fillGraph). Returns local adjacency."""
    nodes = np.asarray(nodes, dtype=np.int64)
    nb = nodes.size
    local = np.zeros((nb, nb), dtype=bool)
    if m <= 0 or nb < 2:
        return local

    th = theta[nodes]
    b = beta[nodes]
    b = b / b.sum()

    dmat = np.abs(np.abs(th)[:, None] - np.abs(th)[None, :])
    if a != 0.0:
        base = np.exp(-a * dmat)
    else:
        base = np.ones_like(dmat)

    m = int(min(m, nb * (nb - 1) // 2))
    stubs = rng.choice(nb, size=m, p=b)
    counts = np.bincount(stubs, minlength=nb)

    for i in np.nonzero(counts)[0]:
        k = counts[i]
        w = b * base[i]
        w[local[i]] = 0.0
        w[i] = 0.0
        js = _gumbel_topk(w, k, rng)
        if js.size:
            local[i, js] = True
            local[js, i] = True
    return local


def cigram(n, avg_deg=None, density=None, sigma_f=1.0, sigma_s=1.0, a=0.0,
           min_degree=1, rng=None, seed=None):
    """Single-community CiGRAM (thesis Section 5.1)."""
    if rng is None:
        rng = np.random.default_rng(seed)
    if density is None:
        if avg_deg is None:
            raise ValueError("give avg_deg or density")
        density = avg_deg / (n - 1)
    m = int(round(density * n * (n - 1) / 2))

    theta = node_positions(n, sigma_f, rng)
    alpha = node_scores(theta, sigma_s)
    beta = alpha / alpha.sum()

    adj = _fill_subgraph(np.arange(n), theta, beta, m, a, rng,
                         min_degree=min_degree)

    if min_degree > 0:
        deg = adj.sum(axis=1)
        iso = np.nonzero(deg < min_degree)[0]
        if iso.size:
            for i in iso:
                w = beta.copy()
                w[i] = 0.0
                w[adj[i]] = 0.0
                j = _gumbel_topk(w, min_degree - deg[i], rng)
                adj[i, j] = True
                adj[j, i] = True
    return CigramGraph(adj, theta, alpha)


# --------------------------------------------------------------------------
# Modular CiGRAM (thesis Section 5.2, Algorithm 3)
# --------------------------------------------------------------------------

def cigram_modular(n, K, avg_deg=None, density=None, sigma_f=1.0, sigma_s=1.0,
                   a=0.0, ek=0.01, p_o=0.0, sigma_cf=None, sigma_cs=None,
                   min_degree=1, rng=None, seed=None):
    """CiGRAM with K heterogeneous, optionally overlapping communities.

    ek    : fraction of edges placed between communities (thesis e_k)
    p_o   : overlap probability modifier (thesis p_o)
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if density is None:
        if avg_deg is None:
            raise ValueError("give avg_deg or density")
        density = avg_deg / (n - 1)
    m = int(round(density * n * (n - 1) / 2))

    sigma_cf = sigma_f if sigma_cf is None else sigma_cf
    sigma_cs = sigma_s if sigma_cs is None else sigma_cs

    theta = node_positions(n, sigma_f, rng)
    alpha = node_scores(theta, sigma_s)
    beta = alpha / alpha.sum()

    # community latent structure (Eq 5.15)
    ctheta = node_positions(K, sigma_cf, rng)
    calpha = node_scores(ctheta, sigma_cs)
    cbeta = calpha / calpha.sum()

    # node -> primary community
    primary = rng.choice(K, size=n, p=cbeta)
    members = [set() for _ in range(n)]
    for i in range(n):
        members[i].add(int(primary[i]))

    # overlap (Eq 5.16): additional memberships, biased to nearby communities
    if p_o > 0:
        cdist = np.abs(np.abs(ctheta)[:, None] - np.abs(ctheta)[None, :])
        for i in range(n):
            c0 = int(primary[i])
            w = cbeta * np.exp(-a * cdist[c0])
            w[c0] = 0.0
            if w.sum() <= 0:
                continue
            extra = rng.random() < p_o
            while extra:
                k = _gumbel_topk(w, 1, rng)[0]
                members[i].add(int(k))
                w[k] = 0.0
                extra = (w.sum() > 0) and (rng.random() < p_o)

    adj = np.zeros((n, n), dtype=bool)

    # --- intra-community edges: (1-ek)*m, allocated proportional to cbeta
    m_intra_total = int(round((1.0 - ek) * m))
    alloc = rng.multinomial(m_intra_total, cbeta)
    owner = np.array([next(iter(s)) for s in members])

    for c in range(K):
        nodes = np.nonzero(owner == c)[0]
        if nodes.size < 2 or alloc[c] == 0:
            continue
        sub = _fill_subgraph(nodes, theta, beta, int(alloc[c]), a, rng,
                             min_degree=0)
        adj[np.ix_(nodes, nodes)] |= sub

    # --- inter-community edges
    cdist = np.abs(np.abs(ctheta)[:, None] - np.abs(ctheta)[None, :])
    m_inter = m - int(adj.sum() // 2)
    if m_inter > 0:
        # community-level affinity gamma (Eq 5.20), normalised per row
        gamma = np.exp(-a * cdist) * cbeta[None, :]
        gamma[np.arange(K), np.arange(K)] = 0.0
        for _ in range(m_inter):
            c1 = rng.choice(K, p=cbeta)
            i_cands = np.nonzero(owner == c1)[0]
            if i_cands.size == 0:
                continue
            i = rng.choice(i_cands, p=beta[i_cands] / beta[i_cands].sum())
            w = beta * np.exp(-a * (np.abs(np.abs(theta[i]) - np.abs(theta))
                                    + cdist[c1][owner]))
            w[i] = 0.0
            w[adj[i]] = 0.0
            # cannot connect within the same community
            w[owner == c1] = 0.0
            j = _gumbel_topk(w, 1, rng)
            if j.size:
                adj[i, j[0]] = True
                adj[j[0], i] = True

    if min_degree > 0:
        deg = adj.sum(axis=1)
        iso = np.nonzero(deg < min_degree)[0]
        for i in iso:
            w = beta.copy()
            w[i] = 0.0
            w[adj[i]] = 0.0
            j = _gumbel_topk(w, 1, rng)
            if j.size:
                adj[i, j[0]] = True
                adj[j[0], i] = True

    return CigramGraph(adj, theta, alpha, communities=owner,
                       community_theta=ctheta)


# --------------------------------------------------------------------------
# Multiplex extension: CiGRAM-M
# --------------------------------------------------------------------------
#
# The thesis has one graph. A multiplex version needs a story about what is
# shared between layers and what is layer-specific. The story here is the one
# observational epidemiology actually forces on you:
#
#   * ONE population of units (patients, genes, whatever), with ONE latent
#     geometry and ONE latent group structure. A patient does not get
#     re-partitioned because you looked at their prescriptions instead of
#     their diagnoses.
#   * Each layer is an OBSERVATION CHANNEL with its own noise, its own
#     degree heterogeneity (sigma_s), its own mixing preference (a) and its
#     own coverage (avg_deg, ek).
#   * rho in [0,1] is the *shared fraction of latent geometry*: rho=1 means a
#     unit's rank is identical in every channel (maximal inter-layer degree
#     coupling); rho=0 means channels are independent.
#   * `utilisation_bias` gamma multiplies the log propensity by a fixed
#     patient-level frailty z. gamma>0 in one layer mimics a data source that
#     oversamples high-utilisers. This is the ascertainment artefact.
#
# That gives four knobs the thesis' single-graph model does not have, and they
# are the four that matter when layers are data sources rather than
# transmission routes.


def _assemble(theta, beta, owner, ctheta, cbeta, n, avg_deg, a, ek, rng):
    """Build one layer's adjacency given fixed latent geometry."""
    density = avg_deg / (n - 1) if avg_deg < n - 1 else 1.0
    m = int(round(density * n * (n - 1) / 2))
    K = int(owner.max()) + 1
    adj = np.zeros((n, n), dtype=bool)

    if K > 1 and ctheta is not None:
        alloc = rng.multinomial(int(round((1.0 - ek) * m)), cbeta)
        for c in range(K):
            nodes = np.nonzero(owner == c)[0]
            if nodes.size < 2 or alloc[c] == 0:
                continue
            sub = _fill_subgraph(nodes, theta, beta, int(alloc[c]), a, rng,
                                 min_degree=0)
            adj[np.ix_(nodes, nodes)] |= sub

        cdist = np.abs(np.abs(ctheta)[:, None] - np.abs(ctheta)[None, :])
        n_inter = m - int(adj.sum() // 2)
        for _ in range(max(0, n_inter)):
            c1 = int(rng.choice(K, p=cbeta))
            cands = np.nonzero(owner == c1)[0]
            if cands.size == 0:
                continue
            i = int(rng.choice(cands, p=beta[cands] / beta[cands].sum()))
            w = beta * np.exp(-a * (np.abs(np.abs(theta[i]) - np.abs(theta))
                                    + cdist[c1][owner]))
            w[i] = 0.0
            w[adj[i]] = 0.0
            w[owner == c1] = 0.0
            j = _gumbel_topk(w, 1, rng)
            if j.size:
                adj[i, j[0]] = True
                adj[j[0], i] = True
    else:
        adj = _fill_subgraph(np.arange(n), theta, beta, m, a, rng,
                             min_degree=1)

    for i in np.nonzero(adj.sum(axis=1) == 0)[0]:
        w = beta.copy()
        w[i] = 0.0
        w[adj[i]] = 0.0
        j = _gumbel_topk(w, 1, rng)
        if j.size:
            adj[i, j[0]] = True
            adj[j[0], i] = True
    return adj


def cigram_multiplex(n, layers, rho=0.7, avg_deg=8, K=1, ek=0.05,
                     sigma_f=1.0, sigma_s=1.0, sigma_cf=None, sigma_cs=None,
                     rng=None, seed=None):
    """Generate an L-layer multiplex graph over a shared population.

    `layers` is a list of dicts, each optionally specifying:
        name, sigma_s, a, ek, avg_deg, utilisation_bias

    Returns (graphs, context) where graphs is a list of CigramGraph sharing
    node indexing and a common `communities` array, and context holds the
    shared latent positions, the frailty vector and the owner assignment.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    sigma_cf = sigma_f if sigma_cf is None else sigma_cf
    sigma_cs = sigma_s if sigma_cs is None else sigma_cs

    theta_shared = wrap(rng.normal(0.0, sigma_f, size=n))
    z = rng.normal(0.0, 1.0, size=n)          # patient-level frailty

    if K > 1:
        ctheta = node_positions(K, sigma_cf, rng)
        calpha = node_scores(ctheta, sigma_cs)
        cbeta = calpha / calpha.sum()
        owner = rng.choice(K, size=n, p=cbeta)
    else:
        ctheta, cbeta = None, None
        owner = np.zeros(n, dtype=int)

    graphs = []
    for spec in layers:
        ss = spec.get("sigma_s", sigma_s)
        al = spec.get("a", 0.0)
        ekl = spec.get("ek", ek)
        deg = spec.get("avg_deg", avg_deg)
        gamma = spec.get("utilisation_bias", 0.0)
        sf = spec.get("sigma_f", sigma_f)

        eta = rng.normal(0.0, 1.0, size=n)
        theta_l = wrap(rho * theta_shared
                       + np.sqrt(max(0.0, 1.0 - rho ** 2)) * sf * eta)

        alpha = node_scores(theta_l, ss)
        logb = np.log(alpha) + gamma * z
        beta = np.exp(logb - logb.max())
        beta = beta / beta.sum()

        adj = _assemble(theta_l, beta, owner, ctheta, cbeta, n, deg, al, ekl, rng)
        g = CigramGraph(adj, theta_l, alpha, communities=owner,
                        community_theta=ctheta)
        g.name = spec.get("name", "layer")
        graphs.append(g)

    ctx = dict(theta_shared=theta_shared, utilisation=z, owner=owner,
               ctheta=ctheta)
    return graphs, ctx
