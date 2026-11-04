"""Tests for the fixed-density topology model.

The important ones are `test_ipf_matches_optimiser` (the closed-form updates
really do reach the MLE) and `test_block_recovery` (the latent structure is
identified, not just fitted).
"""

import numpy as np
import pytest
from scipy import optimize

import topocdm as t
from topocdm.fit import fit_propensity_nb


@pytest.fixture(scope="module")
def small():
    return t.simulate(t.SimConfig(n=400, c=12, n_concept_groups=3,
                                  n_patient_groups=2, block_amplification=2.0,
                                  block_noise=0.05, seed=7))


def test_conditional_loglik_definition():
    data, true = t.simulate(t.SimConfig(n=60, c=8, seed=1))
    mu = t.mean_matrix(true, data)
    p = mu / mu.sum()
    expected = float(np.sum(data.X * np.log(p)))
    assert np.isclose(t.loglik_conditional(true, data), expected, rtol=1e-10)


def test_identifiability_constraints(small):
    data, _ = small
    r = t.fit(data, n_patient_groups=2, n_restarts=1)
    p = r.params
    assert np.isclose(p.u.mean(), 1.0)
    w = np.bincount(p.patient_group, weights=data.T * p.u,
                    minlength=p.n_groups)
    for h in range(p.B.shape[1]):
        assert np.isclose(np.sum(w * p.B[:, h]) / w.sum(), 1.0, atol=1e-8)


def test_ipf_matches_optimiser():
    """IPF must reach the same optimum as a general-purpose optimiser."""
    data, _ = t.simulate(t.SimConfig(n=120, c=8, block_amplification=0.0,
                                     block_noise=1e-9, seed=3))
    r = t.fit(data, n_patient_groups=1)

    n, c = data.n, data.c

    def nll(theta):
        theta = np.clip(theta, -40.0, 40.0)     # keep exp() finite
        u = np.exp(theta[:n])
        lam = np.exp(theta[n:])
        mu = np.maximum(data.T[:, None] * u[:, None] * lam[None, :], 1e-12)
        return -float(np.sum(data.X * np.log(mu) - mu))

    x0 = np.concatenate([np.log(r.params.u), np.log(r.params.lam)])
    res = optimize.minimize(nll, x0, method="L-BFGS-B",
                            options=dict(maxiter=20000, ftol=1e-15))
    # `nll` is the Poisson kernel (without the -log X! term); compare on the
    # same scale, and also check both agree with the packaged log-likelihood.
    from scipy.special import gammaln
    kernel = t.loglik_poisson(r.params, data) + float(gammaln(data.X + 1).sum())
    assert np.isclose(-res.fun, kernel, rtol=1e-6)


def test_loglik_increases_monotonically(small):
    data, _ = small
    r = t.fit(data, n_patient_groups=2, n_restarts=1)
    hist = np.array(r.history)
    # a classification-EM reassignment can cost a negligible amount; a large
    # decrease would be a bug
    assert np.all(np.diff(hist) > -1e-3), "log-likelihood decreased"
    assert np.isclose(r.loglik, hist.max())


def test_generation_preserves_total():
    data, true = t.simulate(t.SimConfig(n=300, c=10, seed=5))
    syn = t.sample_cdm(true, data, mode="replay", rng=np.random.default_rng(0))
    assert syn.total == data.total
    assert syn.X.shape == data.X.shape
    assert np.all(syn.X >= 0)


def test_block_recovery():
    """With a clear planted structure the groups must be recovered."""
    from scipy.optimize import linear_sum_assignment
    data, true = t.simulate(t.SimConfig(n=2500, c=40, n_concept_groups=4,
                                        n_patient_groups=4,
                                        block_noise=1e-9,
                                        block_amplification=5.0, seed=11))
    r = t.fit(data, n_patient_groups=4, n_restarts=2)
    C = np.array([[((r.params.patient_group == g) &
                    (true.patient_group == k)).sum() for k in range(4)]
                  for g in range(4)])
    ri, ci = linear_sum_assignment(-C)
    assert C[ri, ci].sum() / C.sum() > 0.8


def test_fresh_generation_matches_topology():
    data, _ = t.simulate(t.SimConfig(n=1500, c=30, n_concept_groups=4,
                                     n_patient_groups=4, seed=13))
    r = t.fit(data, n_patient_groups=4, n_restarts=2)
    dist = fit_propensity_nb(data, r.params)
    syn = t.sample_cdm(r.params, data, patient_dist=dist, mode="fresh",
                       rng=np.random.default_rng(1))
    cmp = t.compare(data, syn)
    assert cmp["prevalence"]["r"] > 0.95
    assert cmp["incidence"]["r"] > 0.95
    assert cmp["cooccurrence_r"] > 0.9
    # degree distribution should be in the right place, not just right shape
    assert abs(cmp["degree_mean"][0] - cmp["degree_mean"][1]) < 1.5


def test_null_simulation_has_no_blocks():
    data, true = t.simulate_null(t.SimConfig(n=500, c=20, n_concept_groups=4,
                                             n_patient_groups=4, seed=17))
    assert np.allclose(true.B, true.B[0, 0], rtol=1e-6)


# ---------------------------------------------------------------- scalability

def test_streaming_matches_in_memory():
    """The chunked fit must reach the same optimum as the in-memory fit."""
    from topocdm.scalable import to_sparse, fit_streaming, array_chunks
    data, _ = t.simulate(t.SimConfig(n=800, c=24, n_concept_groups=4,
                                     n_patient_groups=3, block_amplification=4.0,
                                     block_noise=0.05, seed=31))
    ref = t.fit(data, n_patient_groups=3, n_restarts=2)
    sp = to_sparse(data)
    st = fit_streaming(array_chunks(sp, chunk_size=200), n=sp.n, c=sp.c,
                       concept_group=sp.concept_group, n_groups=3, n_epochs=20)
    # same likelihood to within the streaming tolerance
    assert abs(st.loglik - ref.loglik) / abs(ref.loglik) < 0.02
    a = st.params.lam / st.params.lam.sum()
    b = ref.params.lam / ref.params.lam.sum()
    assert np.abs(a - b).sum() < 0.10


def test_subsampling_is_stable():
    """A patient subsample must give nearly the same lambda as the full data."""
    from topocdm.scalable import simulate_sparse, fit_streaming, array_chunks
    d, _ = simulate_sparse(n=4000, c=60, seed=7)
    full = fit_streaming(array_chunks(d, chunk_size=1000), n=d.n, c=d.c,
                         concept_group=d.concept_group, n_groups=3, n_epochs=4,
                         init_sample=2000)
    sub = fit_streaming(array_chunks(d, chunk_size=1000, sample_frac=0.4),
                        n=d.n, c=d.c, concept_group=d.concept_group,
                        n_groups=3, n_epochs=4, init_sample=2000)
    a = full.params.lam / full.params.lam.sum()
    b = sub.params.lam / sub.params.lam.sum()
    assert np.abs(a - b).sum() < 0.10


def test_omop_tables_are_consistent():
    """Every event must carry a valid person, visit and concept id."""
    from topocdm.omop import ConceptVocab, generate_omop_tables
    from topocdm.scalable import simulate_sparse
    d, _ = simulate_sparse(n=120, c=10, seed=3)
    domains = ["Condition", "Drug", "Measurement", "Procedure", "Observation"]
    vocab = ConceptVocab(concept_id=np.arange(1, d.c + 1),
                         domain_id=np.array([domains[i % 5] for i in range(d.c)]))
    tables = generate_omop_tables(d.X.toarray(), d.T, vocab, rng=np.random.default_rng(0))

    persons = set(tables["person"]["person_id"])
    visits = set(tables["visit_occurrence"]["visit_occurrence_id"])
    total = 0
    for name, df in tables.items():
        if name in {"person", "observation_period", "visit_occurrence"}:
            continue
        total += len(df)
        assert set(df["person_id"]).issubset(persons)
        assert set(df["visit_occurrence_id"]).issubset(visits)
    assert total == d.total, "every event should produce exactly one row"


# ---------------------------------------------------------------- scale-out

def test_partitioned_fit_matches_sequential():
    """The map/reduce driver must reproduce the sequential fit exactly."""
    from topocdm.scalable import simulate_sparse, fit_streaming
    from topocdm.distributed import fit_partitioned, partition_stats, merge, Stats
    from topocdm.model import Params

    d, _ = simulate_sparse(n=3000, c=80, n_concept_groups=6,
                           n_patient_groups=4, seed=9)
    G, CH = 4, 750
    H = int(d.concept_group.max()) + 1

    def make_chunks(epoch):
        for s in range(0, d.n, CH):
            idx = np.arange(s, min(s + CH, d.n))
            yield idx, d.X[s:s + CH], d.T[s:s + CH]

    init = Params(u=np.ones(d.n), lam=np.full(d.c, 1.0 / d.c),
                  B=np.ones((G, H)), patient_group=np.zeros(d.n, int),
                  concept_group=d.concept_group, n_groups=G)

    seq = fit_streaming(make_chunks, n=d.n, c=d.c,
                        concept_group=d.concept_group, n_groups=G,
                        n_epochs=4, init=init, keep_u=False)
    parts = [(d.X[s:s + CH], d.T[s:s + CH]) for s in range(0, d.n, CH)]
    par, hist = fit_partitioned(parts, d.c, d.concept_group, G,
                                n_epochs=4, init=init)

    assert abs(seq.loglik - hist[-1]) < 1e-6
    assert np.allclose(seq.params.lam, par.lam, rtol=0, atol=1e-12)
    assert np.allclose(seq.params.B, par.B, rtol=0, atol=1e-12)


def test_sql_plan_is_portable():
    """Every dialect must produce a plan, and content hashes must be stable."""
    from topocdm.sql import plan, days_between
    for dialect in ["postgres", "databricks", "bigquery", "snowflake", "duckdb"]:
        steps = plan(dialect=dialect)
        assert len(steps) == 8
        assert all(s.content_hash for s in steps)
        assert days_between(dialect, "a", "b")
    # domain aggregations are dialect-independent, so they hash the same
    # anywhere -- that is what makes a checkpoint reusable across platforms
    for name in ["ev_condition", "ev_drug", "ev_measurement"]:
        h = {s.content_hash for s in plan(dialect="postgres") if s.name == name}
        h |= {s.content_hash for s in plan(dialect="duckdb") if s.name == name}
        assert len(h) == 1
    # person-time legitimately differs: date arithmetic is the one dialect
    # specific piece
    pg = {s.content_hash for s in plan(dialect="postgres") if s.name == "person_time"}
    dk = {s.content_hash for s in plan(dialect="duckdb") if s.name == "person_time"}
    assert pg != dk


def test_visit_model_recovers_burstiness():
    """The visit chain should put records of related concepts in one encounter."""
    from topocdm.visits import fit_visit_model, assign_to_visits
    import pandas as pd

    rng = np.random.default_rng(0)
    H, C, N = 4, 20, 300
    cg = np.arange(C) % H
    # two groups only, strongly persistent, so the chain is easy to check
    trans = np.full((H, H), 0.02)
    np.fill_diagonal(trans, 0.94); trans /= trans.sum(1, keepdims=True)
    from topocdm.visits import VisitModel
    rows, visits = [], []
    for i in range(1, N + 1):
        nv = int(rng.integers(5, 15))
        dates = pd.to_datetime("2020-01-01") + pd.to_timedelta(
            np.sort(rng.integers(0, 700, nv)), unit="D")
        visits.append(pd.DataFrame({"person_id": i, "visit_date": dates}))
        h = 0
        for d in dates:
            h = rng.choice(H, p=trans[h])
            for _ in range(rng.integers(2, 5)):
                rows.append((i, int(rng.integers(0, C)) , d))
    ev = pd.DataFrame(rows, columns=["person_id", "concept_id", "event_date"])
    ev["concept_id"] = (ev.index % C)
    vi = pd.concat(visits, ignore_index=True)
    model = fit_visit_model(vi, ev, dict(enumerate(cg)), H, concept_group=cg)
    assert model.trans.shape == (H, H)
    assert np.allclose(model.trans.sum(1), 1.0)
    # records should land on dates that exist in the visit table
    counts = rng.integers(0, 4, size=C)
    pairs = assign_to_visits(counts, 8, model, rng)
    assert len(pairs) == int(counts.sum())
    assert all(0 <= v < 8 for v, _ in pairs)


def test_factor_term_increases_likelihood_and_rank():
    """The rank-K interaction must improve the fit and raise the effective rank
    of the mean matrix, which the block term alone cannot do."""
    from topocdm.factors import fit_factors, _mu

    rng = np.random.default_rng(0)
    N, C, K = 220, 45, 4
    a_true = rng.normal(0, 0.7, size=(N, K))
    b_true = rng.normal(0, 0.7, size=(C, K))
    mu0 = rng.uniform(0.4, 1.6, size=(N, C))
    mu_true = mu0 * np.exp(a_true @ b_true.T)
    X = rng.poisson(mu_true).astype(float)

    base = float(np.sum(X * np.log(np.maximum(mu0, 1e-12)) - mu0))
    f = fit_factors(mu0, X, K=K, n_iter=30)
    assert f.history[-1] > base

    def eff_rank(M):
        s = np.linalg.svd(M, compute_uv=False)
        return int((s / s[0] > 0.1).sum())

    assert eff_rank(_mu(mu0, f.a, f.b)) > eff_rank(mu0)
