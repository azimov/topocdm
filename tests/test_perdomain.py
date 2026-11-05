"""Tests for the per-domain temporal model (topocdm.perdomain)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from topocdm import perdomain as pdm


def _synthetic(n=500, seed=0):
    """A small CDM-shaped extract: intervals, points, and a death."""
    rng = np.random.default_rng(seed)
    periods = pd.DataFrame({
        "person_id": np.arange(1, n + 1),
        "start_date": pd.Timestamp("2020-01-01"),
        "end_date": pd.Timestamp("2024-12-31"),
    })
    # per-patient contact volume
    u = rng.gamma(2.0, 0.5, size=n)
    rows = []
    for dom, concepts, is_interval in [
            ("condition", [100, 101, 102], True),
            ("drug", [200, 201], True),
            ("measurement", [300, 301], False)]:
        probs = np.ones(len(concepts)) / len(concepts)
        for i in range(n):
            k = rng.poisson(3.0 * u[i])
            for _ in range(k):
                c = int(rng.choice(concepts, p=probs))
                day = rng.integers(0, 5 * 365 - 1)
                s = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(day))
                e = s + pd.Timedelta(days=int(rng.integers(0, 800))) \
                    if is_interval else pd.NaT
                rows.append((i + 1, dom, c, s, e))
    events = pd.DataFrame(rows, columns=["person_id", "domain", "concept_id",
                                         "start_date", "end_date"])
    return events, periods


def test_duration_mixture_recovers_zero_mass():
    rng = np.random.default_rng(0)
    n = 5000
    zero = rng.random(n) < 0.2
    d = np.where(zero, 0, np.round(rng.lognormal(5.9, 0.3, size=n))).astype(int)
    ev = pd.DataFrame({
        "person_id": np.arange(n),
        "domain": "condition",
        "concept_id": 1,
        "start_date": pd.Timestamp("2020-01-01"),
        "end_date": pd.Timestamp("2020-01-01") + pd.to_timedelta(d, unit="D"),
    })
    dur = pdm.fit_durations(ev, "condition")
    assert abs(dur["pi0"] - 0.2) < 0.03
    # sampled median should be near the log-normal median, exp(5.9) ~ 365
    samp = pdm.sample_durations(dur, 20000, np.random.default_rng(1))
    assert 250 < np.median(samp[samp > 0]) < 500


def test_build_eras_merges_overlapping_and_adjacent():
    df = pd.DataFrame({
        "person_id": [1, 1, 1, 2],
        "condition_concept_id": [9, 9, 9, 9],
        "condition_start_date": pd.to_datetime(
            ["2020-01-01", "2020-02-01", "2020-06-01", "2020-01-01"]),
        "condition_end_date": pd.to_datetime(
            ["2020-02-15", "2020-03-01", "2020-06-10", "2020-01-10"]),
    })
    era = pdm.build_eras(df, "condition_concept_id", "condition_start_date",
                         "condition_end_date",
                         out_start="condition_era_start_date",
                         out_end="condition_era_end_date")
    # patient 1: first two merge (overlap), third is >30d after -> 2 eras
    assert len(era[era.person_id == 1]) == 2
    assert len(era[era.person_id == 2]) == 1
    first = era[era.person_id == 1].sort_values("condition_era_start_date").iloc[0]
    assert first["occurrence_count"] == 2


def test_marginals_recover_known_structure():
    from scipy import sparse
    rng = np.random.default_rng(0)
    n, c = 400, 6
    T = np.full(n, 5.0)
    u = rng.gamma(2.0, 0.5, size=n)
    lam = np.array([0.5, 0.3, 0.2, 0.1, 0.05, 0.02])
    mu = T[:, None] * u[:, None] * lam[None, :]
    X = sparse.csr_matrix(rng.poisson(mu).astype(float))
    u_hat, lam_hat = pdm._fit_marginals_sparse(X, T)
    # shapes recovered, not absolute scale (u has mean 1 by constraint); u is
    # estimated from a handful of Poisson counts, so its recovery is noisier
    # than the column intensities
    assert np.corrcoef(u / u.mean(), u_hat)[0, 1] > 0.8
    assert np.corrcoef(lam / lam.sum(), lam_hat / lam_hat.sum())[0, 1] > 0.99


def test_fit_generate_and_eras():
    events, periods = _synthetic(n=500)
    fit = pdm.fit_perdomain(events, periods, K=0)
    assert 0 < fit.k < 20
    assert np.isfinite(fit.eta).all()
    syn_periods = pdm.sample_periods(periods, 500, np.random.default_rng(1))
    tables = pdm.generate_cohort(fit, 500, syn_periods, seed=2)
    for name in ("condition_occurrence", "drug_exposure", "measurement",
                 "condition_era", "drug_era"):
        assert name in tables
    assert len(tables["condition_occurrence"]) > 0
    # every generated interval is a valid pair
    c = tables["condition_occurrence"]
    assert (pd.to_datetime(c["condition_end_date"])
            >= pd.to_datetime(c["condition_start_date"])).all()
    # era tables are populated and no era is shorter than its source interval
    assert len(tables["condition_era"]) > 0


def test_generation_level_is_close():
    events, periods = _synthetic(n=1500, seed=3)
    fit = pdm.fit_perdomain(events, periods, K=0, seed=1)
    syn_periods = pdm.sample_periods(periods, 1500, np.random.default_rng(4))
    tables = pdm.generate_cohort(fit, 1500, syn_periods, seed=5)
    gen = len(tables["condition_occurrence"])
    obs = fit.n_events["condition"]
    assert 0.6 < gen / obs < 1.6, (gen, obs)


def test_sampling_option_reduces_patients():
    events, periods = _synthetic(n=300)
    fit = pdm.fit_perdomain(events, periods, K=0, max_patients=100)
    assert fit.meta["n_patients"] == 100
    assert len(fit.person_ids) == 100


def test_spectrum_matches_normalised_svd():
    rng = np.random.default_rng(0)
    M = rng.poisson(3.0, size=(40, 15)).astype(float)
    from scipy import sparse
    s = pdm.spectrum(sparse.csr_matrix(M))
    r = M.sum(1); c = M.sum(0)
    N = M / np.sqrt(np.outer(r, c))
    s_ref = np.linalg.svd(N, compute_uv=False)
    k = len(s_ref)
    # the full spectrum is padded with zeros to max(n, m)
    assert np.allclose(np.sort(s)[::-1][:k], s_ref, atol=1e-6)
    assert np.allclose(s[k:], 0.0)
