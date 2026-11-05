"""Tests for the per-concept prevalence layer (topocdm.hurdle)."""

from __future__ import annotations

import numpy as np

from topocdm import hurdle


def test_apply_preserves_counts_and_matches_prevalence():
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(300, 5)).astype(float)
    pi = np.array([0.1, 0.2, 0.3, 0.5, 0.8])
    Y = hurdle.apply_prevalence(X, pi)
    # column totals are preserved: incidence is unchanged
    assert np.allclose(X.sum(axis=0), Y.sum(axis=0))
    # the positive-patient share matches the target
    assert np.allclose((Y > 0).mean(axis=0), pi, atol=0.02)


def test_sample_prevalence_is_a_distribution():
    rng = np.random.default_rng(1)
    X = rng.binomial(1, 0.3, size=(100, 8)).astype(float)
    p = hurdle.sample_prevalence(X, rng)
    assert p.shape == (8,)
    assert np.all((p >= 0) & (p <= 1))


def test_concentrating_does_not_invent_records():
    # 20 patients with 2 records each; concentrate to 10 positive patients
    X = np.zeros((100, 1))
    X[:20, 0] = 2.0
    Y = hurdle.apply_prevalence(X, np.array([0.1]))
    assert np.isclose(Y.sum(), 40.0)
    assert abs((Y > 0).mean() - 0.1) < 0.02
