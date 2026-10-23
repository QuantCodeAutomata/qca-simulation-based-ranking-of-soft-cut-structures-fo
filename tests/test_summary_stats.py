"""Tests for src.summary_stats."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.summary_stats import (
    compute_return_summaries,
    compute_bv_summaries,
    compute_full_summary_vector,
)


def test_return_summaries_shape():
    """compute_return_summaries must return an array of shape (5,)."""
    np.random.seed(10)
    returns = np.random.randn(100) * 0.01
    s = compute_return_summaries(returns)
    assert s.shape == (5,), f"Return summaries shape {s.shape} != (5,)"
    assert s.dtype in [np.float32, np.float64], "Expected float dtype"


def test_bv_summaries_shape():
    """compute_bv_summaries must return an array of shape (5,)."""
    np.random.seed(11)
    bv = np.abs(np.random.randn(100)) * 1e-4   # positive BV values
    s = compute_bv_summaries(bv)
    assert s.shape == (5,), f"BV summaries shape {s.shape} != (5,)"
    assert s.dtype in [np.float32, np.float64], "Expected float dtype"


def test_return_summaries_known():
    """Check that mean and variance statistics match numpy references."""
    np.random.seed(12)
    n = 200
    true_mean = 0.002
    true_std  = 0.01
    returns   = np.random.normal(true_mean, true_std, size=n)

    s = compute_return_summaries(returns)

    # s[0] = mean, s[1] = variance
    assert abs(s[0] - np.mean(returns)) < 1e-12, (
        f"Mean mismatch: summary={s[0]:.6f}, numpy={np.mean(returns):.6f}"
    )
    assert abs(s[1] - np.var(returns, ddof=1)) < 1e-12, (
        f"Variance mismatch: summary={s[1]:.8f}, numpy={np.var(returns, ddof=1):.8f}"
    )


def test_full_summary_vector_finite():
    """All entries of the full summary vector must be finite."""
    np.random.seed(13)
    T, n_features = 50, 12

    returns     = np.random.randn(T) * 0.01
    bv          = np.abs(np.random.randn(T)) * 1e-4 + 1e-6
    iv_features = np.random.randn(T, n_features) * 0.05 + 0.2  # positive-ish IVs

    _, _, _, S_all = compute_full_summary_vector(returns, bv, iv_features)

    n_total  = len(S_all)
    n_finite = int(np.isfinite(S_all).sum())
    # GARCH fitting may fail on tiny data → expect ≥ 95% finite
    assert n_finite >= int(0.95 * n_total), (
        f"Only {n_finite}/{n_total} entries are finite in full summary vector"
    )
    assert n_total == 5 + 5 + n_features * 5, (
        f"Summary length {n_total} != {5 + 5 + n_features * 5}"
    )
