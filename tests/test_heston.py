"""Tests for src.heston_simulator."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.heston_simulator import (
    HestonParams,
    simulate_heston_daily,
    compute_bipower_variation,
    compute_daily_returns,
)

# Default small parameters used throughout
_PARAMS = HestonParams(kappa=0.05, theta=0.03, sigma_v=0.03,
                       lam=-0.02, rho=0.0, mu=0.0, V0=0.03, r=0.02)
_T      = 10
_M      = 8   # small intraday grid for speed


def test_simulate_heston_daily_shapes():
    """Output arrays must have shapes (T+1,), (T+1,), (T, M_intra)."""
    np.random.seed(0)
    lp, v, ir = simulate_heston_daily(_PARAMS, T=_T, M_intra=_M, seed=0)

    assert lp.shape == (_T + 1,), f"log_prices shape {lp.shape} != ({_T + 1},)"
    assert v.shape  == (_T + 1,), f"variances shape {v.shape} != ({_T + 1},)"
    assert ir.shape == (_T, _M),  f"intraday_returns shape {ir.shape} != ({_T}, {_M})"


def test_heston_variance_nonnegative():
    """Full-truncation Euler must keep variance ≥ 0 at all times."""
    np.random.seed(1)
    _, v, _ = simulate_heston_daily(_PARAMS, T=_T, M_intra=_M, seed=1)
    assert np.all(v >= 0.0), f"Variance dipped below zero: min={v.min()}"


def test_bipower_variation_positive():
    """Bipower variation must be strictly positive for non-zero returns."""
    np.random.seed(2)
    _, _, ir = simulate_heston_daily(_PARAMS, T=_T, M_intra=_M, seed=2)
    bv = compute_bipower_variation(ir)
    assert bv.shape == (_T,), f"BV shape {bv.shape} != ({_T},)"
    assert np.all(bv > 0.0), f"BV has non-positive entries: min={bv.min()}"


def test_bipower_variation_formula():
    """Check the BPV formula against a manually computed reference.

    For constant intraday returns c over M steps:
        BV = μ₁ · M/(M-1) · (M-1) · c²  = μ₁ · M · c²
    where μ₁ = √(2/π).
    """
    T, M = 1, 4
    c    = 0.01
    ir   = np.full((T, M), c)
    bv   = compute_bipower_variation(ir)

    mu1      = np.sqrt(2.0 / np.pi)
    expected = mu1 * M * c ** 2   # sum of M-1 products, scaled by M/(M-1)
    assert bv.shape == (T,)
    assert abs(bv[0] - expected) < 1e-12, (
        f"BV formula mismatch: got {bv[0]:.8f}, expected {expected:.8f}"
    )


def test_reproducibility_with_seed():
    """Two calls with the same seed must produce identical results."""
    lp1, v1, ir1 = simulate_heston_daily(_PARAMS, T=_T, M_intra=_M, seed=99)
    lp2, v2, ir2 = simulate_heston_daily(_PARAMS, T=_T, M_intra=_M, seed=99)

    assert np.array_equal(lp1, lp2), "log_prices differ between identical seeds"
    assert np.array_equal(v1,  v2),  "variances differ between identical seeds"
    assert np.array_equal(ir1, ir2), "intraday_returns differ between identical seeds"
