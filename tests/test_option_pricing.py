"""Tests for src.option_pricing."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.option_pricing import (
    bs_call_price,
    bs_implied_vol,
    heston_call_price,
    generate_option_grid,
    price_option_panel,
)
from src.heston_simulator import HestonParams

# Common parameters
S, r = 100.0, 0.05
_PARAMS = HestonParams(kappa=0.1, theta=0.04, sigma_v=0.1,
                       lam=-0.01, rho=0.0, mu=0.0, V0=0.04, r=r)


def test_bs_call_price_positive():
    """BS call price must be strictly positive for in-the-money and ATM contracts."""
    np.random.seed(0)
    for K in [80.0, 100.0, 120.0]:
        for sigma in [0.1, 0.3, 0.5]:
            price = bs_call_price(S=S, K=K, tau=1.0, r=r, sigma=sigma)
            assert price > 0.0, (
                f"BS price ≤ 0 for K={K}, sigma={sigma}: got {price}"
            )


def test_bs_implied_vol_roundtrip():
    """Implied vol must recover the original sigma to within tolerance."""
    np.random.seed(1)
    tol   = 1e-5
    cases = [(100.0, 0.20), (90.0, 0.30), (110.0, 0.15), (100.0, 0.05)]
    for K, sigma_true in cases:
        tau   = 0.5
        price = bs_call_price(S=S, K=K, tau=tau, r=r, sigma=sigma_true)
        sigma_rec = bs_implied_vol(price, S=S, K=K, tau=tau, r=r, tol=tol)
        assert sigma_rec is not None, (
            f"bs_implied_vol returned None for K={K}, sigma={sigma_true}"
        )
        assert abs(sigma_rec - sigma_true) < 10 * tol, (
            f"Roundtrip error {abs(sigma_rec - sigma_true):.2e} > {10*tol} "
            f"for K={K}, sigma_true={sigma_true}"
        )


def test_heston_call_price_positive():
    """Heston call price must be strictly positive."""
    np.random.seed(2)
    kappa_rn = _PARAMS.kappa + _PARAMS.lam
    theta_rn = _PARAMS.kappa * _PARAMS.theta / kappa_rn
    for K in [90.0, 100.0, 110.0]:
        price = heston_call_price(
            S=S, K=K, tau=1.0, r=r, V=0.04,
            kappa_rn=kappa_rn, theta_rn=theta_rn,
            sigma_v=_PARAMS.sigma_v, rho=_PARAMS.rho,
        )
        assert price > 0.0, f"Heston price ≤ 0 for K={K}: got {price}"


def test_option_grid_shapes():
    """generate_option_grid must return 147 contracts with default settings."""
    np.random.seed(3)
    strikes, mats = generate_option_grid(V_t=0.04, S_t=S)
    # Default: 7 maturities × 21 strikes = 147
    assert strikes.shape == (147,), f"strikes shape {strikes.shape} != (147,)"
    assert mats.shape    == (147,), f"maturities shape {mats.shape} != (147,)"
    assert np.all(strikes > 0), "Some strikes are non-positive"
    assert np.all(mats > 0),    "Some maturities are non-positive"


def test_heston_call_put_parity():
    """Call price derived put (via parity C - P = S - K·e^{-rτ}) must be ≥ 0."""
    np.random.seed(4)
    kappa_rn = _PARAMS.kappa + _PARAMS.lam
    theta_rn = _PARAMS.kappa * _PARAMS.theta / kappa_rn
    K, tau   = 105.0, 0.5

    C = heston_call_price(
        S=S, K=K, tau=tau, r=r, V=0.04,
        kappa_rn=kappa_rn, theta_rn=theta_rn,
        sigma_v=_PARAMS.sigma_v, rho=_PARAMS.rho,
    )
    # Put-call parity: P = C - S + K*e^{-r*tau}
    P_parity = C - S + K * np.exp(-r * tau)

    # P must be ≥ 0 (no-arbitrage) with small numerical tolerance
    assert P_parity >= -1e-4, (
        f"Put-call parity violated: P_parity = {P_parity:.6f} < 0"
    )
    # Also verify call is within no-arb bounds
    intrinsic = max(S - K * np.exp(-r * tau), 0.0)
    assert C >= intrinsic - 1e-4, (
        f"Call {C:.4f} < intrinsic lower bound {intrinsic:.4f}"
    )
