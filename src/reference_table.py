import numpy as np
from typing import Tuple, Optional, Dict
import pandas as pd

# Reference table size (implementer choice).
# Set to 300 for runtime feasibility (~3 min on a single core); increase for
# higher statistical fidelity.
REFERENCE_TABLE_SIZE = 300

# Implementer choice: price options every OPTION_SUBSAMPLE-th day to cut
# reference-table build time.  Summary statistics are computed over the
# subsampled IV-feature time series.  Value 25 → 20 pricing days per 500-day
# window (~20 ms per contract panel), running time ≈ 2–3 min on a single core.
OPTION_SUBSAMPLE = 25

# Prior distributions (from paper):
#   kappa ~ U(0, 0.1)
#   theta ~ U(0, 0.1)
#   sigma_v | kappa, theta ~ U(0, sqrt(2*kappa*theta))   [Feller-bounded]
#   lambda | kappa ~ U(-kappa, 0)
#   V0 fixed at 0.03 (same as theta_true)
#
# Implementer note: sigma_v ~ U(0, sqrt(2*kappa*theta)) ensures the Feller
# condition can hold; when kappa=0 or theta=0, sigma_v degenerates to 0.
# lambda ~ U(-kappa, 0) is parameterised as lambda = -kappa * u, u ~ U(0,1).

# Fixed simulation settings
V0_FIXED = 0.03
R_RATE = 0.02
MU_DRIFT = 0.0

# Fixed option grid used throughout reference-table generation.
# Log-moneyness m = log(K/S): 5 levels; maturities in years: 3 levels.
_LOG_MONEYNESS = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
_MATURITIES = np.array([30 / 252, 60 / 252, 90 / 252])


def sample_prior(
    n: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample from the joint prior p(Phi).
    Returns: (n, 4) array with columns [kappa, theta, sigma_v, lambda]
    Enforces: sigma_v in [0, sqrt(2*kappa*theta)], lambda in [-kappa, 0]
    """
    rng = np.random.default_rng(seed)

    kappa = rng.uniform(0.0, 0.1, size=n)
    theta = rng.uniform(0.0, 0.1, size=n)

    # sigma_v | kappa, theta ~ U(0, sqrt(2*kappa*theta))
    upper_sv = np.sqrt(2.0 * kappa * theta)
    u_sv = rng.uniform(0.0, 1.0, size=n)
    sigma_v = upper_sv * u_sv  # = 0 when upper_sv = 0

    # lambda | kappa ~ U(-kappa, 0)  =>  lambda = -kappa * u, u ~ U(0,1)
    u_lam = rng.uniform(0.0, 1.0, size=n)
    lam = -kappa * u_lam

    return np.column_stack([kappa, theta, sigma_v, lam])


def generate_reference_table(
    R: int = REFERENCE_TABLE_SIZE,
    T_window: int = 500,
    M_intra: int = 78,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the offline reference table.

    For each i = 1, ..., R:
      1. Sample Phi^(i) from prior
      2. Simulate T_window-day pseudo-dataset using the MISSPECIFIED simulator:
         - Fine-grid BV from intraday simulated returns (no microstructure noise)
         - Theoretical Heston option prices at fixed grid (no pricing-error model)
      3. Compute full summary vector S^(i)

    Sibling module APIs used (see AGENTS.md):
      heston_simulator:   HestonParams, simulate_heston_daily, compute_bipower_variation,
                          compute_daily_returns
      option_pricing:     price_option_panel
      surface_regression: compute_daily_iv_features
      summary_stats:      compute_full_summary_vector

    Returns: (params_table, summaries_table)
      params_table:    (R, 4)   [kappa, theta, sigma_v, lambda]
      summaries_table: (R, D_S) summary vectors
    """
    # Lazy imports: allow module to be imported even when siblings are absent.
    from src.heston_simulator import (
        HestonParams,
        simulate_heston_daily,
        compute_bipower_variation,
        compute_daily_returns,
    )
    from src.option_pricing import price_option_panel_fast      # vectorised
    from src.surface_regression import compute_daily_iv_features
    from src.summary_stats import compute_full_summary_vector

    rng = np.random.default_rng(seed)
    prior_samples = sample_prior(R, seed=int(rng.integers(0, 2**31)))

    # Fixed option grid (unit spot S=1).
    # Build the full Cartesian product: 5 strikes × 3 maturities = 15 contracts.
    # price_option_panel_fast and compute_daily_iv_features both require equal-
    # length (N_contracts,) arrays.
    S0 = 1.0
    _strikes_base = S0 * np.exp(_LOG_MONEYNESS)   # (5,)
    # Each strike pairs with every maturity:  repeat outer, tile inner
    strikes_full = np.repeat(_strikes_base, len(_MATURITIES))  # (15,)
    taus_full    = np.tile(_MATURITIES, len(_strikes_base))     # (15,)

    summaries_list = []
    fallback_coeffs: Optional[np.ndarray] = None  # placeholder until first success

    for i in range(R):
        if verbose and (i % 200 == 0):
            print(f"  Reference table: {i}/{R}")

        kappa_i, theta_i, sv_i, lam_i = prior_samples[i]

        # HestonParams uses field name 'lam' (not 'lambda_'), see AGENTS.md
        params = HestonParams(
            kappa=kappa_i,
            theta=theta_i,
            sigma_v=sv_i,
            lam=lam_i,
            rho=0.0,   # rho not estimated; fixed at 0
            mu=MU_DRIFT,
            V0=V0_FIXED,
            r=R_RATE,
        )

        sim_seed = int(rng.integers(0, 2**31))

        try:
            log_prices, variances, intraday_rets = simulate_heston_daily(
                params=params,
                T=T_window,
                M_intra=M_intra,
                seed=sim_seed,
            )
            daily_returns = compute_daily_returns(log_prices)   # (T_window,)
            daily_bv = compute_bipower_variation(intraday_rets)  # (T_window,)
        except Exception:
            summaries_list.append(None)
            continue

        # Per-day IV surface features via theoretical Heston pricing.
        # Implementer choice: subsample every OPTION_SUBSAMPLE days to cut
        # runtime; summary statistics are computed over the subsampled series.
        sample_days = list(range(0, T_window, OPTION_SUBSAMPLE))
        iv_feature_rows = []
        fallback_coeffs = None  # reset per simulation row

        for t in sample_days:
            V_t = float(variances[t])
            try:
                # Vectorised fast pricer; returns np.ndarray of ivols
                _, ivols = price_option_panel_fast(
                    S=S0,
                    V=V_t,
                    params=params,
                    strikes=strikes_full,
                    maturities_years=taus_full,
                    r=R_RATE,
                )
                iv_arr = np.where(
                    np.isfinite(ivols) & (ivols > 0), ivols, np.nan
                )
                coeffs_t = compute_daily_iv_features(
                    daily_ivols=iv_arr,
                    strikes=strikes_full,
                    maturities_years=taus_full,
                    S=S0,
                )
                if not np.all(np.isfinite(coeffs_t)):
                    raise ValueError("non-finite coefficients")
                fallback_coeffs = coeffs_t
            except Exception:
                if fallback_coeffs is not None:
                    coeffs_t = fallback_coeffs   # carry-forward last good row
                else:
                    coeffs_t = np.zeros(12)      # 12-feature surface (MAT×MON orders)

            iv_feature_rows.append(coeffs_t)

        iv_features = np.array(iv_feature_rows)  # (n_sample_days, n_features)

        try:
            _, _, _, S_all = compute_full_summary_vector(
                returns=daily_returns,
                bv=daily_bv,
                iv_features=iv_features,
            )
            S_all = np.where(np.isfinite(S_all), S_all, 0.0)
            summaries_list.append(S_all)
        except Exception:
            summaries_list.append(None)

    # Drop failed rows
    valid_mask = [s is not None for s in summaries_list]
    params_table = prior_samples[valid_mask]
    summaries_table = np.array([s for s in summaries_list if s is not None])

    if verbose:
        print(f"Reference table complete: {len(params_table)}/{R} valid rows.")

    return params_table, summaries_table


def save_reference_table(
    params: np.ndarray,
    summaries: np.ndarray,
    path: str,
) -> None:
    """Save reference table to npz file."""
    np.savez(path, params=params, summaries=summaries)


def load_reference_table(
    path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load reference table from npz file."""
    data = np.load(path)
    return data['params'], data['summaries']
