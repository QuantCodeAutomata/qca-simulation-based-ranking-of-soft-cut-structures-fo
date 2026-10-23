"""Generate synthetic Heston dataset for all experiments.

True parameters: theta=0.03, kappa=0.05, sigma_v=0.03, lambda=-0.02, rho=0, mu=0
T=1000 days, M_intra=78 intraday steps per day.

Saves to data/synthetic_dataset.npz with keys:
  log_prices      : (T+1,) daily log-prices
  variances       : (T+1,) daily end-of-day variances
  intraday_returns: (T, M_intra) intraday log-returns
  daily_returns   : (T,) daily log-returns
  daily_bv        : (T,) daily bipower variation
  iv_features     : (T, 12) daily IV surface coefficients (fixed 15-contract grid)
"""

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
from src.option_pricing import price_option_panel
from src.surface_regression import compute_daily_iv_features
from src.utils import set_seed, get_logger, Timer

logger = get_logger(__name__)

# -------------------------------------------------------------------
# Canonical experiment settings (mirror these exactly across all exp/)
# -------------------------------------------------------------------
TRUE_PARAMS = HestonParams(
    kappa=0.05, theta=0.03, sigma_v=0.03,
    lam=-0.02,  rho=0.0,    mu=0.0,
    V0=0.03,    r=0.02,
)
T = 1000       # total trading days
M_INTRA = 78   # 5-minute bars per day
SEED = 42

# Fixed 15-contract option grid (same as reference_table.py)
_LOG_MONEYNESS = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
_MATURITIES    = np.array([30 / 252, 60 / 252, 90 / 252])
S_FIXED        = 1.0   # normalised spot price for IV feature computation
_STRIKES       = S_FIXED * np.exp(_LOG_MONEYNESS)           # (5,)
_TAUS_GRID     = np.tile(_MATURITIES, len(_LOG_MONEYNESS))  # (15,) — repeats maturities
_STRIKES_GRID  = np.repeat(_STRIKES, len(_MATURITIES))      # (15,) — repeats strikes


def _price_and_iv_features(V_t: float) -> np.ndarray:
    """Price 15 options at variance V_t (S=1), return 12-dim IV surface coeffs."""
    _, ivols = price_option_panel(
        S=S_FIXED,
        V=V_t,
        params=TRUE_PARAMS,
        strikes=_STRIKES_GRID,
        maturities_years=_TAUS_GRID,
        r=TRUE_PARAMS.r,
    )
    iv_arr = np.where(
        np.isfinite(ivols) & (ivols > 0), ivols, np.nan
    )
    return compute_daily_iv_features(
        daily_ivols=iv_arr,
        strikes=_STRIKES_GRID,
        maturities_years=_TAUS_GRID,
        S=S_FIXED,
    )


def generate(out_path: str | None = None) -> dict:
    """Run the full simulation and return a dict of arrays.

    Parameters
    ----------
    out_path:
        If given, save as ``<out_path>`` (NPZ).  Defaults to
        ``<project_root>/data/synthetic_dataset.npz``.

    Returns
    -------
    dict with keys matching the module docstring.
    """
    set_seed(SEED)
    logger.info("Simulating Heston path: T=%d, M_intra=%d, seed=%d", T, M_INTRA, SEED)

    with Timer("Heston simulation", logger=logger):
        log_prices, variances, intraday_returns = simulate_heston_daily(
            params=TRUE_PARAMS, T=T, M_intra=M_INTRA, seed=SEED
        )

    daily_returns = compute_daily_returns(log_prices)
    daily_bv      = compute_bipower_variation(intraday_returns)

    logger.info("Computing per-day IV surface features (%d days)...", T)
    iv_rows = []
    with Timer("IV feature computation", logger=logger):
        for t in range(T):
            V_t = float(variances[t])     # variance at start of day t+1
            iv_rows.append(_price_and_iv_features(V_t))

    iv_features = np.array(iv_rows)   # (T, 12)

    data = {
        "log_prices":        log_prices,
        "variances":         variances,
        "intraday_returns":  intraday_returns,
        "daily_returns":     daily_returns,
        "daily_bv":          daily_bv,
        "iv_features":       iv_features,
    }

    if out_path is None:
        out_path = str(ROOT / "data" / "synthetic_dataset.npz")

    np.savez(out_path, **data)
    logger.info("Saved synthetic dataset to %s", out_path)
    return data


def load(path: str | None = None) -> dict:
    """Load previously generated synthetic dataset."""
    if path is None:
        path = str(ROOT / "data" / "synthetic_dataset.npz")
    raw = np.load(path)
    return {k: raw[k] for k in raw.files}


if __name__ == "__main__":
    generate()
    print("generate_synthetic.py complete.")
