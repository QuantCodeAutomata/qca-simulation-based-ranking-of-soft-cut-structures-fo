"""Heston stochastic volatility model simulator.

Uses the full-truncation Euler discretisation (Lord et al., 2010) to ensure
variance non-negativity throughout the path.

Model dynamics (log-price form):
    d log S_t = mu dt + sqrt(V_t) dW1_t
    dV_t = kappa*(theta - V_t) dt + sigma_v*sqrt(V_t) dW2_t
    Corr(dW1, dW2) = rho  (default rho=0)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)

_MU_2_OVER_PI = np.sqrt(2.0 / np.pi)  # constant used in bipower variation


@dataclass
class HestonParams:
    """Container for Heston model parameters.

    Attributes
    ----------
    kappa:
        Mean-reversion speed of variance.
    theta:
        Long-run (unconditional) variance.
    sigma_v:
        Volatility-of-volatility.
    lam:
        Market price of variance risk (lambda).  Used to shift to the
        risk-neutral measure: kappa_rn = kappa + lam,
        theta_rn = kappa * theta / (kappa + lam).
    rho:
        Instantaneous correlation between log-price and variance
        Brownian motions.  Defaults to 0 (independent).
    mu:
        Physical drift of log-price.  Defaults to 0.
    V0:
        Initial variance.  Defaults to 0.03.
    r:
        Continuously compounded risk-free rate.  Defaults to 0.02.
    """

    kappa: float
    theta: float
    sigma_v: float
    lam: float
    rho: float = 0.0
    mu: float = 0.0
    V0: float = 0.03
    r: float = 0.02


def simulate_heston_daily(
    params: HestonParams,
    T: int,
    M_intra: int = 78,
    dt_intra: float = 1.0 / 252.0 / 78.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate daily Heston model paths via intraday full-truncation Euler steps.

    Parameters
    ----------
    params:
        Heston model parameters.
    T:
        Number of trading days to simulate.
    M_intra:
        Number of intraday steps per day.  Default 78 corresponds to
        5-minute intervals over a 6.5-hour trading session.
    dt_intra:
        Length of each intraday step in years.  Default 1/(252*78).
    seed:
        Random seed for reproducibility.  If ``None`` the global state
        is not modified.

    Returns
    -------
    log_prices_daily : np.ndarray, shape (T+1,)
        End-of-day log-prices.  ``log_prices_daily[0] = 0`` (normalised).
    variances_daily : np.ndarray, shape (T+1,)
        End-of-day instantaneous variances.
    intraday_returns : np.ndarray, shape (T, M_intra)
        Intraday log-returns for each day and each step within the day.
    """
    if seed is not None:
        np.random.seed(seed)

    kappa = params.kappa
    theta = params.theta
    sigma_v = params.sigma_v
    mu = params.mu
    rho = params.rho
    sqrt_dt = np.sqrt(dt_intra)
    sqrt_1m_rho2 = np.sqrt(max(0.0, 1.0 - rho ** 2))

    # Pre-allocate output arrays
    log_prices_daily = np.empty(T + 1)
    variances_daily = np.empty(T + 1)
    intraday_returns = np.empty((T, M_intra))

    log_prices_daily[0] = 0.0
    variances_daily[0] = params.V0

    V = params.V0
    log_S = 0.0

    # Draw all random shocks at once for efficiency: shape (T*M_intra, 2)
    Z = np.random.standard_normal((T * M_intra, 2))

    step = 0
    for t in range(T):
        day_log_S_start = log_S
        for m in range(M_intra):
            Z1 = Z[step, 0]
            Z2 = Z[step, 1]
            step += 1

            V_pos = max(V, 0.0)  # full-truncation: clamp before drift/diffusion

            # Correlated Brownian increments
            dW1 = Z1 * sqrt_dt
            dW2 = (rho * Z1 + sqrt_1m_rho2 * Z2) * sqrt_dt

            # Log-price update
            log_ret = (mu - 0.5 * V_pos) * dt_intra + np.sqrt(V_pos) * dW1
            intraday_returns[t, m] = log_ret
            log_S += log_ret

            # Variance update (full-truncation Euler)
            V = V + kappa * (theta - V_pos) * dt_intra + sigma_v * np.sqrt(V_pos) * dW2
            V = max(V, 0.0)  # truncate negative values after the step

        log_prices_daily[t + 1] = log_S
        variances_daily[t + 1] = V

    logger.debug(
        "simulate_heston_daily: T=%d, M_intra=%d, final_V=%.6f", T, M_intra, V
    )
    return log_prices_daily, variances_daily, intraday_returns


def compute_bipower_variation(intraday_returns: np.ndarray) -> np.ndarray:
    """Compute daily bipower variation (Barndorff-Nielsen & Shephard, 2004).

    .. math::

        BV_t = \\frac{\\sqrt{2/\\pi} \\cdot M}{M-1}
               \\sum_{i=2}^{M} |r_{t,i}| \\cdot |r_{t,i-1}|

    Parameters
    ----------
    intraday_returns:
        Array of shape ``(T, M)`` containing intraday log-returns.

    Returns
    -------
    np.ndarray, shape (T,)
        Daily bipower variation estimates.
    """
    abs_r = np.abs(intraday_returns)          # (T, M)
    M = abs_r.shape[1]
    scale = _MU_2_OVER_PI * M / (M - 1)
    bv = scale * np.sum(abs_r[:, 1:] * abs_r[:, :-1], axis=1)
    return bv


def compute_daily_returns(log_prices: np.ndarray) -> np.ndarray:
    """Compute daily log-returns from a log-price path.

    Parameters
    ----------
    log_prices:
        Array of shape ``(T+1,)`` of daily log-prices.

    Returns
    -------
    np.ndarray, shape (T,)
        Daily log-returns ``r_t = log(S_t) - log(S_{t-1})``.
    """
    return np.diff(log_prices)
