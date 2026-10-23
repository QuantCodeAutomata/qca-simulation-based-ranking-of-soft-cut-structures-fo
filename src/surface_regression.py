"""Cross-sectional polynomial regression of the implied-volatility surface.

For each trading day the IV surface is parameterised as a bivariate polynomial
in log-moneyness ``k = log(K/S)`` and time-to-maturity ``tau``:

.. math::

    \\hat{\\sigma}(k, \\tau) = \\sum_{j=0}^{J} \\sum_{l=0}^{L} c_{j,l}\\, \\tau^j\\, k^l

where :math:`J = \\text{MAT\\_ORDER} = 2` and :math:`L = \\text{MON\\_ORDER} = 3`,
giving :math:`(J+1)(L+1) = 12` coefficients per day.

Ordering convention: features are stacked as
``(j=0,l=0), (j=0,l=1), …, (j=0,L), (j=1,l=0), …, (j=J,L)``,
i.e. the outer loop is over ``j`` (maturity power) and the inner loop over
``l`` (moneyness power).
"""

from typing import Tuple

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)

# Polynomial orders (choice documented here; changing these propagates everywhere)
MAT_ORDER: int = 2   # maximum power of tau  → powers 0, 1, 2
MON_ORDER: int = 3   # maximum power of k    → powers 0, 1, 2, 3
N_FEATURES: int = (MAT_ORDER + 1) * (MON_ORDER + 1)  # = 12


def get_poly_features(tau: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Build the bivariate polynomial feature matrix.

    Parameters
    ----------
    tau:
        Array of shape ``(N,)`` with times to maturity in years.
    k:
        Array of shape ``(N,)`` with log-moneyness values ``log(K/S)``.

    Returns
    -------
    np.ndarray, shape (N, (MAT_ORDER+1)*(MON_ORDER+1))
        Feature matrix where column ``j*(MON_ORDER+1) + l`` corresponds to
        the basis function :math:`\\tau^j k^l`.
    """
    N = len(tau)
    X = np.empty((N, N_FEATURES))

    col = 0
    for j in range(MAT_ORDER + 1):
        tau_pow = tau ** j
        for l in range(MON_ORDER + 1):
            X[:, col] = tau_pow * k ** l
            col += 1

    return X


def fit_iv_surface(
    tau: np.ndarray,
    k: np.ndarray,
    iv: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Fit a cross-sectional polynomial OLS regression to the IV surface.

    Parameters
    ----------
    tau:
        Array of shape ``(N,)`` with times to maturity in years.
    k:
        Array of shape ``(N,)`` with log-moneyness values.
    iv:
        Array of shape ``(N,)`` with observed implied volatilities.

    Returns
    -------
    coefficients : np.ndarray, shape (N_FEATURES,)
        OLS coefficient vector ordered as described in the module docstring.
    r_squared : float
        In-sample coefficient of determination :math:`R^2`.

    Raises
    ------
    ValueError
        If ``iv`` contains only NaN values (nothing to fit).
    """
    mask = np.isfinite(iv) & np.isfinite(tau) & np.isfinite(k)
    if mask.sum() == 0:
        raise ValueError("All IV values are NaN; cannot fit surface.")

    X = get_poly_features(tau[mask], k[mask])
    y = iv[mask]

    coeffs, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)

    # Compute R²
    y_hat = X @ coeffs
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    if rank < N_FEATURES:
        # Rank deficiency is expected with the fixed 5×3 moneyness-maturity grid;
        # lstsq handles it via pseudoinverse and the solution is still valid.
        logger.debug(
            "fit_iv_surface: rank-deficient system (rank=%d < %d) — using lstsq pseudoinverse.",
            rank,
            N_FEATURES,
        )

    return coeffs, float(r2)


def compute_daily_iv_features(
    daily_ivols: np.ndarray,
    strikes: np.ndarray,
    maturities_years: np.ndarray,
    S: float,
) -> np.ndarray:
    """Fit the IV surface polynomial for a single day and return coefficients.

    Computes log-moneyness ``k = log(K / S)`` for each contract, then fits
    a bivariate polynomial OLS regression to the IV surface.

    Parameters
    ----------
    daily_ivols:
        Array of shape ``(N_contracts,)`` with implied volatilities for the day.
        NaN entries are excluded from the fit.
    strikes:
        Array of shape ``(N_contracts,)`` with strike prices.
    maturities_years:
        Array of shape ``(N_contracts,)`` with times to maturity in years.
    S:
        Current spot price used to compute log-moneyness.

    Returns
    -------
    np.ndarray, shape (N_FEATURES,)
        OLS coefficient vector ``[c_{0,0}, c_{0,1}, …, c_{2,3}]``.
        A zero vector is returned (with a warning) if fitting fails.
    """
    k = np.log(strikes / S)

    try:
        coeffs, r2 = fit_iv_surface(maturities_years, k, daily_ivols)
    except ValueError as exc:
        logger.warning("compute_daily_iv_features: %s — returning zero vector.", exc)
        return np.zeros(N_FEATURES)

    logger.debug("compute_daily_iv_features: R²=%.4f", r2)
    return coeffs
