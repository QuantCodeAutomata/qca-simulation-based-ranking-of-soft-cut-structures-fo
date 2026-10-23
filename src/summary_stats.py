import numpy as np
from scipy import stats
from typing import Tuple
from arch import arch_model


# GARCH(1,1) score statistic:
# Fit GARCH(1,1) to returns, then compute score as:
#   score = mean(e_t^2 / h_t - 1)
# where e_t = r_t - mu (raw residual), h_t = conditional variance.
# Returns are scaled by 100 for numerical stability before fitting;
# the ratio e^2/h is scale-invariant.


def _lag1_autocorr(x: np.ndarray) -> float:
    """Lag-1 autocorrelation via Pearson r of (x[:-1], x[1:]), NaN-safe."""
    if len(x) < 3:
        return 0.0
    mask = np.isfinite(x)
    # need at least two consecutive valid observations
    pair_ok = mask[:-1] & mask[1:]
    if pair_ok.sum() < 3:
        return 0.0
    a, b = x[:-1][pair_ok], x[1:][pair_ok]
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def compute_return_summaries(
    returns: np.ndarray,  # (T,) daily returns
) -> np.ndarray:
    """
    Compute return summary statistics:
    1. mean(r_t)
    2. variance(r_t)
    3. skewness(r_t)
    4. kurtosis(r_t) (excess)
    5. GARCH(1,1) score: mean(e_t^2/h_t - 1) from fitted GARCH
    Returns: (5,) array
    """
    mu = float(np.mean(returns))
    var = float(np.var(returns, ddof=1))
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=True))  # excess kurtosis

    # GARCH(1,1) score
    try:
        scaled = returns * 100.0
        am = arch_model(scaled, vol='GARCH', p=1, q=1, dist='normal', mean='Constant')
        res = am.fit(disp='off', show_warning=False)
        eps = res.resid
        h = res.conditional_volatility ** 2
        safe_h = np.where(h > 1e-12, h, 1e-12)
        garch_score = float(np.mean(eps ** 2 / safe_h - 1.0))
        if not np.isfinite(garch_score):
            garch_score = 0.0
    except Exception:
        garch_score = 0.0

    return np.array([mu, var, skew, kurt, garch_score])


def compute_bv_summaries(
    bv: np.ndarray,  # (T,) daily bipower variation
) -> np.ndarray:
    """
    Compute BV summary statistics:
    1. mean(BV_t)
    2. variance(BV_t)
    3. skewness(BV_t)
    4. kurtosis(BV_t)
    5. lag-1 autocorrelation of BV_t
    Returns: (5,) array
    """
    mu = float(np.mean(bv))
    var = float(np.var(bv, ddof=1))
    skew = float(stats.skew(bv))
    kurt = float(stats.kurtosis(bv, fisher=True))
    ac1 = _lag1_autocorr(bv)

    return np.array([mu, var, skew, kurt, ac1])


def compute_iv_surface_summaries(
    iv_features: np.ndarray,  # (T, n_features) daily IV surface coefficients
) -> np.ndarray:
    """
    For each feature dimension, compute:
    1. mean
    2. variance
    3. skewness
    4. kurtosis
    5. lag-1 autocorrelation
    Returns: (n_features * 5,) array

    NaN rows (non-sampled days) are silently dropped so this function is
    safe whether iv_features contains all T days (with NaN padding) or
    only the subsampled valid rows.
    """
    T, n_features = iv_features.shape
    result = np.zeros(n_features * 5)
    for k in range(n_features):
        col = iv_features[:, k]
        valid = col[np.isfinite(col)]
        idx = k * 5
        if valid.size < 2:
            # Not enough data; leave zeros
            continue
        result[idx + 0] = float(np.mean(valid))
        result[idx + 1] = float(np.var(valid, ddof=min(1, valid.size - 1)))
        result[idx + 2] = float(stats.skew(valid))
        result[idx + 3] = float(stats.kurtosis(valid, fisher=True))
        result[idx + 4] = _lag1_autocorr(col)   # NaN-safe, uses full original col

    return result


def compute_full_summary_vector(
    returns: np.ndarray,
    bv: np.ndarray,
    iv_features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute all summaries and return (S_r, S_HF, S_O, S_all)
    where S_r = return summaries (5,)
          S_HF = BV summaries (5,)
          S_O = IV surface summaries (n_features*5,)
          S_all = concatenation
    """
    S_r = compute_return_summaries(returns)
    S_HF = compute_bv_summaries(bv)
    S_O = compute_iv_surface_summaries(iv_features)
    S_all = np.concatenate([S_r, S_HF, S_O])
    return S_r, S_HF, S_O, S_all
