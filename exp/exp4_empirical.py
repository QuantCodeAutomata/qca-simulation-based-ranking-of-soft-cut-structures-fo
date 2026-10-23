"""Experiment 4 – Semi-empirical application with a rolling 500-day window.

Since real S&P 500 option data (CBOE DataShop) is unavailable, this experiment
uses synthetic data generated from a realistic parameter set that mimics the
empirical characteristics of S&P 500 data (higher vol-of-vol, negative skew
correlation, and a meaningful equity risk premium).

Note: The parameter values below are chosen to match stylised facts of
S&P 500 data (Bakshi, Cao & Chen, 1997; Eraker, 2004) rather than the
idealised θ_true used in Experiments 1–3.

Procedure
---------
1. Generate 2 500 days of semi-empirical Heston data.
2. Rolling estimation: for each window w = 0, 1, 2, …
     - Estimation window: days [w × STEP, w × STEP + 500)
     - Evaluation day:    w × STEP + 500
   Total rolls: (2500 - 500) / STEP
3. For each roll:
   a. Compute S_obs from the 500-day window.
   b. Use the pre-built reference table and GMM to draw posteriors.
   c. Propagate the particle cloud one step to predict the next day's variance.
   d. Check if the true option price lies within the predictive interval.
4. Aggregate coverage over all rolls.

Outputs (results/)
------------------
exp4_coverage_table.csv
    window, maturity, nominal_level, empirical_coverage
exp4_coverage_by_maturity.png
exp4_rolling_parameters.png
    Time series of posterior-mean parameters across rolling windows.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.heston_simulator import (
    HestonParams, simulate_heston_daily,
    compute_bipower_variation, compute_daily_returns,
)
from src.option_pricing import price_option_panel_fast
from src.surface_regression import compute_daily_iv_features
from src.summary_stats import compute_full_summary_vector
from src.reference_table import (
    generate_reference_table, REFERENCE_TABLE_SIZE,
    save_reference_table, load_reference_table,
    OPTION_SUBSAMPLE, _LOG_MONEYNESS as _REF_LOG_MONEYNESS, _MATURITIES as _REF_MATURITIES,
)
from src.gmm_abi import fit_gmm_bic
from src.cut_structures import (
    enumerate_feasible_cuts,
    compute_cut_posterior_samples,
    compute_posterior_mass,
    rank_cuts_by_posterior_mass,
)
from src.particle_filter import bootstrap_particle_filter, propagate_particles_one_step
from src.utils import set_seed, get_logger, Timer

logger = get_logger(__name__)

# -------------------------------------------------------------------
# Realistic (semi-empirical) parameters
# -------------------------------------------------------------------
# Chosen to replicate S&P 500 stylised facts:
#   negative vol-return correlation (leverage effect), higher vol-of-vol,
#   negative market price of variance risk (upward-sloping IV term structure)
EMPIRICAL_PARAMS = HestonParams(
    kappa=1.15,    # faster mean-reversion than synthetic baseline
    theta=0.04,    # slightly higher long-run variance
    sigma_v=0.39,  # large vol-of-vol (fat tails)
    lam=-0.40,     # large negative risk premium (steep IV skew)
    rho=-0.70,     # strong leverage effect
    mu=0.06,       # annual drift ≈ 6% (equity risk premium)
    V0=0.04,
    r=0.02,
)

T_LONG     = 2500   # total semi-empirical days
T_WINDOW   = 500    # estimation window
ROLL_STEP  = 100    # days between consecutive windows
SEED       = 42
N_POSTERIOR = 50
N_PARTICLES = 100
MAX_GMM_COMPS = 12
NOMINAL_LEVELS = [0.80, 0.90, 0.95]
REF_MATURITIES = {"short": 30/252, "medium": 60/252, "long": 90/252}
N_PARAMS    = 4
BLOCK_SIZES = {"S_r": 5, "S_HF": 5, "S_O": 60}

_LOG_MONEYNESS = _REF_LOG_MONEYNESS
_MATURITIES    = _REF_MATURITIES
S_FIXED        = 1.0
_STRIKES       = S_FIXED * np.exp(_LOG_MONEYNESS)
_TAUS          = np.tile(_MATURITIES, len(_LOG_MONEYNESS))
_STRIKES_FULL  = np.repeat(_STRIKES, len(_MATURITIES))

RESULTS_DIR = ROOT / "results"
DATA_DIR    = ROOT / "data"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _window_summary(log_prices, variances, intraday_returns,
                    start: int, length: int, params: HestonParams) -> np.ndarray:
    """Compute summary from a window [start, start+length) with option subsampling."""
    end = start + length
    rets = compute_daily_returns(log_prices[start:end + 1])
    bv   = compute_bipower_variation(intraday_returns[start:end])
    # Only append rows for sampled days — matches reference_table.py exactly
    iv_rows = []
    _fallback_iv = None
    for t in range(0, length, OPTION_SUBSAMPLE):
        V_t = float(variances[start + t])
        try:
            _, ivols = price_option_panel_fast(
                S=S_FIXED, V=V_t, params=params,
                strikes=_STRIKES_FULL, maturities_years=_TAUS, r=params.r,
            )
            iv_arr = np.where(np.isfinite(ivols) & (ivols > 0), ivols, np.nan)
            coeffs = compute_daily_iv_features(iv_arr, _STRIKES_FULL, _TAUS, S_FIXED)
            if np.all(np.isfinite(coeffs)):
                _fallback_iv = coeffs
        except Exception:
            coeffs = _fallback_iv if _fallback_iv is not None else np.zeros(12)
        iv_rows.append(coeffs)
    iv_features = np.array(iv_rows)   # (n_sampled, 12)
    _, _, _, S_all = compute_full_summary_vector(rets, bv, iv_features)
    return np.where(np.isfinite(S_all), S_all, 0.0)


def _fit_normaliser(joint):
    mu  = joint.mean(axis=0)
    std = joint.std(axis=0).clip(1e-8)
    return mu, std


def _normalise(x, mu, std):
    return (x - mu) / std


# -------------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------------

def run_exp4():
    set_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Generate long semi-empirical dataset
    # ----------------------------------------------------------------
    emp_dataset_path = DATA_DIR / "empirical_dataset.npz"
    if emp_dataset_path.exists():
        logger.info("Loading semi-empirical dataset…")
        raw = np.load(emp_dataset_path)
        log_prices, variances, intraday_returns = (
            raw["log_prices"], raw["variances"], raw["intraday_returns"]
        )
    else:
        logger.info("Generating semi-empirical dataset (T=%d)…", T_LONG)
        with Timer("semi-empirical simulation", logger=logger):
            log_prices, variances, intraday_returns = simulate_heston_daily(
                params=EMPIRICAL_PARAMS, T=T_LONG, M_intra=78, seed=SEED
            )
        np.savez(emp_dataset_path,
                 log_prices=log_prices, variances=variances,
                 intraday_returns=intraday_returns)
        logger.info("Saved semi-empirical dataset.")

    # ----------------------------------------------------------------
    # 2. Reference table + GMM (re-use or build)
    # ----------------------------------------------------------------
    ref_path = DATA_DIR / "reference_table.npz"
    if ref_path.exists():
        params_tbl, summaries_tbl = load_reference_table(str(ref_path))
    else:
        params_tbl, summaries_tbl = generate_reference_table(
            R=REFERENCE_TABLE_SIZE, T_window=T_WINDOW, M_intra=78,
            seed=SEED, verbose=True,
        )
        save_reference_table(params_tbl, summaries_tbl, str(ref_path))

    joint = np.hstack([params_tbl, summaries_tbl])
    mu_norm, std_norm = _fit_normaliser(joint)
    joint_norm = _normalise(joint, mu_norm, std_norm)
    gmm = fit_gmm_bic(joint_norm, max_components=MAX_GMM_COMPS, seed=SEED)

    # Get best cut structure
    exp1_csv = RESULTS_DIR / "exp1_cut_rankings.csv"
    cuts_all  = enumerate_feasible_cuts()
    if exp1_csv.exists():
        df_ranks   = pd.read_csv(exp1_csv)
        best_cut_id = int(df_ranks.iloc[0]["cut_id"])
    else:
        # Fallback: use first estimation window to rank
        S_obs0 = _window_summary(log_prices, variances, intraday_returns,
                                  0, T_WINDOW, EMPIRICAL_PARAMS)
        S_obs0_norm = _normalise(
            np.concatenate([np.zeros(N_PARAMS), S_obs0]), mu_norm, std_norm
        )[N_PARAMS:]
        mass_vals = []
        for cut in cuts_all:
            try:
                samps = compute_cut_posterior_samples(
                    cut=cut, gmm=gmm, S_obs_full=S_obs0_norm,
                    n_params=N_PARAMS, summary_block_sizes=BLOCK_SIZES,
                    n_samples=200, seed=SEED,
                )
                samps_raw = samps * std_norm[:N_PARAMS] + mu_norm[:N_PARAMS]
                mass_vals.append(compute_posterior_mass(samps_raw))
            except Exception:
                mass_vals.append(0.0)
        ranked     = rank_cuts_by_posterior_mass(cuts_all, mass_vals)
        best_cut_id = ranked[0][2].cut_id

    best_cut = next(c for c in cuts_all if c.cut_id == best_cut_id)
    logger.info("Using cut_id=%d for rolling experiment.", best_cut_id)

    # ----------------------------------------------------------------
    # 3. Rolling windows
    # ----------------------------------------------------------------
    n_rolls = (T_LONG - T_WINDOW) // ROLL_STEP
    rng = np.random.default_rng(SEED)
    coverage_records = []
    param_means      = []   # for the parameter time series plot

    logger.info("Running %d rolling windows…", n_rolls)

    for w in range(n_rolls):
        start = w * ROLL_STEP
        eval_day = start + T_WINDOW   # one-step-ahead evaluation day

        logger.debug("Window %d/%d: days %d–%d, eval day %d",
                     w + 1, n_rolls, start, start + T_WINDOW - 1, eval_day)

        # --- Observed summaries from this window ---
        try:
            S_obs_w = _window_summary(log_prices, variances, intraday_returns,
                                       start, T_WINDOW, EMPIRICAL_PARAMS)
        except Exception as exc:
            logger.warning("Window %d summary failed: %s", w, exc)
            continue

        S_obs_w_norm = _normalise(
            np.concatenate([np.zeros(N_PARAMS), S_obs_w]), mu_norm, std_norm
        )[N_PARAMS:]

        # --- Posterior draws ---
        try:
            samps_norm = compute_cut_posterior_samples(
                cut=best_cut, gmm=gmm, S_obs_full=S_obs_w_norm,
                n_params=N_PARAMS, summary_block_sizes=BLOCK_SIZES,
                n_samples=N_POSTERIOR, seed=int(rng.integers(0, 2**31)),
            )
            param_draws = samps_norm * std_norm[:N_PARAMS] + mu_norm[:N_PARAMS]
            param_draws[:, 0] = np.clip(param_draws[:, 0], 1e-4, 5.0)
            param_draws[:, 1] = np.clip(param_draws[:, 1], 1e-4, 0.5)
            param_draws[:, 2] = np.clip(param_draws[:, 2], 1e-4, 1.0)
            param_draws[:, 3] = np.clip(param_draws[:, 3], -2.0, 0.0)
        except Exception as exc:
            logger.warning("Window %d posterior failed: %s", w, exc)
            continue

        param_means.append({
            "window": w,
            "start_day": start,
            "kappa_mean": float(param_draws[:, 0].mean()),
            "theta_mean": float(param_draws[:, 1].mean()),
            "sigma_v_mean": float(param_draws[:, 2].mean()),
            "lam_mean": float(param_draws[:, 3].mean()),
        })

        # --- One-step-ahead predictive coverage ---
        train_rets = compute_daily_returns(log_prices[start:start + T_WINDOW + 1])
        true_V_eval = float(variances[eval_day])

        pf_finals = []
        for j in range(len(param_draws)):
            kp, th, sv, lm = param_draws[j]
            p_j = HestonParams(kappa=kp, theta=th, sigma_v=sv, lam=lm,
                               rho=0.0, mu=EMPIRICAL_PARAMS.mu,
                               V0=EMPIRICAL_PARAMS.V0, r=EMPIRICAL_PARAMS.r)
            fv, _ = bootstrap_particle_filter(train_rets, p_j, N_PARTICLES,
                                              seed=int(rng.integers(0, 2**31)))
            pf_finals.append(fv[-1].copy())

        pred_Vs = []
        for j in range(len(param_draws)):
            kp, th, sv, lm = param_draws[j]
            p_j = HestonParams(kappa=kp, theta=th, sigma_v=sv, lam=lm,
                               rho=0.0, mu=EMPIRICAL_PARAMS.mu,
                               V0=EMPIRICAL_PARAMS.V0, r=EMPIRICAL_PARAMS.r)
            nxt = propagate_particles_one_step(pf_finals[j], p_j,
                                               seed=int(rng.integers(0, 2**31)))
            pred_Vs.append(float(np.mean(nxt)))

        pred_ivs = np.sqrt(np.maximum(pred_Vs, 1e-10))
        true_iv  = np.sqrt(max(true_V_eval, 1e-10))

        for mat_name in REF_MATURITIES:
            for alpha in NOMINAL_LEVELS:
                lo = np.quantile(pred_ivs, (1 - alpha) / 2)
                hi = np.quantile(pred_ivs, 1 - (1 - alpha) / 2)
                coverage_records.append({
                    "window": w, "maturity": mat_name,
                    "nominal": alpha,
                    "covered": int(lo <= true_iv <= hi),
                    "width": hi - lo,
                })

    # ----------------------------------------------------------------
    # 4. Save and plot
    # ----------------------------------------------------------------
    df_cov = pd.DataFrame(coverage_records)
    summary = (df_cov.groupby(["maturity", "nominal"])
               .agg(empirical_coverage=("covered", "mean"),
                    avg_width=("width", "mean"))
               .reset_index())
    summary.insert(0, "dataset", "semi-empirical")

    csv_path = RESULTS_DIR / "exp4_coverage_table.csv"
    summary.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)
    print(summary.to_string(index=False))

    # Plot 1 – coverage by maturity
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle("Exp 4 – Semi-Empirical Predictive Coverage (Rolling Windows)")
    for ax, mat_name in zip(axes, REF_MATURITIES):
        sub = summary[summary["maturity"] == mat_name]
        x   = np.arange(len(NOMINAL_LEVELS))
        ax.bar(x, sub["empirical_coverage"], color="steelblue", alpha=0.8,
               label="Empirical")
        for i, a in enumerate(NOMINAL_LEVELS):
            ax.bar(x[i], a, color="none", edgecolor="red",
                   linewidth=2, label="Nominal" if i == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(a*100)}%" for a in NOMINAL_LEVELS])
        ax.set_title(f"Maturity: {mat_name}")
        ax.set_ylabel("Coverage")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp4_coverage_by_maturity.png", dpi=120)
    plt.close()

    # Plot 2 – rolling parameter means
    if param_means:
        df_pm = pd.DataFrame(param_means)
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        fig.suptitle("Exp 4 – Posterior-Mean Parameters Across Rolling Windows")
        for ax, col, label, true_val in zip(
            axes.flat,
            ["kappa_mean", "theta_mean", "sigma_v_mean", "lam_mean"],
            ["κ", "θ", "σ_v", "λ"],
            [EMPIRICAL_PARAMS.kappa, EMPIRICAL_PARAMS.theta,
             EMPIRICAL_PARAMS.sigma_v, EMPIRICAL_PARAMS.lam],
        ):
            ax.plot(df_pm["start_day"], df_pm[col], label=f"Posterior mean {label}")
            ax.axhline(true_val, color="red", linestyle="--",
                       label=f"True {label}={true_val}")
            ax.set_xlabel("Window start day")
            ax.set_ylabel(label)
            ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "exp4_rolling_parameters.png", dpi=120)
        plt.close()

    logger.info("Experiment 4 complete.")
    return summary


if __name__ == "__main__":
    run_exp4()
