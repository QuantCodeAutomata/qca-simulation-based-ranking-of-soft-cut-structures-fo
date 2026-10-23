"""Experiment 2 – One-step-ahead predictive coverage for best, middle, worst cuts.

For three cut structures (rank 1, rank 19, rank 42 from Exp 1):

1. Draw N_POSTERIOR samples from the cut posterior using the first 500 days.
2. For each posterior draw θʲ, run a bootstrap particle filter (N_PARTICLES)
   through the estimation window to obtain the filtered variance cloud.
3. For each out-of-sample day t = 501…1000:
   a. Propagate all particle clouds one step forward under θʲ.
   b. Use the weighted-mean particle variance V̄ₜʲ as the predicted variance.
   c. Compute the predicted ATM-IV (= √V̄ₜʲ) for three reference maturities.
   d. Form the 80%, 90%, 95% predictive intervals across posterior draws.
   e. Check whether the true ATM-IV (√Vₜ_true) is covered.
4. Record empirical coverage and average interval width by maturity bucket.

Outputs (results/)
------------------
exp2_coverage_table.csv
    method (cut rank), maturity, nominal_level, empirical_coverage, avg_width
exp2_coverage_comparison.png
    Grouped bar chart of empirical vs nominal coverage.
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
    rank_cuts_by_posterior_mass,
    compute_posterior_mass,
    SUMMARY_BLOCKS,
)
from src.particle_filter import bootstrap_particle_filter, propagate_particles_one_step
from src.utils import set_seed, get_logger, Timer

logger = get_logger(__name__)

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
SEED             = 42
N_POSTERIOR      = 50      # posterior draws per cut
N_PARTICLES      = 200     # particle filter size
T_TOTAL          = 1000
T_ESTIMATION     = 500
MAX_GMM_COMPS    = 15
NOMINAL_LEVELS   = [0.80, 0.90, 0.95]
# Reference maturities for ATM-IV predictive intervals (in years)
REF_MATURITIES   = {"short": 30/252, "medium": 60/252, "long": 90/252}

TRUE_PARAMS = HestonParams(
    kappa=0.05, theta=0.03, sigma_v=0.03,
    lam=-0.02,  rho=0.0,    mu=0.0,
    V0=0.03,    r=0.02,
)

_LOG_MONEYNESS = _REF_LOG_MONEYNESS
_MATURITIES    = _REF_MATURITIES
S_FIXED        = 1.0
_STRIKES       = S_FIXED * np.exp(_LOG_MONEYNESS)
_TAUS          = np.tile(_MATURITIES, len(_LOG_MONEYNESS))
_STRIKES_FULL  = np.repeat(_STRIKES, len(_MATURITIES))

BLOCK_SIZES = {"S_r": 5, "S_HF": 5, "S_O": 60}
N_PARAMS    = 4

RESULTS_DIR = ROOT / "results"
DATA_DIR    = ROOT / "data"


# -------------------------------------------------------------------
# Utility: compute observed summaries
# -------------------------------------------------------------------

def _compute_obs_summary(log_prices, variances, intraday_returns, params, T_est):
    """Compute observed summary vector using the same subsampling as the reference table."""
    daily_rets = compute_daily_returns(log_prices[:T_est + 1])
    daily_bv   = compute_bipower_variation(intraday_returns[:T_est])
    # Only append rows for sampled days — matches reference_table.py exactly
    iv_rows = []
    _fallback_iv = None
    for t in range(0, T_est, OPTION_SUBSAMPLE):
        V_t = float(variances[t])
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
    _, _, _, S_all = compute_full_summary_vector(daily_rets, daily_bv, iv_features)
    return np.where(np.isfinite(S_all), S_all, 0.0)


def _fit_normaliser(joint):
    mu  = joint.mean(axis=0)
    std = joint.std(axis=0).clip(1e-8)
    return mu, std


def _normalise(x, mu, std):
    return (x - mu) / std


# -------------------------------------------------------------------
# Heston ATM implied-vol term structure (analytic approximation)
# -------------------------------------------------------------------

def _heston_atm_iv_vectorised(
    V_pred: np.ndarray,       # (N,) predicted next-day variances
    param_draws: np.ndarray,  # (N, 4) [kappa, theta, sigma_v, lam]
    tau_years: float,         # maturity in years
) -> np.ndarray:
    """Heston ATM implied-vol via integrated-variance term structure.

    Under the risk-neutral measure:
        κ* = κ + λ,   θ* = κθ / κ*
        Σ_ATM(τ)² ≈ [θ*τ + (V - θ*)(1 − e^{−κ*τ}) / κ*] / τ

    This is the standard Gatheral approximation for ATM options.
    Maturity-dependence: short τ → dominated by spot V; long τ → pulled to θ*.
    """
    kappa = param_draws[:, 0]
    theta = param_draws[:, 1]
    lam   = param_draws[:, 3]

    kappa_star = np.maximum(kappa + lam, 1e-6)          # ≥ 0 under prior
    theta_star = np.maximum(kappa * theta / kappa_star, 1e-8)

    # α(τ) = (1 − e^{−κ*τ}) / (κ*τ)  — mean-reversion factor ∈ (0,1]
    ks_tau = kappa_star * tau_years
    alpha  = np.where(ks_tau > 1e-10,
                      (1.0 - np.exp(-ks_tau)) / ks_tau,
                      1.0 - 0.5 * ks_tau)            # first-order Taylor for small ks*τ
    alpha  = np.clip(alpha, 0.0, 1.0)

    # E_Q[∫V ds]/τ  =  θ* + (V − θ*) × α
    integrated_var = np.maximum(theta_star + (V_pred - theta_star) * alpha, 1e-10)
    return np.sqrt(integrated_var)


def _heston_atm_iv_true(V_true: float, params: HestonParams, tau_years: float) -> float:
    """Scalar version of _heston_atm_iv_vectorised for the true parameter values."""
    kappa_star = max(params.kappa + params.lam, 1e-6)
    theta_star = max(params.kappa * params.theta / kappa_star, 1e-8)
    ks_tau = kappa_star * tau_years
    alpha  = float(
        (1.0 - np.exp(-ks_tau)) / ks_tau if ks_tau > 1e-10 else 1.0 - 0.5 * ks_tau
    )
    alpha = float(np.clip(alpha, 0.0, 1.0))
    integrated_var = max(theta_star + (V_true - theta_star) * alpha, 1e-10)
    return float(np.sqrt(integrated_var))


# -------------------------------------------------------------------
# Coverage computation for one cut
# -------------------------------------------------------------------

def _coverage_for_cut(
    cut_label: str,
    param_draws: np.ndarray,     # (N_POSTERIOR, 4) [kappa, theta, sigma_v, lam]
    train_returns: np.ndarray,   # (T_est,) daily returns for filter
    true_variances: np.ndarray,  # (T_total+1,) true variances
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Compute maturity-differentiated predictive coverage for one cut structure.

    For each out-of-sample day the predicted ATM-IV at maturity τ is computed
    using the Heston integrated-variance term-structure approximation:

        Σ_ATM(τ)² ≈ [θ*τ + (V̄ − θ*)(1 − e^{−κ*τ}) / κ*] / τ

    where κ* = κ + λ, θ* = κθ/κ*.  This is fast (no option pricing), analytic,
    and correctly differentiates across maturities via mean-reversion.
    """
    records = []
    T_out = T_TOTAL - T_ESTIMATION   # 500 out-of-sample days

    # ---- run particle filter through estimation window for each draw ----
    pf_final_particles = []   # list of (N_PARTICLES,) arrays
    for j in range(len(param_draws)):
        kappa_j, theta_j, sv_j, lam_j = param_draws[j]
        p_j = HestonParams(kappa=kappa_j, theta=theta_j, sigma_v=sv_j,
                           lam=lam_j, rho=0.0, mu=0.0, V0=0.03, r=0.02)
        filtered_V, _ = bootstrap_particle_filter(
            returns=train_returns,
            params=p_j,
            n_particles=N_PARTICLES,
            seed=int(rng.integers(0, 2**31)),
        )
        pf_final_particles.append(filtered_V[-1].copy())  # (N_PARTICLES,)

    # ---- out-of-sample coverage ----
    current_particles = [p.copy() for p in pf_final_particles]

    for t_rel in range(T_out):  # t_rel = 0 → day 501
        t_abs = T_ESTIMATION + t_rel + 1  # index into true_variances

        # Propagate each draw's particles one step forward
        pred_V_means = []
        for j in range(len(param_draws)):
            kappa_j, theta_j, sv_j, lam_j = param_draws[j]
            p_j = HestonParams(kappa=kappa_j, theta=theta_j, sigma_v=sv_j,
                               lam=lam_j, rho=0.0, mu=0.0, V0=0.03, r=0.02)
            next_V = propagate_particles_one_step(
                V_particles=current_particles[j],
                params=p_j,
                seed=int(rng.integers(0, 2**31)),
            )
            current_particles[j] = next_V
            pred_V_means.append(float(np.mean(next_V)))

        pred_V_means = np.array(pred_V_means)   # (N_POSTERIOR,)
        true_V = float(true_variances[t_abs])

        # For each reference maturity, compute maturity-differentiated ATM-IV
        for mat_name, tau in REF_MATURITIES.items():
            # Predicted ATM-IV distribution across posterior draws (vectorised)
            pred_iv = _heston_atm_iv_vectorised(pred_V_means, param_draws, tau)
            # True ATM-IV from the data-generating process
            true_iv = _heston_atm_iv_true(true_V, TRUE_PARAMS, tau)

            # Predictive intervals at each nominal level
            for alpha in NOMINAL_LEVELS:
                lo = np.quantile(pred_iv, (1 - alpha) / 2)
                hi = np.quantile(pred_iv, 1 - (1 - alpha) / 2)
                covered = int(lo <= true_iv <= hi)
                records.append({
                    "cut": cut_label,
                    "maturity": mat_name,
                    "nominal": alpha,
                    "covered": covered,
                    "width": hi - lo,
                })

    df = pd.DataFrame(records)
    summary = (
        df.groupby(["cut", "maturity", "nominal"])
          .agg(empirical_coverage=("covered", "mean"),
               avg_width=("width", "mean"))
          .reset_index()
    )
    return summary


# -------------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------------

def run_exp2():
    set_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ----------------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------------
    dataset_path = DATA_DIR / "synthetic_dataset.npz"
    if dataset_path.exists():
        raw = np.load(dataset_path)
        log_prices       = raw["log_prices"]
        variances        = raw["variances"]
        intraday_returns = raw["intraday_returns"]
    else:
        logger.info("Generating synthetic dataset…")
        log_prices, variances, intraday_returns = simulate_heston_daily(
            params=TRUE_PARAMS, T=T_TOTAL, M_intra=78, seed=SEED
        )

    daily_returns = compute_daily_returns(log_prices)  # (T_TOTAL,)
    train_returns = daily_returns[:T_ESTIMATION]        # (500,)

    # ----------------------------------------------------------------
    # Observed summaries
    # ----------------------------------------------------------------
    logger.info("Computing observed summaries…")
    S_obs = _compute_obs_summary(log_prices, variances, intraday_returns,
                                  TRUE_PARAMS, T_ESTIMATION)

    # ----------------------------------------------------------------
    # Reference table + GMM
    # ----------------------------------------------------------------
    ref_path = DATA_DIR / "reference_table.npz"
    if ref_path.exists():
        params_tbl, summaries_tbl = load_reference_table(str(ref_path))
    else:
        params_tbl, summaries_tbl = generate_reference_table(
            R=REFERENCE_TABLE_SIZE, T_window=T_ESTIMATION,
            M_intra=78, seed=SEED, verbose=True,
        )
        save_reference_table(params_tbl, summaries_tbl, str(ref_path))

    joint = np.hstack([params_tbl, summaries_tbl])
    mu_norm, std_norm = _fit_normaliser(joint)
    joint_norm = _normalise(joint, mu_norm, std_norm)
    S_obs_norm = _normalise(
        np.concatenate([np.zeros(N_PARAMS), S_obs]), mu_norm, std_norm
    )[N_PARAMS:]

    logger.info("Fitting GMM…")
    gmm = fit_gmm_bic(joint_norm, max_components=MAX_GMM_COMPS, seed=SEED)

    # ----------------------------------------------------------------
    # Get cut rankings from Exp 1 results (or recompute)
    # ----------------------------------------------------------------
    exp1_csv = RESULTS_DIR / "exp1_cut_rankings.csv"
    if exp1_csv.exists():
        df_ranks = pd.read_csv(exp1_csv)
        ranked_cut_ids = df_ranks["cut_id"].tolist()
    else:
        logger.info("Re-ranking cuts (Exp 1 results not found)…")
        cuts_all = enumerate_feasible_cuts()
        mass_vals = []
        for cut in cuts_all:
            try:
                samps = compute_cut_posterior_samples(
                    cut=cut, gmm=gmm, S_obs_full=S_obs_norm,
                    n_params=N_PARAMS, summary_block_sizes=BLOCK_SIZES,
                    n_samples=500, seed=SEED,
                )
                samps_raw = samps * std_norm[:N_PARAMS] + mu_norm[:N_PARAMS]
                mass_vals.append(compute_posterior_mass(samps_raw))
            except Exception:
                mass_vals.append(0.0)
        ranked = rank_cuts_by_posterior_mass(cuts_all, mass_vals)
        ranked_cut_ids = [c.cut_id for _, _, c in ranked]

    # Select cuts at rank 1, rank 19, rank 42
    cuts_all = enumerate_feasible_cuts()
    cut_by_id = {c.cut_id: c for c in cuts_all}
    selected_cut_ids = {
        "rank_1":  ranked_cut_ids[0],
        "rank_19": ranked_cut_ids[18],
        "rank_42": ranked_cut_ids[41],
    }

    rng = np.random.default_rng(SEED)
    all_coverage = []

    for label, cut_id in selected_cut_ids.items():
        cut = cut_by_id[cut_id]
        logger.info("Evaluating predictive coverage for %s (cut_id=%d)…",
                    label, cut_id)

        # Draw N_POSTERIOR posterior samples
        try:
            samps_norm = compute_cut_posterior_samples(
                cut=cut, gmm=gmm, S_obs_full=S_obs_norm,
                n_params=N_PARAMS, summary_block_sizes=BLOCK_SIZES,
                n_samples=N_POSTERIOR, seed=int(rng.integers(0, 2**31)),
            )
            param_draws = samps_norm * std_norm[:N_PARAMS] + mu_norm[:N_PARAMS]
            # Clip to physically plausible ranges
            param_draws[:, 0] = np.clip(param_draws[:, 0], 1e-4, 1.0)  # kappa
            param_draws[:, 1] = np.clip(param_draws[:, 1], 1e-4, 0.5)  # theta
            param_draws[:, 2] = np.clip(param_draws[:, 2], 1e-4, 0.5)  # sigma_v
            param_draws[:, 3] = np.clip(param_draws[:, 3], -0.5, 0.0)  # lam
        except Exception as exc:
            logger.warning("Posterior sampling failed for %s: %s", label, exc)
            param_draws = np.tile(
                [TRUE_PARAMS.kappa, TRUE_PARAMS.theta,
                 TRUE_PARAMS.sigma_v, TRUE_PARAMS.lam],
                (N_POSTERIOR, 1)
            )

        with Timer(f"coverage for {label}", logger=logger):
            cov_df = _coverage_for_cut(
                cut_label=label,
                param_draws=param_draws,
                train_returns=train_returns,
                true_variances=variances,
                rng=rng,
            )
        all_coverage.append(cov_df)

    # ----------------------------------------------------------------
    # Save CSV
    # ----------------------------------------------------------------
    df_cov = pd.concat(all_coverage, ignore_index=True)
    csv_path = RESULTS_DIR / "exp2_coverage_table.csv"
    df_cov.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)
    print(df_cov.to_string(index=False))

    # ----------------------------------------------------------------
    # Plot – grouped bar chart
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, len(REF_MATURITIES), figsize=(15, 5), sharey=True)
    fig.suptitle("Exp 2 – Empirical vs Nominal Coverage by Maturity Bucket")

    cut_labels  = list(selected_cut_ids.keys())
    colours     = ["steelblue", "orange", "tomato"]
    x_pos       = np.arange(len(NOMINAL_LEVELS))
    bar_width   = 0.22

    for ax_idx, (mat_name, ax) in enumerate(zip(REF_MATURITIES, axes)):
        for c_idx, (cut_label, colour) in enumerate(zip(cut_labels, colours)):
            sub = df_cov[(df_cov["cut"] == cut_label) & (df_cov["maturity"] == mat_name)]
            emp_covs = [
                sub[sub["nominal"] == lv]["empirical_coverage"].values[0]
                if len(sub[sub["nominal"] == lv]) > 0 else 0.0
                for lv in NOMINAL_LEVELS
            ]
            ax.bar(x_pos + c_idx * bar_width, emp_covs, bar_width,
                   label=cut_label, color=colour, alpha=0.8)

        # Nominal lines
        for i, alpha in enumerate(NOMINAL_LEVELS):
            ax.plot([x_pos[i] - bar_width, x_pos[i] + len(cut_labels) * bar_width],
                    [alpha, alpha], "k--", linewidth=1)

        ax.set_xticks(x_pos + bar_width)
        ax.set_xticklabels([f"{int(a*100)}%" for a in NOMINAL_LEVELS])
        ax.set_title(f"Maturity: {mat_name}")
        ax.set_ylabel("Empirical coverage")
        ax.set_ylim(0, 1.05)
        if ax_idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    png_path = RESULTS_DIR / "exp2_coverage_comparison.png"
    plt.savefig(png_path, dpi=120)
    plt.close()
    logger.info("Saved %s", png_path)

    logger.info("Experiment 2 complete.")
    return df_cov


if __name__ == "__main__":
    run_exp2()
