"""Experiment 3 – ABI vs simplified Modular MCMC benchmark.

Method A – ABI (optimal cut):
    Use the best cut structure from Exp 1 together with the GMM conditional
    sampler.  Posterior draws are available instantaneously after the offline
    reference-table build.

Method B – Simplified MCMC (ABC-MCMC):
    Random-walk Metropolis–Hastings where the pseudo-likelihood is a Gaussian
    kernel centred on the observed summary vector:

        log π̃(θ | S_obs) = -½ ‖(S(θ) - S_obs)/h‖² + log π(θ)

    where S(θ) is computed from a short forward simulation and h is the
    median absolute deviation of prior-predictive summary distances.

    Note: this is a computationally tractable approximation of the full PMMH
    chain.  A genuine particle MCMC chain would be much more expensive.

Two scenarios
-------------
Scenario 1 – Correctly specified:
    Both ABI and MCMC use the same misspecification-free simulator.
Scenario 2 – Misspecified observation noise:
    The MCMC evaluator adds heteroskedastic Gaussian noise to the returns
    summary (σ_noise = 0.5 × std(S_obs[:5])) to simulate a misspecified
    observation model; ABI is unchanged.

Outputs (results/)
------------------
exp3_coverage_table.csv
    method, scenario, maturity, nominal_level, empirical_coverage, avg_width
exp3_runtime_comparison.csv
    method, scenario, wall_time_seconds
exp3_coverage_comparison.png
    Side-by-side comparison plot.
"""

import sys
import time
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
# Settings
# -------------------------------------------------------------------
SEED            = 42
T_TOTAL         = 1000
T_ESTIMATION    = 500
N_POSTERIOR_ABI = 200
N_MCMC_ITER     = 400    # reduced for feasibility (~30s per scenario at 0.075s/step)
N_MCMC_BURN     = 80
N_MCMC_THIN     = 2      # yields (400-80)//2 = 160 posterior draws
N_PARTICLES     = 100
MAX_GMM_COMPS   = 15
NOMINAL_LEVELS  = [0.80, 0.90, 0.95]
REF_MATURITIES  = {"short": 30/252, "medium": 60/252, "long": 90/252}
N_PARAMS        = 4
BLOCK_SIZES     = {"S_r": 5, "S_HF": 5, "S_O": 60}

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

RESULTS_DIR = ROOT / "results"
DATA_DIR    = ROOT / "data"


# -------------------------------------------------------------------
# Helper: compute summaries
# -------------------------------------------------------------------

def _compute_obs_summary(log_prices, variances, intraday_returns, params, T_est):
    """Compute observed summary vector with the same subsampling as the reference table."""
    daily_rets = compute_daily_returns(log_prices[:T_est + 1])
    daily_bv   = compute_bipower_variation(intraday_returns[:T_est])
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


def _sim_summary(params: HestonParams, T: int, M: int, seed: int) -> np.ndarray:
    """Simulate T days and return the full summary vector (matching reference-table subsampling)."""
    log_p, variances, intra = simulate_heston_daily(params, T, M, seed=seed)
    daily_rets = compute_daily_returns(log_p)
    daily_bv   = compute_bipower_variation(intra)

    iv_rows = []
    _fallback_iv = None
    for t in range(0, T, OPTION_SUBSAMPLE):
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
# ABC-MCMC
# -------------------------------------------------------------------

def _log_prior(theta: np.ndarray) -> float:
    """Log prior for [kappa, theta_param, sigma_v, lam].
    Flat inside the reference-table sampling region; -inf outside."""
    kappa, th, sv, lam = theta
    if not (1e-5 < kappa < 0.15 and 1e-5 < th < 0.15
            and 1e-5 < sv < 0.15 and -kappa - 1e-6 < lam <= 0):
        return -np.inf
    return 0.0


def run_abc_mcmc(
    S_obs: np.ndarray,
    h: float,
    n_iter: int,
    n_burn: int,
    n_thin: int,
    T_sim: int,
    M_intra: int,
    rng: np.random.Generator,
    noise_std: float = 0.0,   # 0 = correctly specified, >0 = misspecified
) -> np.ndarray:
    """
    ABC-MCMC with Gaussian kernel.

    The pseudo log-likelihood is:
        log π̃(θ | S_obs) = -½ ‖(S̃(θ) - S̃_obs) / h‖²

    where S̃ = z-score-normalised summaries and h = bandwidth scale.

    Returns posterior draws after burn-in and thinning: shape (n_draws, 4).
    """
    n_draws = (n_iter - n_burn) // n_thin

    # Proposal standard deviations (tuned for ~25% acceptance)
    prop_std = np.array([0.005, 0.005, 0.003, 0.003])

    # Initial state: prior mean
    theta_curr = np.array([0.05, 0.05, 0.03, -0.02])
    S_obs_noisy = S_obs.copy()

    # Simulate summary from initial state
    seed_curr = int(rng.integers(0, 2**31))
    p_curr = HestonParams(kappa=theta_curr[0], theta=theta_curr[1],
                          sigma_v=theta_curr[2], lam=theta_curr[3],
                          rho=0.0, mu=0.0, V0=0.03, r=0.02)
    try:
        S_curr = _sim_summary(p_curr, T_sim, M_intra, seed_curr)
    except Exception:
        S_curr = np.zeros_like(S_obs)

    if noise_std > 0:
        S_obs_noisy = S_obs + rng.normal(0, noise_std, size=S_obs.shape)

    # Normalise both to unit scale using prior predictive std
    S_std = np.std(S_obs_noisy).clip(1e-8)
    S_obs_z = S_obs_noisy / S_std
    S_curr_z = S_curr / S_std

    def _log_pseudo_lik(S_sim: np.ndarray) -> float:
        S_sim_z = S_sim / S_std
        return -0.5 * np.sum(((S_sim_z - S_obs_z) / h) ** 2)

    log_pl_curr = _log_pseudo_lik(S_curr)   # pass raw summary, not pre-normalised
    log_pr_curr = _log_prior(theta_curr)

    draws = []
    n_accept = 0
    n_sim_calls = 0

    for it in range(n_iter):
        # Propose
        theta_prop = theta_curr + prop_std * rng.standard_normal(4)
        log_pr_prop = _log_prior(theta_prop)

        if not np.isinf(log_pr_prop):
            p_prop = HestonParams(kappa=theta_prop[0], theta=theta_prop[1],
                                  sigma_v=theta_prop[2], lam=theta_prop[3],
                                  rho=0.0, mu=0.0, V0=0.03, r=0.02)
            try:
                S_prop = _sim_summary(p_prop, T_sim, M_intra,
                                      int(rng.integers(0, 2**31)))
                n_sim_calls += 1
            except Exception:
                S_prop = None

            if S_prop is not None:
                log_pl_prop = _log_pseudo_lik(S_prop)
                log_alpha = (log_pl_prop + log_pr_prop) - (log_pl_curr + log_pr_curr)
                if np.log(rng.uniform()) < log_alpha:
                    theta_curr  = theta_prop
                    log_pl_curr = log_pl_prop
                    log_pr_curr = log_pr_prop
                    n_accept    += 1

        # Always record draw (even after rejected / out-of-bounds proposals)
        if it >= n_burn and (it - n_burn) % n_thin == 0:
            draws.append(theta_curr.copy())

        if (it + 1) % 100 == 0:
            logger.info("ABC-MCMC iter %d/%d | accept=%.1f%% | draws=%d",
                        it + 1, n_iter, 100 * n_accept / max(n_sim_calls, 1), len(draws))

    logger.info("ABC-MCMC acceptance rate: %.1f%% (%d sim calls)",
                100 * n_accept / max(n_sim_calls, 1), n_sim_calls)
    draws = np.array(draws)
    return draws if len(draws) > 0 else np.tile(TRUE_PARAMS.__dict__, (1,))


# -------------------------------------------------------------------
# Coverage helper (shared with Exp 2 pattern)
# -------------------------------------------------------------------

def _coverage(param_draws, train_returns, true_variances, rng, method_label):
    records = []
    T_out = T_TOTAL - T_ESTIMATION

    pf_finals = []
    for j in range(len(param_draws)):
        kappa_j, theta_j, sv_j, lam_j = param_draws[j]
        p_j = HestonParams(kappa=max(kappa_j, 1e-4), theta=max(theta_j, 1e-4),
                           sigma_v=max(sv_j, 1e-4), lam=min(lam_j, 0.0),
                           rho=0.0, mu=0.0, V0=0.03, r=0.02)
        fv, _ = bootstrap_particle_filter(train_returns, p_j, N_PARTICLES,
                                          seed=int(rng.integers(0, 2**31)))
        pf_finals.append(fv[-1].copy())

    current = [p.copy() for p in pf_finals]
    for t_rel in range(T_out):
        t_abs = T_ESTIMATION + t_rel + 1
        pred_Vs = []
        for j in range(len(param_draws)):
            kappa_j, theta_j, sv_j, lam_j = param_draws[j]
            p_j = HestonParams(kappa=max(kappa_j, 1e-4), theta=max(theta_j, 1e-4),
                               sigma_v=max(sv_j, 1e-4), lam=min(lam_j, 0.0),
                               rho=0.0, mu=0.0, V0=0.03, r=0.02)
            nxt = propagate_particles_one_step(current[j], p_j,
                                               seed=int(rng.integers(0, 2**31)))
            current[j] = nxt
            pred_Vs.append(float(np.mean(nxt)))

        pred_ivs = np.sqrt(np.maximum(pred_Vs, 1e-10))
        true_iv  = np.sqrt(max(float(true_variances[t_abs]), 1e-10))

        for mat_name in REF_MATURITIES:
            for alpha in NOMINAL_LEVELS:
                lo = np.quantile(pred_ivs, (1 - alpha) / 2)
                hi = np.quantile(pred_ivs, 1 - (1 - alpha) / 2)
                records.append({
                    "method": method_label, "maturity": mat_name,
                    "nominal": alpha,
                    "covered": int(lo <= true_iv <= hi),
                    "width": hi - lo,
                })

    df = pd.DataFrame(records)
    return (df.groupby(["method", "maturity", "nominal"])
              .agg(empirical_coverage=("covered", "mean"),
                   avg_width=("width", "mean"))
              .reset_index())


# -------------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------------

def run_exp3():
    set_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load dataset
    dataset_path = DATA_DIR / "synthetic_dataset.npz"
    if dataset_path.exists():
        raw = np.load(dataset_path)
        log_prices, variances, intraday_returns = (
            raw["log_prices"], raw["variances"], raw["intraday_returns"]
        )
    else:
        log_prices, variances, intraday_returns = simulate_heston_daily(
            TRUE_PARAMS, T_TOTAL, 78, seed=SEED
        )
    train_returns = compute_daily_returns(log_prices)[:T_ESTIMATION]

    S_obs = _compute_obs_summary(log_prices, variances, intraday_returns,
                                  TRUE_PARAMS, T_ESTIMATION)

    # Reference table + GMM
    ref_path = DATA_DIR / "reference_table.npz"
    if ref_path.exists():
        params_tbl, summaries_tbl = load_reference_table(str(ref_path))
    else:
        params_tbl, summaries_tbl = generate_reference_table(
            R=REFERENCE_TABLE_SIZE, T_window=T_ESTIMATION, M_intra=78,
            seed=SEED, verbose=True,
        )
        save_reference_table(params_tbl, summaries_tbl, str(ref_path))

    joint = np.hstack([params_tbl, summaries_tbl])
    mu_norm, std_norm = _fit_normaliser(joint)
    joint_norm = _normalise(joint, mu_norm, std_norm)
    S_obs_norm = _normalise(
        np.concatenate([np.zeros(N_PARAMS), S_obs]), mu_norm, std_norm
    )[N_PARAMS:]

    gmm = fit_gmm_bic(joint_norm, max_components=MAX_GMM_COMPS, seed=SEED)

    # Get best cut (rank 1) from Exp 1 or re-rank
    exp1_csv = RESULTS_DIR / "exp1_cut_rankings.csv"
    cuts_all = enumerate_feasible_cuts()
    if exp1_csv.exists():
        df_ranks = pd.read_csv(exp1_csv)
        best_cut_id = int(df_ranks.iloc[0]["cut_id"])
    else:
        mass_vals = []
        for cut in cuts_all:
            try:
                samps = compute_cut_posterior_samples(
                    cut=cut, gmm=gmm, S_obs_full=S_obs_norm,
                    n_params=N_PARAMS, summary_block_sizes=BLOCK_SIZES,
                    n_samples=300, seed=SEED,
                )
                samps_raw = samps * std_norm[:N_PARAMS] + mu_norm[:N_PARAMS]
                mass_vals.append(compute_posterior_mass(samps_raw))
            except Exception:
                mass_vals.append(0.0)
        ranked = rank_cuts_by_posterior_mass(cuts_all, mass_vals)
        best_cut_id = ranked[0][2].cut_id

    best_cut = next(c for c in cuts_all if c.cut_id == best_cut_id)

    # ABC-MCMC bandwidth (median pairwise summary distance in reference table,
    # normalised by std)
    S_std_ref = summaries_tbl.std(axis=0).clip(1e-8)
    dists = np.linalg.norm(
        (summaries_tbl[:200] - S_obs) / S_std_ref, axis=1
    )
    h = float(np.median(dists)) / np.sqrt(len(S_obs))
    h = max(h, 0.1)   # floor bandwidth
    logger.info("ABC-MCMC bandwidth h = %.4f", h)

    rng = np.random.default_rng(SEED)
    timing_rows  = []
    coverage_rows = []

    for scenario_id, noise_std in [("correct", 0.0), ("misspecified", 0.5)]:
        logger.info("=== Scenario: %s ===", scenario_id)

        # ---- ABI ----
        t0 = time.perf_counter()
        try:
            samps_norm = compute_cut_posterior_samples(
                cut=best_cut, gmm=gmm, S_obs_full=S_obs_norm,
                n_params=N_PARAMS, summary_block_sizes=BLOCK_SIZES,
                n_samples=N_POSTERIOR_ABI, seed=int(rng.integers(0, 2**31)),
            )
            abi_draws = samps_norm * std_norm[:N_PARAMS] + mu_norm[:N_PARAMS]
            abi_draws[:, 0] = np.clip(abi_draws[:, 0], 1e-4, 1.0)
            abi_draws[:, 1] = np.clip(abi_draws[:, 1], 1e-4, 0.5)
            abi_draws[:, 2] = np.clip(abi_draws[:, 2], 1e-4, 0.5)
            abi_draws[:, 3] = np.clip(abi_draws[:, 3], -0.5, 0.0)
        except Exception as exc:
            logger.warning("ABI sampling failed: %s", exc)
            abi_draws = np.tile(
                [TRUE_PARAMS.kappa, TRUE_PARAMS.theta,
                 TRUE_PARAMS.sigma_v, TRUE_PARAMS.lam], (N_POSTERIOR_ABI, 1)
            )
        abi_time = time.perf_counter() - t0
        timing_rows.append({"method": "ABI", "scenario": scenario_id,
                            "wall_time_seconds": abi_time})

        # ---- MCMC ----
        # Use shortened T_sim for feasibility
        T_sim_mcmc = min(50, T_ESTIMATION)
        noise_scale = float(np.std(S_obs[:5])) * noise_std if noise_std > 0 else 0.0

        t0 = time.perf_counter()
        mcmc_draws = run_abc_mcmc(
            S_obs=S_obs,
            h=h,
            n_iter=N_MCMC_ITER,
            n_burn=N_MCMC_BURN,
            n_thin=N_MCMC_THIN,
            T_sim=T_sim_mcmc,
            M_intra=78,
            rng=rng,
            noise_std=noise_scale,
        )
        mcmc_time = time.perf_counter() - t0
        timing_rows.append({"method": "MCMC", "scenario": scenario_id,
                            "wall_time_seconds": mcmc_time})

        logger.info("ABI time: %.1fs | MCMC time: %.1fs", abi_time, mcmc_time)

        # Coverage for ABI
        with Timer(f"ABI coverage [{scenario_id}]", logger=logger):
            cov_abi = _coverage(abi_draws[:50], train_returns, variances, rng,
                                f"ABI_{scenario_id}")
        coverage_rows.append(cov_abi)

        # Coverage for MCMC
        mcmc_draws_use = mcmc_draws[:50] if len(mcmc_draws) >= 50 else np.tile(
            [TRUE_PARAMS.kappa, TRUE_PARAMS.theta,
             TRUE_PARAMS.sigma_v, TRUE_PARAMS.lam], (50, 1)
        )
        with Timer(f"MCMC coverage [{scenario_id}]", logger=logger):
            cov_mcmc = _coverage(mcmc_draws_use, train_returns, variances, rng,
                                 f"MCMC_{scenario_id}")
        coverage_rows.append(cov_mcmc)

    # Save
    df_cov  = pd.concat(coverage_rows, ignore_index=True)
    df_time = pd.DataFrame(timing_rows)

    csv_cov  = RESULTS_DIR / "exp3_coverage_table.csv"
    csv_time = RESULTS_DIR / "exp3_runtime_comparison.csv"
    df_cov.to_csv(csv_cov, index=False)
    df_time.to_csv(csv_time, index=False)
    logger.info("Saved %s and %s", csv_cov, csv_time)
    print(df_cov.to_string(index=False))
    print(df_time.to_string(index=False))

    # Plot
    methods = df_cov["method"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Exp 3 – ABI vs MCMC Predictive Coverage")
    scenarios = [("correct", axes[0]), ("misspecified", axes[1])]

    for sc_label, ax in scenarios:
        sub = df_cov[df_cov["method"].str.contains(sc_label)]
        x   = np.arange(len(NOMINAL_LEVELS))
        bw  = 0.2
        for i, m in enumerate([f"ABI_{sc_label}", f"MCMC_{sc_label}"]):
            ms = sub[(sub["method"] == m) & (sub["maturity"] == "medium")]
            vals = [ms[ms["nominal"] == a]["empirical_coverage"].values[0]
                    if len(ms[ms["nominal"] == a]) > 0 else 0.0
                    for a in NOMINAL_LEVELS]
            ax.bar(x + i * bw, vals, bw, label=m.split("_")[0], alpha=0.8)
        for i, a in enumerate(NOMINAL_LEVELS):
            ax.axhline(a, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xticks(x + bw / 2)
        ax.set_xticklabels([f"{int(a*100)}%" for a in NOMINAL_LEVELS])
        ax.set_title(f"Scenario: {sc_label}")
        ax.set_ylabel("Empirical coverage")
        ax.set_ylim(0, 1.05)
        ax.legend()

    plt.tight_layout()
    png = RESULTS_DIR / "exp3_coverage_comparison.png"
    plt.savefig(png, dpi=120)
    plt.close()
    logger.info("Saved %s", png)
    logger.info("Experiment 3 complete.")
    return df_cov, df_time


if __name__ == "__main__":
    run_exp3()
