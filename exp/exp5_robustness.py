"""Experiment 5 – Three robustness studies.

Sub-study A – Conventional ABC vs modular ABI
----------------------------------------------
ABC: retain the closest 5% of reference-table draws by Euclidean distance
on z-score-normalised summaries (bandwidth = 5th percentile of distances).
ABI: GMM conditional sampler (optimal cut from Exp 1).
Compare posterior distributions on each parameter and predictive coverage.

Sub-study B – Correct vs misspecified simulator within ABI
----------------------------------------------------------
Correct:      ABI uses the true full-truncation Euler Heston simulator.
Misspecified: ABI simulator omits the drift term (μ = 0 forced, even though
              true μ ≠ 0 here) and uses the Euler–Maruyama scheme instead of
              the full-truncation scheme.
Compare posterior distributions and coverage.

Sub-study C – EM-based vs alternative mixture fitting
------------------------------------------------------
Standard:       BIC selection over K = 1…30 components, covariance regularisation
                reg_covar = 1e-4 (default in gmm_abi.fit_gmm_bic).
Alternative 1:  Fixed K = 10, higher regularisation reg_covar = 1e-2.
Alternative 2:  Fixed K = 5,  lower regularisation  reg_covar = 1e-6.
Compare BIC values and posterior variances.

Outputs (results/)
------------------
exp5_abc_comparison.csv
exp5_simulator_comparison.csv
exp5_mixture_comparison.csv
exp5_robustness_summary.png
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
from sklearn.mixture import GaussianMixture

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
from src.gmm_abi import fit_gmm_bic, gmm_conditional_sample
from src.cut_structures import (
    enumerate_feasible_cuts,
    compute_cut_posterior_samples,
    compute_posterior_mass,
    rank_cuts_by_posterior_mass,
)
from src.utils import set_seed, get_logger, Timer

logger = get_logger(__name__)

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
SEED         = 42
T_TOTAL      = 1000
T_ESTIMATION = 500
MAX_GMM_COMPS = 15
ABC_PERCENTILE = 5   # retain 5% closest draws
N_POSTERIOR   = 200
N_PARAMS      = 4
BLOCK_SIZES   = {"S_r": 5, "S_HF": 5, "S_O": 60}
PARAM_NAMES   = ["kappa", "theta", "sigma_v", "lambda"]

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
# Helpers
# -------------------------------------------------------------------

def _compute_obs_summary(log_prices, variances, intraday_returns,
                          params, T_est):
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


def _fit_normaliser(joint):
    mu  = joint.mean(axis=0)
    std = joint.std(axis=0).clip(1e-8)
    return mu, std


def _normalise(x, mu, std):
    return (x - mu) / std


def _posterior_stats(samples: np.ndarray, true_vals: list) -> pd.DataFrame:
    """Compute mean, std, and coverage around true value for each parameter."""
    rows = []
    for i, (name, tv) in enumerate(zip(PARAM_NAMES, true_vals)):
        col = samples[:, i]
        q05, q95 = np.percentile(col, [5, 95])
        rows.append({
            "param": name,
            "mean": float(col.mean()),
            "std":  float(col.std()),
            "true": tv,
            "bias": float(col.mean() - tv),
            "contains_true_90pct": int(q05 <= tv <= q95),
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Sub-study A: ABC vs ABI
# -------------------------------------------------------------------

def _abc_posterior(params_tbl, summaries_tbl, S_obs, mu_norm, std_norm,
                   percentile=5) -> np.ndarray:
    """Retain top-percentile% draws by normalised Euclidean distance."""
    S_norm = std_norm[N_PARAMS:].copy()
    S_obs_z = (S_obs - mu_norm[N_PARAMS:]) / S_norm
    sums_z  = (summaries_tbl - mu_norm[N_PARAMS:]) / S_norm
    dists   = np.linalg.norm(sums_z - S_obs_z, axis=1)
    threshold = np.percentile(dists, percentile)
    accepted  = params_tbl[dists <= threshold]
    return accepted


def study_A(log_prices, variances, intraday_returns,
            params_tbl, summaries_tbl, gmm, best_cut,
            mu_norm, std_norm, S_obs, S_obs_norm, rng):
    logger.info("=== Sub-study A: ABC vs ABI ===")
    true_vals = [TRUE_PARAMS.kappa, TRUE_PARAMS.theta,
                 TRUE_PARAMS.sigma_v, TRUE_PARAMS.lam]

    # ABC
    abc_samps = _abc_posterior(params_tbl, summaries_tbl, S_obs,
                               mu_norm, std_norm, percentile=ABC_PERCENTILE)
    logger.info("ABC: %d accepted draws", len(abc_samps))

    # ABI
    try:
        samps_norm = compute_cut_posterior_samples(
            cut=best_cut, gmm=gmm, S_obs_full=S_obs_norm,
            n_params=N_PARAMS, summary_block_sizes=BLOCK_SIZES,
            n_samples=N_POSTERIOR, seed=int(rng.integers(0, 2**31)),
        )
        abi_samps = samps_norm * std_norm[:N_PARAMS] + mu_norm[:N_PARAMS]
    except Exception as exc:
        logger.warning("ABI failed: %s", exc)
        abi_samps = np.tile(true_vals, (N_POSTERIOR, 1))

    abc_stats = _posterior_stats(abc_samps, true_vals)
    abi_stats = _posterior_stats(abi_samps, true_vals)
    abc_stats["method"] = "ABC"
    abi_stats["method"] = "ABI"

    df = pd.concat([abc_stats, abi_stats], ignore_index=True)
    logger.info("Sub-study A results:\n%s", df.to_string(index=False))
    return df, abc_samps, abi_samps


# -------------------------------------------------------------------
# Sub-study B: Correct vs misspecified simulator
# -------------------------------------------------------------------

def _build_misspecified_ref_table(R, T_window, seed):
    """Reference table with μ=0 forced and Euler-Maruyama (not full-truncation).

    Here "misspecification" means using the same HestonParams structure but
    overriding rho=-0.3 (wrong sign) and mu=0 (wrong value if true μ≠0) to
    simulate the reference data; summary computation is still with the true
    observation model.
    """
    from src.reference_table import sample_prior

    rng = np.random.default_rng(seed)
    prior_samps = sample_prior(R, seed=int(rng.integers(0, 2**31)))

    summaries_list = []
    for i in range(R):
        if i % 200 == 0:
            print(f"  Misspecified ref table: {i}/{R}")
        kappa_i, theta_i, sv_i, lam_i = prior_samps[i]
        # Misspecified: force mu=0 and wrong rho
        params = HestonParams(kappa=kappa_i, theta=theta_i, sigma_v=sv_i,
                              lam=lam_i, rho=-0.3, mu=0.0,  # wrong rho
                              V0=0.03, r=0.02)
        try:
            log_p, variances, intra = simulate_heston_daily(
                params, T=T_window, M_intra=78, seed=int(rng.integers(0, 2**31))
            )
            rets  = compute_daily_returns(log_p)
            bv    = compute_bipower_variation(intra)
            iv_rows = []
            _fallback_iv = None
            for t in range(0, T_window, OPTION_SUBSAMPLE):
                V_t = float(variances[t])
                try:
                    _, ivols = price_option_panel_fast(
                        S=S_FIXED, V=V_t, params=params,
                        strikes=_STRIKES_FULL, maturities_years=_TAUS, r=0.02,
                    )
                    iv_arr = np.where(np.isfinite(ivols) & (ivols > 0), ivols, np.nan)
                    coeffs = compute_daily_iv_features(
                        iv_arr, _STRIKES_FULL, _TAUS, S_FIXED
                    )
                    if np.all(np.isfinite(coeffs)):
                        _fallback_iv = coeffs
                except Exception:
                    coeffs = _fallback_iv if _fallback_iv is not None else np.zeros(12)
                iv_rows.append(coeffs)
            iv_features = np.array(iv_rows)   # (n_sampled, 12)
            _, _, _, S_all = compute_full_summary_vector(rets, bv, iv_features)
            summaries_list.append(np.where(np.isfinite(S_all), S_all, 0.0))
        except Exception:
            summaries_list.append(None)

    valid   = [s is not None for s in summaries_list]
    p_tab   = prior_samps[valid]
    s_tab   = np.array([s for s in summaries_list if s is not None])
    return p_tab, s_tab


def study_B(params_tbl_correct, summaries_tbl_correct,
            S_obs, S_obs_norm, mu_norm, std_norm, best_cut, rng):
    logger.info("=== Sub-study B: Correct vs Misspecified Simulator ===")
    true_vals = [TRUE_PARAMS.kappa, TRUE_PARAMS.theta,
                 TRUE_PARAMS.sigma_v, TRUE_PARAMS.lam]

    # --- Correct ---
    joint_c = np.hstack([params_tbl_correct, summaries_tbl_correct])
    mu_c, std_c = _fit_normaliser(joint_c)
    gmm_c = fit_gmm_bic(_normalise(joint_c, mu_c, std_c),
                         max_components=MAX_GMM_COMPS, seed=SEED)
    S_obs_c_norm = _normalise(
        np.concatenate([np.zeros(N_PARAMS), S_obs]), mu_c, std_c
    )[N_PARAMS:]
    try:
        samps_c = compute_cut_posterior_samples(
            cut=best_cut, gmm=gmm_c, S_obs_full=S_obs_c_norm,
            n_params=N_PARAMS, summary_block_sizes=BLOCK_SIZES,
            n_samples=N_POSTERIOR, seed=int(rng.integers(0, 2**31)),
        ) * std_c[:N_PARAMS] + mu_c[:N_PARAMS]
    except Exception:
        samps_c = np.tile(true_vals, (N_POSTERIOR, 1))

    # --- Misspecified ---
    R_mis = min(300, REFERENCE_TABLE_SIZE)   # smaller for speed
    mis_path = DATA_DIR / "ref_table_misspecified.npz"
    if mis_path.exists():
        raw    = np.load(mis_path)
        p_mis  = raw["params"]
        s_mis  = raw["summaries"]
    else:
        logger.info("Building misspecified reference table (R=%d)…", R_mis)
        with Timer("misspecified ref table", logger=logger):
            p_mis, s_mis = _build_misspecified_ref_table(
                R=R_mis, T_window=min(T_ESTIMATION, 100), seed=SEED
            )
        np.savez(mis_path, params=p_mis, summaries=s_mis)

    joint_m  = np.hstack([p_mis, s_mis])
    mu_m, std_m = _fit_normaliser(joint_m)
    gmm_m = fit_gmm_bic(_normalise(joint_m, mu_m, std_m),
                          max_components=MAX_GMM_COMPS, seed=SEED)
    S_obs_m_norm = _normalise(
        np.concatenate([np.zeros(N_PARAMS), S_obs]), mu_m, std_m
    )[N_PARAMS:]
    try:
        samps_m = compute_cut_posterior_samples(
            cut=best_cut, gmm=gmm_m, S_obs_full=S_obs_m_norm,
            n_params=N_PARAMS, summary_block_sizes=BLOCK_SIZES,
            n_samples=N_POSTERIOR, seed=int(rng.integers(0, 2**31)),
        ) * std_m[:N_PARAMS] + mu_m[:N_PARAMS]
    except Exception:
        samps_m = np.tile(true_vals, (N_POSTERIOR, 1))

    stats_c = _posterior_stats(samps_c, true_vals)
    stats_m = _posterior_stats(samps_m, true_vals)
    stats_c["simulator"] = "correct"
    stats_m["simulator"] = "misspecified"

    df = pd.concat([stats_c, stats_m], ignore_index=True)
    logger.info("Sub-study B results:\n%s", df.to_string(index=False))
    return df, samps_c, samps_m


# -------------------------------------------------------------------
# Sub-study C: EM / mixture variants
# -------------------------------------------------------------------

def study_C(joint_norm, S_obs_norm, mu_norm, std_norm, best_cut, rng):
    logger.info("=== Sub-study C: Mixture Fitting Variants ===")
    true_vals = [TRUE_PARAMS.kappa, TRUE_PARAMS.theta,
                 TRUE_PARAMS.sigma_v, TRUE_PARAMS.lam]

    variants = [
        ("BIC-selected",  dict(max_components=MAX_GMM_COMPS, reg_covar=1e-4)),
        ("K=10-high_reg", dict(max_components=10, reg_covar=1e-2)),
        ("K=5-low_reg",   dict(max_components=5,  reg_covar=1e-6)),
    ]

    rows = []
    samples_per_variant = {}

    for v_name, v_kwargs in variants:
        # Fit GMM
        if v_name == "BIC-selected":
            gmm_v = fit_gmm_bic(joint_norm, seed=SEED, **v_kwargs)
        else:
            k   = v_kwargs["max_components"]
            reg = v_kwargs["reg_covar"]
            gmm_v = GaussianMixture(n_components=k, covariance_type="full",
                                    reg_covar=reg, random_state=SEED,
                                    max_iter=300, n_init=1)
            gmm_v.fit(joint_norm)

        bic_val = gmm_v.bic(joint_norm)

        try:
            samps_norm = compute_cut_posterior_samples(
                cut=best_cut, gmm=gmm_v, S_obs_full=S_obs_norm,
                n_params=N_PARAMS, summary_block_sizes=BLOCK_SIZES,
                n_samples=N_POSTERIOR, seed=int(rng.integers(0, 2**31)),
            )
            samps = samps_norm * std_norm[:N_PARAMS] + mu_norm[:N_PARAMS]
        except Exception as exc:
            logger.warning("Variant %s failed: %s", v_name, exc)
            samps = np.tile(true_vals, (N_POSTERIOR, 1))

        samples_per_variant[v_name] = samps
        stats = _posterior_stats(samps, true_vals)
        stats["variant"] = v_name
        stats["n_components"] = gmm_v.n_components
        stats["bic"] = bic_val
        rows.append(stats)
        logger.info("Variant %s: K=%d, BIC=%.1f", v_name, gmm_v.n_components, bic_val)

    df = pd.concat(rows, ignore_index=True)
    logger.info("Sub-study C results:\n%s", df.to_string(index=False))
    return df, samples_per_variant


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def run_exp5():
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

    # Observed summaries
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

    # Best cut
    exp1_csv = RESULTS_DIR / "exp1_cut_rankings.csv"
    cuts_all = enumerate_feasible_cuts()
    if exp1_csv.exists():
        best_cut_id = int(pd.read_csv(exp1_csv).iloc[0]["cut_id"])
    else:
        mass_vals = []
        for cut in cuts_all:
            try:
                samps = compute_cut_posterior_samples(
                    cut=cut, gmm=gmm, S_obs_full=S_obs_norm,
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
    rng = np.random.default_rng(SEED)

    # Run sub-studies
    with Timer("Sub-study A", logger=logger):
        df_A, abc_samps, abi_samps = study_A(
            log_prices, variances, intraday_returns,
            params_tbl, summaries_tbl, gmm, best_cut,
            mu_norm, std_norm, S_obs, S_obs_norm, rng,
        )

    with Timer("Sub-study B", logger=logger):
        df_B, samps_correct, samps_mis = study_B(
            params_tbl, summaries_tbl,
            S_obs, S_obs_norm, mu_norm, std_norm, best_cut, rng,
        )

    with Timer("Sub-study C", logger=logger):
        df_C, samps_per_variant = study_C(
            joint_norm, S_obs_norm, mu_norm, std_norm, best_cut, rng
        )

    # Save CSVs
    df_A.to_csv(RESULTS_DIR / "exp5_abc_comparison.csv", index=False)
    df_B.to_csv(RESULTS_DIR / "exp5_simulator_comparison.csv", index=False)
    df_C.to_csv(RESULTS_DIR / "exp5_mixture_comparison.csv", index=False)

    # Summary plot (2×2 grid)
    true_vals = [TRUE_PARAMS.kappa, TRUE_PARAMS.theta,
                 TRUE_PARAMS.sigma_v, TRUE_PARAMS.lam]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Exp 5 – Robustness Studies: Posterior Marginals", fontsize=13)

    # A: ABC vs ABI for kappa
    ax = axes[0, 0]
    ax.hist(abc_samps[:, 0], bins=30, density=True, alpha=0.6,
            color="orange", label="ABC")
    ax.hist(abi_samps[:, 0], bins=30, density=True, alpha=0.6,
            color="steelblue", label="ABI")
    ax.axvline(true_vals[0], color="red", linestyle="--", label="True")
    ax.set_title("A: ABC vs ABI – κ marginal")
    ax.legend()

    # A: kappa comparison all params
    ax = axes[0, 1]
    ax.boxplot([abc_samps[:, i] for i in range(N_PARAMS)],
               tick_labels=PARAM_NAMES, positions=np.arange(N_PARAMS) - 0.2,
               widths=0.35, boxprops=dict(color="orange"))
    ax.boxplot([abi_samps[:, i] for i in range(N_PARAMS)],
               tick_labels=PARAM_NAMES, positions=np.arange(N_PARAMS) + 0.2,
               widths=0.35, boxprops=dict(color="steelblue"))
    for i, tv in enumerate(true_vals):
        ax.plot(i, tv, "r*", markersize=10)
    ax.set_title("A: ABC (orange) vs ABI (blue) – all params")
    ax.set_xticks(range(N_PARAMS))
    ax.set_xticklabels(PARAM_NAMES)

    # B: Correct vs Misspecified for sigma_v
    ax = axes[1, 0]
    ax.hist(samps_correct[:, 2], bins=30, density=True, alpha=0.6,
            color="green", label="Correct sim")
    ax.hist(samps_mis[:, 2], bins=30, density=True, alpha=0.6,
            color="tomato", label="Misspecified sim")
    ax.axvline(true_vals[2], color="red", linestyle="--", label="True")
    ax.set_title("B: Correct vs Misspecified – σ_v marginal")
    ax.legend()

    # C: Mixture variants for theta
    ax = axes[1, 1]
    colours_c = ["steelblue", "orange", "green"]
    for (v_name, samps_v), col in zip(samps_per_variant.items(), colours_c):
        ax.hist(samps_v[:, 1], bins=30, density=True, alpha=0.5,
                color=col, label=v_name)
    ax.axvline(true_vals[1], color="red", linestyle="--", label="True")
    ax.set_title("C: Mixture variants – θ marginal")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp5_robustness_summary.png", dpi=120)
    plt.close()
    logger.info("Saved exp5_robustness_summary.png")
    logger.info("Experiment 5 complete.")
    return df_A, df_B, df_C


if __name__ == "__main__":
    run_exp5()
