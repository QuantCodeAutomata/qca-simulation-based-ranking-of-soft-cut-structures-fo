"""Experiment 1 – Rank all 42 feasible cut structures by relative posterior mass.

Pipeline
--------
1. Load (or generate) the 1 000-day synthetic Heston dataset.
2. Extract the first 500 days as the "estimation window".
3. Compute the 70-dimensional observed summary vector S_obs.
4. Build a reference table of R = 2 000 (params, summary) pairs.
5. Z-score-normalise the joint (params, summaries) table.
6. Fit a GMM with BIC-selected number of components (1–30).
7. Enumerate all 42 feasible cut structures.
8. For each cut draw N_POSTERIOR_SAMPLES from the cut posterior and compute
   the posterior-mass criterion.
9. Rank cuts by descending posterior mass.
10. Save CSV results and two diagnostic plots.

Outputs (results/)
------------------
exp1_cut_rankings.csv
    rank, cut_id, phi1_params, phi2_params, s1_blocks, s2_blocks,
    posterior_mass
exp1_top_cuts_posteriors.png
    Posterior marginals for the top-3 cuts vs. the true parameter values.
exp1_posterior_mass_distribution.png
    Bar chart of all 42 cuts ordered by posterior mass.
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
import seaborn as sns

from src.heston_simulator import (
    HestonParams, simulate_heston_daily,
    compute_bipower_variation, compute_daily_returns,
)
from src.option_pricing import price_option_panel_fast      # vectorised fast pricer
from src.surface_regression import compute_daily_iv_features
from src.summary_stats import compute_full_summary_vector
from src.reference_table import (
    generate_reference_table, REFERENCE_TABLE_SIZE,
    save_reference_table, load_reference_table,
    OPTION_SUBSAMPLE,                          # shared subsampling constant
    _LOG_MONEYNESS,                            # keep option grid consistent
    _MATURITIES,
)
from src.gmm_abi import fit_gmm_bic
from src.cut_structures import (
    enumerate_feasible_cuts,
    compute_cut_posterior_samples,
    compute_posterior_mass,
    rank_cuts_by_posterior_mass,
    get_summary_indices,
    SUMMARY_BLOCKS,
)
from src.utils import set_seed, get_logger, Timer

logger = get_logger(__name__)

# -------------------------------------------------------------------
# Experiment settings
# -------------------------------------------------------------------
SEED             = 42
N_POSTERIOR_SAMPLES = 2000
T_TOTAL          = 1000
T_ESTIMATION     = 500
MAX_GMM_COMPS    = 15    # search 1..MAX_GMM_COMPS components

TRUE_PARAMS = HestonParams(
    kappa=0.05, theta=0.03, sigma_v=0.03,
    lam=-0.02,  rho=0.0,    mu=0.0,
    V0=0.03,    r=0.02,
)

# Fixed option grid for IV features – must be identical to reference_table.py.
# Build the Cartesian product: 5 strikes × 3 maturities = 15 contracts.
S_FIXED       = 1.0
_STRIKES_BASE = S_FIXED * np.exp(_LOG_MONEYNESS)          # (5,)
_STRIKES_FULL = np.repeat(_STRIKES_BASE, len(_MATURITIES)) # (15,)
_TAUS_FULL    = np.tile(_MATURITIES, len(_STRIKES_BASE))   # (15,)

# Summary block sizes that mirror compute_full_summary_vector output
BLOCK_SIZES = {"S_r": 5, "S_HF": 5, "S_O": 60}
N_PARAMS    = 4   # [kappa, theta, sigma_v, lambda]
PARAM_NAMES = ["kappa", "theta", "sigma_v", "lambda"]

RESULTS_DIR = ROOT / "results"
DATA_DIR    = ROOT / "data"


# -------------------------------------------------------------------
# Helper: compute observed summary for T_ESTIMATION days
# -------------------------------------------------------------------

def _compute_observed_summaries(
    log_prices: np.ndarray,
    variances:  np.ndarray,
    intraday_returns: np.ndarray,
    params: HestonParams,
    T_est: int,
) -> np.ndarray:
    """Compute 70-dim summary from the first T_est days.

    Uses the same subsampled option-pricing schedule and Cartesian grid as
    reference_table.generate_reference_table so that observed and simulated
    summary vectors are directly comparable.
    """
    daily_rets = compute_daily_returns(log_prices[: T_est + 1])
    daily_bv   = compute_bipower_variation(intraday_returns[:T_est])

    # Subsample option days to match reference_table.py (every OPTION_SUBSAMPLE days)
    sample_days = list(range(0, T_est, OPTION_SUBSAMPLE))

    # We accumulate one coefficient vector per sampled day, then form a
    # (len(sample_days), 12) array for compute_full_summary_vector.
    iv_rows: dict[int, np.ndarray] = {}
    fallback_coeffs: np.ndarray | None = None

    for t in sample_days:
        V_t = float(variances[t])
        try:
            _, ivols = price_option_panel_fast(
                S=S_FIXED, V=V_t, params=params,
                strikes=_STRIKES_FULL, maturities_years=_TAUS_FULL, r=params.r,
            )
        except Exception:
            ivols = np.full(len(_STRIKES_FULL), np.nan)

        iv_arr = np.where(np.isfinite(ivols) & (ivols > 0), ivols, np.nan)
        coeffs = compute_daily_iv_features(iv_arr, _STRIKES_FULL, _TAUS_FULL, S_FIXED)
        if np.any(np.isfinite(coeffs)):
            fallback_coeffs = coeffs
        else:
            coeffs = fallback_coeffs if fallback_coeffs is not None else np.zeros_like(coeffs)
        iv_rows[t] = coeffs

    # Build dense subsampled feature matrix
    iv_features = np.array([iv_rows[t] for t in sample_days])   # (n_sampled, 12)
    _, _, _, S_all = compute_full_summary_vector(daily_rets, daily_bv, iv_features)
    return S_all


# -------------------------------------------------------------------
# Normalisation helpers
# -------------------------------------------------------------------

def _fit_normaliser(joint: np.ndarray):
    """Return (mean, std) for z-score normalisation, clipping std at 1e-8."""
    mu  = joint.mean(axis=0)
    std = joint.std(axis=0).clip(1e-8)
    return mu, std


def _normalise(x: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mu) / std


# -------------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------------

def run_exp1():
    set_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Load or generate the full synthetic dataset
    # ----------------------------------------------------------------
    dataset_path = DATA_DIR / "synthetic_dataset.npz"
    if dataset_path.exists():
        logger.info("Loading synthetic dataset from %s", dataset_path)
        raw = np.load(dataset_path)
        log_prices        = raw["log_prices"]
        variances         = raw["variances"]
        intraday_returns  = raw["intraday_returns"]
    else:
        logger.info("Generating synthetic dataset (T=%d)…", T_TOTAL)
        with Timer("Heston simulation", logger=logger):
            log_prices, variances, intraday_returns = simulate_heston_daily(
                params=TRUE_PARAMS, T=T_TOTAL, M_intra=78, seed=SEED
            )

    # ----------------------------------------------------------------
    # 2. Observed summary from estimation window
    # ----------------------------------------------------------------
    logger.info("Computing observed summary vector (T_est=%d days)…", T_ESTIMATION)
    with Timer("observed summaries", logger=logger):
        S_obs = _compute_observed_summaries(
            log_prices, variances, intraday_returns, TRUE_PARAMS, T_ESTIMATION
        )
    logger.info("S_obs shape: %s, finite: %d/%d",
                S_obs.shape, np.isfinite(S_obs).sum(), len(S_obs))
    S_obs = np.where(np.isfinite(S_obs), S_obs, 0.0)

    # ----------------------------------------------------------------
    # 3. Build or load reference table
    # ----------------------------------------------------------------
    ref_path = DATA_DIR / "reference_table.npz"
    if ref_path.exists():
        logger.info("Loading reference table from %s", ref_path)
        params_tbl, summaries_tbl = load_reference_table(str(ref_path))
    else:
        logger.info("Generating reference table (R=%d)…", REFERENCE_TABLE_SIZE)
        with Timer("reference table", logger=logger):
            params_tbl, summaries_tbl = generate_reference_table(
                R=REFERENCE_TABLE_SIZE,
                T_window=T_ESTIMATION,
                M_intra=78,
                seed=SEED,
                verbose=True,
            )
        save_reference_table(params_tbl, summaries_tbl, str(ref_path))

    R = len(params_tbl)
    logger.info("Reference table: %d valid rows, summaries shape: %s",
                R, summaries_tbl.shape)

    # ----------------------------------------------------------------
    # 4. Normalise and fit GMM
    # ----------------------------------------------------------------
    joint = np.hstack([params_tbl, summaries_tbl])
    mu_norm, std_norm = _fit_normaliser(joint)
    joint_norm = _normalise(joint, mu_norm, std_norm)
    S_obs_norm = _normalise(
        np.concatenate([np.zeros(N_PARAMS), S_obs]),  # dummy params prefix
        mu_norm, std_norm
    )[N_PARAMS:]  # extract normalised summary part

    logger.info("Fitting GMM (max_components=%d) on %d × %d joint…",
                MAX_GMM_COMPS, *joint_norm.shape)
    with Timer("GMM fitting", logger=logger):
        gmm = fit_gmm_bic(joint_norm, max_components=MAX_GMM_COMPS, seed=SEED)
    logger.info("GMM selected %d components (BIC-optimal).", gmm.n_components)

    # ----------------------------------------------------------------
    # 5. Enumerate cuts and compute posterior mass for each
    # ----------------------------------------------------------------
    cuts = enumerate_feasible_cuts()
    assert len(cuts) == 42, f"Expected 42 cuts, got {len(cuts)}"
    logger.info("Evaluating %d cuts…", len(cuts))

    mass_values = []
    posterior_samples_per_cut = {}

    for i, cut in enumerate(cuts):
        try:
            samples_norm = compute_cut_posterior_samples(
                cut=cut,
                gmm=gmm,
                S_obs_full=S_obs_norm,
                n_params=N_PARAMS,
                summary_block_sizes=BLOCK_SIZES,
                n_samples=N_POSTERIOR_SAMPLES,
                seed=SEED + i,
            )
            # De-normalise parameters only
            param_mu  = mu_norm[:N_PARAMS]
            param_std = std_norm[:N_PARAMS]
            samples_raw = samples_norm * param_std + param_mu
        except Exception as exc:
            logger.warning("Cut %d failed: %s", i, exc)
            samples_raw = np.zeros((N_POSTERIOR_SAMPLES, N_PARAMS))

        mass = compute_posterior_mass(samples_raw)
        mass_values.append(mass)
        posterior_samples_per_cut[i] = samples_raw

        if (i + 1) % 10 == 0:
            logger.info("  … %d/42 cuts evaluated.", i + 1)

    # ----------------------------------------------------------------
    # 6. Rank cuts
    # ----------------------------------------------------------------
    ranked = rank_cuts_by_posterior_mass(cuts, mass_values)

    # ----------------------------------------------------------------
    # 7. Save CSV results
    # ----------------------------------------------------------------
    rows = []
    for rank, mass, cut in ranked:
        phi1_names = [PARAM_NAMES[p] for p in cut.phi1_params]
        phi2_names = [PARAM_NAMES[p] for p in cut.phi2_params]
        s1_names   = [SUMMARY_BLOCKS[b] for b in cut.s1_blocks]
        s2_names   = [SUMMARY_BLOCKS[b] for b in cut.s2_blocks]
        rows.append({
            "rank":           rank,
            "cut_id":         cut.cut_id,
            "phi1_params":    "|".join(phi1_names),
            "phi2_params":    "|".join(phi2_names),
            "s1_blocks":      "|".join(s1_names),
            "s2_blocks":      "|".join(s2_names),
            "posterior_mass": mass,
        })

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "exp1_cut_rankings.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved cut rankings to %s", csv_path)
    print(df.to_string(index=False))

    # ----------------------------------------------------------------
    # 8. Plot 1 – posterior marginals for top-3 cuts
    # ----------------------------------------------------------------
    true_vals = [TRUE_PARAMS.kappa, TRUE_PARAMS.theta,
                 TRUE_PARAMS.sigma_v, TRUE_PARAMS.lam]
    top3 = ranked[:3]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle("Exp 1 – Posterior Marginals: Top-3 Cut Structures", fontsize=13)

    for row_idx, (rank, mass, cut) in enumerate(top3):
        samps = posterior_samples_per_cut[cut.cut_id]
        phi1_names = [PARAM_NAMES[p] for p in cut.phi1_params]
        phi2_names = [PARAM_NAMES[p] for p in cut.phi2_params]
        cut_label = f"Rank {rank}  φ₁={{{','.join(phi1_names)}}}  mass={mass:.3f}"

        for col_idx, (pname, ptrue) in enumerate(zip(PARAM_NAMES, true_vals)):
            ax = axes[row_idx, col_idx]
            ax.hist(samps[:, col_idx], bins=40, density=True,
                    alpha=0.7, color=f"C{row_idx}")
            ax.axvline(ptrue, color="red", linestyle="--", linewidth=1.5,
                       label="True")
            ax.set_xlabel(pname)
            ax.set_ylabel("Density" if col_idx == 0 else "")
            if col_idx == 0:
                ax.set_title(cut_label, fontsize=8, loc="left")
            if row_idx == 0 and col_idx == 3:
                ax.legend(fontsize=8)

    plt.tight_layout()
    png1 = RESULTS_DIR / "exp1_top_cuts_posteriors.png"
    plt.savefig(png1, dpi=120)
    plt.close()
    logger.info("Saved %s", png1)

    # ----------------------------------------------------------------
    # 9. Plot 2 – bar chart of all 42 masses
    # ----------------------------------------------------------------
    sorted_masses = [m for _, m, _ in ranked]
    sorted_ids    = [f"#{c.cut_id}" for _, _, c in ranked]

    fig, ax = plt.subplots(figsize=(18, 5))
    colours = ["steelblue"] * 42
    colours[0]  = "gold"   # best
    colours[18] = "orange" # ~middle
    colours[-1] = "tomato" # worst
    ax.bar(range(42), sorted_masses, color=colours)
    ax.set_xticks(range(42))
    ax.set_xticklabels(sorted_ids, rotation=90, fontsize=7)
    ax.set_xlabel("Cut (ranked by mass)")
    ax.set_ylabel("Posterior mass criterion")
    ax.set_title("Exp 1 – All 42 Cut Structures Ranked by Posterior Mass")
    plt.tight_layout()
    png2 = RESULTS_DIR / "exp1_posterior_mass_distribution.png"
    plt.savefig(png2, dpi=120)
    plt.close()
    logger.info("Saved %s", png2)

    logger.info("Experiment 1 complete.")
    return df


if __name__ == "__main__":
    run_exp1()
