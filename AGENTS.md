
### `src/summary_stats.py`
- `compute_return_summaries(returns)` -> `(5,)`: [mean, var, skew, kurt, GARCH(1,1) score]
- `compute_bv_summaries(bv)` -> `(5,)`: [mean, var, skew, kurt, lag-1 autocorr]
- `compute_iv_surface_summaries(iv_features)` -> `(n_features*5,)`: 5 stats per IV surface coeff
- `compute_full_summary_vector(returns, bv, iv_features)` -> tuple `(S_r, S_HF, S_O, S_all)` where `S_all` dim=70 (5+5+60)
- Summary block sizes: `{'S_r': 5, 'S_HF': 5, 'S_O': 60}` when `iv_features` has 12 columns

### `src/reference_table.py`
- `REFERENCE_TABLE_SIZE = 2000`
- `sample_prior(n, seed)` -> `(n, 4)` [kappa, theta, sigma_v, lambda]
  - Prior: kappa,theta~U(0,0.1); sigma_v|kappa,theta~U(0,sqrt(2*kappa*theta)); lam|kappa~U(-kappa,0)
- `generate_reference_table(R, T_window, M_intra, seed, verbose)` -> `(params_table, summaries_table)`
  - Fixed 15-contract grid: 5 log-moneyness x 3 maturities (30d,60d,90d); S=1 fixed
- `save_reference_table` / `load_reference_table` -> NPZ I/O

### `src/gmm_abi.py`
- `fit_gmm_bic(data, max_components=30, reg_covar=1e-4, seed=42)` -> `GaussianMixture` (BIC-selected)
- `compute_conditional_weights(gmm, x_obs, x_obs_idx)` -> `(K,)` normalised weights
- `gmm_conditional_sample(gmm, x_obs, x_obs_idx, x_target_idx, n_samples, seed)` -> `(n_samples, d_target)`
- `gmm_batch_conditional_sample(gmm, x_obs_batch, x_obs_idx, x_target_idx, rng)` -> `(N, d_target)`

### `src/cut_structures.py`
- `PARAM_NAMES = ['kappa','theta','sigma_v','lambda']`; `SUMMARY_BLOCKS = ['S_r','S_HF','S_O']`
- `enumerate_feasible_cuts()` -> list of exactly 42 CutStructure objects
  - Constraint: lambda in phi1 <=> S_O in s1 (lambda-option pairing)
  - 42 = 7*3 + 7*3 (with/without lambda in phi1, each paired with 3 block split choices)
- `get_summary_indices(block_indices, summary_block_sizes)` -> flat index array
- `compute_cut_posterior_samples(cut, gmm, S_obs_full, n_params, summary_block_sizes, n_samples, seed)` -> `(n_samples, 4)`
- `compute_posterior_mass(samples)` -> float in [0,1] (mean of 4 interval indicators)
- `rank_cuts_by_posterior_mass(cuts, mass_values)` -> sorted list of (rank, mass, cut)

### `src/particle_filter.py`
- `bootstrap_particle_filter(returns, params, n_particles, seed)` -> `(filtered_vars, log_weights)`
  - filtered_vars: `(T, N)`, log_weights: `(T, N)`
  - Init: V_0^i ~ |N(theta, 0.005^2)|; obs model: r_t | V_t ~ N(mu*dt, V_t*dt)
  - Systematic resampling when ESS < 0.5*N
- `propagate_particles_one_step(V_particles, params, dt, seed)` -> `(N,)` next variance
- `get_filtered_variance(filtered_vars, log_weights)` -> `(T,)` weighted-mean variance

## Experiment Modules

### `data/generate_synthetic.py`
- TRUE_PARAMS: kappa=0.05, theta=0.03, sigma_v=0.03, lam=-0.02, rho=0.0, mu=0.0, V0=0.03, r=0.02
- T=1000, M_INTRA=78, SEED=42
- Saves `data/synthetic_dataset.npz`: log_prices, variances, intraday_returns, daily_returns, daily_bv, iv_features
- IV features use fixed 15-contract grid identical to reference_table.py

### Experiment Scripts
- exp1: Ranks all 42 cuts; saves exp1_cut_rankings.csv, two PNGs
- exp2: ATM-IV predictive coverage for ranks 1/19/42; saves exp2_coverage_table.csv, PNG
- exp3: ABI vs ABC-MCMC; correct/misspecified scenarios; saves exp3_coverage_table.csv, runtime CSV, PNG
- exp4: Semi-empirical rolling windows (T_LONG=2500, STEP=100); saves exp4_coverage_table.csv, two PNGs
- exp5: Sub-studies A/B/C (ABC vs ABI, misspecified sim, GMM variants); saves 3 CSVs + summary PNG

## Test Suite
All tests are self-contained with small synthetic data, use np.random.seed for determinism,
and contain no real market data. Run with: `python -m pytest tests/ -v`

- test_heston.py (5 tests): shapes, non-negativity, BV formula, reproducibility
- test_option_pricing.py (5 tests): BS positivity, IV roundtrip, Heston price, grid shape, parity
- test_summary_stats.py (4 tests): output shapes, known-input check, finiteness
- test_cut_structures.py (5 tests): 42 count, constraint, partition validity, mass range, uniqueness
- test_gmm_abi.py (4 tests): returns GaussianMixture, weights sum to 1, sample shape, finiteness

## Known Issues / Debugging Notes

### Matplotlib ≥ 3.9 Breaking Change
`ax.boxplot(data, labels=...)` → use `tick_labels=` instead. All boxplot calls updated in exp5.

### ABC-MCMC in exp3
- Pass raw (un-normalised) summary stats to `_log_pseudo_lik()` in init; normalisation done inside.
- With T_sim=50 (vs reference-table T_window=500), MCMC coverage is degraded (~17%) because 
  the summary-space mismatch. This is a feasibility tradeoff documented in results.
- Bandwidth h computed as median distance in reference-table summary space / sqrt(dim_S).

### exp5 misspecified simulator
Severe misspecification (noise_std factor = large) causes σ_v to become negative/zero in the 
posterior — clip results or flag as invalid. This is expected and shows the limits of ABI.

### Reference-table size R=300
The paper uses R=10,000. With R=300, BIC typically selects 2 GMM components (vs higher with 
more data). Posterior masses are lower in absolute terms but relative rankings are preserved.

### Option pricing with scipy.integrate.quad
Heston characteristic function integration uses `quad` with `limit=200`. For extreme parameter 
values (κ very small, σ_v very large), integration may fail → always clip to prior support.
