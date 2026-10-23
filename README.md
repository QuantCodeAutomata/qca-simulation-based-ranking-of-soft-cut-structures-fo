# Modular ABI Framework for the Heston Stochastic Volatility Model

A modular Approximate Bayesian Inference (ABI) framework that exploits the
cut-posterior decomposition of the Heston (1993) model, separating inference
on physical-measure parameters from risk-neutral parameters via a structured
partition of summary statistics.

## Overview

The Heston model couples a log-price process with a CIR variance process:

```
dS_t = μ S_t dt + √V_t S_t dW_t^S
dV_t = κ(θ - V_t) dt + σ_v √V_t dW_t^V,   corr(dW^S, dW^V) = ρ dt
```

Parameters: `κ` (mean-reversion speed), `θ` (long-run variance), `σ_v` (vol-of-vol),
`λ` (market price of variance risk), `ρ` (leverage), `μ` (drift).

Under the modular ABI framework, the six parameters are partitioned into two
modules. Module 1 is informed by return and bipower-variation (BV) summaries;
Module 2 adds option-implied-volatility (IV) surface features to identify the
risk-neutral parameters, with the constraint that `λ` always resides in the
option module. Across 6 parameter splits × 7 option-maturity splits this
yields **42 feasible cut structures**, which are ranked by relative posterior mass.

## Repository Layout

```
project/
├── data/
│   ├── __init__.py
│   └── generate_synthetic.py   # generate and serialise the master dataset
├── exp/
│   ├── __init__.py
│   ├── exp1_cut_ranking.py     # rank all 42 cuts by posterior mass
│   ├── exp2_predictive_coverage.py  # out-of-sample option-price coverage
│   ├── exp3_abi_vs_mcmc.py     # ABI vs simplified MCMC benchmark
│   ├── exp4_empirical.py       # semi-empirical rolling-window study
│   └── exp5_robustness.py      # robustness: ABC / misspecification / EM
├── results/                    # outputs — gitignored
├── src/
│   ├── __init__.py
│   ├── utils.py                # set_seed, get_logger, Timer
│   ├── heston_simulator.py     # full-truncation Euler + BV
│   ├── option_pricing.py       # Heston char-func, BS-IV, option grid
│   ├── surface_regression.py   # polynomial IV surface (12 features)
│   ├── summary_stats.py        # 46-dim summary vector
│   ├── reference_table.py      # prior sampling + batch simulation
│   ├── gmm_abi.py              # GMM with BIC selection + conditional sampler
│   └── cut_structures.py       # 42 cut definitions + posterior mass
└── tests/
    ├── __init__.py
    ├── test_heston.py
    ├── test_option_pricing.py
    ├── test_summary_stats.py
    ├── test_cut_structures.py
    └── test_gmm_abi.py
```

## Summary Statistics (46-dimensional vector)

| Block | Indices | Content |
|-------|---------|---------|
| B1 – Returns      | 0–4   | mean, std, skew, kurt, AR(1) of daily log-returns |
| B2 – Bipower Var  | 5–9   | mean, std, skew, kurt, AR(1) of daily BV |
| B3 – Short IV     | 10–21 | 12 polynomial surface coefficients (maturities ≤ 42 d) |
| B4 – Medium IV    | 22–33 | 12 polynomial surface coefficients (63–84 d) |
| B5 – Long IV      | 34–45 | 12 polynomial surface coefficients (126–252 d) |

## Cut Structures (42 total = 6 × 7)

**Parameter splits** (6) – ways to assign `{κ, θ, σ_v, ρ, μ}` to module 1
(`λ` is always in module 2 with the option summaries):

| ID | φ₁ | φ₂ |
|----|----|----|
| 0 | {κ, θ} | {σ_v, λ, ρ, μ} |
| 1 | {σ_v} | {κ, θ, λ, ρ, μ} |
| 2 | {ρ, μ} | {κ, θ, σ_v, λ} |
| 3 | {κ, θ, σ_v} | {λ, ρ, μ} |
| 4 | {κ, θ, ρ, μ} | {σ_v, λ} |
| 5 | {σ_v, ρ, μ} | {κ, θ, λ} |

**Option-maturity splits** (7) – which IV maturity blocks go into module 2:

| ID | s₂ option blocks |
|----|-----------------|
| 0 | short (21–42 d) |
| 1 | medium (63–84 d) |
| 2 | long (126–252 d) |
| 3 | short + medium |
| 4 | short + long |
| 5 | medium + long |
| 6 | short + medium + long |

Module 1 always receives the return and BV summaries plus any unassigned IV blocks.

## Option Grid

`generate_option_grid` produces **147 contracts** (7 maturities × 21 strikes):

- Maturities (trading days): 21, 42, 63, 84, 126, 189, 252
- Log-moneyness grid: power-law warped `k_norm = sign(x)|x|^0.7` over `x∈[-1,1]`,
  spanning ±3√(V_t τ)
- Risk-neutral mapping: `κ_rn = κ + λ`, `θ_rn = κθ/(κ+λ)`

## True Parameters (Synthetic Experiments)

```python
TRUE_PARAMS = HestonParams(
    kappa=0.05, theta=0.03, sigma_v=0.03,
    lam=-0.02,  rho=0.0,    mu=0.0,
    V0=0.03,    r=0.02
)
T = 1000      # trading days
M_INTRA = 78  # 5-minute bars per day (6.5 h session)
```

## Installation

```bash
pip install numpy scipy scikit-learn matplotlib seaborn pandas tqdm
# clone and run from project root
cd /workspace/project
python data/generate_synthetic.py
python exp/exp1_cut_ranking.py
python exp/exp2_predictive_coverage.py
python exp/exp3_abi_vs_mcmc.py
python exp/exp4_empirical.py
python exp/exp5_robustness.py
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Experiments Summary

| Script | Description | Key Output |
|--------|-------------|------------|
| `exp1_cut_ranking.py` | Rank 42 cuts by posterior mass | `exp1_cut_rankings.csv` |
| `exp2_predictive_coverage.py` | Out-of-sample option-price coverage for best / middle / worst cut | `exp2_coverage_table.csv` |
| `exp3_abi_vs_mcmc.py` | ABI vs simplified MCMC; correct vs misspecified noise | `exp3_coverage_table.csv` |
| `exp4_empirical.py` | Semi-empirical rolling 500-day window on realistic synthetic data | `exp4_coverage_table.csv` |
| `exp5_robustness.py` | ABC vs ABI; misspecified simulator; EM regularisation variants | `exp5_robustness_summary.png` |

## References

- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility.
  *Review of Financial Studies*, 6(2), 327–343.
- Barndorff-Nielsen, O. E., & Shephard, N. (2004). Power and bipower variation with
  stochastic volatility and jumps. *Journal of Financial Econometrics*, 2(1), 1–37.
- Jacob, P. E., Murray, L. M., Holmes, C. C., & Robert, C. P. (2017). Better together?
  Statistical learning in models made of modules. *arXiv:1708.08719*.
