import numpy as np
from typing import Tuple, Optional

# Implementer choices (documented):
#   N_PARTICLES = 500        number of particles
#   RESAMPLE_THRESHOLD = 0.5 ESS/N threshold; resample when ESS < threshold * N
#   dt = 1/252               one trading day (annual convention)
#   V0 initialisation: V_0^(i) ~ |N(theta, 0.005^2)| (folded normal, ensures positivity)
#   Variance propagation: full-truncation Euler — max(V_t, 0) used in drift AND diffusion

N_PARTICLES = 500
RESAMPLE_THRESHOLD = 0.5
_DT = 1.0 / 252.0


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Systematic resampling.
    weights: (N,) normalised probability weights
    Returns: (N,) integer index array
    """
    N = len(weights)
    positions = (rng.uniform(0.0, 1.0) + np.arange(N)) / N
    cumsum = np.cumsum(weights)
    indices = np.searchsorted(cumsum, positions)
    return np.clip(indices, 0, N - 1)


def _effective_sample_size(log_weights: np.ndarray) -> float:
    """ESS = 1 / sum(w_i^2) from normalised weights derived from log_weights."""
    log_w = log_weights - log_weights.max()
    w = np.exp(log_w)
    w /= w.sum()
    return float(1.0 / np.sum(w ** 2))


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def propagate_particles_one_step(
    V_particles: np.ndarray,  # (N,) current variance particles
    params,                   # HestonParams (duck-typed)
    dt: float = _DT,
    seed: Optional[int] = None,
    n_steps: int = 1,
) -> np.ndarray:
    """
    Propagate variance particles forward one day using full-truncation Euler.

    Full-truncation Euler discretisation of:
        dV_t = kappa*(theta - V_t)*dt + sigma_v*sqrt(V_t)*dW2_t

    At each sub-step:
        V_next = V + kappa*(theta - V_+)*dt + sigma_v*sqrt(V_+)*sqrt(dt)*Z
    where V_+ = max(V, 0).

    Returns: (N,) next-day variance particles (non-negative)
    """
    rng = np.random.default_rng(seed)
    kappa = params.kappa
    theta = params.theta
    sigma_v = params.sigma_v
    sub_dt = dt / n_steps

    V = V_particles.copy()
    for _ in range(n_steps):
        V_pos = np.maximum(V, 0.0)
        Z = rng.standard_normal(len(V))
        V = V + kappa * (theta - V_pos) * sub_dt + sigma_v * np.sqrt(V_pos * sub_dt) * Z
        # Full truncation: do NOT floor here; keep raw value (can go negative),
        # but use V_pos inside the step. The log-likelihood uses the raw V.
        V = np.maximum(V, 0.0)  # absorbing barrier at 0 (alternative: reflect)

    return V


def bootstrap_particle_filter(
    returns: np.ndarray,   # (T,) observed daily returns
    params,                # HestonParams (duck-typed)
    n_particles: int = N_PARTICLES,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap particle filter for Heston latent variance.

    State model:  dV_t = kappa*(theta-V_t)*dt + sigma_v*sqrt(V_t)*dW2_t
    Observation:  r_t | V_t ~ N(mu*dt, V_t*dt)

    Initialisation:
        V_0^(i) ~ |N(theta, 0.005^2)|   (folded normal ensures V_0 > 0)

    For t = 1, ..., T:
        Predict: V_t^(i) = propagate(V_{t-1}^(i))
        Update:  log w_t^(i) = log N(r_t; mu*dt, V_t^(i)*dt)
        Resample: systematic resampling when ESS < RESAMPLE_THRESHOLD * N

    Returns:
        filtered_vars: (T, n_particles)  particle clouds for V_t
        log_weights:   (T, n_particles)  log (unnormalised) weights
    """
    rng = np.random.default_rng(seed)
    T = len(returns)
    N = n_particles

    kappa = params.kappa
    theta = params.theta
    sigma_v = params.sigma_v
    mu = getattr(params, 'mu', 0.0)
    dt = _DT

    # Initialise particles: |N(theta, 0.005^2)|
    V = np.abs(rng.normal(loc=theta, scale=0.005, size=N))

    filtered_vars = np.empty((T, N))
    log_weights_out = np.empty((T, N))

    log_w = np.zeros(N)  # equal weights initially

    for t in range(T):
        # Predict: propagate each particle one step
        V = propagate_particles_one_step(
            V_particles=V,
            params=params,
            dt=dt,
            seed=int(rng.integers(0, 2**31)),
        )

        # Update: p(r_t | V_t^(i)) = N(r_t; mu*dt, V_t^(i)*dt)
        obs_var = np.maximum(V * dt, 1e-12)   # prevent zero variance
        obs_std = np.sqrt(obs_var)
        log_lik = -0.5 * (
            np.log(2.0 * np.pi * obs_var)
            + (returns[t] - mu * dt) ** 2 / obs_var
        )
        log_w = log_w + log_lik

        # Normalise for storage and ESS
        log_w_stable = log_w - log_w.max()
        w_norm = np.exp(log_w_stable)
        w_norm /= w_norm.sum()

        filtered_vars[t] = V.copy()
        log_weights_out[t] = log_w.copy()

        # Resample if ESS is too low
        ess = 1.0 / np.sum(w_norm ** 2)
        if ess < RESAMPLE_THRESHOLD * N:
            idx = _systematic_resample(w_norm, rng)
            V = V[idx]
            log_w = np.zeros(N)  # reset weights after resampling

    return filtered_vars, log_weights_out


def get_filtered_variance(
    filtered_vars: np.ndarray,  # (T, N)
    log_weights: np.ndarray,    # (T, N)
) -> np.ndarray:
    """
    Compute the weighted-mean variance estimate from the particle cloud.

    For each time step t:
        V_hat_t = sum_i w_t^(i) * V_t^(i)   (normalised weights)

    Returns: (T,) mean variance estimates
    """
    # Numerically stable normalisation row by row
    log_w = log_weights - log_weights.max(axis=1, keepdims=True)
    w = np.exp(log_w)
    w /= w.sum(axis=1, keepdims=True)
    return np.sum(w * filtered_vars, axis=1)
