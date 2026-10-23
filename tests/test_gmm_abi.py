"""Tests for src.gmm_abi."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.mixture import GaussianMixture

from src.gmm_abi import (
    fit_gmm_bic,
    compute_conditional_weights,
    gmm_conditional_sample,
)


def _make_synthetic_joint(n=300, d_phi=2, d_s=4, seed=0):
    """Create a small synthetic (phi, s) joint dataset.

    phi ~ N(0, I_{d_phi}), s | phi ~ N(A @ phi, 0.1*I_{d_s}),
    where A is a fixed random projection.  The joint is a 2-component GMM
    in practice (no labels needed; we just need a well-behaved dataset).
    """
    rng = np.random.default_rng(seed)
    A   = rng.standard_normal((d_s, d_phi))
    phi = rng.standard_normal((n, d_phi))
    s   = phi @ A.T + 0.1 * rng.standard_normal((n, d_s))
    return np.hstack([phi, s]), d_phi, d_s, A


def test_fit_gmm_bic_returns_gmm():
    """fit_gmm_bic must return a fitted GaussianMixture object."""
    np.random.seed(30)
    data, _, _, _ = _make_synthetic_joint(n=200, seed=30)

    gmm = fit_gmm_bic(data, max_components=5, seed=30)

    assert isinstance(gmm, GaussianMixture), (
        f"Expected GaussianMixture, got {type(gmm)}"
    )
    assert 1 <= gmm.n_components <= 5, (
        f"Selected components {gmm.n_components} outside [1, 5]"
    )
    # Means and covariances must be correctly shaped
    d = data.shape[1]
    assert gmm.means_.shape    == (gmm.n_components, d)
    assert gmm.covariances_.shape == (gmm.n_components, d, d)
    assert gmm.weights_.shape  == (gmm.n_components,)


def test_conditional_weights_sum_to_one():
    """Component weights conditioned on x_obs must sum to 1."""
    np.random.seed(31)
    data, d_phi, d_s, _ = _make_synthetic_joint(n=200, seed=31)

    gmm = fit_gmm_bic(data, max_components=5, seed=31)

    # Condition on the summary part (indices d_phi .. d_phi+d_s-1)
    s_obs_idx = np.arange(d_phi, d_phi + d_s)
    s_obs     = data[0, d_phi:]   # use first sample's summary as observation

    weights = compute_conditional_weights(gmm, s_obs, s_obs_idx)

    assert weights.shape == (gmm.n_components,), (
        f"Weights shape {weights.shape} != ({gmm.n_components},)"
    )
    assert abs(weights.sum() - 1.0) < 1e-10, (
        f"Weights sum to {weights.sum():.8f} != 1.0"
    )
    assert np.all(weights >= 0.0), "Some weights are negative"


def test_conditional_sample_shape():
    """gmm_conditional_sample must return array of shape (n_samples, d_target)."""
    np.random.seed(32)
    n_samples = 50
    data, d_phi, d_s, _ = _make_synthetic_joint(n=250, seed=32)

    gmm = fit_gmm_bic(data, max_components=5, seed=32)

    phi_idx = np.arange(d_phi)              # target: phi indices
    s_idx   = np.arange(d_phi, d_phi + d_s) # conditioning: s indices
    s_obs   = data[5, d_phi:]               # use 6th row's summary

    samples = gmm_conditional_sample(
        gmm=gmm,
        x_obs=s_obs,
        x_obs_idx=s_idx,
        x_target_idx=phi_idx,
        n_samples=n_samples,
        seed=32,
    )

    assert samples.shape == (n_samples, d_phi), (
        f"Sample shape {samples.shape} != ({n_samples}, {d_phi})"
    )


def test_gmm_conditional_finite():
    """Conditional samples must all be finite (no NaN or ±Inf)."""
    np.random.seed(33)
    data, d_phi, d_s, _ = _make_synthetic_joint(n=300, seed=33)

    gmm = fit_gmm_bic(data, max_components=8, seed=33)

    phi_idx = np.arange(d_phi)
    s_idx   = np.arange(d_phi, d_phi + d_s)
    s_obs   = data[10, d_phi:]

    samples = gmm_conditional_sample(
        gmm=gmm,
        x_obs=s_obs,
        x_obs_idx=s_idx,
        x_target_idx=phi_idx,
        n_samples=100,
        seed=33,
    )

    n_finite = int(np.isfinite(samples).all(axis=1).sum())
    assert n_finite == len(samples), (
        f"Only {n_finite}/{len(samples)} rows are fully finite"
    )
