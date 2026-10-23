import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from typing import Tuple, Optional, List

# Maximum GMM components to try
MAX_COMPONENTS = 30

# Minimum eigenvalue for conditional covariance regularisation
_MIN_EIGVAL = 1e-10


def fit_gmm_bic(
    data: np.ndarray,  # (N, D) joint (params, summaries)
    max_components: int = MAX_COMPONENTS,
    reg_covar: float = 1e-4,
    seed: int = 42,
    verbose: bool = False,
) -> GaussianMixture:
    """
    Fit Gaussian mixture using EM with BIC-based model selection.
    Tries 1..max_components components and selects best by BIC.
    Returns the fitted GaussianMixture object.
    """
    best_bic = np.inf
    best_gmm: Optional[GaussianMixture] = None

    for k in range(1, max_components + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            reg_covar=reg_covar,
            random_state=seed,
            max_iter=300,
            n_init=1,
        )
        gmm.fit(data)
        bic = gmm.bic(data)
        if verbose:
            print(f"  GMM k={k}: BIC={bic:.2f}")
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm

    return best_gmm


def compute_conditional_weights(
    gmm: GaussianMixture,
    x_obs: np.ndarray,   # (D_obs,)
    x_obs_idx: np.ndarray,
) -> np.ndarray:
    """
    Compute posterior mixture component weights given observed x_obs.
    w_l_posterior ∝ w_l * N(x_obs; mu_l_obs, Sigma_l_obs)
    Returns normalised weight vector (K,)
    """
    K = gmm.n_components
    log_weights = np.empty(K)

    for l in range(K):
        mu_l_o = gmm.means_[l][x_obs_idx]
        Sigma_l_oo = gmm.covariances_[l][np.ix_(x_obs_idx, x_obs_idx)]
        log_p = multivariate_normal.logpdf(x_obs, mean=mu_l_o, cov=Sigma_l_oo, allow_singular=True)
        log_weights[l] = np.log(gmm.weights_[l] + 1e-300) + log_p

    # Numerically stable softmax
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    weights /= weights.sum()
    return weights


def compute_conditional_params(
    gmm: GaussianMixture,
    x_obs: np.ndarray,       # (D_obs,)
    x_obs_idx: np.ndarray,
    x_target_idx: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Compute conditional means, covariances, and weights for all components.

    For each component l:
      mu_l_t|o  = mu_l_t + Sigma_l_to * Sigma_l_oo^{-1} * (x_obs - mu_l_o)
      Sigma_l_t|o = Sigma_l_tt - Sigma_l_to * Sigma_l_oo^{-1} * Sigma_l_ot

    Returns: (cond_means, cond_covs, cond_weights)
      cond_means: list of K arrays, each (D_target,)
      cond_covs:  list of K arrays, each (D_target, D_target)
      cond_weights: (K,) normalised posterior weights
    """
    K = gmm.n_components
    cond_weights = compute_conditional_weights(gmm, x_obs, x_obs_idx)
    cond_means = []
    cond_covs = []

    for l in range(K):
        mu_l = gmm.means_[l]
        Sigma_l = gmm.covariances_[l]

        mu_l_t = mu_l[x_target_idx]
        mu_l_o = mu_l[x_obs_idx]

        Sigma_l_tt = Sigma_l[np.ix_(x_target_idx, x_target_idx)]
        Sigma_l_oo = Sigma_l[np.ix_(x_obs_idx, x_obs_idx)]
        Sigma_l_to = Sigma_l[np.ix_(x_target_idx, x_obs_idx)]

        # Solve Sigma_l_oo * A^T = Sigma_l_to^T  =>  A = Sigma_l_to @ inv(Sigma_l_oo)
        try:
            A_l = np.linalg.solve(Sigma_l_oo.T, Sigma_l_to.T).T
        except np.linalg.LinAlgError:
            A_l = Sigma_l_to @ np.linalg.pinv(Sigma_l_oo)

        cond_mean = mu_l_t + A_l @ (x_obs - mu_l_o)
        cond_cov = Sigma_l_tt - A_l @ Sigma_l_to.T

        # Regularise: clip negative eigenvalues
        cond_cov = _regularise_cov(cond_cov)

        cond_means.append(cond_mean)
        cond_covs.append(cond_cov)

    return cond_means, cond_covs, cond_weights


def _regularise_cov(cov: np.ndarray) -> np.ndarray:
    """Symmetrise and clip negative eigenvalues to _MIN_EIGVAL."""
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, _MIN_EIGVAL, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def gmm_conditional_sample(
    gmm: GaussianMixture,
    x_obs: np.ndarray,       # (D_obs,) observed conditioning variables
    x_obs_idx: np.ndarray,   # indices of conditioning variables in joint
    x_target_idx: np.ndarray,  # indices of target variables in joint
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample from the conditional distribution p(x_target | x_obs) using
    Gaussian mixture conditional distributions.

    Algorithm:
      1. Compute posterior component weights w_l ∝ w_l * N(x_obs; mu_l_o, Sigma_l_oo)
      2. Sample component indices according to weights
      3. For each component, sample from the conditional Gaussian
         N(mu_l_t|o, Sigma_l_t|o) with eigenvalue-clipped covariance

    Returns: (n_samples, len(x_target_idx)) samples from conditional
    """
    rng = np.random.default_rng(seed)
    cond_means, cond_covs, cond_weights = compute_conditional_params(
        gmm, x_obs, x_obs_idx, x_target_idx
    )

    # Sample component indices
    comp_indices = rng.choice(len(cond_weights), size=n_samples, p=cond_weights)

    D_t = len(x_target_idx)
    samples = np.empty((n_samples, D_t))

    for l in range(gmm.n_components):
        mask = comp_indices == l
        n_l = mask.sum()
        if n_l == 0:
            continue
        samples[mask] = rng.multivariate_normal(
            mean=cond_means[l],
            cov=cond_covs[l],
            size=n_l,
        )

    return samples


def gmm_batch_conditional_sample(
    gmm: GaussianMixture,
    x_obs_batch: np.ndarray,   # (N, D_obs)
    x_obs_idx: np.ndarray,
    x_target_idx: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    For each row in x_obs_batch draw exactly one sample from q(x_target | x_obs).

    Precomputes the component-invariant matrices (A_l, Sigma_l_cond, Cholesky
    factors of Sigma_l_oo) outside the per-sample loop so that the inner work
    per sample is O(K * D).

    Returns: (N, D_target)
    """
    K = gmm.n_components
    N = len(x_obs_batch)
    D_t = len(x_target_idx)

    # Precompute per-component static quantities
    A_list = []          # A_l = Sigma_l_to @ inv(Sigma_l_oo), shape (D_t, D_o)
    cond_cov_list = []   # Sigma_l_t|o, shape (D_t, D_t)
    mu_t_list = []       # mu_l_t, shape (D_t,)
    mu_o_list = []       # mu_l_o, shape (D_o,)
    L_oo_list = []       # chol(Sigma_l_oo) for log-likelihood, shape (D_o, D_o)
    Sigma_oo_list = []   # Sigma_l_oo, shape (D_o, D_o)

    for l in range(K):
        mu_l = gmm.means_[l]
        Sigma_l = gmm.covariances_[l]

        mu_l_t = mu_l[x_target_idx]
        mu_l_o = mu_l[x_obs_idx]
        Sigma_l_tt = Sigma_l[np.ix_(x_target_idx, x_target_idx)]
        Sigma_l_oo = Sigma_l[np.ix_(x_obs_idx, x_obs_idx)]
        Sigma_l_to = Sigma_l[np.ix_(x_target_idx, x_obs_idx)]

        try:
            A_l = np.linalg.solve(Sigma_l_oo.T, Sigma_l_to.T).T
        except np.linalg.LinAlgError:
            A_l = Sigma_l_to @ np.linalg.pinv(Sigma_l_oo)

        cond_cov = _regularise_cov(Sigma_l_tt - A_l @ Sigma_l_to.T)

        try:
            L_oo = np.linalg.cholesky(Sigma_l_oo)
        except np.linalg.LinAlgError:
            L_oo = np.linalg.cholesky(Sigma_l_oo + np.eye(len(x_obs_idx)) * 1e-8)

        A_list.append(A_l)
        cond_cov_list.append(cond_cov)
        mu_t_list.append(mu_l_t)
        mu_o_list.append(mu_l_o)
        L_oo_list.append(L_oo)
        Sigma_oo_list.append(Sigma_l_oo)

    # Compute log-weights for all samples and all components: (N, K)
    log_pi = np.log(gmm.weights_ + 1e-300)
    log_w_matrix = np.full((N, K), -np.inf)

    for l in range(K):
        diff = x_obs_batch - mu_o_list[l]  # (N, D_o)
        # Mahalanobis via triangular solve: L_oo @ z = diff^T  => z = L_oo^{-1} diff^T
        z = np.linalg.solve(L_oo_list[l], diff.T)  # (D_o, N)
        mah2 = np.sum(z ** 2, axis=0)               # (N,)
        D_o = len(x_obs_idx)
        log_det = 2.0 * np.sum(np.log(np.diag(L_oo_list[l])))
        log_lik = -0.5 * (D_o * np.log(2.0 * np.pi) + log_det + mah2)
        log_w_matrix[:, l] = log_pi[l] + log_lik

    # Numerically stable row-wise softmax
    log_w_matrix -= log_w_matrix.max(axis=1, keepdims=True)
    w_matrix = np.exp(log_w_matrix)
    w_matrix /= w_matrix.sum(axis=1, keepdims=True)

    # Sample component for each observation
    cumw = np.cumsum(w_matrix, axis=1)
    u = rng.uniform(0.0, 1.0, size=N)
    comp_indices = (u[:, None] > cumw).sum(axis=1)
    comp_indices = np.clip(comp_indices, 0, K - 1)

    # Sample one value per observation from its conditional Gaussian
    samples = np.empty((N, D_t))
    for l in range(K):
        mask = comp_indices == l
        n_l = int(mask.sum())
        if n_l == 0:
            continue
        # Conditional means for this component and these observations
        diff_l = x_obs_batch[mask] - mu_o_list[l]  # (n_l, D_o)
        mu_cond = mu_t_list[l] + diff_l @ A_list[l].T  # (n_l, D_t)
        # Sample from N(mu_cond, Sigma_l_cond) by drawing z ~ N(0,I) and
        # computing mu_cond + L_cond @ z  (L_cond = chol of cond_cov_list[l])
        try:
            L_cond = np.linalg.cholesky(cond_cov_list[l])
        except np.linalg.LinAlgError:
            L_cond = np.linalg.cholesky(
                cond_cov_list[l] + np.eye(D_t) * 1e-8
            )
        z = rng.standard_normal((n_l, D_t))
        samples[mask] = mu_cond + z @ L_cond.T

    return samples
