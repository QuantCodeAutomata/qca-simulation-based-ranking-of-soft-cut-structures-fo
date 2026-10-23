import numpy as np
from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple, Dict, Optional
from sklearn.mixture import GaussianMixture

from src.gmm_abi import gmm_conditional_sample, gmm_batch_conditional_sample

# Parameter index map (position in the 4-element parameter vector)
PARAM_NAMES = ['kappa', 'theta', 'sigma_v', 'lambda']
# Summary block index map
SUMMARY_BLOCKS = ['S_r', 'S_HF', 'S_O']

N_PARAMS = 4
N_BLOCKS = 3

# Specific indices referenced by the constraint logic
_LAMBDA_IDX = 3   # lambda is param index 3
_SO_BLOCK_IDX = 2  # S_O is block index 2


@dataclass
class CutStructure:
    phi1_params: Tuple[int, ...]  # indices of params in module 1
    phi2_params: Tuple[int, ...]  # indices of params in module 2
    s1_blocks: Tuple[int, ...]    # summary block indices in module 1
    s2_blocks: Tuple[int, ...]    # summary block indices in module 2
    cut_id: int


def enumerate_feasible_cuts() -> List[CutStructure]:
    """
    Enumerate all feasible two-module cut structures.

    Constraints:
    1. phi1 and phi2 partition {kappa, theta, sigma_v, lambda} (both non-empty)
    2. s1 and s2 partition {S_r, S_HF, S_O} (both non-empty)
    3. Lambda-option constraint: lambda in phi1 <=> S_O in s1

    Returns a list of 42 CutStructure objects.
    """
    all_params = set(range(N_PARAMS))   # {0, 1, 2, 3}
    all_blocks = set(range(N_BLOCKS))   # {0, 1, 2}

    feasible: List[CutStructure] = []
    cut_id = 0

    # Enumerate all non-trivial subsets of params for phi1
    for size in range(1, N_PARAMS):           # phi1 has 1..3 elements
        for phi1 in combinations(range(N_PARAMS), size):
            phi1 = tuple(phi1)
            phi2 = tuple(sorted(all_params - set(phi1)))
            if not phi2:
                continue

            lambda_in_phi1 = (_LAMBDA_IDX in phi1)

            # Enumerate all non-trivial subsets of blocks for s1
            for bsize in range(1, N_BLOCKS):      # s1 has 1..2 blocks
                for s1 in combinations(range(N_BLOCKS), bsize):
                    s1 = tuple(s1)
                    s2 = tuple(sorted(all_blocks - set(s1)))
                    if not s2:
                        continue

                    so_in_s1 = (_SO_BLOCK_IDX in s1)

                    # Lambda-option constraint: lambda in phi1 <=> S_O in s1
                    if lambda_in_phi1 != so_in_s1:
                        continue

                    feasible.append(CutStructure(
                        phi1_params=phi1,
                        phi2_params=phi2,
                        s1_blocks=s1,
                        s2_blocks=s2,
                        cut_id=cut_id,
                    ))
                    cut_id += 1

    return feasible


def get_summary_indices(
    block_indices: Tuple[int, ...],
    summary_block_sizes: Dict[str, int],
) -> np.ndarray:
    """
    Convert summary block indices to actual positions in the full summary vector.

    Block 0 (S_r)  -> first summary_block_sizes['S_r'] entries
    Block 1 (S_HF) -> next  summary_block_sizes['S_HF'] entries
    Block 2 (S_O)  -> next  summary_block_sizes['S_O']  entries

    Returns index array into the full summary vector.
    """
    block_names = SUMMARY_BLOCKS  # ['S_r', 'S_HF', 'S_O']
    block_sizes = [summary_block_sizes[b] for b in block_names]

    # Start offsets for each block
    offsets = np.concatenate([[0], np.cumsum(block_sizes[:-1])]).astype(int)

    indices = []
    for b in block_indices:
        start = offsets[b]
        size = block_sizes[b]
        indices.extend(range(start, start + size))

    return np.array(indices, dtype=int)


def compute_cut_posterior_samples(
    cut: CutStructure,
    gmm: GaussianMixture,
    S_obs_full: np.ndarray,      # full observed summary vector (D_S,)
    n_params: int,                # number of parameters (4)
    summary_block_sizes: Dict[str, int],  # sizes of S_r, S_HF, S_O
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Compute cut posterior samples for a given cut structure.

    In the GMM the joint vector is ordered as [params (4), summaries (D_S)].

    Module 1 posterior: q(phi1 | S1)
      Condition on S1 summaries to draw phi1 samples.

    Module 2 posterior: q(phi2 | phi1, S_all)
      For each phi1 sample, condition on (phi1, S_all) to draw phi2.

    Returns: (n_samples, 4) posterior samples [kappa, theta, sigma_v, lambda]
    """
    rng = np.random.default_rng(seed)
    D_S = len(S_obs_full)

    # ------------------------------------------------------------------ #
    # Index bookkeeping                                                    #
    # In the joint vector: params at 0..n_params-1, summaries at n_params..
    # ------------------------------------------------------------------ #
    phi1_joint_idx = np.array(cut.phi1_params, dtype=int)
    phi2_joint_idx = np.array(cut.phi2_params, dtype=int)

    # Summary block indices for s1 and s2
    s1_local_idx = get_summary_indices(cut.s1_blocks, summary_block_sizes)
    # All summary indices in joint space (offset by n_params)
    all_summary_joint_idx = np.arange(n_params, n_params + D_S, dtype=int)
    s1_joint_idx = s1_local_idx + n_params

    # ------------------------------------------------------------------ #
    # Module 1: sample phi1 from q(phi1 | S1)                            #
    # ------------------------------------------------------------------ #
    phi1_samples = gmm_conditional_sample(
        gmm=gmm,
        x_obs=S_obs_full[s1_local_idx],
        x_obs_idx=s1_joint_idx,
        x_target_idx=phi1_joint_idx,
        n_samples=n_samples,
        seed=int(rng.integers(0, 2**31)),
    )
    # phi1_samples: (n_samples, len(phi1_params))

    # ------------------------------------------------------------------ #
    # Module 2: for each phi1^(i) sample phi2 from q(phi2 | phi1, S_all) #
    # ------------------------------------------------------------------ #
    # Build the conditioning block: [phi1_values, S_obs_full]
    # x_obs_idx for module 2: phi1 positions + all summary positions
    mod2_obs_idx = np.concatenate([phi1_joint_idx, all_summary_joint_idx])

    # Construct x_obs_batch: (n_samples, D_phi1 + D_S)
    # First columns = phi1 values; remaining = S_obs_full (same for all)
    S_obs_repeated = np.tile(S_obs_full, (n_samples, 1))       # (n_samples, D_S)
    x_obs_batch = np.concatenate([phi1_samples, S_obs_repeated], axis=1)

    phi2_samples = gmm_batch_conditional_sample(
        gmm=gmm,
        x_obs_batch=x_obs_batch,
        x_obs_idx=mod2_obs_idx,
        x_target_idx=phi2_joint_idx,
        rng=rng,
    )
    # phi2_samples: (n_samples, len(phi2_params))

    # ------------------------------------------------------------------ #
    # Assemble full parameter vector                                       #
    # ------------------------------------------------------------------ #
    out = np.zeros((n_samples, N_PARAMS))
    out[:, cut.phi1_params] = phi1_samples
    out[:, cut.phi2_params] = phi2_samples

    return out


def compute_posterior_mass(
    samples: np.ndarray,  # (n_samples, 4) [kappa, theta, sigma_v, lambda]
) -> float:
    """
    Compute relative posterior mass criterion:
    [P(theta in [0.02,0.04]) + P(kappa in [0.025,0.075]) +
     P(sigma_v in [0.025,0.035]) + P(lambda in [-0.03,0])] / 4
    """
    kappa_mass = np.mean((samples[:, 0] >= 0.025) & (samples[:, 0] <= 0.075))
    theta_mass = np.mean((samples[:, 1] >= 0.02) & (samples[:, 1] <= 0.04))
    sv_mass = np.mean((samples[:, 2] >= 0.025) & (samples[:, 2] <= 0.035))
    lam_mass = np.mean((samples[:, 3] >= -0.03) & (samples[:, 3] <= 0.0))
    return float((theta_mass + kappa_mass + sv_mass + lam_mass) / 4.0)


def rank_cuts_by_posterior_mass(
    cuts: List[CutStructure],
    mass_values: List[float],
) -> List[Tuple[int, float, CutStructure]]:
    """
    Rank cuts by posterior mass (descending).
    Returns sorted list of (rank, mass, cut) tuples (1-indexed rank).
    """
    paired = sorted(zip(mass_values, cuts), key=lambda x: x[0], reverse=True)
    return [(rank + 1, mass, cut) for rank, (mass, cut) in enumerate(paired)]
