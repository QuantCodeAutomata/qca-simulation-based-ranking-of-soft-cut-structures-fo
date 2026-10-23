"""Tests for src.cut_structures."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.cut_structures import (
    enumerate_feasible_cuts,
    get_summary_indices,
    compute_posterior_mass,
    rank_cuts_by_posterior_mass,
    CutStructure,
    N_PARAMS,
    N_BLOCKS,
    _LAMBDA_IDX,
    _SO_BLOCK_IDX,
)


def test_enumerate_cuts_count():
    """There must be exactly 42 feasible cut structures."""
    cuts = enumerate_feasible_cuts()
    assert len(cuts) == 42, f"Expected 42 cuts, got {len(cuts)}"


def test_lambda_option_constraint():
    """For every cut: lambda in phi1 ↔ S_O in s1 (lambda-option pairing)."""
    cuts = enumerate_feasible_cuts()
    for cut in cuts:
        lam_in_phi1 = (_LAMBDA_IDX in cut.phi1_params)
        so_in_s1    = (_SO_BLOCK_IDX in cut.s1_blocks)
        assert lam_in_phi1 == so_in_s1, (
            f"Cut {cut.cut_id}: lambda_in_phi1={lam_in_phi1} "
            f"but so_in_s1={so_in_s1} (constraint violated)"
        )


def test_cuts_are_valid_partitions():
    """phi1 ∪ phi2 = all params, s1 ∪ s2 = all blocks; no overlap."""
    all_params = set(range(N_PARAMS))
    all_blocks = set(range(N_BLOCKS))
    cuts = enumerate_feasible_cuts()

    for cut in cuts:
        p1, p2 = set(cut.phi1_params), set(cut.phi2_params)
        b1, b2 = set(cut.s1_blocks),   set(cut.s2_blocks)

        assert p1 | p2 == all_params, (
            f"Cut {cut.cut_id}: params union {p1|p2} != {all_params}"
        )
        assert p1 & p2 == set(), (
            f"Cut {cut.cut_id}: params overlap {p1&p2}"
        )
        assert b1 | b2 == all_blocks, (
            f"Cut {cut.cut_id}: blocks union {b1|b2} != {all_blocks}"
        )
        assert b1 & b2 == set(), (
            f"Cut {cut.cut_id}: blocks overlap {b1&b2}"
        )
        assert len(p1) > 0 and len(p2) > 0, "Empty partition not allowed"
        assert len(b1) > 0 and len(b2) > 0, "Empty partition not allowed"


def test_posterior_mass_range():
    """Posterior mass criterion must lie in [0, 1]."""
    np.random.seed(20)
    # Samples near the true parameter values → high mass
    samples_good = np.column_stack([
        np.random.normal(0.05, 0.005, 200),    # kappa ≈ 0.05
        np.random.normal(0.03, 0.003, 200),    # theta ≈ 0.03
        np.random.normal(0.03, 0.003, 200),    # sigma_v ≈ 0.03
        np.random.normal(-0.02, 0.005, 200),   # lambda ≈ -0.02
    ])
    # Samples far from true values → low mass
    samples_bad = np.column_stack([
        np.random.normal(0.50, 0.01, 200),
        np.random.normal(0.20, 0.01, 200),
        np.random.normal(0.20, 0.01, 200),
        np.random.normal(-0.40, 0.01, 200),
    ])

    mass_good = compute_posterior_mass(samples_good)
    mass_bad  = compute_posterior_mass(samples_bad)

    assert 0.0 <= mass_good <= 1.0, f"mass_good={mass_good} outside [0,1]"
    assert 0.0 <= mass_bad  <= 1.0, f"mass_bad={mass_bad} outside [0,1]"
    assert mass_good > mass_bad, (
        f"Expected mass_good={mass_good:.3f} > mass_bad={mass_bad:.3f}"
    )


def test_cut_uniqueness():
    """No two cut structures should be identical (same phi1, phi2, s1, s2)."""
    cuts = enumerate_feasible_cuts()
    seen = set()
    for cut in cuts:
        key = (cut.phi1_params, cut.phi2_params, cut.s1_blocks, cut.s2_blocks)
        assert key not in seen, f"Duplicate cut: {key}"
        seen.add(key)
    assert len(seen) == 42
