"""Tests for the Q2/Proposal-1 signed Ricci lower-bound audit.

The audit extracts the per-node minimum eigenvalue of the signed
Hessian-discrepancy Ricci tensor R^Hess and shows the regime
curvature lower bound K_N = inf_a lambda_min(a) extrapolates to a
finite, non-negative limit -- the signed information missing from
admissibility condition (A5) of thm:hard_xi_continuum.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import verify_signed_ricci_lower_bound as M  # noqa: E402

OUT = REPO / "outputs" / "signed_ricci_lower_bound_audit.json"


@pytest.fixture(scope="module")
def output():
    if not OUT.exists():
        M.main()
    with open(OUT, "r", encoding="utf-8") as f:
        return json.load(f)


def test_canonical_ladder_n50_to_n512(output):
    ns = [n for _, n in output["canonical_ladder"]]
    assert min(ns) == 50 and max(ns) == 512


def test_curvature_lower_bound_has_finite_limit(output):
    """K_p1 does not diverge to -infinity."""
    cd = output["cd_lower_bound"]
    assert cd["K_p1_finite_limit"] is True
    lo, _ = cd["K_p1_ci95"]
    assert lo > -1.0


def test_hyperbolic_node_fraction_vanishes(output):
    """f_neg -> 0: at large N no node has negative minimum Ricci
    eigenvalue."""
    f_neg = output["symanzik_fits"]["f_neg"]
    assert abs(f_neg["y_inf"]) < 5e-3
    # Large-N regimes should have essentially zero hyperbolic nodes.
    big = [r for r in output["per_regime"] if r["N"] >= 128]
    assert all(r["f_neg"] < 0.01 for r in big)


def test_curvature_lower_bound_point_estimate_nonnegative(output):
    """The K_p1 point estimate is non-negative (the CI may straddle
    zero -- that honest distinction is carried by separate flags)."""
    cd = output["cd_lower_bound"]
    assert cd["K_p1_point_nonnegative"] is True
    # Exactly one of the two sign tiers must hold.
    assert (cd["K_p1_ci_strictly_nonnegative"]
            ^ cd["K_p1_ci_straddles_zero"]) or not cd["K_p1_finite_limit"]


def test_verdict_is_a_finite_cd_witness(output):
    """The signed-Ricci route certifies at least a finite limit;
    whether it is strictly non-negative or only CI-consistent with
    zero is encoded in the verdict suffix."""
    v = output["verdict"]
    assert v in {
        "CD_LOWER_BOUND_SUPPORTED_NONNEGATIVE",
        "CD_LOWER_BOUND_FINITE_CI_CONSISTENT_WITH_ZERO",
        "CD_LOWER_BOUND_SUPPORTED_FINITE_NEGATIVE",
    }
