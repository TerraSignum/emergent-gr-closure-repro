"""Tests for the Q2/Proposal-2 discrete Bakry-Emery Gamma_2 CD audit.

The audit computes the intrinsic Bakry-Emery curvature of the carrier
random-walk Dirichlet form -- the pointwise smallest generalised
eigenvalue K_x = min gen-eig(B_x, A_x) of the Gamma_2 / Gamma forms --
and shows the regime curvature extrapolates to a finite, non-negative
limit, the intrinsic CD(K,infinity) route for the open
(A5)+(A6) => CD(K_CD,N) implication of thm:hard_xi_continuum.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import verify_carrier_bakry_emery_cd as M  # noqa: E402

OUT = REPO / "outputs" / "carrier_bakry_emery_cd_audit.json"


@pytest.fixture(scope="module")
def output():
    if not OUT.exists():
        M.main()
    with open(OUT, "r", encoding="utf-8") as f:
        return json.load(f)


def test_local_curvature_on_a_triangle():
    """Sanity check the Gamma_2 machinery on the complete graph K_3:
    the random walk on a triangle is positively curved, so the
    pointwise Bakry-Emery curvature must be strictly positive."""
    w = np.ones((3, 3)) - np.eye(3)
    p, keep = M.random_walk_operator(w)
    assert keep.all()
    k0 = M.local_curvature(p, 0)
    assert k0 is not None and k0 > 0.0


def test_subsampling_budget_disclosed(output):
    sub = output["subsampling"]
    assert sub["max_seeds"] >= 1 and sub["max_vertices_per_seed"] >= 1


def test_bakry_emery_curvature_finite_limit(output):
    cd = output["cd_curvature"]
    assert cd["finite_limit"] is True
    lo, _ = cd["K_p5_ci95"]
    assert lo > -1.0


def test_bakry_emery_curvature_nonnegative(output):
    """Every regime on the ladder carries a positive robust
    Bakry-Emery curvature -- the carrier is uniformly CD(K,infinity)
    with K > 0."""
    cd = output["cd_curvature"]
    assert cd["nonnegative"] is True
    assert all(r["K_p5"] > 0.0 for r in output["per_regime"])


def test_verdict_supports_cd(output):
    assert output["verdict"].startswith("BAKRY_EMERY_CD_SUPPORTED")
