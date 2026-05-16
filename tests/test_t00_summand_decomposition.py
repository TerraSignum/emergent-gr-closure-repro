"""Tests for the Q1/Proposal-1 per-summand T_00 decomposition audit.

The audit instruments the canonical Galerkin per-node T_00 scalar into
its four physical summands (edge-Xi row-variance, amplitude variance,
gradient energy, K/Q recoil), Symanzik-extrapolates each on the
canonical-physics ladder N in [50,512], and checks the per-summand
System-R identifications and their internal consistency against the
independent direct T_00 fit.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import verify_t00_summand_decomposition as M  # noqa: E402

OUT = REPO / "outputs" / "t00_summand_decomposition_audit.json"


@pytest.fixture(scope="module")
def output():
    if not OUT.exists():
        M.main()
    with open(OUT, "r", encoding="utf-8") as f:
        return json.load(f)


def test_canonical_ladder_includes_n50_to_n512(output):
    ns = [n for _, n in output["canonical_ladder"]]
    assert min(ns) == 50 and max(ns) == 512
    assert len(ns) >= 8


def test_summands_S2_S3_vanish_in_continuum(output):
    """Amplitude variance and gradient energy vanish in the limit."""
    for key in ("S2_var_amp", "S3_grad_psi_sq"):
        rep = output["summand_symanzik"][key]
        assert rep["structural_in_ci95"] is True
        lo, hi = rep["bootstrap_ci95"]
        assert lo <= 1e-3 and abs(rep["y_inf"]) < 5e-3


def test_S1_is_alpha_xi_sq_gamma_sq(output):
    """The leading summand S1 = 0.5 var_xi -> alpha_xi^2 * gamma^2."""
    rep = output["summand_symanzik"]["S1_half_var_xi"]
    assert rep["structural_id"] == "alpha_xi^2 * gamma^2"
    assert rep["structural_in_ci95"] is True
    assert rep["structural_rel_err"] < 0.025


def test_all_summand_structural_ids_within_ci(output):
    assert output["reconstruction"]["all_structural_ids_within_ci95"] is True


def test_decomposition_reproduces_direct_T00_fit(output):
    """The per-summand structural sum lands inside the independent
    direct T_00 Symanzik fit's bootstrap CI."""
    ic = output["internal_consistency"]
    assert ic["structural_sum_within_direct_ci95"] is True
    lo, hi = ic["direct_T00_bootstrap_ci95"]
    assert lo <= ic["structural_sum_value"] <= hi


def test_verdict_is_a_consistent_decomposition(output):
    assert output["verdict"].startswith("DECOMPOSITION_CONSISTENT")
