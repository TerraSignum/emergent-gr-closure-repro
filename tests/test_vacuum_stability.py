"""Tests for the bundled vacuum-stability + reheating closures."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import verify_vacuum_stability as M  # noqa: E402


@pytest.fixture(scope="module")
def bundle():
    return M.load_bundle()


@pytest.fixture(scope="module")
def output(bundle):
    M.main()
    out_path = REPO / "outputs" / "vacuum_stability_recompute.json"
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_three_regimes_all_absolutely_stable(bundle):
    v = bundle["vacuum_stability_closure"]
    regimes = v["regime_results"]
    assert len(regimes) == 3
    for r in regimes:
        assert r["verdict"] == "absolutely stable"
        assert r["B_action"] == "infinity"
        assert r["tunnelling_channel"] == "none"
        assert "monoton" in r["barrier_shape"]


def test_vacuum_stability_tier_derived(bundle):
    assert bundle["vacuum_stability_closure"]["tier"] == "DERIVED"


def test_reheating_T_RH_PRECISE(bundle):
    """T_RH ratio to Planck 3-sigma anchor must be inside PRECISE 2.5%."""
    r = bundle["reheating_closure"]
    ratio = r["T_RH_ratio_to_anchor"]
    assert abs(ratio - 1.0) <= 0.025
    assert r["tier"] == "PRECISE"


def test_reheating_universal_gravitational_channel(bundle):
    r = bundle["reheating_closure"]
    formula = r["Gamma_grav_formula"]
    assert "N_dof" in formula
    assert "m_phi" in formula
    assert "M_Pl" in formula
    assert r["Gamma_grav_GeV"] == pytest.approx(1.89e15, rel=1e-3)


def test_reheating_T_RH_value(bundle):
    """T_RH = 6.61e16 GeV is the bundled prediction."""
    r = bundle["reheating_closure"]
    assert r["T_RH_predicted_GeV"] == pytest.approx(6.61e16, rel=1e-3)


def test_zero_fitted_parameters(bundle):
    assert bundle["summary"]["fitted_parameters"] == 0


def test_recompute_output_passes(output):
    assert output["verdict"] == "PASS"
    assert output["all_regimes_absolutely_stable"] is True
    assert output["B_infinite_in_all_regimes"] is True
    assert output["T_RH_PRECISE"] is True
