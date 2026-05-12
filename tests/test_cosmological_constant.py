"""Tests for the cosmological-constant 9-layer closure (122-OoM hierarchy)."""

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import verify_cosmological_constant as M  # noqa: E402


@pytest.fixture(scope="module")
def bundle():
    return M.load_bundle()


@pytest.fixture(scope="module")
def output(bundle):
    M.main()
    out_path = REPO / "outputs" / "cosmological_constant_recompute.json"
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_122_orders_of_magnitude_hierarchy(bundle):
    h = bundle["hierarchy_to_close"]
    assert h["orders_of_magnitude"] == 122


def test_six_dressing_layers_present(bundle):
    layers = bundle["six_dressing_layers"]
    assert len(layers) == 6
    names = [L["name"] for L in layers]
    assert "EW-hierarchy suppression" in names
    assert "Gamow vacuum sequestering" in names
    assert "Spectral-dimension IR screening" in names


def test_three_corrective_layers_present(bundle):
    cor = bundle["three_corrective_layers"]
    assert len(cor) == 3
    layer_ids = [L["layer"] for L in cor]
    assert layer_ids == ["H195", "H196", "H197"]


def test_closure_within_PRECISE_band(bundle):
    """Residual must be inside +/-0.1 OoM (i.e. within ~25% on rho)."""
    c = bundle["closure_result"]
    assert abs(c["residual_log10_orders"]) <= 0.1
    assert 0.5 <= c["ratio_predicted_over_observed"] <= 2.0
    assert c["tier"] == "PRECISE"


def test_zero_fitted_parameters(bundle):
    assert bundle["closure_result"]["fitted_parameters"] == 0
    assert bundle["summary"]["fitted_parameters"] == 0


def test_planck_anchor_value(bundle):
    """Planck rho_Lambda is 2.49e-47 GeV^4 (cosmological-constant value)."""
    obs = bundle["rho_observed"]
    assert obs["value_GeV4"] == pytest.approx(2.49e-47, rel=1e-2)


def test_canonical_residual_OoM_below_0p05(bundle):
    """Canonical-regime residual must be inside +/-0.05 orders of magnitude."""
    c = bundle["closure_result"]
    assert abs(c["residual_log10_orders"]) < 0.05


def test_extended_regime_explicitly_inconsistent(bundle):
    """The extended regime is a stress-test that should fail at ~7 OoM."""
    e = bundle["extended_regime_residual"]
    assert e["residual_log10_orders"] > 5.0  # explicitly large, regime-filter signal


def test_recompute_output_passes(output):
    assert output["verdict"] == "PASS"
    assert output["hierarchy_orders"] == 122
    assert output["fitted_parameters"] == 0
    assert output["n_layers_total"] == 9
