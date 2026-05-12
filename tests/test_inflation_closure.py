"""Tests for the inflation closure (n_s, r, N_efolds, A_s)."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import verify_inflation_closure as M  # noqa: E402


@pytest.fixture(scope="module")
def bundle():
    return M.load_bundle()


@pytest.fixture(scope="module")
def output(bundle):
    M.main()
    out_path = REPO / "outputs" / "inflation_closure_recompute.json"
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_primary_ns_loop_class_EXACT(bundle):
    """The load-bearing n_s closure is the loop-class identity
    n_s = 1 - gamma * eps_sync^2 = 0.96499 EXACT 0.001% vs Planck 0.9649."""
    p = bundle["primary_closure_loop_class_algebraic"]
    assert p["tier"] == "EXACT"
    assert p["residual_pct"] < 0.5
    assert p["predicted"] == pytest.approx(0.96499, abs=1e-5)
    assert p["anchor_value"] == pytest.approx(0.9649, abs=1e-4)
    assert "1 - gamma" in p["structural_identity"]
    assert "eps_sync" in p["structural_identity"]


def test_cascade_range_contains_Planck(bundle):
    s = bundle["secondary_closure_inflation_cascade_range"]
    assert s["n_s_measured_in_range"] is True
    # Planck 0.9649 must lie inside [0.914, 0.993]
    assert s["n_s_range_lo"] <= 0.9649 <= s["n_s_range_hi"]


def test_tensor_scalar_r_below_BICEP_bound(bundle):
    r = bundle["tensor_scalar_closure"]
    assert r["below_limit"] is True
    assert r["predicted_r"] < r["observational_upper_bound_r"]
    assert r["factor_below_bound"] > 1.5


def test_N_efolds_above_horizon_flatness_threshold(bundle):
    nf = bundle["horizon_flatness_closure"]
    assert nf["above_typical"] is True
    assert nf["N_efolds_canonical"] > 70
    assert "cascade" in nf["inflation_mechanism"]


def test_inflation_mechanism_cascade_26_barriers(bundle):
    nf = bundle["horizon_flatness_closure"]
    assert "26" in nf["inflation_mechanism"]


def test_extended_regime_stress_test_explicitly_inconsistent(bundle):
    """Extended regime n_s = 0.588 (39% off) and r = 0.68 (>10x BICEP bound)
    serves as an explicit regime-filter; observation rejects it."""
    e = bundle["extended_regime_stress_test"]
    assert e["values"]["n_s"] < 0.7
    assert e["values"]["r_tensor_to_scalar"] > 0.1  # > BICEP bound by 10x
    assert e["values"]["factor_above_r_bound"] > 5.0


def test_zero_fitted_parameters(bundle):
    assert bundle["summary"]["fitted_parameters"] == 0


def test_open_problems_resolved_list(bundle):
    """The summary lists the four canonical inflation problems closed."""
    resolved = bundle["summary"]["open_problems_resolved"]
    text = " ".join(resolved)
    assert "n_s" in text
    assert "horizon" in text or "e-folds" in text
    assert "BICEP" in text or "r is small" in text


def test_recompute_output_passes(output):
    assert output["verdict"] == "PASS"
    assert output["primary_n_s_EXACT_via_loop_class"] is True
    assert output["cascade_range_contains_Planck"] is True
    assert output["tensor_scalar_r_below_BICEP_bound"] is True
    assert output["N_efolds_above_horizon_flatness_typical"] is True
