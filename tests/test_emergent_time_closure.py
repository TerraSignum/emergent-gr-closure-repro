"""Tests for the bundled emergent-time + Lorentz-signature closure."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import verify_emergent_time as M  # noqa: E402


@pytest.fixture(scope="module")
def bundle():
    return M.load_bundle()


@pytest.fixture(scope="module")
def output(bundle):
    M.main()
    out_path = REPO / "outputs" / "emergent_time_recompute.json"
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_lorentz_signature_derived(bundle):
    l = bundle["lorentz_signature_closure"]
    assert l["tier"] == "DERIVED"
    assert "c_Xi^2" in l["structural_identity"]
    assert "o(k^2)" in l["lorentz_violation_residual"]


def test_time_dilation_formula(bundle):
    t = bundle["time_dilation_closure"]
    assert t["tier"] == "DERIVED"
    formula = t["structural_identity"]
    assert "dtau" in formula
    assert "alpha_K" in formula
    assert "alpha_R" in formula
    assert "sqrt" in formula


def test_time_dilation_recovers_weak_field_redshift(bundle):
    t = bundle["time_dilation_closure"]
    interp = t["interpretation"]
    assert "Phi_N" in interp
    assert "weak-field" in interp


def test_macroscopic_time_rate_lipschitz(bundle):
    r = bundle["macroscopic_time_rate_closure"]
    assert r["tier"] == "DERIVED"
    assert "q(x,t)" in r["structural_identity"]
    assert "Lipschitz" in r["lipschitz_bound"]


def test_irreversibility_five_channels_sum_to_100pct(bundle):
    ir = bundle["irreversibility_channels"]
    assert ir["n_channels"] == 5
    total = sum(c["share_pct"] for c in ir["channels"])
    assert abs(total - 100.0) <= 0.5


def test_gamow_channel_dominates(bundle):
    ir = bundle["irreversibility_channels"]
    gamow = ir["channels"][0]
    assert "Gamow" in gamow["name"]
    assert gamow["share_pct"] > 90.0


def test_zero_fitted_parameters(bundle):
    assert bundle["summary"]["fitted_parameters"] == 0


def test_recompute_output_passes(output):
    assert output["verdict"] == "PASS"
    assert output["lorentz_signature_DERIVED"] is True
    assert output["time_dilation_DERIVED"] is True
    assert output["macroscopic_time_rate_DERIVED"] is True
    assert output["Gamow_channel_dominates"] is True
