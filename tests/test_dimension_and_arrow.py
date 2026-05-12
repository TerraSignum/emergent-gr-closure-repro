"""Tests for the spacetime-dimension + arrow-of-time closure."""

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import verify_dimension_and_arrow as M  # noqa: E402


@pytest.fixture(scope="module")
def bundle():
    return M.load_bundle()


@pytest.fixture(scope="module")
def output(bundle):
    M.main()
    out_path = REPO / "outputs" / "dimension_and_arrow_recompute.json"
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_spectral_dim_3_17_rounds_to_3(bundle):
    v = bundle["spacetime_dimension_closure"]["values"]
    assert v["d_spectral_eff"] == pytest.approx(3.170, abs=1e-3)
    assert round(v["d_spectral_eff"]) == 3
    assert v["d_spatial_nearest_integer"] == 3


def test_d_spacetime_equals_4(bundle):
    v = bundle["spacetime_dimension_closure"]["values"]
    assert v["d_spacetime"] == 4
    assert v["spacetime_matches_GR"] is True


def test_light_cone_fuzziness_matches_residual(bundle):
    v = bundle["spacetime_dimension_closure"]["values"]
    expected = abs(v["d_spectral_eff"] - round(v["d_spectral_eff"]))
    assert v["light_cone_fuzziness"] == pytest.approx(expected, abs=1e-4)


def test_lorentzian_possible(bundle):
    v = bundle["spacetime_dimension_closure"]["values"]
    assert v["lorentzian_possible"] is True


def test_arrow_of_time_asymmetry_above_30_orders(bundle):
    """exp(2 S_bounce) with S_bounce ~ 38 must be >> 10^30."""
    av = bundle["arrow_of_time_closure"]["values"]
    s_bounce = av["S_bounce_canonical"]
    ratio = math.exp(2 * s_bounce)
    assert ratio > 1e30
    # log10(ratio) = 2 * S_bounce / ln(10)
    expected_oom = 2 * s_bounce / math.log(10)
    assert av["asymmetry_orders_of_magnitude"] == pytest.approx(
        expected_oom, abs=1.0
    )


def test_arrow_of_time_S_bounce_matches_paper1(bundle):
    """S_bounce in this bundle must match Paper 1's canonical value 37.99."""
    av = bundle["arrow_of_time_closure"]["values"]
    assert av["S_bounce_canonical"] == pytest.approx(37.99, abs=0.01)


def test_two_closures_zero_fitted_parameters(bundle):
    s = bundle["summary"]
    assert s["n_closures"] == 2
    assert s["fitted_parameters"] == 0


def test_recompute_output_passes(output):
    assert output["verdict"] == "PASS"
    assert output["d_spacetime_equals_4"] is True
    assert output["d_spatial_equals_3"] is True
    assert output["lorentzian_possible"] is True
    assert output["P_forward_over_P_backward_recomputed"] > 1e30
