"""Tests for the curvature-fixed-point certificate.

The curvature-fixed-point theorem (Theorem 4.10 of the program's proof
collection) requires four operative-closure-rule conditions
(Corollary 4.11): uniform boundedness, summable scale-deviation,
Cauchy convergence to a fixed point R_*, and scheme independence
across coarse-graining families. This test asserts each of the four
on the bundled coarse-graining sequence.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import verify_curvature_fixed_point as M  # noqa: E402


@pytest.fixture(scope="module")
def output():
    M.main()
    out_path = REPO / "outputs" / "curvature_fixed_point_certificate.json"
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_R_scalar_loaded_from_bundled_data():
    R = M.load_R_scalar()
    assert R == pytest.approx(79.34, abs=0.01)


def test_CFcanonical_uniform_boundedness(output):
    """sup_n |R(r_n)| < infinity."""
    for scheme in output["schemes"].values():
        assert scheme["uniformly_bounded"] is True
        assert scheme["sup_norm"] < float("inf")


def test_CFP2_summable_scale_deviation(output):
    """sum_n Delta_curv(r_n) < infinity."""
    for scheme in output["schemes"].values():
        assert scheme["summable_scale_deviation"] is True
        assert scheme["sum_scale_deviations"] < float("inf")
        assert scheme["sum_scale_deviations"] > 0.0


def test_CFP3_cauchy_convergence(output):
    """R(r_n) -> R_* with Cauchy tail < 1e-3 at n_steps=20."""
    for scheme in output["schemes"].values():
        assert scheme["cauchy_converges"] is True
        assert scheme["cauchy_tail"] < 1e-3


def test_CFP4_scheme_independence(output):
    """|R_*(b=1/2) - R_*(b=1/3)| < 1e-3 (scheme independence)."""
    diff = output["scheme_independence"]["diff"]
    assert output["scheme_independence"]["passes"] is True
    assert diff < 1e-3


def test_all_four_conditions_pass(output):
    assert output["all_four_conditions_pass"] is True


def test_certificate_output_has_expected_keys(output):
    expected = {"criterion", "theorem", "R_star_lu", "schemes",
                "scheme_independence", "all_four_conditions_pass"}
    assert expected.issubset(set(output.keys()))
    assert "b = 1/2" in output["schemes"]
    assert "b = 1/3" in output["schemes"]


def test_geometric_decay_rate_consistent_with_b(output):
    """Cauchy residuals must decay geometrically at rate b."""
    sch = output["schemes"]["b = 1/2"]
    residuals = sch["cauchy_residuals"]
    # Take ratios of successive non-zero residuals; should approach 0.5.
    ratios = [residuals[n + 1] / residuals[n]
              for n in range(len(residuals) - 1)
              if residuals[n] > 1e-12]
    avg_ratio = sum(ratios) / len(ratios)
    assert avg_ratio == pytest.approx(0.5, abs=1e-6)
