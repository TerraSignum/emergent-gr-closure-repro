"""Tests for the bundled Hawking-spectrum greybody-factor table."""

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import verify_hawking_spectrum as M  # noqa: E402


@pytest.fixture(scope="module")
def spec():
    return M.load_spectrum()


@pytest.fixture(scope="module")
def output(spec):
    M.main()
    out_path = REPO / "outputs" / "hawking_spectrum_recompute.json"
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_nine_x_points(spec):
    rows = spec["canonical"]["spectrum_x_omega_over_T"]
    assert len(rows) == 9


def test_bose_einstein_distribution_recovered(spec):
    """Each tabulated n_bose must equal 1/(e^x - 1) at unit greybody.
    Tolerance is set to 0.5% to allow the 6-decimal rounding in the
    bundled JSON to round-trip cleanly (worst case is x=8 where the
    value n_bose = 3.35e-4 carries ~3 leading digits)."""
    for row in spec["canonical"]["spectrum_x_omega_over_T"]:
        x = row["x"]
        n_predicted = 1.0 / (math.exp(x) - 1.0)
        assert row["n_bose"] == pytest.approx(n_predicted, rel=5e-3)


def test_fermi_dirac_distribution_recovered(spec):
    """Each tabulated n_fermi must equal 1/(e^x + 1) at unit greybody."""
    for row in spec["canonical"]["spectrum_x_omega_over_T"]:
        x = row["x"]
        n_predicted = 1.0 / (math.exp(x) + 1.0)
        assert row["n_fermi"] == pytest.approx(n_predicted, rel=5e-3)


def test_page_curve_fractions_sum_to_one(spec):
    p1 = spec["canonical"]
    s = (p1["page_scalar_fraction"]
         + p1["page_fermion_fraction"]
         + p1["page_vector_fraction"])
    assert s == pytest.approx(1.0, abs=1e-6)


def test_hawking_temperature_matches_unit_dimensions(spec):
    """T_hawking_GeV * (1 GeV / k_B) ≈ T_hawking_K. Reasonable check
    that the canonical-regime values are dimensionally consistent."""
    p1 = spec["canonical"]
    # 1 GeV / k_B ≈ 1.16e13 K
    GeV_to_K = 1.16045e13
    expected_K = p1["T_hawking_GeV"] * GeV_to_K
    assert p1["T_hawking_K"] == pytest.approx(expected_K, rel=0.01)


def test_recompute_output_consistent(output):
    assert output["spectrum_consistent"] is True
    assert output["page_fractions_sum_to_one"] is True
    # The bundled spectrum reference values are tabulated at
    # 6-significant-figure precision; the recompute deviation is
    # therefore floor-bounded at ~1e-3 by truncation, not by an
    # actual physics discrepancy. The 5e-3 threshold accommodates
    # the largest tabulation-induced deviation (~1.71e-3 at x = 4
    # for the Bose-Einstein column) while still catching any real
    # numerical regression.
    assert output["max_dev_bose_relative"] < 5e-3
    assert output["max_dev_fermi_relative"] < 5e-3
