"""Tests for the Schwarzschild-defect recompute (Section 3 of the manuscript).

These tests assert the load-bearing numbers that the manuscript references:
the Newton-constant ratio (residual <= 0.5%), the Planck-length ratio
(residual <= 0.001%), the effective spectral dimension (~3.170), the
emergent spatial / spacetime dimensions (3 / 4), and the metric-quality
match together with the Jacobson-Einstein 3/3 area-law-derivation checks
in both the canonical and extended regimes.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import recompute_schwarzschild_defect as M  # noqa: E402


@pytest.fixture(scope="module")
def sw():
    return M.load_schwarzschild()


@pytest.fixture(scope="module")
def output(sw):
    M.main()
    out_path = REPO / "outputs" / "schwarzschild_defect_recompute.json"
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_newton_constant_ratio(sw):
    nc = sw["newton_constant"]
    assert nc["G_N_ratio"] == pytest.approx(1.0, abs=0.005), (
        f"G_N ratio off unity by more than 0.5%: ratio = {nc['G_N_ratio']}"
    )
    assert nc["residual_pct"] <= 0.5, (
        f"G_N residual {nc['residual_pct']}% exceeds 0.5%"
    )
    assert nc["tier"] in ("EXACT", "PRECISE")


def test_planck_length_ratio(sw):
    nc = sw["newton_constant"]
    assert nc["ell_Planck_ratio"] == pytest.approx(1.0, abs=1e-5), (
        f"Planck-length ratio off unity by more than 0.001%: "
        f"{nc['ell_Planck_ratio']}"
    )


def test_emergent_spatial_and_spacetime_dimension(sw):
    ed = sw["emergent_spacetime_dimension"]
    assert ed["d_spatial_nearest_integer"] == 3
    assert ed["d_spacetime"] == 4
    assert ed["d_spectral_eff"] == pytest.approx(3.170, abs=0.005), (
        f"d_eff^spec = {ed['d_spectral_eff']} != 3.170"
    )
    assert ed["spatial_dimension_derived"] is True
    assert ed["spacetime_matches_GR"] is True


def test_metric_quality_passes_both_regimes(sw):
    p1 = sw["metric_quality_canonical"]
    p2 = sw["metric_quality_extended"]
    assert p1["metric_quality_pass"] is True
    assert p2["metric_quality_pass"] is True


def test_jacobson_einstein_3_of_3_both_regimes(sw):
    je = sw["jacobson_einstein"]
    assert je["checks_passed_canonical"] == je["checks_total_canonical"] == 3
    assert je["checks_passed_extended"] == 3
    assert je["einstein_equation_status_canonical"] == "EMERGENT_GR"
    assert je["einstein_equation_status_extended"] == "EMERGENT_GR"


def test_recompute_output_has_expected_keys(output):
    expected = {
        "G_N_ratio", "G_N_residual_pct",
        "G_N_tier_label", "G_N_tier_recomputed",
        "G_N_tier_label_consistent", "tier_thresholds_pct",
        "d_eff_spec", "d_spatial", "d_spacetime",
        "metric_canonical", "metric_extended", "jacobson_einstein",
    }
    assert expected.issubset(set(output.keys())), (
        f"Missing keys in recompute output: "
        f"{expected - set(output.keys())}"
    )
    assert output["d_spacetime"] == 4
    assert output["d_spatial"] == 3
    assert output["G_N_residual_pct"] <= 0.5


def test_G_N_tier_label_matches_recomputed_tier(output):
    """The tier string in the data file must agree with the tier
    derived strictly from the residual via program-wide thresholds.
    Catches silent drift where the data file says EXACT but the
    residual exceeds the EXACT threshold."""
    assert output["G_N_tier_label_consistent"] is True, (
        f"G_N tier label {output['G_N_tier_label']!r} disagrees with "
        f"residual-derived tier {output['G_N_tier_recomputed']!r}"
    )


def test_jacobson_einstein_status_label_matches_check_count(output):
    """EMERGENT_GR status in the data file must agree with the
    check_count-derived status."""
    je = output["jacobson_einstein"]
    assert je["canonical_status_label_consistent"] is True
    assert je["extended_status_label_consistent"] is True
    assert je["canonical_status_recomputed"] == "EMERGENT_GR"
    assert je["extended_status_recomputed"] == "EMERGENT_GR"
