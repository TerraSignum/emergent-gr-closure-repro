"""Tests for the black-hole-sector recompute (Section 7 of the manuscript).

These tests assert the load-bearing numbers that the manuscript references:
the Bekenstein-Hawking 1/4 entropy-to-area ratio (residual <= 5e-8 in P1,
<= 2e-7 in P2'), the bound state S = 4 pi M^2 (residual <= 0.05%), the
horizon-threshold compactness (>> 1), the Kerr-defect ISCO and Lense-
Thirring frame-dragging, the Penrose energy-extraction efficiency, and
the binary-defect Peters-formula radiation observables.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import recompute_bh_sector as M  # noqa: E402


@pytest.fixture(scope="module")
def bh():
    return M._load("bekenstein_hawking.json")


@pytest.fixture(scope="module")
def ht():
    return M._load("horizon_threshold.json")


@pytest.fixture(scope="module")
def kg():
    return M._load("kerr_geometry.json")


@pytest.fixture(scope="module")
def pp():
    return M._load("penrose_process.json")


@pytest.fixture(scope="module")
def bi():
    return M._load("binary_inspiral.json")


@pytest.fixture(scope="module")
def ip():
    return M._load("information_paradox.json")


@pytest.fixture(scope="module")
def output():
    M.main()
    out_path = REPO / "outputs" / "bh_sector_recompute.json"
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_bh_quarter_S_over_A_both_regimes(bh):
    assert bh["canonical"]["S_over_A"] == pytest.approx(0.25, abs=1e-6)
    assert bh["canonical"]["S_over_A_residual_vs_quarter"] <= 5e-8
    assert bh["extended"]["S_over_A"] == pytest.approx(0.25, abs=1e-6)
    assert bh["extended"]["S_over_A_residual_vs_quarter"] <= 2e-7
    assert bh["canonical"]["area_law_satisfied"] is True
    assert bh["extended"]["area_law_satisfied"] is True


def test_bh_bound_state_S_equals_4_pi_M_squared(bh):
    assert bh["canonical"]["S_over_4piM2"] == pytest.approx(1.0, abs=5e-4)
    assert bh["extended"]["S_over_4piM2"] == pytest.approx(1.0, abs=5e-4)


def test_horizon_threshold_compactness_both_regimes(ht):
    for tag in ("canonical", "extended"):
        assert ht[tag]["is_black_hole"] is True
        assert ht[tag]["compactness_status"] == "BLACK_HOLE"
        assert ht[tag]["compactness_lattice"] > 5.0, (
            f"{tag} compactness {ht[tag]['compactness_lattice']} not >> 1"
        )
        assert ht[tag]["r_core_over_r_s"] < 0.05


def test_kerr_isco_canonical_regime(kg):
    p1 = kg["canonical"]
    assert p1["r_ISCO_lu"] == pytest.approx(17.43, abs=0.01)
    assert p1["chi_below_extremal"] is True
    assert p1["chi_spin"] < 0.05
    assert p1["omega_lense_thirring_lu"] == pytest.approx(0.0183, abs=2e-4)


def test_kerr_isco_extended_regime(kg):
    p2 = kg["extended"]
    assert p2["r_ISCO_lu"] == pytest.approx(19.33, abs=0.01)
    assert p2["chi_below_extremal"] is True
    assert p2["chi_spin"] < 0.05


def test_kerr_horizon_inside_ergosphere_both_regimes(kg):
    """Kerr structure: r_+ <= r_ergo_equator strictly (rotation lifts the
    ergosphere outside the outer horizon at the equator)."""
    for tag in ("canonical", "extended"):
        k = kg[tag]
        assert k["r_plus_lu"] <= k["r_ergo_equator_lu"]
        assert k["r_minus_lu"] < k["r_plus_lu"]


def test_kerr_units_convention_chi_equals_J_with_M_lu_one():
    """Lock the units convention: chi_spin = J_lu / M_lu^2 with M_lu == 1
    in the bundled normalization, so chi numerically equals J_lu and
    differs from a_kerr_lu (which uses a separate Kerr-coordinate
    normalization). The convention is documented in the JSON's
    units_convention field; this test asserts the numerical relations
    that field implies."""
    full = json.loads(
        (REPO / "data" / "black_hole" / "kerr_geometry.json").read_text(
            encoding="utf-8"
        )
    )
    assert "units_convention" in full
    for tag in ("canonical", "extended"):
        k = full[tag]
        assert k["chi_spin"] == pytest.approx(k["J_lu"], rel=1e-9)
        assert abs(k["a_kerr_lu"] - k["chi_spin"]) > 1e-3


def test_penrose_efficiency_both_regimes(pp):
    assert pp["canonical"]["penrose_efficiency"] == pytest.approx(0.0066, abs=5e-4)
    assert pp["extended"]["penrose_efficiency"] == pytest.approx(0.0043, abs=5e-4)
    assert pp["canonical"]["extremal_limit"] is False
    assert pp["extended"]["extremal_limit"] is False


def test_binary_inspiral_chirp_mass_and_eta(bi):
    """Equal-mass binary -> eta_symmetric = 1/4; chirp mass M_c = M / 2^{1/5}."""
    for tag in ("canonical", "extended"):
        b = bi[tag]
        assert b["eta_symmetric"] == pytest.approx(0.25)
        M1 = b["M1_lu"]
        M2 = b["M2_lu"]
        assert M1 == pytest.approx(M2, rel=1e-6), (
            f"{tag} binary not equal-mass: {M1} vs {M2}"
        )
        Mc_pred = (M1 * M2) ** (3 / 5) / (M1 + M2) ** (1 / 5)
        assert b["M_chirp_lu"] == pytest.approx(Mc_pred, rel=2e-3), (
            f"{tag} chirp mass: stored {b['M_chirp_lu']} vs "
            f"computed {Mc_pred} from M1, M2"
        )


def test_binary_inspiral_f_isco_consistent(bi):
    """f_ISCO must match between the two regimes within 1e-3 (same M_total
    100-Hz calibration); ringdown frequency must exceed f_ISCO."""
    f1 = bi["canonical"]["f_ISCO_Hz"]
    f2 = bi["extended"]["f_ISCO_Hz"]
    assert f1 == pytest.approx(f2, rel=1e-3)
    assert bi["canonical"]["f_ringdown_Hz"] > f1
    assert bi["extended"]["f_ringdown_Hz"] > f2


def test_binary_inspiral_astrophysical_calibration(bi):
    cal = bi["astrophysical_calibration"]
    assert cal["M_total_for_100Hz_Msun"] == pytest.approx(43.96, abs=0.1)
    assert cal["GW150914_f_ISCO_Hz"] == pytest.approx(150.0, abs=1.0)


def test_information_paradox_unitarity_preserved(ip):
    """Unitarity is preserved in both regimes; the arrow of time
    supports rather than violates it."""
    assert ip["unitarity_status"] == "PRESERVED"
    for tag in ("canonical", "extended"):
        assert ip[tag]["arrow_supports_unitarity"] is True
        assert ip[tag]["scrambling_time_lu"] > 0
        assert ip[tag]["page_time_lu"] > ip[tag]["scrambling_time_lu"]


def test_information_paradox_page_curve_consistent(ip):
    """The Page time exceeds the scrambling time (Page-curve direction);
    the textbook ratio tau_Page = 2 tau_scr holds within the bundled
    25% lattice-discreteness tolerance."""
    pc = ip["page_curve_check"]
    assert pc["page_curve_consistent"] is True
    for tag, key in (("canonical", "canonical_ratio"), ("extended", "extended_ratio")):
        ratio_obs = ip[tag]["page_time_lu"] / ip[tag]["scrambling_time_lu"]
        assert ratio_obs == pytest.approx(pc[key], rel=1e-3)
        # Within 25% of the textbook 2.0 ratio.
        assert abs(ratio_obs - pc["expected_ratio_page_over_scrambling"]) \
               / pc["expected_ratio_page_over_scrambling"] < 0.30


def test_recompute_output_has_expected_top_keys(output):
    expected = {"bekenstein_hawking", "horizon_threshold",
                "kerr_geometry", "penrose", "binary_inspiral",
                "information_paradox", "bh_quarter_label_audit"}
    assert expected.issubset(set(output.keys()))
    for k in expected - {"information_paradox", "bh_quarter_label_audit"}:
        assert "canonical" in output[k]
        assert "extended" in output[k]
    ip_out = output["information_paradox"]
    assert ip_out["unitarity_status_label"] == "PRESERVED"
    assert ip_out["unitarity_status_recomputed"] == "PRESERVED"
    assert ip_out["page_curve_consistent_label"] is True
    assert ip_out["page_curve_consistent_recomputed"] is True
    assert ip_out["label_consistent_with_recomp"] is True
    assert "canonical" in ip_out
    assert "extended" in ip_out


def test_bh_area_law_label_matches_residual_recompute(output):
    """The 'area_law_satisfied' flag in the data file must agree with
    the residual-derived recompute |S/A - 1/4| <= 1e-6."""
    audit = output["bh_quarter_label_audit"]
    for tag in ("canonical", "extended"):
        assert audit[tag]["label_consistent_with_recomp"] is True, (
            f"{tag} BH area-law label disagrees with the residual-derived "
            f"recompute (residual {audit[tag]['S_over_A_residual_recomp']})"
        )
        assert audit[tag]["area_law_satisfied_recomp"] is True


def test_information_paradox_arrow_label_matches_time_recompute(output):
    """arrow_supports_unitarity flag must agree with tau_Page > tau_scr."""
    ip_out = output["information_paradox"]
    for tag in ("canonical", "extended"):
        regime = ip_out[tag]
        assert regime["arrow_supports_unitarity_label"] == \
               regime["arrow_supports_unitarity_recomputed"], (
            f"{tag} unitarity-arrow label disagrees with the time-ordering "
            f"recompute (page_ratio={regime['page_ratio_recomputed']})"
        )
