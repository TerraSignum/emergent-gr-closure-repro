"""Tests for the Lorentzian 3+1 causal-ordering recompute (Section 3).

The bundled file `data/lorentzian_3plus1.json` carries the load-bearing
prerequisites of the discrete-to-continuum bridge:
  - 3+1 spatial-temporal decomposition;
  - Lorentzian-metric existence in both regimes;
  - clean causal partial order with finite light-cone fuzziness;
  - strong arrow of time via the bounce action;
  - Planck-time consistency to within ~2e-5;
  - Tolman thermodynamic-time consistency.

These tests assert each of the load-bearing claims and the
three-axis metric-quality table (sigma, f_LC, Delta_E).
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import recompute_lorentzian_3plus1 as M  # noqa: E402


@pytest.fixture(scope="module")
def d():
    return M.load_lorentzian_3plus1()


@pytest.fixture(scope="module")
def output(d):
    M.main()
    out_path = REPO / "outputs" / "lorentzian_3plus1_recompute.json"
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_three_plus_one_decomposition_and_dimensions(d):
    co = d["causal_ordering"]
    assert co["d_spacetime"] == 4
    assert co["d_spatial"] == 3
    assert co["decomposition"] == "3+1"
    assert co["lorentzian_possible"] is True


def test_lorentzian_metric_exists_in_both_regimes(d):
    co = d["causal_ordering"]
    assert co["lorentzian_metric_exists_canonical"] is True
    assert co["lorentzian_metric_exists_extended"] is True


def test_causal_structure_clean_and_massless_modes(d):
    co = d["causal_ordering"]
    assert co["causal_structure_clean"] is True
    assert co["causal_modes_present"] is True
    assert co["causal_mode_mass"] == pytest.approx(0.0, abs=1e-12)


def test_proper_time_well_defined_in_both_regimes(d):
    co = d["causal_ordering"]
    assert co["proper_time_well_defined_canonical"] is True
    assert co["proper_time_well_defined_extended"] is True


def test_light_cone_fuzziness_consistent_across_regimes(d):
    """Both regimes report the same f_LC = d_eff_residual ~ 0.170."""
    co = d["causal_ordering"]
    assert co["light_cone_fuzziness_canonical"] == pytest.approx(0.170, abs=0.005)
    assert co["light_cone_fuzziness_extended"] == pytest.approx(0.170, abs=0.005)


def test_strong_arrow_of_time_in_both_regimes(d):
    """The bounce action S_bounce is large enough that the time-asymmetry
    ratio exp(2 S_bounce) exceeds 1e30 in both regimes (EXTREMELY_STRONG)."""
    aot = d["arrow_of_time"]
    assert aot["S_bounce_canonical"] > 30.0
    assert aot["S_bounce_extended"] > 30.0
    assert aot["time_asymmetry_ratio_canonical"] > 1e30
    assert aot["time_asymmetry_ratio_extended"] > 1e30
    assert aot["arrow_strength_canonical"] == "EXTREMELY_STRONG"
    assert aot["arrow_strength_extended"] == "EXTREMELY_STRONG"


def test_planck_time_derived_matches_measured_within_two_e_minus_five(d):
    pt = d["proper_time"]
    assert pt["t_Planck_ratio"] == pytest.approx(1.0, abs=2e-5)


def test_tolman_consistency_both_regimes(d):
    tt = d["thermodynamic_time"]
    assert tt["tolman_consistency_canonical"] is True
    assert tt["tolman_consistency_extended"] is True
    # Tolman redshift factor in (0, 1] for a static thermal observer.
    for f in (tt["grav_redshift_factor_canonical"], tt["grav_redshift_factor_extended"]):
        assert 0.0 < f <= 1.0


def test_three_axis_metric_quality_table_consistent(d):
    """sigma > f_LC > Delta_E (canonical regime), all positive."""
    summ = d["metric_quality_axes_summary"]
    axes = {a["name"]: a for a in summ["axes"]}
    sigma = axes["Riemannian-embedding stress (spatial only)"]["value_canonical"]
    f_LC = axes["Lorentzian light-cone fuzziness (3+1 causal-ordering)"]["value_canonical"]
    Delta_E = axes["Einstein-identity gap"]["value_canonical"]
    assert sigma > f_LC > Delta_E > 0.0
    # The "missing chain" of canonical regime is sigma - f_LC ~ 0.11.
    assert summ["differences"]["sigma_minus_lightcone_canonical"] == \
        pytest.approx(sigma - f_LC, abs=1e-6)
    # And f_LC - Delta_E ~ 0.03 at canonical.
    assert summ["differences"]["lightcone_minus_einstein_gap_canonical"] == \
        pytest.approx(f_LC - Delta_E, abs=1e-6)


def test_recompute_output_has_expected_keys(output):
    expected = {
        "criterion", "lorentzian_metric_exists", "proper_time_well_defined",
        "decomposition", "d_spacetime", "d_spatial",
        "light_cone_fuzziness", "arrow_of_time", "planck_time_ratio",
        "tolman_consistency", "metric_quality_table",
    }
    assert expected.issubset(set(output.keys()))
    assert output["d_spacetime"] == 4
    assert output["d_spatial"] == 3
    assert output["decomposition"] == "3+1"
