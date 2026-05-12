"""Tests for the five-point chirality-deviation Einstein-gap-exponent fit.

The bundled `data/einstein_gap_5point_fit.json` carries five lattice
sizes N in {28, 30, 36, 42, 50} with chirality-balance deviations.
The script `verify_einstein_gap_5point_fit.py` recomputes the
log-log linear fit and recovers alpha_fit ~ 0.636, in agreement with
the analytical Delta_E(N) <= C_0 N^{-2/3} bound to within 4.7%.
This complements the two-point Richardson construction at the
canonical and extended anchor regimes.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import verify_einstein_gap_5point_fit as M  # noqa: E402


@pytest.fixture(scope="module")
def fit():
    return M.load_5point_fit()


@pytest.fixture(scope="module")
def output(fit):
    M.main()
    out_path = REPO / "outputs" / "einstein_gap_5point_recompute.json"
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_five_lattice_sizes(fit):
    Ns = fit["scan"]["N_values"]
    assert Ns == [28, 30, 36, 42, 50]


def test_alpha_fit_close_to_two_thirds(fit):
    """alpha_fit should be within 5% of the analytical 2/3 target."""
    alpha = fit["fit_result"]["alpha_fit"]
    assert abs(alpha - 2.0 / 3.0) / (2.0 / 3.0) < 0.05


def test_log_log_recompute_matches_bundled():
    """Recomputing the log-log linear fit on the bundled scan must
    reproduce the bundled alpha_fit, C_fit, and R^2 to ~ 1e-6."""
    fit = M.load_5point_fit()
    Ns = fit["scan"]["N_values"]
    devs = fit["scan"]["deviation_means"]
    slope, intercept, r2 = M.loglog_linear_fit(Ns, devs)
    import math
    alpha_recomp = -slope
    C_recomp = math.exp(intercept)
    assert alpha_recomp == pytest.approx(fit["fit_result"]["alpha_fit"], abs=1e-9)
    assert C_recomp == pytest.approx(fit["fit_result"]["C_fit"], abs=1e-9)
    assert r2 == pytest.approx(fit["fit_result"]["r2_loglog"], abs=1e-9)


def test_r_squared_reasonable(fit):
    """R^2 of the log-log linear fit must be at least 0.7
    (reasonable five-point regression)."""
    r2 = fit["fit_result"]["r2_loglog"]
    assert r2 > 0.70


def test_deviation_decreasing_with_N(fit):
    """Chirality deviation must decrease with N (non-strict
    monotonicity allowed for noise)."""
    devs = fit["scan"]["deviation_means"]
    assert devs[0] > devs[-1]


def test_recompute_output_consistent(output):
    assert output["consistent_with_bundled"] is True
    assert output["tier_alpha"] == "PRECISE"


def test_asymptotic_deviation_below_threshold(output):
    """Extrapolated deviation at N = 10^6 must lie comfortably
    below the closure-domain threshold of 0.05."""
    assert output["deviation_at_N_1e6"] < 0.001
