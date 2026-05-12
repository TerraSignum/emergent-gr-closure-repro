"""Verify the Einstein-gap two-point Richardson extrapolation passes
the closure-domain threshold under all three candidate exponents.

The honest data set at the audit date contains two clean lattice
points (N1=1534, N2=2254). A multi-point fit that independently
fixes alpha requires >=3 points; that data is in flight in
parallel lattice runs.

The test exercises:
  - structural target alpha = 2/3 declared
  - exactly two clean points
  - convergence direction (gap decreases as N increases)
  - all three Richardson candidate exponents land gap_inf below 0.05
  - the empirical 2-point fit slope reproduces the stored alpha=0.8477
"""

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _gap():
    with open(REPO / "data" / "einstein_gap_results.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_target_exponent_is_two_thirds():
    g = _gap()
    assert g["alpha_target_for_universality_claim"] == pytest.approx(2 / 3, abs=1e-3)
    assert g["alpha_target_form"] == "2/3"


def test_two_clean_points():
    g = _gap()
    pts = g["honest_two_point_data"]
    assert len(pts) == 2
    Ns = sorted(p["N"] for p in pts)
    assert Ns == [1534, 2254]


def test_delta_E_decreases_with_N():
    g = _gap()
    pts = sorted(g["honest_two_point_data"], key=lambda p: p["N"])
    for a, b in zip(pts[:-1], pts[1:]):
        assert b["Delta_E"] <= a["Delta_E"], (
            f"Non-monotonic gap at N={b['N']}"
        )
    diag = g["two_point_diagnostics"]
    assert diag["convergence_direction_correct"] is True


def test_three_richardson_candidates_all_pass_005():
    g = _gap()
    cands = g["richardson_candidates"]
    assert len(cands) == 3
    expected_alphas = sorted([0.6667, 1.0, 0.8477])
    got_alphas = sorted(c["alpha"] for c in cands)
    for got, exp in zip(got_alphas, expected_alphas):
        assert abs(got - exp) < 1e-3, f"got alpha={got}, expected {exp}"
    for c in cands:
        assert c["passes_005"] is True, (
            f"candidate alpha={c['alpha']} fails the 0.05 threshold"
        )
        assert abs(c["gap_inf"]) <= 0.05, (
            f"|gap_inf| for alpha={c['alpha']} is {c['gap_inf']}, > 0.05"
        )


def test_empirical_alpha_recomputed_from_two_points():
    """The 2-point empirical fit alpha = log(g1/g2) / log(N2/N1)
    should reproduce the stored 0.8477 within numerical roundoff."""
    g = _gap()
    pts = sorted(g["honest_two_point_data"], key=lambda p: p["N"])
    N1, gap1 = pts[0]["N"], pts[0]["Delta_E"]
    N2, gap2 = pts[1]["N"], pts[1]["Delta_E"]
    # gap = a * N^(-alpha)  =>  alpha = log(gap1/gap2) / log(N2/N1)
    alpha_recomputed = math.log(gap1 / gap2) / math.log(N2 / N1)
    # Find the 'empirical' candidate
    emp = next(c for c in g["richardson_candidates"]
               if c["exponent_name"] == "empirical 2-point fit")
    assert abs(alpha_recomputed - emp["alpha"]) < 1e-3, (
        f"recomputed alpha={alpha_recomputed:.4f} vs stored {emp['alpha']}"
    )


def test_data_poverty_caveat_is_present():
    """The honest data set must explicitly carry the >=3 N-points
    caveat so that no consumer mis-reads the 2-point construction
    as a definitive single-exponent identification."""
    g = _gap()
    assert "data_poverty_caveat" in g
    cav = g["data_poverty_caveat"]
    assert ">=3" in cav or ">= 3" in cav or "3 N-points" in cav
