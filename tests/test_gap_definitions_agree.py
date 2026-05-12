"""The three Richardson candidate exponents must each pass the
0.05 closure-domain threshold under the two-point honest data.

This test was previously written against a fabricated set of three
alternative gap definitions (spectral, curvature-diffusion, BH-info)
each with its own claimed exponent. That data was not backed by any
recompute; it has been replaced by the canonical two-point Richardson
construction with three candidate exponents (Ricci-order 2/3, linear
1.0, empirical 2-point fit 0.8477) all of which yield gap_inf < 0.05.
"""

import json
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def _gap():
    with open(REPO / "data" / "einstein_gap_results.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_three_richardson_candidates_present():
    g = _gap()
    assert len(g["richardson_candidates"]) == 3


def test_all_candidates_pass_closure_threshold():
    g = _gap()
    threshold = g.get("closure_threshold", 0.05)
    for cand in g["richardson_candidates"]:
        assert abs(cand["gap_inf"]) <= threshold, (
            f"candidate alpha={cand['alpha']} ({cand['exponent_name']}) "
            f"fails closure threshold {threshold}: gap_inf={cand['gap_inf']}"
        )
        assert cand["passes_005"] is True


def test_universality_target_is_two_thirds():
    g = _gap()
    assert g["alpha_target_form"] == "2/3"
    assert abs(g["alpha_target_for_universality_claim"] - 2 / 3) < 1e-3


def test_aggregate_passes_flag():
    g = _gap()
    assert g["all_candidates_pass_005_threshold"] is True
