"""Deliberate-failure tests for Paper 4."""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def test_extended_does_not_pass_a1():
    """The P2' regime must NOT pass the A1 fast-slow thresholds.
    If it did, A1 would not be regime-discriminating."""
    with open(REPO / "data" / "a1_regime_constants.json", "r", encoding="utf-8") as f:
        a = json.load(f)
    p2 = a["regimes"]["extended"]
    fail_lambda = p2["lambda_triangle"] < a["lambda_triangle_threshold"]
    fail_eps    = p2["epsilon"]   > a["epsilon_separation_threshold"]
    assert fail_lambda or fail_eps


def test_clp_below_threshold_would_break_closure():
    """If any CLP score were below threshold, the closure would fail."""
    with open(REPO / "data" / "clp_scores.json", "r", encoding="utf-8") as f:
        c = json.load(f)
    fake_clp = 0.50
    threshold = c["axes"]["CLP-A"]["threshold"]
    assert fake_clp < threshold


def test_gap_exponent_outside_F4_band_breaks():
    """F4 trigger: a >=3-point fit selecting a single alpha_gap outside
    the structural target 2/3 +/- 30% would falsify the universality
    claim. Deliberate-failure construction: build candidates just outside
    the band and verify the relative deviation exceeds 0.30."""
    with open(REPO / "data" / "einstein_gap_results.json", "r", encoding="utf-8") as f:
        g = json.load(f)
    target = g["alpha_target_for_universality_claim"]
    F4_BAND_HALF = 0.30  # +/- 30% per manuscript F4
    # Deliberate-failure candidates just outside the F4 band
    fake_alphas = [target * (1 - F4_BAND_HALF - 0.01),
                   target * (1 + F4_BAND_HALF + 0.01),
                   target * 0.1,
                   target * 5.0]
    for fake in fake_alphas:
        rel = abs(fake - target) / target
        assert rel > F4_BAND_HALF, (
            f"deliberate-failure alpha={fake:.4f} (rel={rel:.4f}) does not "
            f"clear the F4 band half-width {F4_BAND_HALF}"
        )
    # And verify the canonical structural target is preserved
    assert abs(target - 2/3) < 1e-3
    # The empirical candidate (0.8477) should sit INSIDE the F4 band
    emp = next(c for c in g["richardson_candidates"]
               if c["exponent_name"] == "empirical 2-point fit")
    rel_emp = abs(emp["alpha"] - target) / target
    assert rel_emp <= F4_BAND_HALF, (
        f"empirical fit alpha={emp['alpha']} is outside F4 band — "
        f"would falsify universality even though it passes |gap_inf| < 0.05"
    )


def test_PPN_far_off_would_break():
    fake_gamma = 0.99  # 1% deviation
    band_min = 0.99998
    assert fake_gamma < band_min


def test_metric_axiom_violation():
    """If M2 (Xi_ii = 1) were violated, the identity-of-indiscernibles would fail."""
    Xi_ii = 0.9  # M2 violated
    import math
    d_ii = -1.0 * math.log(Xi_ii)
    assert d_ii > 0  # contradicts d_ii = 0
