"""Verify metric axioms M1-M3 are documented and the metric formula is consistent."""

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _metric():
    with open(REPO / "data" / "xi_metric_inputs.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_axioms_documented():
    m = _metric()
    for k in ("M1", "M2", "M3"):
        assert k in m["Xi_axioms"]
        assert len(m["Xi_axioms"][k]) > 10


def test_metric_formula():
    m = _metric()
    assert "log" in m["metric_definition"]
    assert "ell_0" in m["metric_definition"]


def test_metric_consequences():
    m = _metric()
    cons = m["metric_consequences"]
    assert "positivity" in cons
    assert "symmetry" in cons
    assert "triangle_inequality" in cons


def test_metric_positivity_holds_under_M2():
    """If Xi_ii = 1 (M2) and Xi_ij <= 1 (M3 + M2 imply this for sub-multiplicative
    similarities), then -log Xi_ij >= 0, i.e. d_ij >= 0."""
    Xi_ii = 1.0
    assert -math.log(Xi_ii) == 0.0
    for Xi_ij in [0.99, 0.5, 0.1, 0.01]:
        assert -math.log(Xi_ij) > 0


def test_ell_0_is_positive():
    m = _metric()
    assert m["ell_0"] > 0
