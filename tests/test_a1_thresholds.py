"""Verify A1 fast-slow regime thresholds are met on the canonical regime P1."""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _a1():
    with open(REPO / "data" / "a1_regime_constants.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_lambda_threshold_canonical():
    a = _a1()
    p1 = a["regimes"]["canonical"]
    threshold = a["lambda_triangle_threshold"]
    assert p1["lambda_triangle"] >= threshold


def test_epsilon_threshold_canonical():
    a = _a1()
    p1 = a["regimes"]["canonical"]
    threshold = a["epsilon_separation_threshold"]
    assert p1["epsilon"] <= threshold


def test_canonical_status_pass():
    a = _a1()
    p1 = a["regimes"]["canonical"]
    assert p1["status"] == "PASS"


def test_extended_drift_marked():
    a = _a1()
    p2 = a["regimes"]["extended"]
    assert "DRIFT" in p2["status"]


def test_fast_slow_equation_documented():
    a = _a1()
    eq = a["fast_slow_equation"]
    assert "lambda_triangle" in eq
    assert "epsilon" in eq
