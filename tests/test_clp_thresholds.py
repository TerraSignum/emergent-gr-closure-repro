"""All four CLP families must pass their pre-registered family-level thresholds.

Dual-threshold protocol: CLP-A, CLP-C, CLP-D carry the family-level
closure-domain threshold 0.70; CLP-B/B4 carries the heuristic
per-sub-component threshold 0.50 (operator-convergence sub-family).
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _clp():
    with open(REPO / "data" / "clp_scores.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_clp_axes_present():
    c = _clp()
    assert set(c["axes"].keys()) == {"CLP-A", "CLP-B", "CLP-C", "CLP-D"}
    for name, ax in c["axes"].items():
        assert "value" in ax and "threshold" in ax, f"{name} missing fields"


def test_data_source_caveat_documents_aggregation():
    """CLP-A/C/D family scores are aggregate; CLP-B is per-sub-component."""
    c = _clp()
    cav = c.get("data_source_caveat", "")
    assert "aggregate" in cav.lower()


def test_clp_a_above_threshold():
    c = _clp()
    ax = c["axes"]["CLP-A"]
    assert ax["value"] >= ax["threshold"]


def test_clp_b_above_per_sub_threshold():
    """CLP-B carries the per-sub-component 0.50 heuristic threshold,
    not the family-level 0.70 closure-domain threshold."""
    c = _clp()
    ax = c["axes"]["CLP-B"]
    assert ax["threshold"] == 0.50
    assert ax["value"] >= ax["threshold"]


def test_clp_c_above_threshold():
    c = _clp()
    ax = c["axes"]["CLP-C"]
    assert ax["value"] >= ax["threshold"]


def test_clp_d_above_threshold():
    c = _clp()
    ax = c["axes"]["CLP-D"]
    assert ax["value"] >= ax["threshold"]


def test_dual_threshold_pass():
    """All four families pass their pre-registered (dual) thresholds:
    A/C/D >= 0.70, B (per-sub-component) >= 0.50."""
    c = _clp()
    for name, ax in c["axes"].items():
        assert ax["value"] >= ax["threshold"], f"{name} below threshold"
