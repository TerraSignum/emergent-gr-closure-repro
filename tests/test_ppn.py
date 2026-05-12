"""PPN parameters gamma_PPN = beta_PPN = 1 within experimental constraints."""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _ppn():
    with open(REPO / "data" / "ppn_results.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_gamma_PPN_is_one():
    p = _ppn()
    g = p["predictions"]["gamma_PPN"]["value"]
    assert g == pytest.approx(1.0, abs=1e-4)


def test_beta_PPN_is_one():
    p = _ppn()
    b = p["predictions"]["beta_PPN"]["value"]
    assert b == pytest.approx(1.0, abs=1e-4)


def test_gamma_PPN_inside_experimental_band():
    p = _ppn()
    g = p["predictions"]["gamma_PPN"]["value"]
    band = p["predictions"]["gamma_PPN"]["experimental_constraint"]
    assert band[0] <= g <= band[1]


def test_beta_PPN_inside_experimental_band():
    p = _ppn()
    b = p["predictions"]["beta_PPN"]["value"]
    band = p["predictions"]["beta_PPN"]["experimental_constraint"]
    assert band[0] <= b <= band[1]


def test_tier_is_exact():
    p = _ppn()
    assert "EXACT" in p["tier"]
