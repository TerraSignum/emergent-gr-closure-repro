"""Tests for the Q1/Proposal-2 variance-reactivity identity audit.

The audit checks the closed-form System-R identity
sigma^2_Xi -> 2 * alpha_xi^2 * gamma^2 = 81/5000 for the edge-Xi
row-variance on the canonical-physics ladder N in [50,512].
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import verify_variance_reactivity_identity as M  # noqa: E402

OUT = REPO / "outputs" / "variance_reactivity_identity_audit.json"


@pytest.fixture(scope="module")
def output():
    if not OUT.exists():
        M.main()
    with open(OUT, "r", encoding="utf-8") as f:
        return json.load(f)


def test_sigma2_target_is_81_over_5000(output):
    s2 = output["sigma2_xi"]
    assert s2["target_fraction"] == "81/5000"
    assert s2["target_value"] == pytest.approx(0.0162, abs=1e-6)


def test_variance_reactivity_identity_within_ci(output):
    """sigma^2_Xi -> 81/5000 lies inside the seed-bootstrap CI."""
    s2 = output["sigma2_xi"]
    assert s2["target_within_ci95"] is True
    lo, hi = s2["bootstrap_ci95"]
    assert lo <= s2["target_value"] <= hi
    assert s2["rel_err"] < 0.025


def test_sigma2_fit_quality(output):
    """The row-variance Symanzik fit is well-determined (unlike the
    noisy T_00 aggregate it feeds)."""
    assert output["sigma2_xi"]["r_squared"] > 0.5


def test_verdict_confirmed(output):
    assert output["verdict"] == "VARIANCE_REACTIVITY_IDENTITY_CONFIRMED"
