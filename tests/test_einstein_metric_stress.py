"""Tests for the discrete Riemannian-embedding-stress series.

The bundled 7-point series stores the geometric coherence c_g(N) and
the Einstein-metric-stress sigma(N) at seven lattice sizes. The
manuscript-relevant claims are:

  (1) sigma(N) = 1 - c_g(N) holds to floating-point precision on every
      row (algebraic identity by definition of the MDS coherence);
  (2) the regime profile is non-monotonic with sigma in [0.26, 0.34],
      i.e. the lattice retains robust but imperfect geometric
      coherence across the full available range;
  (3) two-point Richardson under N^(-alpha) ansatze does not extrapolate
      sigma to zero (sigma_inf ~ 0.30 at all three canonical exponents),
      which is consistent with the canonical multi-N tensor-closure-gap
      series and is qualitatively distinct from the narrow Einstein-
      identity-gap axis.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import verify_einstein_metric_stress as M  # noqa: E402


@pytest.fixture(scope="module")
def series():
    return M.load_series()


@pytest.fixture(scope="module")
def output():
    M.main()
    out_path = REPO / "outputs" / "einstein_metric_stress_certificate.json"
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_seven_points_in_series(series):
    rows = series["raw_series"]
    assert len(rows) == 7
    Ns = [r["N"] for r in rows]
    assert min(Ns) == pytest.approx(409.5, abs=0.1)
    assert max(Ns) == pytest.approx(14181.0, abs=0.1)


def test_identity_sigma_equals_one_minus_coherence(series):
    """sigma(N) = 1 - c_g(N) must hold to floating-point precision."""
    rows = M.identity_residuals(series)
    for r in rows:
        assert r["identity_passes"] is True, (
            f"identity violated at N={r['N']}: |1 - c_g - sigma| = "
            f"{r['identity_residual']:.2e}"
        )


def test_geometric_coherence_in_unit_interval(series):
    """The coherence is a 1/(1+stress) score and must lie strictly in (0,1)."""
    for row in series["raw_series"]:
        assert 0.0 < row["geometric_coherence"] < 1.0
        assert 0.0 < row["einstein_metric_stress"] < 1.0


def test_minimum_coherence_is_at_intermediate_N(series):
    """The maximum stress occurs at intermediate N (not at the endpoints).
    This is the load-bearing physical observation: the discrete-Riemannian
    obstruction has a non-monotonic profile."""
    prof = M.regime_profile(series)
    Nmin, Nmax = prof["N_range"]
    N_at_max = prof["N_at_max_stress"]
    assert N_at_max > Nmin
    assert N_at_max < Nmax
    # Specifically at N approximately 4000 (the third regime out of seven).
    assert 3000 < N_at_max < 7000


def test_sigma_floor_around_0_30_under_two_point_richardson(series):
    """All three canonical Richardson ansatze (alpha = 2/3, 1, 0.848)
    extrapolate to a positive sigma_inf, not to zero. This is the
    qualitative-distinction claim: the broad Riemannian-embedding axis
    has a finite-resolution stress floor at the available lattice sizes."""
    rich = M.richardson_endpoints_at_alpha(series)
    for cand in rich["candidates"]:
        assert cand["sigma_inf"] > 0.20, (
            f"alpha = {cand['alpha']}: sigma_inf = {cand['sigma_inf']} "
            f"unexpectedly close to or below 0.20"
        )
        assert cand["sigma_inf"] < 0.40, (
            f"alpha = {cand['alpha']}: sigma_inf = {cand['sigma_inf']} "
            f"unexpectedly above 0.40"
        )


def test_certificate_output_has_expected_keys(output):
    expected = {
        "criterion", "definition", "identity_check",
        "all_identity_pass", "regime_profile",
        "richardson_endpoints", "verdict",
    }
    assert expected.issubset(set(output.keys()))
    assert output["all_identity_pass"] is True
    assert output["regime_profile"]["n_points"] == 7
    assert len(output["richardson_endpoints"]["candidates"]) == 3


def test_geometric_coherence_at_smallest_N_above_two_thirds(series):
    """At the smallest available N, the lattice is robustly coherent:
    c_g > 2/3, i.e. the embedding stress is below 50%. This anchors
    the lower end of the observed regime."""
    rows = sorted(series["raw_series"], key=lambda r: r["N"])
    smallest = rows[0]
    assert smallest["geometric_coherence"] > 2.0 / 3.0
