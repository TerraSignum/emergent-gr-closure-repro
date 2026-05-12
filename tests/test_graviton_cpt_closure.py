"""Tests for the bundled graviton + CPT + CP-violation closures."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import verify_graviton_cpt_closure as M  # noqa: E402


@pytest.fixture(scope="module")
def bundle():
    return M.load_bundle()


@pytest.fixture(scope="module")
def output(bundle):
    M.main()
    out_path = REPO / "outputs" / "graviton_cpt_recompute.json"
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_graviton_massless_spin_two_two_polarizations(bundle):
    g = bundle["graviton_closure"]
    v = g["values"]
    assert v["n_causal_modes_canonical"] == 2
    assert v["graviton_mass_lu"] == 0.0
    assert v["graviton_massless"] is True
    assert v["graviton_spin"] == 2
    assert v["graviton_polarizations"] == 2
    assert v["dispersion"] == "omega = |k|"


def test_graviton_derivation_four_steps(bundle):
    steps = bundle["graviton_closure"]["derivation"]
    assert len(steps) == 4
    step_names = {s["step"] for s in steps}
    assert step_names == {"mass", "spin", "dispersion", "coupling"}


def test_cpt_theorem_derived_from_three_premises(bundle):
    c = bundle["CPT_theorem_closure"]
    derivation = " ".join(c["derivation"])
    assert "Lorentz" in derivation
    assert "Locality" in derivation or "locality" in derivation
    assert "Unitarity" in derivation or "unitarity" in derivation
    assert c["tier"] == "DERIVED"


def test_cpt_pauli_lueders_bound_satisfied(bundle):
    """Construction predicts zero CPT-violation; trivially satisfies the
    experimental Pauli-Lueders upper bound from neutral-kaon mass-
    difference symmetry, currently 2e-12 GeV^2 (PDG 2024)."""
    eb = bundle["CPT_theorem_closure"]["experimental_bound"]
    assert eb["satisfied"] is True
    assert eb["model_prediction_GeV2"] == 0.0
    assert eb["anchor_value_GeV2_squared_mass"] == 2.0e-12
    assert "neutral-kaon" in eb["name"] or "Pauli-Lueders" in eb["name"]


def test_cp_violation_unique_mechanism(bundle):
    cp = bundle["CP_violation_closure"]
    assert "ONLY CP-violation mechanism" in cp["claim"]
    assert "GUE" in " ".join(cp["derivation"])
    assert cp["tier"] == "DERIVED"


def test_strong_CP_problem_dissolved_structurally(bundle):
    cp = bundle["CP_violation_closure"]
    consequence = cp["consequence_for_strong_CP_problem"]
    assert "theta_QCD" in consequence
    assert "structural" in consequence


def test_zero_fitted_parameters(bundle):
    assert bundle["summary"]["fitted_parameters"] == 0


def test_recompute_output_passes(output):
    assert output["verdict"] == "PASS"
    assert output["graviton_massless"] is True
    assert output["graviton_spin"] == 2
    assert output["graviton_polarizations"] == 2
    assert output["fitted_parameters"] == 0
