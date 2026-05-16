"""Pin presence of branch_metadata on the 201 tagged JSONs.

Round-2 peer-review finding R2-C3: the `branch_metadata` block on
outputs/*.json is injected post-hoc by `peer_reviews/iter7_branch_tag_apply.py`
using an external manifest. Re-running an audit script from scratch
silently drops the tag. This test catches the silent drop.

If a NEW outputs/*.json is added that legitimately has no branch
metadata (because it's not a regime-tagged audit), add its path to
LEGITIMATELY_UNTAGGED below.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"
SCHEMA = "worldformula/branch_metadata/v1"


# JSONs that legitimately don't carry branch_metadata (e.g. cross-regime
# aggregates, anchor files that are corpus-wide rather than regime-specific).
# Enumerated 2026-05-14 (Round 4 R4-S1 fix). When adding a NEW outputs/*.json,
# either ensure peer_reviews/iter7_branch_tag_apply.py tags it, OR add its
# basename here after confirming the JSON is NOT a per-regime audit output.
# The strict set ensures any silent tag-drop on a previously-tagged file
# fails the test (Round-2 R2-C3 anti-regression purpose).
LEGITIMATELY_UNTAGGED = {
    "audit_omega_m_dual_form_corpus_anchors.json",
    "bh_sector_recompute.json",
    "carrier_bakry_emery_cd_audit.json",
    "clp_c_gamma_convergence_detailed_audit.json",
    "clp_full_report.json",
    "cosmological_constant_recompute.json",
    "curvature_fixed_point_certificate.json",
    "dimension_and_arrow_recompute.json",
    "einstein_gap_5point_recompute.json",
    "einstein_metric_stress_certificate.json",
    "emergent_time_recompute.json",
    "graviton_cpt_recompute.json",
    "hawking_spectrum_recompute.json",
    "inflation_closure_recompute.json",
    "lorentzian_3plus1_recompute.json",
    "regime_invariance_audit.json",
    "schwarzschild_defect_recompute.json",
    "signed_ricci_lower_bound_audit.json",
    "stage6f_full_tensor_norm_audit.json",
    "stage6f_signed_dm_hypothesis_test.json",
    "sup_aware_envelope_refinement.json",
    "t00_summand_decomposition_audit.json",
    "vacuum_stability_recompute.json",
    "verify_admissibility_counterexample_and_spectral_gap.json",
    "verify_chirality_flip_cross_sector.json",
    "verify_clp_b_architectural_ceiling.json",
    "verify_clp_b_redundancy_audit.json",
    "verify_cone_tightness_extended_ladder.json",
    "verify_cone_tightness_p5_ladder.json",
    "verify_density_contrast_a0_structural.json",
    "verify_gap_monotonicity_p5_ladder.json",
    "verify_lemma_B_M_F_empirical_off_diagonal_extraction.json",
    "verify_lemma_B_M_F_off_diagonal_identification.json",
    "verify_lemma_B_alpha_xi_master_identity.json",
    "verify_lemma_B_branch_resolved_fit.json",
    "verify_lemma_B_carrier_spectral_synthesis.json",
    "verify_lemma_B_defect_condensation_test.json",
    "verify_lemma_B_degree_concentration.json",
    "verify_lemma_B_edge_correlation.json",
    "verify_lemma_B_edge_weight_structure.json",
    "verify_lemma_B_equitable_partition.json",
    "verify_lemma_B_gap_statistical_fingerprint.json",
    "verify_lemma_B_family_factor_p1p2prime.json",
    "verify_lemma_B_family_factor_p5n_canonical.json",
    "verify_lemma_B_fiedler_halo_test.json",
    "verify_lemma_B_fiedler_overlap.json",
    "verify_lemma_B_fiedler_vs_corpus_matter_core.json",
    "verify_lemma_B_fiedler_vs_t00.json",
    "verify_lemma_B_matter_branch_universality.json",
    "verify_lemma_B_p0p8_cross_validation.json",
    "verify_lemma_B_percentile_decomposition.json",
    "verify_lemma_B_radial_hypothesis.json",
    "verify_lemma_B_skeleton_diameter.json",
    "verify_lemma_B_skeleton_laplacian.json",
    "verify_lemma_B_spectral_fingerprint.json",
    "verify_lemma_B_threshold_sweep.json",
    "verify_lemma_B_uniform_poincare.json",
    "verify_lemma_B_universal_X_minus_1_over_X_pattern.json",
    "variance_reactivity_identity_audit.json",
    "xi_graph_topology_deep.json",
}


def _all_jsons():
    return sorted(OUTPUTS.glob("*.json"))


def _is_tagged(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(d, dict) and "branch_metadata" in d


@pytest.mark.skipif(not OUTPUTS.is_dir(), reason="outputs/ directory absent")
def test_branch_metadata_present_on_known_tagged():
    """Strict allowlist (R4-S1 fix): every outputs/*.json must either
    carry branch_metadata OR appear in LEGITIMATELY_UNTAGGED. Any single
    silent drop of a tagged file fails the test immediately, which is
    the iter7_branch_tag_apply post-hoc injection anti-regression
    purpose of R2-C3."""
    jsons = _all_jsons()
    untagged = {p.name for p in jsons if not _is_tagged(p)}
    new_untagged = untagged - LEGITIMATELY_UNTAGGED
    assert not new_untagged, (
        f"Branch-metadata drop detected on previously-tagged file(s): "
        f"{sorted(new_untagged)}. Either re-run "
        f"peer_reviews/iter7_branch_tag_apply.py to re-tag, OR if the "
        f"file legitimately should not be tagged (e.g. it is a "
        f"corpus-wide aggregate, not a per-regime audit), add its "
        f"basename to LEGITIMATELY_UNTAGGED."
    )


@pytest.mark.skipif(not OUTPUTS.is_dir(), reason="outputs/ directory absent")
def test_tagged_jsons_use_canonical_schema():
    """Every JSON that does carry branch_metadata must use the canonical
    v1 schema -- silent schema drift is also a regression."""
    bad = []
    for p in _all_jsons():
        try:
            with p.open("r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        bm = d.get("branch_metadata") if isinstance(d, dict) else None
        if not isinstance(bm, dict):
            continue
        schema = bm.get("schema")
        if schema != SCHEMA:
            bad.append((p.name, schema))
    assert not bad, (
        f"Non-canonical branch_metadata.schema values found: {bad[:10]}. "
        f"Expected schema={SCHEMA!r} on every tagged JSON."
    )
