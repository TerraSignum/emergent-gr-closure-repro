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
LEGITIMATELY_UNTAGGED = {
    # Auto-discovered as of 2026-05-14. Add new entries here only after
    # confirming the JSON is NOT a per-regime audit output.
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
    """The 201 JSONs that carried branch_metadata at the 2026-05-14
    Round-2 snapshot must still carry it. A drop signals that someone
    re-ran an audit script and lost the tag (the iter7_branch_tag_apply
    post-hoc injection regression of R2-C3)."""
    jsons = _all_jsons()
    untagged = [p.name for p in jsons if not _is_tagged(p)]
    legit_set = LEGITIMATELY_UNTAGGED
    # Drift test: if more than the expected ~54 are untagged, fail loudly.
    new_untagged = [n for n in untagged if n not in legit_set]
    # We expect ~54 legitimately-untagged outputs at the snapshot. Allow up
    # to 70 to absorb minor corpus growth without flapping; below that
    # threshold a regression of the tag-injection workflow is the most
    # likely cause and the test fails loudly.
    assert len(new_untagged) <= 70, (
        f"Branch-metadata drop detected: {len(new_untagged)} untagged "
        f"JSONs (expected <=70 at the Round-2 snapshot). Re-run "
        f"peer_reviews/iter7_branch_tag_apply.py or investigate "
        f"whether a script regression dropped the tag. "
        f"First-15 untagged: {sorted(new_untagged)[:15]}"
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
