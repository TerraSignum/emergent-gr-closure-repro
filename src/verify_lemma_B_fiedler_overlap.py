"""Lemma B S-loop: where are the spectrally-active vertices?

The R-loop established that ~30% of vertices are spectrally
active in the Fiedler mode of L_w (PR/N -> 0.31). The
natural follow-up: WHERE structurally are those 30% located?
Are they the matter-core nodes (top by row-sum)? The bulk
nodes (bottom by row-sum)? Both mixed evenly?

For each snapshot we:
  1. Compute the Fiedler vector |v_2|.
  2. Select the top-30% by |v_2| (the "spectrally active set").
  3. Compute Jaccard overlap with:
     - Top-30% by row-sum (matter-core proxy)
     - Bottom-30% by row-sum (bulk proxy)
     - Top-50% by row-sum
     - Bottom-50% by row-sum
  4. Report per-snapshot overlap; cross-seed mean per regime.

If Fiedler-active set has high Jaccard overlap with
matter-core, the spectral gap is matter-core-driven; if it
overlaps with bulk, bulk-driven; if it has ~50% overlap with
each (random), the active set is regime-independent.

Output: outputs/verify_lemma_B_fiedler_overlap.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REPO_ROOT = REPO.parent
OUT = REPO / "outputs" / "verify_lemma_B_fiedler_overlap.json"

LADDER = [
    ("P5",     50,  "results_d1_fix17/d1_p5.npz",                       "xi_seedK"),
    ("P5N64",  64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz",     "edge_xi_snapshots"),
    ("P5N72",  72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz",     "edge_xi_snapshots"),
    ("P5N84",  84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz",     "edge_xi_snapshots"),
    ("P5N100", 100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz",   "edge_xi_snapshots"),
    ("P5N128", 128, "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz",  "edge_xi_snapshots"),
    ("P5N200", 200, "results_d1_p5n200_8seeds/P5N200.snapshots.npz",    "edge_xi_snapshots"),
    ("P5N256", 256, "results_d1_p5n256_12seeds/P5N256.snapshots.npz",   "edge_xi_snapshots"),
    ("P5N300", 300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz",   "edge_xi_snapshots"),
    ("P5N512", 512, "results_d1_p5n512_12seeds/P5N512.snapshots.npz",   "edge_xi_snapshots"),
]


def load_all_xi(npz_path: Path, hint: str) -> list[np.ndarray]:
    if not npz_path.exists():
        return []
    z = np.load(npz_path, allow_pickle=True)
    matrices: list[np.ndarray] = []
    if hint == "edge_xi_snapshots" and "edge_xi_snapshots" in z.files:
        snaps = np.asarray(z["edge_xi_snapshots"])
        last = snaps.shape[1] - 1
        for s in range(snaps.shape[0]):
            xi = np.asarray(snaps[s, last], dtype=float).copy()
            np.fill_diagonal(xi, 1.0)
            matrices.append(xi)
        return matrices
    if hint == "xi_seedK":
        n_seeds = sum(1 for k in z.files if k.startswith("xi_seed"))
        for s in range(n_seeds):
            key = f"xi_seed{s}"
            if key not in z.files:
                continue
            xi = np.asarray(z[key], dtype=float).copy()
            np.fill_diagonal(xi, 1.0)
            matrices.append(xi)
        return matrices
    return matrices


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def per_snapshot_overlap(xi: np.ndarray) -> dict | None:
    n = xi.shape[0]
    w = xi - np.eye(n)
    w = np.maximum(w, 0.0)
    deg = w.sum(axis=1)
    if np.any(deg <= 1e-12):
        return None
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    norm = w * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    lap = np.eye(n) - norm
    lap = 0.5 * (lap + lap.T)
    eigvals, eigvecs = np.linalg.eigh(lap)
    fiedler = np.abs(eigvecs[:, 1])

    # Top-30% by Fiedler amplitude
    k30 = max(3, int(np.round(0.30 * n)))
    k50 = max(3, int(np.round(0.50 * n)))

    fiedler_top30 = set(np.argsort(-fiedler)[:k30])
    rowsum_top30 = set(np.argsort(-deg)[:k30])
    rowsum_bot30 = set(np.argsort(deg)[:k30])
    rowsum_top50 = set(np.argsort(-deg)[:k50])
    rowsum_bot50 = set(np.argsort(deg)[:k50])

    # Random baseline: expected overlap of two random k30-sets =
    # k30 / n (for set_a fixed of size k30, random set_b: prob a node
    # is in both = k30/n).
    random_expected_jaccard = (k30 / n) / (2 - k30 / n)
    # = k30 * 0.3 / (n - k30 * 0.3) when len(b) = len(a) = k30...
    # actually the proper formula: E[|A ∩ B|] = k30 * (k30/n) for
    # random B; E[|A ∪ B|] = k30 + k30 - k30*(k30/n) = k30*(2 - k30/n)
    # E[J] = (k30 * k30/n) / (k30 * (2 - k30/n)) = (k30/n)/(2-k30/n).

    return {
        "n_nodes": n,
        "k_30pct": k30,
        "jaccard_fiedler_vs_core_top30":
            jaccard(fiedler_top30, rowsum_top30),
        "jaccard_fiedler_vs_bulk_bot30":
            jaccard(fiedler_top30, rowsum_bot30),
        "jaccard_fiedler_vs_core_top50":
            jaccard(fiedler_top30, rowsum_top50),
        "jaccard_fiedler_vs_bulk_bot50":
            jaccard(fiedler_top30, rowsum_bot50),
        "jaccard_random_baseline_30_30":
            random_expected_jaccard,
    }


def audit_regime(regime, n_lat, rel, hint):
    npz = REPO_ROOT / rel
    xis = load_all_xi(npz, hint)
    if not xis:
        return {"regime": regime, "N": n_lat,
                "n_seeds_loaded": 0,
                "status": "SNAPSHOT_NOT_AVAILABLE"}
    diags = [per_snapshot_overlap(xi) for xi in xis]
    diags = [d for d in diags if d is not None]
    if not diags:
        return {"regime": regime, "N": n_lat,
                "n_seeds_loaded": len(xis),
                "status": "ALL_DEGENERATE"}
    out = {"regime": regime, "N": n_lat,
           "n_seeds_loaded": len(xis),
           "n_seeds_valid": len(diags),
           "status": "OK"}
    keys = ["jaccard_fiedler_vs_core_top30",
            "jaccard_fiedler_vs_bulk_bot30",
            "jaccard_fiedler_vs_core_top50",
            "jaccard_fiedler_vs_bulk_bot50",
            "jaccard_random_baseline_30_30"]
    for key in keys:
        vals = [d[key] for d in diags]
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def main():
    per_regime = [audit_regime(*row) for row in LADDER]
    out = {
        "headline": ("Lemma B S-loop: structural identification "
                     "of the spectrally-active 30% set "
                     "(Fiedler-amplitude top-30%) via Jaccard "
                     "overlap with row-sum (matter-core/bulk) "
                     "classification."),
        "per_regime": per_regime,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_summary(per_regime)
    return 0


def print_summary(per_regime):
    print("=" * 100)
    print("Lemma B S-loop: Fiedler-active set vs matter-core/bulk Jaccard overlap")
    print("=" * 100)
    print(f"{'Regime':<8} {'N':>4} {'seeds':>6} "
          f"{'J(F,core30%)':>12} {'J(F,bulk30%)':>12} "
          f"{'J(F,core50%)':>12} {'J(F,bulk50%)':>12} "
          f"{'random_30':>10}")
    print("-" * 100)
    for r in per_regime:
        if r["status"] != "OK":
            continue
        print(f"{r['regime']:<8} {r['N']:>4} {r['n_seeds_valid']:>6} "
              f"{r['jaccard_fiedler_vs_core_top30_mean']:>12.4f} "
              f"{r['jaccard_fiedler_vs_bulk_bot30_mean']:>12.4f} "
              f"{r['jaccard_fiedler_vs_core_top50_mean']:>12.4f} "
              f"{r['jaccard_fiedler_vs_bulk_bot50_mean']:>12.4f} "
              f"{r['jaccard_random_baseline_30_30_mean']:>10.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
