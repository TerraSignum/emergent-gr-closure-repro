"""Lemma B T-loop: seed-stability of the Fiedler-active set.

S-loop established that the ~30% Fiedler-active set
correlates with the matter-core/bulk Interface. The natural
follow-up: is this set the SAME nodes across different seeds
within one regime (structural), or different sets each time
(statistical)?

For each regime:
  - Take each pair (s_a, s_b) of seeds.
  - Compute the Fiedler-top-30% set for each.
  - Jaccard overlap between the two sets.
Cross-seed-pair-mean Jaccard per regime tells us the
structural stability:
  - J ~ 1.0: same nodes always (structural)
  - J ~ 0.3 = random baseline (k/n for k=0.3n): statistical
  - J in between: partially structural

Output: outputs/verify_lemma_B_fiedler_seed_stability.json
"""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REPO_ROOT = REPO.parent
OUT = REPO / "outputs" / "verify_lemma_B_fiedler_seed_stability.json"

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


def fiedler_top_k(xi: np.ndarray, k: int) -> set | None:
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
    return set(np.argsort(-fiedler)[:k])


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def audit_regime(regime, n_lat, rel, hint):
    npz = REPO_ROOT / rel
    xis = load_all_xi(npz, hint)
    if not xis:
        return {"regime": regime, "N": n_lat,
                "status": "SNAPSHOT_NOT_AVAILABLE"}
    n = xis[0].shape[0]
    k = max(3, int(np.round(0.30 * n)))
    sets = [fiedler_top_k(xi, k) for xi in xis]
    sets = [s for s in sets if s is not None]
    if len(sets) < 2:
        return {"regime": regime, "N": n_lat,
                "n_seeds": len(sets),
                "status": "INSUFFICIENT_SEEDS"}

    # All pairwise Jaccard
    pairs = list(combinations(range(len(sets)), 2))
    js = [jaccard(sets[i], sets[j]) for i, j in pairs]

    # Random baseline: two random k-subsets of {0..n-1}
    # E[J] = k*(k/n) / (2k - k*(k/n)) = (k/n) / (2 - k/n)
    k_over_n = k / n
    random_baseline = k_over_n / (2 - k_over_n)

    return {
        "regime": regime, "N": n_lat,
        "n_seeds": len(sets),
        "k_30pct": k,
        "n_pairs": len(pairs),
        "jaccard_pairwise_mean": float(np.mean(js)),
        "jaccard_pairwise_std": float(np.std(js, ddof=1)) if len(js) > 1 else 0.0,
        "jaccard_pairwise_min": float(np.min(js)),
        "jaccard_pairwise_max": float(np.max(js)),
        "random_baseline_jaccard": float(random_baseline),
        "jaccard_over_baseline":
            float(np.mean(js) / random_baseline),
        "status": "OK",
    }


def main():
    per_regime = [audit_regime(*row) for row in LADDER]
    out = {
        "headline": ("Lemma B T-loop: seed-stability of the "
                     "Fiedler-active set. Pairwise Jaccard of "
                     "Fiedler-top-30% sets across all seed pairs "
                     "within each regime."),
        "per_regime": per_regime,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_summary(per_regime)
    return 0


def print_summary(per_regime):
    print("=" * 90)
    print("Lemma B T-loop: Fiedler-active set seed-stability")
    print("=" * 90)
    print(f"{'Regime':<8} {'N':>4} {'seeds':>6} {'pairs':>6} "
          f"{'mean J':>9} {'std':>8} {'min':>7} {'max':>7} "
          f"{'random':>8} {'ratio':>7}")
    print("-" * 90)
    for r in per_regime:
        if r["status"] != "OK":
            print(f"{r['regime']:<8} {r['N']:>4}  {r['status']}")
            continue
        print(f"{r['regime']:<8} {r['N']:>4} "
              f"{r['n_seeds']:>6} {r['n_pairs']:>6} "
              f"{r['jaccard_pairwise_mean']:>9.4f} "
              f"{r['jaccard_pairwise_std']:>8.4f} "
              f"{r['jaccard_pairwise_min']:>7.4f} "
              f"{r['jaccard_pairwise_max']:>7.4f} "
              f"{r['random_baseline_jaccard']:>8.4f} "
              f"{r['jaccard_over_baseline']:>7.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
