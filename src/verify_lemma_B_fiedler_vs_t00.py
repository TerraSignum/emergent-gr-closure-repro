"""Lemma B W-loop: Fiedler-vs-t_00 cross-check using the
corpus's canonical Galerkin pipeline.

User correction: my earlier matter-core audits used
row-variance(Xi) as a *proxy* for matter-core. The corpus's
PRIMARY classifier is t_00(a), the per-node Galerkin
energy-density, computed via per_seed_galerkin from
psi/K/Q fields. AUC 0.85, enrichment 11.3x for the matter-
core membership prediction (P4 §local-RG-window, anisotropic
companion paper).

The snapshot npz files contain ALL keys needed:
  - edge_xi_snapshots (24 seeds x 17 timesteps x N x N)
  - psi_real_snapshots, psi_imag_snapshots
  - k_snapshots, q_snapshots (24 x 17 x N x N)
  - ff_K_seedK, ff_Q_seedK (final K, Q per seed)

This audit:
  1. Loads psi, K, Q, Xi at the last timestep for each seed.
  2. Runs per_seed_galerkin (corpus-standard) to get t_00
     per node.
  3. Identifies matter-core as top-10% by |t_00|.
  4. Computes Jaccard overlap with Fiedler-top-30%.
  5. Compares to the row-variance proxy used earlier
     (verify_lemma_B_fiedler_vs_corpus_matter_core.py).

This is the canonical t_00-based test of the
Fiedler-vs-matter-core hypothesis.

Output: outputs/verify_lemma_B_fiedler_vs_t00.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REPO_ROOT = REPO.parent
OUT = REPO / "outputs" / "verify_lemma_B_fiedler_vs_t00.json"

sys.path.insert(0, str(REPO / "src"))


class _BlockCupy:
    def find_spec(self, name, path=None, target=None):
        if name == "cupy" or name.startswith("cupy."):
            raise ImportError("cupy disabled")
        return None

    def load_module(self, name):
        raise ImportError("cupy disabled")


sys.meta_path.insert(0, _BlockCupy())

from verify_galerkin_runner_A_hessian_ricci import per_seed_galerkin

LADDER = [
    ("P5N64",  64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz"),
    ("P5N72",  72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz"),
    ("P5N84",  84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz"),
    ("P5N100", 100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz"),
    ("P5N128", 128, "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz"),
    ("P5N200", 200, "results_d1_p5n200_8seeds/P5N200.snapshots.npz"),
    ("P5N256", 256, "results_d1_p5n256_12seeds/P5N256.snapshots.npz"),
    ("P5N300", 300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz"),
    ("P5N512", 512, "results_d1_p5n512_12seeds/P5N512.snapshots.npz"),
]
# Throttle seeds for the heavy Galerkin pipeline (O(N^3) per snapshot).
MAX_SEEDS_PER_REGIME = 6


def load_snapshots(npz_path: Path, n_lat: int, max_seeds: int):
    """Yield (xi, psi, k_field, q_field) for each seed,
    taking the last timestep."""
    if not npz_path.exists():
        return
    z = np.load(npz_path, allow_pickle=True)
    if "edge_xi_snapshots" not in z.files:
        return
    xi_snaps = np.asarray(z["edge_xi_snapshots"])
    last = xi_snaps.shape[1] - 1
    n_seeds_total = xi_snaps.shape[0]
    n_seeds = min(max_seeds, n_seeds_total)
    psi_re = np.asarray(z["psi_real_snapshots"])
    psi_im = np.asarray(z["psi_imag_snapshots"])
    k_snaps = np.asarray(z["k_snapshots"])
    q_snaps = np.asarray(z["q_snapshots"])
    for s in range(n_seeds):
        xi = np.asarray(xi_snaps[s, last], dtype=float).copy()
        np.fill_diagonal(xi, 1.0)
        psi = psi_re[s, last] + 1j * psi_im[s, last]
        k_field = np.asarray(k_snaps[s, last], dtype=float)
        q_field = np.asarray(q_snaps[s, last], dtype=float)
        yield xi, psi, k_field, q_field


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def per_snapshot_audit(xi: np.ndarray, psi: np.ndarray,
                        k_field: np.ndarray, q_field: np.ndarray
                        ) -> dict | None:
    n = xi.shape[0]
    # Fiedler from weighted Laplacian on Xi (off-diag, clamp neg)
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
    k30 = max(3, int(np.round(0.30 * n)))
    fiedler_set = set(np.argsort(-fiedler)[:k30])

    # Corpus-canonical t_00 via Galerkin
    try:
        prep = per_seed_galerkin(xi, psi, k_field, q_field, n, np)
    except Exception:
        return None
    t00 = np.asarray(prep["t00"], dtype=float)
    if not np.all(np.isfinite(t00)):
        t00 = np.where(np.isfinite(t00), t00, 0.0)
    abs_t00 = np.abs(t00)

    # Matter-core proxies
    k10 = max(3, int(np.round(0.10 * n)))    # top-10% standard
    k01 = max(3, int(np.round(0.01 * n)))    # top-1% = C99
    k05 = max(3, int(np.round(0.05 * n)))    # top-5% = C95
    k30b = k30
    t00_top10 = set(np.argsort(-abs_t00)[:k10])
    t00_top30 = set(np.argsort(-abs_t00)[:k30b])
    t00_top5 = set(np.argsort(-abs_t00)[:k05])
    t00_top1 = set(np.argsort(-abs_t00)[:k01])

    # row-variance proxy (for cross-check vs earlier audits)
    deg_var = np.array([np.var(w[i, :]) for i in range(n)])
    rowvar_top30 = set(np.argsort(-deg_var)[:k30])
    rowvar_top10 = set(np.argsort(-deg_var)[:k10])

    def er(a: int, b: int, total: int) -> float:
        """Expected Jaccard for random disjoint subsets of sizes a, b."""
        return (a * b / total) / (a + b - a * b / total)

    return {
        "n_nodes": n,
        # Fiedler vs t_00 (corpus-canonical proxy)
        "J_fiedler30_vs_t00_top10": jaccard(fiedler_set, t00_top10),
        "J_fiedler30_vs_t00_top30": jaccard(fiedler_set, t00_top30),
        "J_fiedler30_vs_t00_top5_C95": jaccard(fiedler_set, t00_top5),
        "J_fiedler30_vs_t00_top1_C99": jaccard(fiedler_set, t00_top1),
        # Fiedler vs row-variance (earlier audit's proxy)
        "J_fiedler30_vs_rowvar_top30": jaccard(fiedler_set, rowvar_top30),
        "J_fiedler30_vs_rowvar_top10": jaccard(fiedler_set, rowvar_top10),
        # row-variance vs t_00 (proxy quality)
        "J_rowvar30_vs_t00_top30": jaccard(rowvar_top30, t00_top30),
        "J_rowvar10_vs_t00_top10": jaccard(rowvar_top10, t00_top10),
        # Random baselines
        "rand_30_10": er(k30, k10, n),
        "rand_30_30": er(k30, k30b, n),
        "rand_30_5":  er(k30, k05, n),
        "rand_30_1":  er(k30, k01, n),
    }


def audit_regime(regime: str, n_lat: int, rel: str):
    npz = REPO_ROOT / rel
    results = []
    for xi, psi, k_field, q_field in load_snapshots(
            npz, n_lat, MAX_SEEDS_PER_REGIME):
        r = per_snapshot_audit(xi, psi, k_field, q_field)
        if r is not None:
            results.append(r)
    if not results:
        return {"regime": regime, "N": n_lat,
                "status": "NO_DATA"}
    out = {"regime": regime, "N": n_lat,
           "n_seeds": len(results),
           "status": "OK"}
    keys = [k for k in results[0].keys() if k != "n_nodes"]
    for key in keys:
        vals = [r[key] for r in results]
        out[f"{key}_mean"] = float(np.mean(vals))
    return out


def main():
    per_regime = [audit_regime(*row) for row in LADDER]
    out = {
        "headline": ("Lemma B W-loop: Fiedler-top-30% vs "
                     "corpus-canonical t_00 (Galerkin), with "
                     "cross-check vs row-variance proxy from "
                     "earlier audit."),
        "ladder": [reg for reg, *_ in LADDER],
        "max_seeds_per_regime": MAX_SEEDS_PER_REGIME,
        "per_regime": per_regime,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_summary(per_regime)
    return 0


def print_summary(per_regime):
    print("=" * 115)
    print("Lemma B W-loop: Fiedler vs t_00 (corpus-canonical) vs row-variance (proxy)")
    print("=" * 115)
    print(f"{'Regime':<8} {'N':>4} {'seeds':>5} "
          f"{'J(F,t00_10)':>12} {'J(F,t00_30)':>12} "
          f"{'J(F,t00_5)':>11} {'J(F,t00_1)':>11} "
          f"{'J(F,rv_30)':>11} {'J(rv,t00_30)':>13}")
    print("-" * 115)
    for r in per_regime:
        if r["status"] != "OK":
            print(f"{r['regime']:<8} {r['N']:>4}  {r['status']}")
            continue
        print(f"{r['regime']:<8} {r['N']:>4} {r['n_seeds']:>5} "
              f"{r.get('J_fiedler30_vs_t00_top10_mean', 0):>12.4f} "
              f"{r.get('J_fiedler30_vs_t00_top30_mean', 0):>12.4f} "
              f"{r.get('J_fiedler30_vs_t00_top5_C95_mean', 0):>11.4f} "
              f"{r.get('J_fiedler30_vs_t00_top1_C99_mean', 0):>11.4f} "
              f"{r.get('J_fiedler30_vs_rowvar_top30_mean', 0):>11.4f} "
              f"{r.get('J_rowvar30_vs_t00_top30_mean', 0):>13.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
