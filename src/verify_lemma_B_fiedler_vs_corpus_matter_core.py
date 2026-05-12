"""Lemma B T-loop corrected: Fiedler-active set vs corpus-validated
matter-core proxies.

User correction: the S-loop audit used row-sum(Xi) as matter-core
proxy, but the corpus has an established matter-core
classification with validated AUC scores (P4 lines ~700, anisotropic
companion paper):

  - Per-node t_00(a) top-decile:  AUC 0.85, enrichment 11.3x  (PRIMARY)
  - Per-node closure-residual Δ(a) top-decile: AUC 0.62      (secondary)
  - Per-node |w(a)|>1/2 topological winding:                  (tertiary)
  - Row-VARIANCE(Xi):  AUC 0.83-0.90 in LORO 8/8              (top per-node candidate)

Computing t_00(a) requires loading psi/K/Q fields and running the
per-seed Galerkin construction — a heavy pipeline. The
row-variance(Xi) proxy is the cheap-and-cheerful alternative
with corpus-validated AUC 0.83-0.90.

This audit:
  - Re-runs the Fiedler-overlap analysis with row-variance as
    matter-core proxy (vs row-sum from the S-loop).
  - Compares the two proxies for the same Fiedler-active set.
  - Tests robustness: do Fiedler hot-spots correlate with the
    BETTER matter-core proxy?

Output: outputs/verify_lemma_B_fiedler_vs_corpus_matter_core.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REPO_ROOT = REPO.parent
OUT = REPO / "outputs" / "verify_lemma_B_fiedler_vs_corpus_matter_core.json"

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


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def per_snapshot_audit(xi: np.ndarray) -> dict | None:
    """Per-snapshot: Fiedler top-p% vs different matter-core proxies."""
    n = xi.shape[0]
    w = xi - np.eye(n)
    w = np.maximum(w, 0.0)
    deg_sum = w.sum(axis=1)
    # row-variance(Xi) is the corpus-validated proxy
    # (use the full Xi row including diagonal=1, which is constant)
    # so equivalently variance of off-diagonal Xi entries per row
    deg_var = np.array([np.var(w[i, :]) for i in range(n)])
    if np.any(deg_sum <= 1e-12):
        return None
    d_inv_sqrt = 1.0 / np.sqrt(deg_sum)
    norm = w * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    lap = np.eye(n) - norm
    lap = 0.5 * (lap + lap.T)
    eigvals, eigvecs = np.linalg.eigh(lap)
    fiedler = np.abs(eigvecs[:, 1])

    # Top-30% by various criteria
    k30 = max(3, int(np.round(0.30 * n)))
    k01 = max(3, int(np.round(0.01 * n)))
    k05 = max(3, int(np.round(0.05 * n)))

    fiedler_top30 = set(np.argsort(-fiedler)[:k30])
    rowsum_top30 = set(np.argsort(-deg_sum)[:k30])
    rowvar_top30 = set(np.argsort(-deg_var)[:k30])

    # Also: C95 (top-5%) and C99 (top-1%) layers per corpus-standard
    rowvar_C95 = set(np.argsort(-deg_var)[:k05])
    rowvar_C99 = set(np.argsort(-deg_var)[:k01])

    # Random baseline for top-30% sets of size k30 in n nodes
    k_over_n = k30 / n
    random_30 = k_over_n / (2 - k_over_n)
    # For asymmetric sizes (e.g., k30 vs k05) Jaccard:
    # E[|A∩B|] = a*(b/n), |A∪B| = a + b - a*(b/n)
    # E[J] = (a*b/n) / (a+b-a*b/n)
    def asym_random(a, b, total):
        return (a * b / total) / (a + b - a * b / total)

    return {
        "n_nodes": n,
        "k_30pct": k30,
        # vs row-sum (my naive S-loop proxy)
        "J_fiedler_vs_rowsum_top30": jaccard(fiedler_top30, rowsum_top30),
        # vs row-variance (corpus-validated proxy, AUC 0.83-0.90)
        "J_fiedler_vs_rowvar_top30": jaccard(fiedler_top30, rowvar_top30),
        # vs corpus-standard layers C95, C99 (asymmetric set sizes)
        "J_fiedler_vs_C95_rowvar": jaccard(fiedler_top30, rowvar_C95),
        "J_fiedler_vs_C99_rowvar": jaccard(fiedler_top30, rowvar_C99),
        # Random baselines
        "random_30_30": random_30,
        "random_30_5": asym_random(k30, k05, n),
        "random_30_1": asym_random(k30, k01, n),
        # Coupling of the two proxies themselves
        "J_rowsum_vs_rowvar_top30": jaccard(rowsum_top30, rowvar_top30),
    }


def audit_regime(regime, n_lat, rel, hint):
    npz = REPO_ROOT / rel
    xis = load_all_xi(npz, hint)
    if not xis:
        return {"regime": regime, "N": n_lat,
                "status": "SNAPSHOT_NOT_AVAILABLE"}
    diags = [per_snapshot_audit(xi) for xi in xis]
    diags = [d for d in diags if d is not None]
    if not diags:
        return {"regime": regime, "N": n_lat,
                "status": "ALL_DEGENERATE"}
    out = {"regime": regime, "N": n_lat,
           "n_seeds": len(diags),
           "status": "OK"}
    keys = ["J_fiedler_vs_rowsum_top30", "J_fiedler_vs_rowvar_top30",
            "J_fiedler_vs_C95_rowvar", "J_fiedler_vs_C99_rowvar",
            "random_30_30", "random_30_5", "random_30_1",
            "J_rowsum_vs_rowvar_top30"]
    for key in keys:
        vals = [d[key] for d in diags]
        out[f"{key}_mean"] = float(np.mean(vals))
    return out


def main():
    per_regime = [audit_regime(*row) for row in LADDER]
    out = {
        "headline": ("Lemma B T-loop corrected: Fiedler-active set "
                     "vs corpus-validated matter-core proxies "
                     "(row-variance + C95/C99 layers)."),
        "per_regime": per_regime,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_summary(per_regime)
    return 0


def print_summary(per_regime):
    print("=" * 110)
    print("Lemma B T-loop: Fiedler vs row-sum (S-loop) vs row-variance (corpus-validated) vs C95/C99")
    print("=" * 110)
    print(f"{'Regime':<8} {'N':>4} {'seeds':>6} "
          f"{'J(F,rowsum)':>11} {'J(F,rowvar)':>11} "
          f"{'J(F,C95)':>9} {'J(F,C99)':>9} "
          f"{'J(rs,rv)':>9} {'rnd30':>7}")
    print("-" * 110)
    for r in per_regime:
        if r["status"] != "OK":
            continue
        print(f"{r['regime']:<8} {r['N']:>4} {r['n_seeds']:>6} "
              f"{r['J_fiedler_vs_rowsum_top30_mean']:>11.4f} "
              f"{r['J_fiedler_vs_rowvar_top30_mean']:>11.4f} "
              f"{r['J_fiedler_vs_C95_rowvar_mean']:>9.4f} "
              f"{r['J_fiedler_vs_C99_rowvar_mean']:>9.4f} "
              f"{r['J_rowsum_vs_rowvar_top30_mean']:>9.4f} "
              f"{r['random_30_30_mean']:>7.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
