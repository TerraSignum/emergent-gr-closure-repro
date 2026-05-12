"""Lemma B R-loop: per-node-percentile and bulk/matter-core
decomposition of lambda_2(L_w).

User question: are the Phase-2 negative findings (radial low-rank
falsified, degree concentration falsified, etc.) running on the
correct regimes / percentiles? The Q-loop threshold-sweep already
showed lambda_skel(tau) is threshold-specific; this R-loop audit
goes further:

  (R1) Per-node-percentile sub-Laplacian audit. For each
       percentile p in {50, 90, 95, 99, sup-equivalent},
       restrict the Xi-weighted graph to the top-p% of nodes
       by row-sum (i.e., most-coupled nodes), compute lambda_2
       of the restricted Laplacian, and fit Symanzik-1 across
       the 10-regime ladder. If lambda_inf varies with p, the
       3/8 conjecture is bulk-vs-core sensitive.

  (R2) Complementary: bottom-p% sub-Laplacian, isolating the
       BULK (low-row-sum, weakly-coupled nodes). If lambda_inf
       differs between top-p% (matter-core) and bottom-p%
       (bulk), the carrier has a non-trivial regime structure
       not captured by the global lambda_2.

  (R3) Fiedler-participation diagnostic: |v_2(i)|, the
       amplitude of the lambda_2 eigenvector on node i.
       Percentile-distribution of |v_2| reveals whether
       lambda_2 is driven by a few "hot-spot" nodes or
       uniformly across all nodes.

The output captures per-regime, per-percentile mean lambda_2
plus N-scaling fit per percentile.

Output: outputs/verify_lemma_B_percentile_decomposition.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REPO_ROOT = REPO.parent
OUT = REPO / "outputs" / "verify_lemma_B_percentile_decomposition.json"

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
    ("P5N600", 600, "results_d1_p5n600_12seeds/P5N600.snapshots.npz",   "edge_xi_snapshots"),
    ("P5N700", 700, "results_d1_p5n700_12seeds/P5N700.snapshots.npz",   "edge_xi_snapshots"),
    ("P5N800", 800, "results_d1_p5n800_12seeds/P5N800.snapshots.npz",   "edge_xi_snapshots"),
]

# Top-p% by row-sum = matter-core-concentrated; bottom-p% = bulk.
# We sweep both ends.
PERCENTILES_TOP = [0.50, 0.20, 0.10, 0.05, 0.01]  # top 50%, 20%, ..., 1%
PERCENTILES_BOT = [0.50, 0.20, 0.10, 0.05, 0.01]  # bottom 50%, ..., 1%


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


def lambda2_normalised(w: np.ndarray) -> float | None:
    """lambda_2 of L = I - D^{-1/2} W D^{-1/2}.
    Returns None if any row sums to zero."""
    n = w.shape[0]
    if n < 3:
        return None
    deg = w.sum(axis=1)
    if np.any(deg <= 1e-12):
        return None
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    norm = w * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    lap = np.eye(n) - norm
    lap = 0.5 * (lap + lap.T)
    eigs = np.linalg.eigvalsh(lap)
    return float(eigs[1])


def fiedler_participation(xi: np.ndarray) -> np.ndarray | None:
    """Return |v_2(i)| for each node i (Fiedler-vector amplitude).
    Returns None if degenerate."""
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
    # Second eigenvector (smallest non-zero) is the Fiedler
    return np.abs(eigvecs[:, 1])


def lambda2_sub(xi: np.ndarray, idx: np.ndarray) -> float | None:
    """Sub-Laplacian lambda_2 on the row-sum-induced subgraph."""
    if idx.size < 3:
        return None
    w_full = xi - np.eye(xi.shape[0])
    w_full = np.maximum(w_full, 0.0)
    w_sub = w_full[np.ix_(idx, idx)]
    return lambda2_normalised(w_sub)


def per_snapshot(xi: np.ndarray) -> dict | None:
    """Per-snapshot percentile-decomposition diagnostics."""
    n = xi.shape[0]
    w = xi - np.eye(n)
    w = np.maximum(w, 0.0)
    deg = w.sum(axis=1)
    if np.any(deg <= 1e-12):
        return None

    # Full lambda_2 (reference)
    lam_full = lambda2_normalised(w)
    if lam_full is None:
        return None

    # Sort nodes by row-sum descending
    order = np.argsort(-deg)

    results: dict = {"lambda2_full": lam_full,
                     "n_nodes_full": n}

    # Top-p% (matter-core) sub-Laplacian
    for p in PERCENTILES_TOP:
        k = max(3, int(np.round(p * n)))
        top_idx = order[:k]
        lam_top = lambda2_sub(xi, top_idx)
        results[f"lambda2_top_{int(p*100)}pct"] = lam_top
        results[f"n_top_{int(p*100)}pct"] = k

    # Bottom-p% (bulk) sub-Laplacian
    for p in PERCENTILES_BOT:
        k = max(3, int(np.round(p * n)))
        bot_idx = order[-k:]
        lam_bot = lambda2_sub(xi, bot_idx)
        results[f"lambda2_bot_{int(p*100)}pct"] = lam_bot
        results[f"n_bot_{int(p*100)}pct"] = k

    # Fiedler participation percentiles
    fiedler = fiedler_participation(xi)
    if fiedler is not None:
        fiedler_p = np.percentile(fiedler, [50, 90, 95, 99])
        results["fiedler_p50"] = float(fiedler_p[0])
        results["fiedler_p90"] = float(fiedler_p[1])
        results["fiedler_p95"] = float(fiedler_p[2])
        results["fiedler_p99"] = float(fiedler_p[3])
        results["fiedler_max"] = float(fiedler.max())
        # Fiedler participation ratio (concentration metric)
        f_sq = fiedler ** 2
        results["fiedler_PR"] = float((f_sq.sum() ** 2)
                                      / np.sum(f_sq ** 2))
    return results


def fit_symanzik1(n_arr, y_arr):
    valid = [(n, y) for n, y in zip(n_arr, y_arr)
             if y is not None and np.isfinite(y)]
    if len(valid) < 3:
        return None
    n_a = np.array([v[0] for v in valid], dtype=float)
    y_a = np.array([v[1] for v in valid], dtype=float)
    a_mat = np.column_stack([np.ones_like(n_a), 1.0 / n_a])
    sol, *_ = np.linalg.lstsq(a_mat, y_a, rcond=None)
    return float(sol[0]), float(sol[1])


def main():
    per_regime = []
    for reg, n_lat, rel, hint in LADDER:
        xis = load_all_xi(REPO_ROOT / rel, hint)
        if not xis:
            per_regime.append({"regime": reg, "N": n_lat,
                               "n_seeds_loaded": 0,
                               "status": "SNAPSHOT_NOT_AVAILABLE"})
            continue
        diags = [per_snapshot(xi) for xi in xis]
        diags = [d for d in diags if d is not None]
        if not diags:
            per_regime.append({"regime": reg, "N": n_lat,
                               "n_seeds_loaded": len(xis),
                               "status": "ALL_SEEDS_DEGENERATE"})
            continue
        out = {"regime": reg, "N": n_lat,
               "n_seeds_loaded": len(xis),
               "n_seeds_valid": len(diags),
               "status": "OK"}
        keys = ["lambda2_full"]
        for p in PERCENTILES_TOP:
            keys.append(f"lambda2_top_{int(p*100)}pct")
        for p in PERCENTILES_BOT:
            keys.append(f"lambda2_bot_{int(p*100)}pct")
        keys += ["fiedler_p50", "fiedler_p90", "fiedler_p95",
                 "fiedler_p99", "fiedler_max", "fiedler_PR"]
        for key in keys:
            vals = [d.get(key) for d in diags
                    if d.get(key) is not None]
            if vals:
                out[f"{key}_mean"] = float(np.mean(vals))
        per_regime.append(out)

    # Symanzik-1 N-scaling per percentile cut
    percentile_fits = {}
    keys_for_fit = ["lambda2_full"]
    for p in PERCENTILES_TOP:
        keys_for_fit.append(f"lambda2_top_{int(p*100)}pct")
    for p in PERCENTILES_BOT:
        keys_for_fit.append(f"lambda2_bot_{int(p*100)}pct")
    keys_for_fit += ["fiedler_PR"]

    for key in keys_for_fit:
        n_arr = [r["N"] for r in per_regime if r.get("status") == "OK"]
        y_arr = [r.get(f"{key}_mean") for r in per_regime
                 if r.get("status") == "OK"]
        fit = fit_symanzik1(n_arr, y_arr)
        if fit:
            percentile_fits[key] = {"c_inf": fit[0], "a": fit[1]}

    out = {
        "headline": ("Lemma B R-loop: per-node-percentile + "
                     "bulk/matter-core sub-Laplacian lambda_2 "
                     "decomposition."),
        "ladder": [reg for reg, *_ in LADDER],
        "percentiles_top": PERCENTILES_TOP,
        "percentiles_bot": PERCENTILES_BOT,
        "per_regime": per_regime,
        "symanzik_fits": percentile_fits,
        "reference_3_over_8": 3.0 / 8.0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_summary(per_regime, percentile_fits)
    return 0


def print_summary(per_regime, fits):
    print("=" * 100)
    print("Lemma B R-loop: per-node-percentile sub-Laplacian decomposition")
    print("=" * 100)
    # Show per-regime: full lambda_2 + top-50/20/10/5/1 + bot-50/20/10/5/1
    cols = (["lam2_full"]
            + [f"top{int(p*100)}" for p in PERCENTILES_TOP]
            + [f"bot{int(p*100)}" for p in PERCENTILES_BOT])
    header = f"{'Regime':<8} {'N':>4} " + " ".join(f"{c:>8}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in per_regime:
        if r.get("status") != "OK":
            continue
        line = f"{r['regime']:<8} {r['N']:>4}"
        vals = [r.get("lambda2_full_mean", None)]
        for p in PERCENTILES_TOP:
            vals.append(r.get(f"lambda2_top_{int(p*100)}pct_mean"))
        for p in PERCENTILES_BOT:
            vals.append(r.get(f"lambda2_bot_{int(p*100)}pct_mean"))
        for v in vals:
            if v is None:
                line += "   --- "
            else:
                line += f" {v:>8.4f}"
        print(line)
    print()
    print("Symanzik-1 fits (lambda_inf per percentile cut):")
    print(f"  3/8 = {3/8:.5f}")
    for key, p in fits.items():
        print(f"  {key:<26s} lambda_inf = {p['c_inf']:>8.5f}, a = {p['a']:>9.3f}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
