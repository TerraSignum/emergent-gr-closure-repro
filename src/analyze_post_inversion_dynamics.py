r"""Deep diagnostic of the post-inversion regime (N > N_inv ~ 591-600).

Tests the xi-saturation-cascade hypothesis against the existing
N=1024 iter=1000 long-trajectory snapshot data
(results_d1_p5n1024_longiter_3seeds).

Diagnostics computed per snapshot:
  (D1) xi-distribution histogram + cumulative Gini coefficient
  (D2) Fraction of xi entries above 0.95*xi_max (saturation fraction)
  (D3) Top-percentile lambda_2 (matter-core sub-Laplacian)
  (D4) Bottom-percentile lambda_2 (bulk sub-Laplacian)
  (D5) Mean degree / sigma_degree (heterogeneity collapse?)
  (D6) Entropy of normalised xi (i.e. -sum p log p, p = xi/sum)

Output: trajectory of all 6 diagnostics over iter 0..1000, three seeds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
REPO = SRC.parent
OUTPUTS = REPO / "outputs"

XI_MAX = 1.0   # default in worldformula
SATURATION_THRESHOLD = 0.95  # |xi| > 0.95 * xi_max => "saturated"


def gini_coefficient(values: np.ndarray) -> float:
    """Standard Gini coefficient for a 1D non-negative array."""
    if values.size == 0:
        return 0.0
    v = np.sort(np.abs(values).astype(np.float64))
    n = v.size
    cum = np.cumsum(v)
    if cum[-1] <= 0:
        return 0.0
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def shannon_entropy_nat(p: np.ndarray) -> float:
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


def lambda2_normalised(adj: np.ndarray) -> float | None:
    deg = np.maximum(adj.sum(axis=1), 1e-12)
    d_inv = 1.0 / np.sqrt(deg)
    L = np.eye(adj.shape[0]) - (d_inv[:, None] * adj * d_inv[None, :])
    L = 0.5 * (L + L.T)
    try:
        eigs = np.linalg.eigvalsh(L)
        return float(eigs[1])
    except np.linalg.LinAlgError:
        return None


def per_snapshot_diagnostics(xi: np.ndarray) -> dict:
    n = xi.shape[0]
    upper_mask = np.triu(np.ones_like(xi, dtype=bool), k=1)
    xi_off = xi[upper_mask]  # N*(N-1)/2 off-diagonal entries

    # (D1+D6) distribution histogram + entropy
    hist, edges = np.histogram(xi_off, bins=20, range=(0.0, XI_MAX))
    hist_p = hist / max(hist.sum(), 1)
    entropy = shannon_entropy_nat(hist_p)

    # (D2) saturation fraction
    sat_frac = float((xi_off >= SATURATION_THRESHOLD * XI_MAX).mean())

    # (D3, D4) top/bottom-percentile sub-Laplacian
    w = xi.copy()
    np.fill_diagonal(w, 0.0)
    deg = w.sum(axis=1)
    order = np.argsort(-deg)
    top10 = order[: max(3, n // 10)]
    bot50 = order[-max(3, n // 2):]
    lam_top = lambda2_normalised(w[np.ix_(top10, top10)]) or float("nan")
    lam_bot = lambda2_normalised(w[np.ix_(bot50, bot50)]) or float("nan")

    # (D5) degree statistics
    mean_deg = float(deg.mean())
    std_deg = float(deg.std(ddof=1))

    # Full carrier lambda_2 (reference)
    lam_full = lambda2_normalised(w) or float("nan")

    # Gini of xi
    gini = gini_coefficient(xi_off)

    return {
        "lambda2_full": lam_full,
        "lambda2_top10pct": lam_top,
        "lambda2_bot50pct": lam_bot,
        "xi_saturation_fraction": sat_frac,
        "xi_gini": gini,
        "xi_entropy_nat": entropy,
        "mean_degree": mean_deg,
        "std_degree": std_deg,
        "deg_cv": float(std_deg / max(mean_deg, 1e-12)),
        "xi_mean": float(xi_off.mean()),
        "xi_std": float(xi_off.std(ddof=1)),
        "xi_min": float(xi_off.min()),
        "xi_max": float(xi_off.max()),
    }


def main():
    print("=" * 78)
    print("Post-inversion deep dynamics diagnostic (N=1024 iter=0..1000)")
    print("=" * 78)
    npz = Path("results_d1_p5n1024_longiter_3seeds/P5N1024.snapshots.npz")
    npz = (REPO.parent / npz).resolve()
    if not npz.is_file():
        npz = REPO.parent / "results_d1_p5n1024_longiter_3seeds/P5N1024.snapshots.npz"
    print(f"Loading: {npz}")
    d = np.load(npz, allow_pickle=True)
    xi_snaps = d["edge_xi_snapshots"]  # (n_seeds, n_snapshots, N, N)
    steps = d["snapshot_steps"]
    n_seeds, n_snaps = xi_snaps.shape[:2]

    per_step = []
    for snap_idx in range(n_snaps):
        diag_per_seed = []
        for s in range(n_seeds):
            xi = xi_snaps[s, snap_idx].astype(np.float64)
            diag_per_seed.append(per_snapshot_diagnostics(xi))
        # Average across seeds
        mean_diag = {}
        for k in diag_per_seed[0]:
            vals = [d[k] for d in diag_per_seed if np.isfinite(d[k])]
            mean_diag[k] = float(np.mean(vals)) if vals else float("nan")
        per_step.append({"step": int(steps[snap_idx]), **mean_diag})

    print()
    print(f"  {'step':>5} {'lam_full':>10} {'lam_top10':>10} {'sat_frac':>9} "
          f"{'gini':>7} {'mean_deg':>9} {'deg_cv':>7}")
    for row in per_step:
        print(f"  {row['step']:>5} {row['lambda2_full']:>10.5f} "
              f"{row['lambda2_top10pct']:>10.5f} "
              f"{row['xi_saturation_fraction']:>9.4f} "
              f"{row['xi_gini']:>7.4f} {row['mean_degree']:>9.2f} "
              f"{row['deg_cv']:>7.4f}")

    out = {
        "title": "Post-inversion deep dynamics diagnostic",
        "source_npz": str(npz),
        "n_seeds": n_seeds,
        "n_snapshots": n_snaps,
        "per_step": per_step,
    }
    out_path = OUTPUTS / "analyze_post_inversion_dynamics.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # Summary headlines
    print()
    print("-" * 78)
    print("Headlines:")
    print("-" * 78)
    initial = per_step[0]
    final = per_step[-1]
    print(f"  saturation_fraction: {initial['xi_saturation_fraction']:.4f} "
          f"-> {final['xi_saturation_fraction']:.4f} "
          f"(delta {final['xi_saturation_fraction'] - initial['xi_saturation_fraction']:+.4f})")
    print(f"  xi_gini:             {initial['xi_gini']:.4f} -> {final['xi_gini']:.4f} "
          f"(delta {final['xi_gini'] - initial['xi_gini']:+.4f})")
    print(f"  xi_entropy_nat:      {initial['xi_entropy_nat']:.4f} -> {final['xi_entropy_nat']:.4f} "
          f"(delta {final['xi_entropy_nat'] - initial['xi_entropy_nat']:+.4f})")
    print(f"  degree_CV:           {initial['deg_cv']:.4f} -> {final['deg_cv']:.4f} "
          f"(delta {final['deg_cv'] - initial['deg_cv']:+.4f})")
    print(f"  lambda2_full:        {initial['lambda2_full']:.5f} -> {final['lambda2_full']:.5f}")
    print(f"  lambda2_top10pct:    {initial['lambda2_top10pct']:.5f} -> {final['lambda2_top10pct']:.5f}")
    print(f"  mean_degree:         {initial['mean_degree']:.2f} -> {final['mean_degree']:.2f} "
          f"(N=1024, max possible {1023})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
