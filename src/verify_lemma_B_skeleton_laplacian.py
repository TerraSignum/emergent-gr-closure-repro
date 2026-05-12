"""Lemma B Phase-2 Step 3b: direct skeleton-Laplacian audit.

Step 3a (`verify_lemma_B_edge_weight_structure.py`) identified
tau = 0.10 as the unique threshold for which the threshold-
dependent effective degree d_eff(tau=0.10) -> 12 is consistent
with the empirical Laplacian gap lambda_inf = 0.3789 via
Alon-Boppana (lambda_AB(12) = 0.4499, ratio 0.84, near-
Ramanujan). The interpretation: an UNWEIGHTED skeleton

  A_skel(i, j) = 1[Xi_N(i, j) > 0.10]

is the analytical handle for the uniform spectral gap. This
audit tests whether the skeleton's own normalised graph
Laplacian has a spectral gap that matches the weighted
Laplacian gap of prop:sg_empirical:

  lambda_2(L_skel) ≟ lambda_2(L_weighted) ≈ 0.3789  ?

If yes, the analytical question reduces to "is the skeleton a
near-Ramanujan expander on the carrier-action construction?"
and the weighted-vs-unweighted Laplacian discrepancy is
controlled by Kahale's irregular-expander bound. The
analytical Phase-2 Step 4 then becomes computable in closed
form from the carrier-action structure.

Per snapshot, computes:
  - skeleton mean degree, std, CV
  - lambda_2(L_skel) of the normalised skeleton Laplacian
  - lambda_2(L_weighted) for comparison (matches earlier audit)
  - ratio R = lambda_2(L_skel) / lambda_2(L_weighted)

Per regime: cross-seed mean + bootstrap-95% CI. Symanzik-1
N-scaling fit on R(N).

Output: outputs/verify_lemma_B_skeleton_laplacian.json
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REPO_ROOT = REPO.parent
OUT = REPO / "outputs" / "verify_lemma_B_skeleton_laplacian.json"

LADDER = [
    ("P5",     50,  "results_d1_fix17/d1_p5.npz",            "xi_seedK"),
    ("P5N64",  64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz", "edge_xi_snapshots"),
    ("P5N72",  72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz", "edge_xi_snapshots"),
    ("P5N84",  84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz", "edge_xi_snapshots"),
    ("P5N100", 100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz", "edge_xi_snapshots"),
    ("P5N128", 128, "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz", "edge_xi_snapshots"),
    ("P5N200", 200, "results_d1_p5n200_8seeds/P5N200.snapshots.npz", "edge_xi_snapshots"),
    ("P5N256", 256, "results_d1_p5n256_12seeds/P5N256.snapshots.npz", "edge_xi_snapshots"),
    ("P5N300", 300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz", "edge_xi_snapshots"),
    ("P5N512", 512, "results_d1_p5n512_12seeds/P5N512.snapshots.npz", "edge_xi_snapshots"),
    ("P5N600", 600, "results_d1_p5n600_12seeds/P5N600.snapshots.npz",   "edge_xi_snapshots"),
    ("P5N700", 700, "results_d1_p5n700_12seeds/P5N700.snapshots.npz",   "edge_xi_snapshots"),
    ("P5N800", 800, "results_d1_p5n800_12seeds/P5N800.snapshots.npz",   "edge_xi_snapshots"),
]

TAU_SKEL = 0.10
BOOT = 1000
RNG_SEED = 42


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


def normalised_laplacian_lambda2(w_sym: np.ndarray) -> float | None:
    """lambda_2 of L = I - D^{-1/2} W D^{-1/2} for symmetric W with
    zero diagonal. Returns None if any row sum vanishes."""
    n = w_sym.shape[0]
    deg = w_sym.sum(axis=1)
    if np.any(deg <= 1e-12):
        return None
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    norm = w_sym * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    laplacian = np.eye(n) - norm
    laplacian = 0.5 * (laplacian + laplacian.T)
    eigs = np.linalg.eigvalsh(laplacian)
    return float(eigs[1])


def per_snapshot(xi: np.ndarray, tau: float) -> dict[str, Any] | None:
    n = xi.shape[0]
    # Weighted W (off-diagonal Xi, clamp neg to zero)
    w_weighted = np.maximum(xi - np.eye(n), 0.0)
    # Skeleton A (0/1)
    a_skel = (w_weighted > tau).astype(float)
    np.fill_diagonal(a_skel, 0.0)
    # Symmetrise
    a_skel = 0.5 * (a_skel + a_skel.T)

    deg_w = w_weighted.sum(axis=1)
    deg_s = a_skel.sum(axis=1)
    if np.any(deg_w <= 1e-12) or np.any(deg_s <= 1e-12):
        return None

    lam2_w = normalised_laplacian_lambda2(w_weighted)
    lam2_s = normalised_laplacian_lambda2(a_skel)
    if lam2_w is None or lam2_s is None:
        return None

    return {
        "lambda2_weighted": lam2_w,
        "lambda2_skeleton": lam2_s,
        "ratio_skel_over_weighted": lam2_s / lam2_w if lam2_w > 0 else None,
        "skel_mean_deg": float(deg_s.mean()),
        "skel_std_deg": float(deg_s.std(ddof=1)) if deg_s.size > 1 else 0.0,
        "skel_CV_deg": float(deg_s.std(ddof=1) / max(deg_s.mean(), 1e-12)),
        "skel_min_deg": float(deg_s.min()),
        "skel_max_deg": float(deg_s.max()),
    }


def bootstrap_ci(values, n_boot, rng):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def audit_regime(regime, n_lat, rel, hint, rng):
    npz = REPO_ROOT / rel
    xis = load_all_xi(npz, hint)
    if not xis:
        return {"regime": regime, "N": n_lat, "n_seeds_loaded": 0,
                "status": "SNAPSHOT_NOT_AVAILABLE"}
    diags = [per_snapshot(xi, TAU_SKEL) for xi in xis]
    diags = [d for d in diags if d is not None]
    if not diags:
        return {"regime": regime, "N": n_lat,
                "n_seeds_loaded": len(xis),
                "status": "ALL_SEEDS_DEGENERATE"}
    out = {"regime": regime, "N": n_lat,
           "n_seeds_loaded": len(xis),
           "n_seeds_valid": len(diags),
           "status": "OK"}
    keys = ["lambda2_weighted", "lambda2_skeleton",
            "ratio_skel_over_weighted",
            "skel_mean_deg", "skel_CV_deg",
            "skel_min_deg", "skel_max_deg"]
    for key in keys:
        vals = [d[key] for d in diags if d[key] is not None]
        if not vals:
            out[f"{key}_mean"] = None
            continue
        out[f"{key}_per_seed"] = vals
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out[f"{key}_min"] = float(np.min(vals))
        out[f"{key}_max"] = float(np.max(vals))
        ci = bootstrap_ci(vals, BOOT, rng)
        out[f"{key}_ci95"] = [ci[0], ci[1]]
    return out


def _aicc(sse, n, k):
    return n * np.log(sse / n + 1e-30) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)


def fit_symanzik(per_regime, key):
    valid = [(r["N"], r[f"{key}_mean"]) for r in per_regime
             if r.get(f"{key}_mean") is not None
             and np.isfinite(r[f"{key}_mean"])]
    if len(valid) < 3:
        return None
    n_arr = np.array([v[0] for v in valid], dtype=float)
    y = np.array([v[1] for v in valid], dtype=float)
    n_pts = len(valid)
    # const
    c = float(y.mean())
    sse_c = float(((y - c) ** 2).sum())
    # Symanzik-1
    A = np.column_stack([np.ones_like(n_arr), 1.0 / n_arr])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    c_inf, a_sym = float(sol[0]), float(sol[1])
    sse_s = float(((y - (c_inf + a_sym / n_arr)) ** 2).sum())
    aicc_c = _aicc(sse_c, n_pts, 1)
    aicc_s = _aicc(sse_s, n_pts, 2)
    best = "const" if aicc_c < aicc_s else "symanzik_1"
    asym = c if best == "const" else c_inf
    return {
        "n_points": n_pts,
        "const": {"c": c, "AICc": aicc_c},
        "symanzik_1": {"c_inf": c_inf, "a": a_sym, "AICc": aicc_s},
        "preferred": best,
        "asymptote": asym,
    }


def main():
    rng = np.random.default_rng(RNG_SEED)
    per_regime = [audit_regime(*row, rng) for row in LADDER]

    fits = {}
    for key in ("lambda2_weighted", "lambda2_skeleton",
                "ratio_skel_over_weighted",
                "skel_mean_deg", "skel_CV_deg"):
        fits[key] = fit_symanzik(per_regime, key)

    # Headline: do the two Laplacian gaps converge to the same value?
    lw_inf = fits["lambda2_weighted"]["asymptote"] if fits["lambda2_weighted"] else None
    ls_inf = fits["lambda2_skeleton"]["asymptote"] if fits["lambda2_skeleton"] else None
    ratio_inf = fits["ratio_skel_over_weighted"]["asymptote"] if fits["ratio_skel_over_weighted"] else None

    out = {
        "headline": ("Lemma B Phase-2 Step 3b: skeleton-Laplacian "
                     "spectral-gap audit. Tests whether the "
                     "unweighted skeleton A_skel = 1[Xi > 0.10] "
                     "has the same uniform spectral gap as the "
                     "weighted Laplacian of prop:sg_empirical."),
        "method": (
            "Per snapshot: skeleton A_skel = 1[Xi(i,j) > 0.10], "
            "symmetrised, zero-diagonal. Normalised graph Laplacians "
            "L_w and L_skel computed on weighted Xi and on A_skel "
            "respectively. lambda_2 = second-smallest eigenvalue. "
            "Per regime: cross-seed mean + bootstrap-95% CI. "
            "Symanzik-1 + const N-scaling fits with AICc selection."),
        "tau_skeleton": TAU_SKEL,
        "boot_resamples": BOOT,
        "rng_seed": RNG_SEED,
        "per_regime": per_regime,
        "scaling_fits": fits,
        "asymptotes": {
            "lambda_weighted_inf": lw_inf,
            "lambda_skeleton_inf": ls_inf,
            "ratio_skel_over_weighted_inf": ratio_inf,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_summary(per_regime, fits)
    return 0


def print_summary(per_regime, fits):
    print("=" * 80)
    print("Lemma B Phase-2 Step 3b: Skeleton-Laplacian audit (tau = 0.10)")
    print("=" * 80)
    print(f"{'Regime':<8} {'N':>4} {'seeds':>6} "
          f"{'lam2_w':>9} {'lam2_skel':>10} {'ratio':>8} "
          f"{'skel_dM':>9} {'CV(d_skel)':>11}")
    print("-" * 80)
    for r in per_regime:
        if r["status"] != "OK":
            print(f"{r['regime']:<8} {r['N']:>4}  {r['status']}")
            continue
        print(f"{r['regime']:<8} {r['N']:>4} "
              f"{r['n_seeds_valid']:>6} "
              f"{r['lambda2_weighted_mean']:>9.4f} "
              f"{r['lambda2_skeleton_mean']:>10.4f} "
              f"{r['ratio_skel_over_weighted_mean']:>8.3f} "
              f"{r['skel_mean_deg_mean']:>9.2f} "
              f"{r['skel_CV_deg_mean']:>11.4f}")
    print()
    print("Symanzik-1 / const fits (preferred per AICc):")
    for k, f in fits.items():
        if f is None:
            continue
        best = f["preferred"]
        if best == "const":
            print(f"  {k:<32s} const c = {f['const']['c']:.4f}")
        else:
            print(f"  {k:<32s} symanzik_1: c_inf = {f['symanzik_1']['c_inf']:.4f}, "
                  f"a = {f['symanzik_1']['a']:.3f}")
    print()
    if fits["lambda2_skeleton"] and fits["lambda2_weighted"]:
        ls = fits["lambda2_skeleton"]["asymptote"]
        lw = fits["lambda2_weighted"]["asymptote"]
        print("Asymptote comparison:")
        print(f"  lambda2_weighted^inf = {lw:.4f} (Phase-1 reproduction)")
        print(f"  lambda2_skeleton^inf = {ls:.4f}  (skeleton at tau=0.10)")
        print(f"  ratio = lambda_skel / lambda_w = {ls/lw:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
