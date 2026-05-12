"""Lemma B regime-sensitivity audit: lambda_skel(tau) sweep across
multiple skeleton thresholds.

User question (Q-loop): the Step-3b lambda_skel^inf = 0.2924 audit
was run only at tau = 0.10. Is 7/24 specific to this threshold, or
does it appear universally across thresholds? If the former, the
conjecture is a matter-core-skeleton property; if the latter, it
is a structural carrier property.

This audit sweeps lambda_2(L_skel(tau)) across
tau in {0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50} and
reports per-regime cross-seed mean + Symanzik-1 asymptote per
threshold. Compares each tau's asymptote against:
  - 7/24 = 0.29167 (Step 3b conjecture)
  - Alon-Boppana(d_eff(tau)) for the corresponding d_eff
  - Saturation ratio lambda_inf / lambda_AB

Output: outputs/verify_lemma_B_threshold_sweep.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REPO_ROOT = REPO.parent
OUT = REPO / "outputs" / "verify_lemma_B_threshold_sweep.json"

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

THRESHOLDS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]


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


def lambda2_skel(xi: np.ndarray, tau: float) -> tuple[float, float] | None:
    """Return (lambda_2(L_skel), mean_deg_skel) for the tau-skeleton,
    or None if disconnected/degenerate."""
    n = xi.shape[0]
    a = ((xi - np.eye(n)) > tau).astype(float)
    np.fill_diagonal(a, 0.0)
    a = 0.5 * (a + a.T)
    deg = a.sum(axis=1)
    if np.any(deg <= 1e-12):
        return None
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    norm = a * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    lap = np.eye(n) - norm
    lap = 0.5 * (lap + lap.T)
    eigs = np.linalg.eigvalsh(lap)
    return float(eigs[1]), float(deg.mean())


def alon_boppana(d: float) -> float:
    if d <= 1:
        return 0.0
    return float(1.0 - 2.0 * np.sqrt(d - 1.0) / d)


def fit_symanzik1(n_arr, y_arr) -> tuple[float, float] | None:
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
    per_threshold: dict[str, dict] = {}
    for tau in THRESHOLDS:
        per_regime = []
        for reg, n_lat, rel, hint in LADDER:
            npz = REPO_ROOT / rel
            xis = load_all_xi(npz, hint)
            if not xis:
                per_regime.append({"regime": reg, "N": n_lat,
                                   "lambda2_mean": None,
                                   "d_skel_mean": None,
                                   "n_seeds_valid": 0})
                continue
            results = [lambda2_skel(xi, tau) for xi in xis]
            valid = [r for r in results if r is not None]
            if not valid:
                per_regime.append({"regime": reg, "N": n_lat,
                                   "lambda2_mean": None,
                                   "d_skel_mean": None,
                                   "n_seeds_valid": 0,
                                   "n_seeds_total": len(xis)})
                continue
            lams = [r[0] for r in valid]
            degs = [r[1] for r in valid]
            per_regime.append({"regime": reg, "N": n_lat,
                               "lambda2_mean": float(np.mean(lams)),
                               "lambda2_std": float(np.std(lams, ddof=1)) if len(lams) > 1 else 0.0,
                               "d_skel_mean": float(np.mean(degs)),
                               "n_seeds_valid": len(valid),
                               "n_seeds_total": len(xis)})
        n_arr = [r["N"] for r in per_regime]
        l_arr = [r["lambda2_mean"] for r in per_regime]
        d_arr = [r["d_skel_mean"] for r in per_regime]
        fit_lambda = fit_symanzik1(n_arr, l_arr)
        fit_deg = fit_symanzik1(n_arr, d_arr)
        lam_inf = fit_lambda[0] if fit_lambda else None
        deg_inf = fit_deg[0] if fit_deg else None
        ab = alon_boppana(deg_inf) if deg_inf and deg_inf > 1 else None
        saturation = (lam_inf / ab) if (lam_inf and ab and ab > 0) else None
        per_threshold[f"tau_{tau}"] = {
            "tau": tau,
            "per_regime": per_regime,
            "lambda_inf_symanzik": lam_inf,
            "d_skel_inf_symanzik": deg_inf,
            "alon_boppana_at_d_inf": ab,
            "saturation_ratio": saturation,
            "match_to_7_24_pct":
                (abs(lam_inf - 7.0/24.0) / (7.0/24.0) * 100.0
                 if lam_inf else None),
        }

    out = {
        "headline": ("Threshold-sweep audit: is lambda_skel^inf = "
                     "7/24 specific to tau=0.10 or universal?"),
        "thresholds": THRESHOLDS,
        "per_threshold": per_threshold,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Console summary
    print("=" * 90)
    print("Lemma B threshold-sweep audit: is lambda_skel = 7/24 tau-specific?")
    print("=" * 90)
    print(f"{'tau':>6} | {'lambda_inf':>11} {'d_skel_inf':>11} "
          f"{'lam_AB(d_inf)':>14} {'sat ratio':>10} {'Delta to 7/24':>14}")
    print("-" * 90)
    for tau in THRESHOLDS:
        d = per_threshold[f"tau_{tau}"]
        li = d["lambda_inf_symanzik"]
        di = d["d_skel_inf_symanzik"]
        ab = d["alon_boppana_at_d_inf"]
        sat = d["saturation_ratio"]
        m = d["match_to_7_24_pct"]
        print(f"{tau:>6.3f} | "
              f"{li:>11.5f} " if li is not None else f"{tau:>6.3f} |    N/A     ",
              end="")
        if li is not None:
            print(f"{di:>11.3f} {ab:>14.5f} {sat:>10.4f} {m:>13.2f}%")
    print()
    print(f"Reference: 7/24 = {7/24:.5f}")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
