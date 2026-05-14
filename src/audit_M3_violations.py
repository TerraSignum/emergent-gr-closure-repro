"""Reproducer: M3 (sub-multiplicative triangle) axiom violation rate.

Computes the fraction of triples (i, j, k) for which
    Xi_ij * Xi_jk > Xi_ik + tol
on the canonical regime relational similarity matrix Xi. M3 is
the multiplicative triangle inequality used in Theorem 4.2 of the
manuscript; the empirical violation rate quantifies the residual
gap that the fast-slow gradient flow drives down.

Reads the canonical-regime similarity matrix from
data/d1_runs/d1_p5.npz (Tier-2 location). Falls back to the
parent program path c:/Users/user/Desktop/Emergence/results_d1_fix17/d1_p5.npz
when the bundled NPZ is absent.

Reports:
- pre-stabilisation violation rate (raw direct similarity, before
  composition closure) per seed mean and aggregate
- post-stabilisation violation rate (closed_xi_seed = direct closed
  under composition envelope) per seed mean and aggregate
- documented manuscript values for cross-check

Usage:
    python ./src/audit_M3_violations.py [--regime P5] [--n-seeds 4] [--tol 1e-12]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def find_d1_npz(regime: str) -> Path | None:
    """Locate per-regime D1 lattice payload.

    Looks first in the bundled location data/d1_runs/ (Tier-2
    reproduction). Other parent-directory locations are fallback
    discovery candidates only and not part of the public package.
    """
    candidates = [REPO / "data" / "d1_runs" / f"d1_{regime.lower()}.npz"]
    parent_d1_dirs = list((REPO.parent).glob("results_d1_*"))
    for d in parent_d1_dirs:
        candidates.append(d / f"d1_{regime.lower()}.npz")
        candidates.append(d / regime.lower() / f"d1_{regime.lower()}.npz")
    for c in candidates:
        if c.exists():
            return c
    return None


def m3_violation_rate(xi_mat: np.ndarray, tol: float = 1e-12) -> dict:
    """Compute three definitions of M3 (sub-multiplicative triangle)
    violation on a relational similarity matrix Xi:

    (a) raw_count: fraction of off-diagonal triples (i, j, k) with
        Xi_ij * Xi_jk > Xi_ik + tol;
    (b) significant_count: same, with the absolute slack
        (Xi_ij * Xi_jk - Xi_ik) > 0.05 (i.e. only material violations);
    (c) penalty_residual: relative L2 norm
        sqrt( sum_triples max(0, Xi_ij*Xi_jk - Xi_ik)^2 / n^3 )
        which is the gradient-flow penalty G_Delta in normalised form.

    Definition (c) is the quantity bounded by epsilon^(can) = 0.07
    in the manuscript; definitions (a, b) are reader-facing diagnostic
    counts.
    """
    n = int(xi_mat.shape[0])
    prod = xi_mat[:, :, None] * xi_mat[None, :, :]
    target = xi_mat[:, None, :]
    slack = prod - target
    diag_mask = np.ones((n, n, n), dtype=bool)
    diag_mask[np.arange(n), np.arange(n), :] = False
    diag_mask[:, np.arange(n), np.arange(n)] = False
    diag_mask[np.arange(n), :, np.arange(n)] = False
    n_valid = float(diag_mask.sum())
    raw = ((slack > tol) & diag_mask).astype(np.float64)
    sig = ((slack > 0.05) & diag_mask).astype(np.float64)
    pen_term = (np.maximum(slack, 0.0) ** 2) * diag_mask
    return {
        "raw_count_rate":         float(raw.sum() / n_valid) if n_valid > 0 else float("nan"),
        "significant_count_rate": float(sig.sum() / n_valid) if n_valid > 0 else float("nan"),
        "penalty_residual_L2":    float(np.sqrt(pen_term.sum() / n_valid)),
        "penalty_residual_max":   float(np.maximum(slack * diag_mask, 0.0).max()),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--regime", default="P5")
    p.add_argument("--n-seeds", type=int, default=4)
    p.add_argument("--tol", type=float, default=1e-12)
    args = p.parse_args(argv)

    npz = find_d1_npz(args.regime)
    if npz is None or not npz.exists():
        print(json.dumps({
            "regime": args.regime,
            "status": "tier2_fallback",
            "reason": "d1_p<regime>.npz not bundled; report documented manuscript values",
            "documented_pre_stabilisation_rate": "~0.04 (4 percent)",
            "documented_post_stabilisation_rate": "<= 0.005 (0.5 percent)",
            "manuscript_reference": "manuscript.tex line 358 (paragraph 'Empirical M3-violation rate')",
            "to_reproduce": "place d1_p5.npz in data/d1_runs/ and rerun",
        }, indent=2))
        return 0

    d = np.load(npz, allow_pickle=True)
    if "direct_xi_seed" not in d.files or "closed_xi_seed" not in d.files:
        print(f"npz {npz.name} missing expected keys; available: {list(d.files)[:8]}")
        return 1
    direct = d["direct_xi_seed"]
    closed = d["closed_xi_seed"]
    n_seeds = min(args.n_seeds, int(direct.shape[0]))

    pre_results = []
    post_results = []
    for s in range(n_seeds):
        xi_pre = direct[s].copy()
        np.fill_diagonal(xi_pre, 1.0)
        pre_results.append(m3_violation_rate(xi_pre, args.tol))
        xi_post = closed[s].copy()
        np.fill_diagonal(xi_post, 1.0)
        post_results.append(m3_violation_rate(xi_post, args.tol))

    def _agg(results, key):
        arr = np.array([r[key] for r in results], dtype=float)
        return float(arr.mean()), float(arr.std())

    print("=" * 76)
    print(f"M3-violation audit on regime {args.regime} (N={direct.shape[1]})")
    print("=" * 76)
    print(f"n_seeds processed: {n_seeds}")
    print()
    print(f"{'definition':<32} {'pre mean':>11} {'post mean':>11}")
    for key in ["raw_count_rate", "significant_count_rate",
                "penalty_residual_L2", "penalty_residual_max"]:
        pm, ps = _agg(pre_results, key)
        qm, qs = _agg(post_results, key)
        print(f"{key:<32} {pm:>11.4f} {qm:>11.4f}")
    print()
    print("Manuscript-documented bands (definition: significant-count "
          "and/or penalty-residual L2):")
    print("  pre-stabilisation:  ~0.04 (4 percent significant violations)")
    print("  post-stabilisation: <= 0.005 (0.5 percent significant)")
    print("  penalty residual L2 bound: <= 0.07 (epsilon^(can))")

    out = {
        "regime": args.regime, "N": int(direct.shape[1]),
        "n_seeds": n_seeds,
        "definitions": {
            "raw_count_rate": "fraction of off-diagonal triples with Xi_ij * Xi_jk > Xi_ik + tol",
            "significant_count_rate": "same, with absolute slack > 0.05",
            "penalty_residual_L2": "sqrt(sum max(0, slack)^2 / n^3) -- gradient-flow penalty",
            "penalty_residual_max": "max slack value across all valid triples",
        },
        "pre_stabilisation":  {key: _agg(pre_results, key) for key in pre_results[0]},
        "post_stabilisation": {key: _agg(post_results, key) for key in post_results[0]},
        "documented_band_significant_pre":  [0.03, 0.05],
        "documented_band_significant_post": [0.0,  0.005],
        "documented_penalty_bound_post":    0.07,
    }
    out_path = REPO / "outputs" / f"audit_M3_violations_{args.regime}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
