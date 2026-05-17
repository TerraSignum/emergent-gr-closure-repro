r"""Extend percentile-decomposition (spatial top-percentile matter-CORE)
to high-N (N=512, 1024, 2048) and re-evaluate against 79/200.

The "matter core" in the original corpus framework was defined via
the top-X% nodes by row-sum (degree) in xi, and the matter-core
lambda_2 was the sub-Laplacian gap on that subgraph. This is DIFFERENT
from the per-seed branch classification: matter-core is a SPATIAL
top-percentile per-seed (within one xi), not a per-seed branch
classification.

The original audit (outputs/verify_lemma_B_percentile_decomposition.json)
only covered N <= 300. This script extends to N <= 2048 using the
auto-discovered ladder.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
REPO = SRC.parent
OUTPUTS = REPO / "outputs"

sys.path.insert(0, str(SRC))
from _d1_ladder_discovery import discover_d1_ladder  # noqa: E402

PERCENTILES_TOP = [0.05, 0.10, 0.20, 0.50]


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


def per_seed_top_percentile(xi: np.ndarray, p: float) -> float | None:
    n = xi.shape[0]
    w = xi - np.eye(n)
    w = np.maximum(w, 0.0)
    deg = w.sum(axis=1)
    if np.any(deg <= 1e-12):
        return None
    order = np.argsort(-deg)
    k = max(3, int(np.round(p * n)))
    top_idx = order[:k]
    if top_idx.size < 3:
        return None
    w_sub = w[np.ix_(top_idx, top_idx)]
    if w_sub.sum() == 0:
        return None
    return lambda2_normalised(w_sub)


def main():
    print("=" * 78)
    print("Matter-core top-percentile sub-Laplacian audit (extended to high N)")
    print("=" * 78)
    ladder = discover_d1_ladder(REPO)
    report = {"per_regime": []}
    for regime, n_lat, npz_path in ladder:
        if n_lat < 100:
            continue
        try:
            d = np.load(npz_path, allow_pickle=True)
        except OSError:
            continue
        if "edge_xi_snapshots" not in d.files:
            continue
        xi_arr = d["edge_xi_snapshots"][:, -1]
        per_seed = {p: [] for p in PERCENTILES_TOP}
        per_seed["full"] = []
        for s in range(xi_arr.shape[0]):
            xi = xi_arr[s].astype(np.float64)
            w = xi - np.eye(xi.shape[0])
            w = np.maximum(w, 0.0)
            full_lam = lambda2_normalised(w)
            if full_lam is not None:
                per_seed["full"].append(full_lam)
            for p in PERCENTILES_TOP:
                lam = per_seed_top_percentile(xi, p)
                if lam is not None:
                    per_seed[p].append(lam)
        # Summarise
        summary = {"regime": regime, "N": n_lat,
                    "n_seeds": xi_arr.shape[0]}
        summary["full_lambda2_mean"] = float(np.mean(per_seed["full"])) \
            if per_seed["full"] else float("nan")
        for p in PERCENTILES_TOP:
            v = per_seed[p]
            if v:
                summary[f"top_{int(p*100)}pct_mean"] = float(np.mean(v))
                summary[f"top_{int(p*100)}pct_std"] = float(np.std(v, ddof=1)) \
                    if len(v) > 1 else 0.0
                summary[f"top_{int(p*100)}pct_n_valid"] = len(v)
            else:
                summary[f"top_{int(p*100)}pct_mean"] = float("nan")
                summary[f"top_{int(p*100)}pct_n_valid"] = 0
        report["per_regime"].append(summary)

    # Print summary table
    print(f"\n{'regime':<10} {'N':>5} {'full':>9} {'top5':>9} {'top10':>9} "
          f"{'top20':>9} {'top50':>9}")
    for r in report["per_regime"]:
        f_val = r.get('full_lambda2_mean', float('nan'))
        t5 = r.get('top_5pct_mean', float('nan'))
        t10 = r.get('top_10pct_mean', float('nan'))
        t20 = r.get('top_20pct_mean', float('nan'))
        t50 = r.get('top_50pct_mean', float('nan'))
        print(f"  {r['regime']:<10} {r['N']:>5} {f_val:>9.5f} "
              f"{t5:>9.5f} {t10:>9.5f} {t20:>9.5f} {t50:>9.5f}")

    out_path = OUTPUTS / "verify_matter_core_top_percentile_high_n.json"
    out_path.write_text(json.dumps(report, indent=2, default=str),
                          encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
