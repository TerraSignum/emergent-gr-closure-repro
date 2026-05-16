r"""Check carrier equilibration: does lambda_2 stabilise within 120 iterations?

Reads the per-snapshot xi for each N and computes lambda_2 at iter 0, 10,
20, ..., 120. If still drifting at iter 120, the system isn't equilibrated
and the "asymptote" we extract is contaminated by transient dynamics.

Particularly important for N=2048 where the trust-region alarm fired
unexpectedly high.
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


def lambda_2_from_xi(xi: np.ndarray) -> float:
    np.fill_diagonal(xi, 0.0)
    deg = np.maximum(xi.sum(axis=1), 1e-12)
    d_inv = 1.0 / np.sqrt(deg)
    L = np.eye(xi.shape[0]) - (d_inv[:, None] * xi * d_inv[None, :])
    L = 0.5 * (L + L.T)
    eigs = np.linalg.eigvalsh(L)
    return float(eigs[1])


def main():
    ladder = discover_d1_ladder(REPO)
    print("Carrier equilibration audit (per-snapshot lambda_2 evolution)")
    print("=" * 78)

    report = {"per_regime": {}}
    for regime, n_lat, npz_path in ladder:
        if n_lat < 100:
            continue
        try:
            d = np.load(npz_path, allow_pickle=True)
        except OSError:
            print(f"  [skip] {regime}: NPZ unreadable")
            continue
        if "edge_xi_snapshots" not in d.files:
            continue
        xi_snaps = d["edge_xi_snapshots"]  # shape (n_seeds, n_snapshots, N, N)
        steps = d["snapshot_steps"]
        n_seeds, n_snaps = xi_snaps.shape[:2]
        max_seeds = min(6, n_seeds)
        # Mean lambda_2 across seeds, per snapshot
        per_step_means = []
        for snap_idx in range(n_snaps):
            lams = []
            for s in range(max_seeds):
                xi = xi_snaps[s, snap_idx].astype(np.float64)
                try:
                    lams.append(lambda_2_from_xi(xi))
                except np.linalg.LinAlgError:
                    continue
            per_step_means.append((int(steps[snap_idx]), float(np.mean(lams))))
        print(f"\n  {regime} (N={n_lat}, {max_seeds} seeds):")
        for step, lam in per_step_means:
            print(f"    iter {step:>4}: lambda_2 = {lam:.5f}")
        # Drift between final 3 snapshots
        if len(per_step_means) >= 3:
            late = [lam for _step, lam in per_step_means[-3:]]
            drift = max(late) - min(late)
            print(f"    final-3 drift: {drift:.5f}")
            converged = drift < 0.005
            print(f"    converged: {converged}")
        report["per_regime"][regime] = {
            "N": n_lat,
            "n_seeds_used": max_seeds,
            "per_step_means": per_step_means,
        }

    out_path = OUTPUTS / "verify_equilibration_per_n.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
