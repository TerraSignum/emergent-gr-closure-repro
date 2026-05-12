"""Test if adding the ADM extrinsic-curvature term reduces the
Delta_time floor that the static G_00 = R_bar/2 leaves behind.

ADM Hamiltonian constraint:
  16 pi G rho = (3)R + K^2 - K_{ij} K^{ij}

where K_{ij} = (1/2) partial_t g_{ij} is the extrinsic curvature.
The current Galerkin pipeline uses G_00 = R_bar/2 (static); a true
3+1D treatment adds the K-trace-square minus K-square term.

Snapshots provide partial_t Ξ via finite-differences; from these we
reconstruct partial_t g_{ij} per node and per snapshot-pair, then
compute K_{ij} K^{ij} - K^2 as a per-node correction to G_00.

Test:
  G_00^(ADM)(a) := R_bar(a)/2 + (1/2) * (K_{ij} K^{ij} - K^2)(a)

Compare residual:
  Delta_time^old(a)  = G_00^(static) + Lambda_t - T_00
  Delta_time^new(a)  = G_00^(ADM)    + Lambda_t - T_00

If |Delta_time^new|_med < |Delta_time^old|_med significantly, the
floor was due to missing extrinsic-curvature term.

Output: outputs/3plus1d_extrinsic_curvature_audit.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


class _BlockCupy:
    def find_module(self, name, path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self

    def load_module(self, name):
        raise ImportError("cupy disabled")

sys.meta_path.insert(0, _BlockCupy())

from verify_galerkin_runner_A_hessian_ricci import (
    edge_to_matrix, per_seed_galerkin)


SNAP_FILES = [
    REPO.parent / "results_d1_p5n100_snap_dense" / "P5N100.snapshots.npz",
    REPO.parent / "results_d1_p5n100_snapshot_16seeds" / "P5N100.snapshots.npz",
    REPO.parent / "results_d1_p5n100_snapshot" / "P5N100.snapshots.npz",
]
LAMBDA_T = 0.81


def main() -> int:
    snap_path = next((p for p in SNAP_FILES if p.exists()), None)
    if snap_path is None:
        print("No snapshot file found, abort")
        return 1
    print(f"Using snapshot file: {snap_path}")

    d = np.load(snap_path, allow_pickle=True)
    edge = d["edge_xi_snapshots"]      # (n_seeds, n_snaps, N, N)
    psi_r = d["psi_real_snapshots"]
    psi_i = d["psi_imag_snapshots"]
    snap_steps = d["snapshot_steps"]
    snap_every_arr = np.asarray(d["snapshot_every"]).flatten()
    snap_every = int(snap_every_arr[0]) if snap_every_arr.size else 1
    n_seeds, n_snaps, N, _ = edge.shape
    print(f"  n_seeds={n_seeds}, n_snaps={n_snaps}, N={N}, snap_every={snap_every}")

    # Compute per-snapshot g_ij and time-derivative K_ij = (1/2) dt g_ij
    # Per pair (t-1, t+1) via central finite difference, t from 1 to n_snaps-2
    pool_R_time_static = []
    pool_R_time_adm = []
    pool_K_term = []
    n_used = 0

    for s in range(min(n_seeds, 4)):
        # Build g_ij sequence for this seed across all snapshots
        prep_seq = []
        for snap_idx in range(n_snaps):
            xi_mat = edge[s, snap_idx].copy()
            np.fill_diagonal(xi_mat, 1.0)
            psi = psi_r[s, snap_idx] + 1j * psi_i[s, snap_idx]
            k_field = np.full((N, N), 0.55)
            q_field = np.full((N, N), 0.45)
            try:
                prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, N, np)
            except Exception as e:
                print(f"  seed {s} snap {snap_idx}: prep failed: {e}")
                prep_seq.append(None)
                continue
            prep_seq.append({
                "g_00":   np.asarray(prep["g_00_h"]).copy(),
                "g_ij":   np.asarray(prep["g_ij_h"]).copy(),
                "t00":    np.asarray(prep["t00"]).copy(),
                "r_bar":  np.asarray(prep["r_bar_h"]).copy(),
            })

        # Central-difference time derivative for snap_idx in [1, n_snaps-2]
        for snap_idx in range(1, n_snaps - 1):
            p_minus = prep_seq[snap_idx - 1]
            p_curr  = prep_seq[snap_idx]
            p_plus  = prep_seq[snap_idx + 1]
            if any(p is None for p in (p_minus, p_curr, p_plus)):
                continue
            # Time step in lattice-step units
            dt = (snap_steps[snap_idx + 1] - snap_steps[snap_idx - 1]) / 2.0
            if dt == 0:
                continue
            # K_ij = (1/2) dt g_ij  (3x3 per node)
            dgdt = (p_plus["g_ij"] - p_minus["g_ij"]) / (2 * dt)
            K_ij = 0.5 * dgdt
            # K^2 - K_ij K^ij  (Hamiltonian-constraint scalar correction)
            K_trace = np.trace(K_ij, axis1=1, axis2=2)              # (N,)
            K_K_full = np.einsum("aij,aji->a", K_ij, K_ij)           # (N,)
            K_correction = 0.5 * (K_trace ** 2 - K_K_full)          # (N,)

            # Static G_00 vs ADM G_00
            G_00_static = p_curr["g_00"]
            G_00_adm = G_00_static + K_correction
            t00 = p_curr["t00"]

            R_time_static = G_00_static + LAMBDA_T - t00
            R_time_adm = G_00_adm + LAMBDA_T - t00

            pool_R_time_static.append(R_time_static)
            pool_R_time_adm.append(R_time_adm)
            pool_K_term.append(K_correction)
            n_used += 1

    R_static = np.concatenate(pool_R_time_static)
    R_adm = np.concatenate(pool_R_time_adm)
    K_t = np.concatenate(pool_K_term)
    print(f"\n  Total snapshot-triples used: {n_used}")
    print(f"  Total per-node residuals: {len(R_static)}")
    print()
    print(f"{'observable':<22} {'median':>10} {'mean':>10} {'std':>10}")
    print("-" * 60)
    print(f"{'|R_time| static':<22} {float(np.median(np.abs(R_static))):>10.5f} "
          f"{float(np.mean(np.abs(R_static))):>10.5f} {float(np.std(R_static)):>10.5f}")
    print(f"{'|R_time| ADM':<22} {float(np.median(np.abs(R_adm))):>10.5f} "
          f"{float(np.mean(np.abs(R_adm))):>10.5f} {float(np.std(R_adm)):>10.5f}")
    print(f"{'K-correction':<22} {float(np.median(np.abs(K_t))):>10.5f} "
          f"{float(np.mean(np.abs(K_t))):>10.5f} {float(np.std(K_t)):>10.5f}")

    delta_med_old = float(np.median(np.abs(R_static)))
    delta_med_new = float(np.median(np.abs(R_adm)))
    delta_mean_old = float(np.mean(np.abs(R_static)))
    delta_mean_new = float(np.mean(np.abs(R_adm)))

    print()
    if delta_med_new < delta_med_old * 0.7:
        verdict = "ADM_K_TERM_LARGELY_REDUCES_FLOOR"
    elif delta_med_new < delta_med_old * 0.95:
        verdict = "ADM_K_TERM_PARTIALLY_REDUCES_FLOOR"
    elif delta_med_new <= delta_med_old * 1.05:
        verdict = "ADM_K_TERM_NEGLIGIBLE"
    else:
        verdict = "ADM_K_TERM_INCREASES_RESIDUAL"
    print(f"VERDICT: {verdict}")
    print(f"  median  R_time: static {delta_med_old:.5f} -> ADM {delta_med_new:.5f}  "
          f"(change: {(delta_med_new - delta_med_old) / delta_med_old * 100:+.1f}%)")
    print(f"  mean    R_time: static {delta_mean_old:.5f} -> ADM {delta_mean_new:.5f}  "
          f"(change: {(delta_mean_new - delta_mean_old) / delta_mean_old * 100:+.1f}%)")

    # Also test with optimal Lambda_t for each (find best constant offset)
    best_lt_static = float(-np.median(R_static - LAMBDA_T))
    best_lt_adm = float(-np.median(R_adm - LAMBDA_T))
    print()
    print(f"Best-fit Lambda_t (median = 0):")
    print(f"  static: {best_lt_static:+.4f}  (System-R: 0.81, deviation {best_lt_static - 0.81:+.4f})")
    print(f"  ADM:    {best_lt_adm:+.4f}  (System-R: 0.81, deviation {best_lt_adm - 0.81:+.4f})")

    out = {
        "method": "ADM_extrinsic_curvature_correction_test",
        "schema_version": "1.0.0",
        "snapshot_file": str(snap_path),
        "n_snapshot_triples": n_used,
        "delta_time_static": {"median": delta_med_old, "mean": delta_mean_old},
        "delta_time_ADM":    {"median": delta_med_new, "mean": delta_mean_new},
        "K_correction_stats": {
            "median_abs": float(np.median(np.abs(K_t))),
            "mean_abs":   float(np.mean(np.abs(K_t))),
            "std":        float(np.std(K_t)),
        },
        "best_lambda_t_static": best_lt_static,
        "best_lambda_t_ADM":    best_lt_adm,
        "verdict": verdict,
    }
    out_path = REPO / "outputs" / "3plus1d_extrinsic_curvature_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
