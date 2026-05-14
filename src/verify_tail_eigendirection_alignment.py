"""Test (A): Eigendirection alignment of the tail residual tensor.

Per tail node a, compute the residual tensor

  R_munu(a) = G_munu(a) + Lambda_munu - 8 pi G T_munu(a)

(in the principal frame of T_munu, so off-diagonal of T are zero
and diagonal eigvalues are sorted). Decompose R into:

  R_time(a) :  scalar time-time residual
  R_diag(a) :  3-vector spatial-diagonal residual
  R_off(a)  :  spatial off-diagonal Frobenius residual

Three alignment measures per tail node:

  1. spatial direction of R_diag: which T-eigenvector index
     carries the largest |R_diag_i|? Distribution -> uniform
     means isotropic (no preferred axis), peaked means anisotropic.

  2. R_time vs |R_diag|: ratio of time-time vs spatial residual.
     If ratio >> 1: closure failure dominantly in time-time.
     If ratio << 1: dominantly spatial.

  3. R_diag direction with T_00 gradient: does R_diag align with
     the lattice gradient of T_00? If yes -> the residual lives in
     the matter-flow direction -> tensorial source identified.

Output: outputs/tail_eigendirection_alignment_audit.json
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

from _d1_npz_discovery import find_d1_npz
from verify_galerkin_runner_A_hessian_ricci import (
    edge_to_matrix, per_seed_galerkin)
from verify_per_eigendirection_residual import (
    per_node_eigendirection_residuals)


REGIMES = [
    ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]
LAMBDA_T = 0.81
LAMBDA_S = -0.005
TAIL_FRAC = 0.10


def per_node_residual_decomposition(prep):
    """Return per-node R_time, R_diag (3-vec sorted by T eigvals
    ascending), R_off, and T eigenvectors."""
    res = per_node_eigendirection_residuals(prep, LAMBDA_T, LAMBDA_S)
    return {
        "R_time":  res["R_time"],
        "R_diag":  res["R_diag"],     # sorted by T-eigval order
        "R_off":   res["R_off"],
        "t_eigvals": res["T_eigvals"],
    }


def gather_tail_nodes(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)

    pool = {"R_time": [], "R_diag": [], "R_off": [], "t_eigvals": [], "t00": []}
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        dec = per_node_residual_decomposition(prep)
        pool["R_time"].append(dec["R_time"])
        pool["R_diag"].append(dec["R_diag"])
        pool["R_off"].append(dec["R_off"])
        pool["t_eigvals"].append(dec["t_eigvals"])
        pool["t00"].append(np.asarray(prep["t00"]))
    R_time = np.concatenate(pool["R_time"])
    R_diag = np.concatenate(pool["R_diag"], axis=0)
    R_off = np.concatenate(pool["R_off"])
    t_eig = np.concatenate(pool["t_eigvals"], axis=0)
    t00 = np.concatenate(pool["t00"])

    # Per-node Delta (relative)
    R_norm = np.sqrt(R_time ** 2 + (R_diag ** 2).sum(axis=1) + R_off ** 2)
    T_norm = np.sqrt(t00 ** 2 + (t_eig ** 2).sum(axis=1))
    delta = R_norm / np.maximum(T_norm, 1e-12)

    n_total = len(delta)
    n_tail = max(1, int(n_total * TAIL_FRAC))
    order = np.argsort(-delta)
    tail = order[:n_tail]
    bulk = order[n_tail:]

    return {
        "regime": reg, "N": n_lat,
        "n_total": int(n_total), "n_tail": int(n_tail),
        "tail": {
            "R_time":   R_time[tail],
            "R_diag":   R_diag[tail],
            "R_off":    R_off[tail],
            "t_eig":    t_eig[tail],
            "t00":      t00[tail],
            "delta":    delta[tail],
        },
        "bulk": {
            "R_time":   R_time[bulk],
            "R_diag":   R_diag[bulk],
            "R_off":    R_off[bulk],
            "t_eig":    t_eig[bulk],
            "t00":      t00[bulk],
            "delta":    delta[bulk],
        },
    }


def alignment_stats(group):
    """Compute alignment metrics for a node group (tail or bulk)."""
    R_t = group["R_time"]
    R_d = group["R_diag"]            # (n, 3) sorted by T eigval ascending
    R_o = group["R_off"]
    t_e = group["t_eig"]              # (n, 3) sorted ascending

    n = len(R_t)
    # 1. Distribution of which |R_diag_i| is the largest
    abs_R_diag = np.abs(R_d)
    arg_max_dir = np.argmax(abs_R_diag, axis=1)   # in {0,1,2}
    counts_dir = [int((arg_max_dir == i).sum()) for i in range(3)]
    # i=0: smallest T-eigval direction; i=2: largest

    # 2. Time vs spatial vs off-diagonal magnitude ratios
    spatial_mag = np.sqrt((R_d ** 2).sum(axis=1))
    R_time_mag = np.abs(R_t)
    ratio_time_to_space = R_time_mag / np.maximum(spatial_mag, 1e-12)
    ratio_off_to_space = R_o / np.maximum(spatial_mag, 1e-12)

    # 3. Sign of R_diag in the direction of largest T-eigval
    # T eigvals sorted ascending so largest is index 2
    R_along_largest = R_d[:, 2]
    sign_largest_pos_frac = float((R_along_largest > 0).sum()) / max(n, 1)

    # 4. Per-direction mean ratio (R_diag_i / spatial_mag) — check
    # if there's a SIGNED preference along a direction
    proj = R_d / np.maximum(spatial_mag[:, None], 1e-12)
    mean_proj_per_dir = proj.mean(axis=0).tolist()

    return {
        "n": int(n),
        "argmax_direction_counts": {
            "i=0_smallest_eigval": counts_dir[0],
            "i=1_middle_eigval":   counts_dir[1],
            "i=2_largest_eigval":  counts_dir[2],
            "fraction_largest":    counts_dir[2] / max(n, 1),
        },
        "time_to_spatial_ratio": {
            "median": float(np.median(ratio_time_to_space)),
            "mean":   float(np.mean(ratio_time_to_space)),
        },
        "off_to_spatial_ratio": {
            "median": float(np.median(ratio_off_to_space)),
            "mean":   float(np.mean(ratio_off_to_space)),
        },
        "sign_R_along_largest_T_eigval_pos_frac": sign_largest_pos_frac,
        "mean_signed_projection_per_direction": [float(x) for x in mean_proj_per_dir],
    }


def main() -> int:
    print("=" * 110)
    print("Tail Residual Eigendirection Alignment Audit")
    print("=" * 110)
    print()

    rows = []
    for reg, n_lat in REGIMES:
        d = gather_tail_nodes(reg, n_lat)
        if d is None:
            continue
        tail_stats = alignment_stats(d["tail"])
        bulk_stats = alignment_stats(d["bulk"])
        rows.append({
            "regime": reg, "N": n_lat,
            "tail": tail_stats, "bulk": bulk_stats,
        })

    print(f"{'reg':<8} {'N':>3} | "
          f"{'tail T:S':>9} {'tail O:S':>9} {'tail argmax_dist':>18} "
          f"{'tail sign+_largest':>20} {'tail mean_proj_largest':>24}")
    print("-" * 110)
    for r in rows:
        t = r["tail"]
        argmax_str = (
            f"[{t['argmax_direction_counts']['i=0_smallest_eigval']:>2}, "
            f"{t['argmax_direction_counts']['i=1_middle_eigval']:>2}, "
            f"{t['argmax_direction_counts']['i=2_largest_eigval']:>2}]"
        )
        print(f"{r['regime']:<8} {r['N']:>3} | "
              f"{t['time_to_spatial_ratio']['median']:>9.4f} "
              f"{t['off_to_spatial_ratio']['median']:>9.4f} "
              f"{argmax_str:>18} "
              f"{t['sign_R_along_largest_T_eigval_pos_frac']:>20.3f} "
              f"{t['mean_signed_projection_per_direction'][2]:>+24.4f}")

    print()
    print("Bulk comparison:")
    print(f"{'reg':<8} {'N':>3} | "
          f"{'bulk T:S':>9} {'bulk O:S':>9} {'bulk argmax_dist':>18} "
          f"{'bulk sign+_largest':>20}")
    print("-" * 90)
    for r in rows:
        b = r["bulk"]
        argmax_str = (
            f"[{b['argmax_direction_counts']['i=0_smallest_eigval']:>3}, "
            f"{b['argmax_direction_counts']['i=1_middle_eigval']:>3}, "
            f"{b['argmax_direction_counts']['i=2_largest_eigval']:>3}]"
        )
        print(f"{r['regime']:<8} {r['N']:>3} | "
              f"{b['time_to_spatial_ratio']['median']:>9.4f} "
              f"{b['off_to_spatial_ratio']['median']:>9.4f} "
              f"{argmax_str:>18} "
              f"{b['sign_R_along_largest_T_eigval_pos_frac']:>20.3f}")

    # Aggregate verdict
    largest_frac_avg = float(np.mean([
        r["tail"]["argmax_direction_counts"]["fraction_largest"] for r in rows
    ]))
    sign_pos_avg = float(np.mean([
        r["tail"]["sign_R_along_largest_T_eigval_pos_frac"] for r in rows
    ]))
    proj_largest_avg = float(np.mean([
        r["tail"]["mean_signed_projection_per_direction"][2] for r in rows
    ]))

    print()
    print(f"Aggregate (averaged over regimes):")
    print(f"  Tail argmax in largest-eigval direction: {largest_frac_avg*100:.1f}% "
          f"(isotropic baseline: 33%)")
    print(f"  Tail sign of R along largest T-eigval > 0: {sign_pos_avg*100:.1f}%")
    print(f"  Tail mean signed projection on largest T-eigval: {proj_largest_avg:+.4f}")

    if largest_frac_avg >= 0.55 and abs(proj_largest_avg) >= 0.3:
        verdict = "TAIL_ANISOTROPIC_ALIGNED_LARGEST_T_EIGVAL"
        rationale = ("Tail residual preferentially aligned with the largest "
                     "T-eigval direction. A tensorial correction of the form "
                     "Lambda_munu - alpha * T_T_largest_eigval projector might close.")
    elif largest_frac_avg >= 0.45:
        verdict = "TAIL_PARTIAL_ANISOTROPY"
        rationale = ("Modest alignment with largest T-eigval direction; "
                     "partial tensorial pathway available but not dominant.")
    else:
        verdict = "TAIL_ISOTROPIC_NO_PREFERRED_AXIS"
        rationale = ("No preferred direction in tail residual; the universal "
                     "magnitude law is the dominant structure and tensorial "
                     "closure cannot be achieved by a single anisotropic term.")
    print()
    print(f"VERDICT: {verdict}")
    print(f"  Rationale: {rationale}")

    out_path = REPO / "outputs" / "tail_eigendirection_alignment_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "tail_residual_eigendirection_alignment",
            "schema_version": "1.0.0",
            "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
            "tail_fraction": TAIL_FRAC,
            "per_regime": rows,
            "aggregate": {
                "tail_argmax_largest_eigval_fraction": largest_frac_avg,
                "tail_sign_positive_fraction": sign_pos_avg,
                "tail_mean_signed_projection_largest": proj_largest_avg,
            },
            "verdict": verdict,
            "rationale": rationale,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
