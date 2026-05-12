"""Per-eigendirection decomposition of the per-node Einstein
residual.

Equation per node a, after diagonalisation of the spatial T_ij in
its own principal frame and rotation of G_ij into the same frame:

  G_00(a) + Lambda_t      - 8 pi G T_00(a)        = R_00(a)
  G_(11)(a) + Lambda_s    - 8 pi G T_(11)(a)      = R_11(a)
  G_(22)(a) + Lambda_s    - 8 pi G T_(22)(a)      = R_22(a)
  G_(33)(a) + Lambda_s    - 8 pi G T_(33)(a)      = R_33(a)

with G_(ii) = (U^T G_ij U)_ii in the T-eigenbasis. The four
scalar residuals R_00, R_11, R_22, R_33 are independent
diagonal-block tests of the closure; the off-diagonal residual
||(U^T G U)_off||_F decays separately as documented in fig11.

We furthermore split the spatial diagonal block into:

  trace contribution:    R_trace = (R_11 + R_22 + R_33)/3
  traceless contribution: R_TF_ii = R_ii - R_trace
                         |R_TF| = sqrt(sum R_TF_ii^2)

The trace equation is sourced by Lambda_s; the traceless
equation has no Lambda contribution (since Lambda is isotropic).
Therefore:

  - if heavy-tail mean lives in the TRACE part, an isotropic
    Lambda-running (O5 with kappa=gamma^2) can absorb it
    structurally;
  - if heavy-tail mean lives in the TRACELESS part, Lambda-running
    cannot help; the closure requires either a higher-order
    Ricci correction (O1) or an anisotropic Lambda_ij;
  - if it lives in the TIME-TIME part, a different
    Lambda_t-running is required.

The audit identifies which contribution dominates and reports it
honestly per regime.

Output: outputs/per_eigendirection_residual_audit.json
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
from verify_higher_order_terms_all8 import LAMBDA_T, LAMBDA_S


REGIMES_TO_TEST = [
    ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]


def per_node_eigendirection_residuals(prep, lambda_t, lambda_s,
                                         eight_pi_G=1.0):
    """Diagonalise spatial T_ij, rotate G_ij into same basis,
    compute the 4 scalar diagonal residuals + traceless / trace
    decomposition + off-diagonal Frobenius norm per node."""
    g_00 = np.asarray(prep["g_00_h"])           # (n,)
    g_ij = np.asarray(prep["g_ij_h"])           # (n, 3, 3)
    t00 = np.asarray(prep["t00"])              # (n,)
    t_ij = np.asarray(prep["t_ij"])            # (n, 3, 3)
    n_lat = g_00.shape[0]

    # Eigendecomposition of T_ij (sanitise NaNs).
    t_clean = np.where(np.isfinite(t_ij), t_ij, 0.0)
    g_clean = np.where(np.isfinite(g_ij), g_ij, 0.0)
    eigvals_T = np.zeros((n_lat, 3))
    eigvecs_T = np.zeros((n_lat, 3, 3))
    for a in range(n_lat):
        try:
            w, V = np.linalg.eigh(t_clean[a])
        except np.linalg.LinAlgError:
            w, V = np.zeros(3), np.eye(3)
        # Sort ascending (already done by eigh but explicit)
        order = np.argsort(w)
        eigvals_T[a] = w[order]
        eigvecs_T[a] = V[:, order]

    # Rotate G into T's eigenframe: G_rot = V^T G V (per node)
    g_rot = np.einsum("aki,akl,alj->aij", eigvecs_T, g_clean, eigvecs_T)

    # Diagonal scalar residuals
    R_time = g_00 + lambda_t - eight_pi_G * t00          # (n,)
    R_diag = (np.diagonal(g_rot, axis1=1, axis2=2)
              + lambda_s - eight_pi_G * eigvals_T)        # (n, 3)

    # Off-diagonal magnitude in T-eigenframe
    g_off = g_rot.copy()
    for i in range(3):
        g_off[:, i, i] = 0.0
    R_off = np.sqrt((g_off ** 2).sum(axis=(1, 2)))        # (n,)

    # Trace-traceless split of spatial diagonal residual
    R_trace = R_diag.mean(axis=1)                         # (n,)
    R_TF = R_diag - R_trace[:, None]                      # (n, 3)
    R_TF_norm = np.sqrt((R_TF ** 2).sum(axis=1))          # (n,)

    return {
        "R_time": R_time,
        "R_diag": R_diag,                  # (n, 3) per spatial direction
        "R_off": R_off,
        "R_trace": R_trace,                # spatial trace residual / 3
        "R_TF": R_TF,                       # traceless 3-vector
        "R_TF_norm": R_TF_norm,             # |traceless|
        "T_eigvals": eigvals_T,
    }


def gather_regime(regime, n_lat, lambda_t=LAMBDA_T, lambda_s=LAMBDA_S):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    # Schema dispatch: legacy d1.npz uses dense_cell_*_values arrays,
    # newer .snapshots.npz uses edge_xi_snapshots / psi_*_snapshots
    # with a shape (n_seeds, n_snapshots, ...). Pick the last snapshot.
    if "edge_xi_snapshots" in d.files:
        edge_snap = d["edge_xi_snapshots"]
        psi_r = d["psi_real_snapshots"]
        psi_i = d["psi_imag_snapshots"]
        n_seeds = min(int(edge_snap.shape[0]), 32)
        def _seed_xi_psi(s):
            xi_mat = np.asarray(edge_snap[s, -1], dtype=np.float64).copy()
            np.fill_diagonal(xi_mat, 1.0)
            psi = (np.asarray(psi_r[s, -1], dtype=np.float64)
                   + 1j * np.asarray(psi_i[s, -1], dtype=np.float64))
            return xi_mat, psi
    else:
        edge_arr = d["dense_cell_edge_xi_values"]
        amp_arr = d["dense_cell_node_amplitude_values"]
        phase_arr = d["dense_cell_node_phase_values"]
        n_seeds = min(int(edge_arr.shape[0]), 32)
        def _seed_xi_psi(s):
            xi_mat = edge_to_matrix(edge_arr[s], n_lat)
            np.fill_diagonal(xi_mat, 1.0)
            psi = amp_arr[s] * np.exp(1j * phase_arr[s])
            return xi_mat, psi

    pool = {"R_time": [], "R_diag1": [], "R_diag2": [], "R_diag3": [],
            "R_off": [], "R_trace": [], "R_TF_norm": [],
            "G_norm": [], "T_norm": []}
    for s in range(n_seeds):
        xi_mat, psi = _seed_xi_psi(s)
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        res = per_node_eigendirection_residuals(
            prep, lambda_t, lambda_s)
        pool["R_time"].append(res["R_time"])
        pool["R_diag1"].append(res["R_diag"][:, 0])
        pool["R_diag2"].append(res["R_diag"][:, 1])
        pool["R_diag3"].append(res["R_diag"][:, 2])
        pool["R_off"].append(res["R_off"])
        pool["R_trace"].append(res["R_trace"])
        pool["R_TF_norm"].append(res["R_TF_norm"])
        # tensor magnitudes for normalisation
        g_norm_a = np.sqrt(np.asarray(prep["g_00_h"]) ** 2
                            + (np.asarray(prep["g_ij_h"]) ** 2).sum(axis=(1, 2)))
        t_norm_a = np.sqrt(np.asarray(prep["t00"]) ** 2
                            + (np.asarray(prep["t_ij"]) ** 2).sum(axis=(1, 2)))
        pool["G_norm"].append(g_norm_a)
        pool["T_norm"].append(t_norm_a)
    arrays = {k: np.concatenate(v) for k, v in pool.items()}

    rec = {"regime": regime, "N": n_lat, "n_nodes_total": int(len(arrays["R_time"]))}
    for key in ("R_time", "R_diag1", "R_diag2", "R_diag3",
                  "R_off", "R_trace", "R_TF_norm"):
        v = arrays[key]
        rec[f"{key}_mean_abs"] = float(np.abs(v).mean())
        rec[f"{key}_median_abs"] = float(np.median(np.abs(v)))
        rec[f"{key}_p90_abs"] = float(np.percentile(np.abs(v), 90))
    # contribution analysis: in mean-of-squared-residual budget
    sq_total = (arrays["R_time"] ** 2 + arrays["R_diag1"] ** 2
                + arrays["R_diag2"] ** 2 + arrays["R_diag3"] ** 2
                + arrays["R_off"] ** 2)
    rec["mean_sq_total"] = float(sq_total.mean())
    rec["mean_sq_R_time"] = float((arrays["R_time"] ** 2).mean())
    rec["mean_sq_R_diag_total"] = float(
        (arrays["R_diag1"] ** 2 + arrays["R_diag2"] ** 2
         + arrays["R_diag3"] ** 2).mean())
    rec["mean_sq_R_off"] = float((arrays["R_off"] ** 2).mean())
    rec["mean_sq_R_trace_x3"] = float(
        3 * (arrays["R_trace"] ** 2).mean())
    rec["mean_sq_R_TF_norm"] = float((arrays["R_TF_norm"] ** 2).mean())
    # fractional budget
    rec["frac_time"] = rec["mean_sq_R_time"] / max(rec["mean_sq_total"], 1e-12)
    rec["frac_spatial_diag"] = rec["mean_sq_R_diag_total"] / max(rec["mean_sq_total"], 1e-12)
    rec["frac_off"] = rec["mean_sq_R_off"] / max(rec["mean_sq_total"], 1e-12)
    # within spatial: trace vs traceless
    rec["frac_within_spatial_trace"] = rec["mean_sq_R_trace_x3"] / max(rec["mean_sq_R_diag_total"], 1e-12)
    rec["frac_within_spatial_TF"] = rec["mean_sq_R_TF_norm"] / max(rec["mean_sq_R_diag_total"], 1e-12)
    return rec


def main():
    print("=" * 110)
    print("Per-eigendirection decomposition of per-node residual")
    print("Equation: G_(ii)(a) + Lambda_i - 8piG * lambda_i(a) = R_(ii)(a)")
    print("=" * 110)
    print()
    print(f"{'reg':<8} {'N':>3} | "
          f"{'|R_time|_med':>12} {'|R_d1|_med':>11} {'|R_d2|_med':>11} {'|R_d3|_med':>11} "
          f"{'|R_off|_med':>12} | {'frac_time':>10} {'frac_spat':>10} {'frac_off':>9}")
    print("-" * 130)
    results = []
    for reg, n_lat in REGIMES_TO_TEST:
        r = gather_regime(reg, n_lat)
        if r is None:
            continue
        results.append(r)
        print(f"{reg:<8} {n_lat:>3} | "
              f"{r['R_time_median_abs']:>12.5f} "
              f"{r['R_diag1_median_abs']:>11.5f} "
              f"{r['R_diag2_median_abs']:>11.5f} "
              f"{r['R_diag3_median_abs']:>11.5f} "
              f"{r['R_off_median_abs']:>12.5f} | "
              f"{r['frac_time']:>10.3f} "
              f"{r['frac_spatial_diag']:>10.3f} "
              f"{r['frac_off']:>9.3f}")

    print()
    print("--- Within spatial diagonal: trace vs traceless ---")
    print(f"{'reg':<8} {'N':>3} | "
          f"{'|R_trace|_med':>14} {'|R_TF|_med':>11} | "
          f"{'frac_trace':>11} {'frac_TF':>9}")
    print("-" * 70)
    for r in results:
        print(f"{r['regime']:<8} {r['N']:>3} | "
              f"{r['R_trace_median_abs']:>14.5f} "
              f"{r['R_TF_norm_median_abs']:>11.5f} | "
              f"{r['frac_within_spatial_trace']:>11.3f} "
              f"{r['frac_within_spatial_TF']:>9.3f}")

    print()
    print("=" * 110)
    print("VERDICT: which component dominates the mean residual?")
    print("=" * 110)
    for r in results:
        dominant = max([
            ("time-time", r["frac_time"]),
            ("spatial-diag", r["frac_spatial_diag"]),
            ("off-diag", r["frac_off"]),
        ], key=lambda x: x[1])
        sub_dom = ""
        if dominant[0] == "spatial-diag":
            sub = max([
                ("trace (Lambda-absorbable)", r["frac_within_spatial_trace"]),
                ("traceless (NOT Lambda-absorbable)", r["frac_within_spatial_TF"]),
            ], key=lambda x: x[1])
            sub_dom = f"  -> within spatial: {sub[0]} ({sub[1]*100:.1f}%)"
        print(f"  {r['regime']:<8} N={r['N']:>3}: dominant = {dominant[0]} "
              f"({dominant[1]*100:.1f}%){sub_dom}")

    out = REPO / "outputs" / "per_eigendirection_residual_audit.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "method": "per_eigendirection_decomposition",
            "schema_version": "1.0.0",
            "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
            "per_regime": results,
        }, f, indent=2)
    print()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
