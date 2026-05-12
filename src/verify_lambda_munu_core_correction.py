"""Test scheme D: Lambda_munu spatial-running correction.

If the matter-core residual cannot be absorbed into T_munu
(scheme C failed), test instead a node-local cosmological-tensor
correction:

  Lambda_t(a)^core = Lambda_t + D * log(T_00(a)/<T_00>)
  Lambda_s(a)^core = Lambda_s   (unchanged)

This is the time-time component of an inhomogeneous Lambda-tensor
that depends on the local matter density. It is structurally
equivalent to:

  Lambda_munu^cosmo(a) = Lambda_struct * g_munu + D * log(T_00/<T_00>) * u_mu u_nu
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                   matter-induced anisotropic correction

where u^mu = (1, 0, 0, 0) is the lattice rest-frame.

Sweep D in a grid; check if any single D drives the raw mean
residual <= 0.05 at all N>=60.

Output: outputs/lambda_munu_core_correction_audit.json
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
D_VALUES = [-0.5, -0.25, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.25, 0.5]


def per_node_delta_lambda_corrected(prep, D):
    """Recompute per-direction relative residual with Lambda_t per-node."""
    t00 = np.asarray(prep["t00"])
    t00_mean = float(np.mean(t00))
    log_ratio = np.log(np.maximum(t00 / max(t00_mean, 1e-12), 1e-10))
    lambda_t_a = LAMBDA_T + D * log_ratio   # per-node Lambda_t

    g_00 = np.asarray(prep["g_00_h"])
    g_ij = np.asarray(prep["g_ij_h"])
    t_ij = np.asarray(prep["t_ij"])

    # Eigendecomposition of T_ij (sanitise NaNs)
    t_clean = np.where(np.isfinite(t_ij), t_ij, 0.0)
    g_clean = np.where(np.isfinite(g_ij), g_ij, 0.0)
    n_lat = g_00.shape[0]
    eigvals_T = np.zeros((n_lat, 3))
    eigvecs_T = np.zeros((n_lat, 3, 3))
    for a in range(n_lat):
        try:
            w, V = np.linalg.eigh(t_clean[a])
        except np.linalg.LinAlgError:
            w, V = np.zeros(3), np.eye(3)
        order = np.argsort(w)
        eigvals_T[a] = w[order]
        eigvecs_T[a] = V[:, order]

    g_rot = np.einsum("aki,akl,alj->aij", eigvecs_T, g_clean, eigvecs_T)

    # Time-time residual with per-node Lambda_t
    R_time = g_00 + lambda_t_a - t00          # (n,)
    # Spatial diagonal residual (uses isotropic Lambda_s)
    R_diag = (np.diagonal(g_rot, axis1=1, axis2=2)
              + LAMBDA_S - eigvals_T)         # (n, 3)

    # Off-diagonal Frobenius
    g_off = g_rot.copy()
    for i in range(3):
        g_off[:, i, i] = 0.0
    R_off = np.sqrt((g_off ** 2).sum(axis=(1, 2)))

    R_norm = np.sqrt(R_time ** 2 + (R_diag ** 2).sum(axis=1) + R_off ** 2)
    T_norm = np.sqrt(t00 ** 2 + (eigvals_T ** 2).sum(axis=1))
    return R_norm / np.maximum(T_norm, 1e-12)


def gather_preps(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)
    preps = []
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        preps.append(prep)
    return preps


def main() -> int:
    print("=" * 110)
    print("Lambda_munu(a) core-correction sweep (scheme D)")
    print("Lambda_t(a)^core = Lambda_t + D * log(T_00(a)/<T_00>)")
    print("=" * 110)
    print()

    regime_preps = []
    for reg, n_lat in REGIMES:
        ps = gather_preps(reg, n_lat)
        if ps is None:
            continue
        regime_preps.append({"regime": reg, "N": n_lat, "preps": ps})

    print(f"{'D':>6} | " + " | ".join([
        f"{rp['regime']:<6} med/mean" for rp in regime_preps
    ]))
    print("-" * (12 + 17 * len(regime_preps)))

    sweep = {}
    for D in D_VALUES:
        per_reg = []
        line = f"{D:>+6.3f} |"
        for rp in regime_preps:
            ds = []
            for prep in rp["preps"]:
                ds.append(per_node_delta_lambda_corrected(prep, D))
            d_all = np.concatenate(ds)
            med = float(np.median(d_all))
            mean = float(np.mean(d_all))
            per_reg.append({"regime": rp["regime"], "N": rp["N"],
                            "median": med, "mean": mean})
            line += f" {med:.3f}/{mean:.3f} |"
        sweep[f"D={D}"] = {"D": D, "per_regime": per_reg}
        print(line)

    best_D, best_avg = None, float("inf")
    for tag, info in sweep.items():
        ms = [r["mean"] for r in info["per_regime"] if r["N"] >= 60]
        if not ms:
            continue
        avg = float(np.mean(ms))
        if avg < best_avg:
            best_avg = avg
            best_D = info["D"]

    print()
    print(f"BEST D (smallest mean averaged over N>=60): D = {best_D:+.4f}")
    print(f"  averaged mean delta (N>=60) = {best_avg:.4f}")

    info_best = sweep[f"D={best_D}"]
    n_strict = n_relax = n_total = 0
    for r in info_best["per_regime"]:
        if r["N"] < 60:
            continue
        n_total += 1
        if r["median"] <= 0.05 and r["mean"] <= 0.05:
            n_strict += 1
        if r["median"] <= 0.05 and r["mean"] <= 0.10:
            n_relax += 1
    print(f"  STRICT  (med<=0.05 AND mean<=0.05): {n_strict}/{n_total} regimes pass")
    print(f"  RELAXED (med<=0.05 AND mean<=0.10): {n_relax}/{n_total} regimes pass")

    if n_strict == n_total:
        verdict = "LAMBDA_CORE_FULL_TENSOR_CLOSURE_HOLDS_STRICT"
    elif n_relax == n_total:
        verdict = "LAMBDA_CORE_FULL_TENSOR_CLOSURE_HOLDS_RELAXED"
    else:
        verdict = f"LAMBDA_CORE_PARTIAL_{n_relax}_OF_{n_total}_RELAXED"
    print(f"VERDICT: {verdict}")

    out_path = REPO / "outputs" / "lambda_munu_core_correction_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "Lambda_munu_node_local_log_T_correction_sweep",
            "schema_version": "1.0.0",
            "lambda_t_struct": LAMBDA_T, "lambda_s_struct": LAMBDA_S,
            "correction_form": "Lambda_t(a)^core = Lambda_t + D * log(T_00(a)/<T_00>)",
            "D_values_swept": D_VALUES,
            "per_D": sweep,
            "best_D": best_D,
            "best_avg_mean_N_geq_60": best_avg,
            "verdict": verdict,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
