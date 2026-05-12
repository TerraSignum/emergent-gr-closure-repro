"""Test scheme E: joint (Lambda_t, Lambda_s) per-node core correction.

Lambda_t(a)^core = Lambda_t + D_t * log(T_00(a)/<T_00>)
Lambda_s(a)^core = Lambda_s + D_s * log(T_00(a)/<T_00>)

If neither D_t alone (scheme D) nor D_s correction alone closes
the raw mean residual, the JOINT correction may. This is
structurally equivalent to the full anisotropic Lambda_munu(a)
tensor running with local matter density.

Sweep (D_t, D_s) jointly on a 5x5 grid. Find best pair.

Output: outputs/joint_lambda_correction_audit.json
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


REGIMES = [
    ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]
LAMBDA_T = 0.81
LAMBDA_S = -0.005
D_T_VALUES = [-0.05, 0.0, 0.05, 0.1, 0.2]
D_S_VALUES = [-0.05, -0.02, 0.0, 0.02, 0.05]


def per_node_delta_joint(prep, D_t, D_s):
    t00 = np.asarray(prep["t00"])
    g_00 = np.asarray(prep["g_00_h"])
    g_ij = np.asarray(prep["g_ij_h"])
    t_ij = np.asarray(prep["t_ij"])

    t00_mean = float(np.mean(t00))
    log_ratio = np.log(np.maximum(t00 / max(t00_mean, 1e-12), 1e-10))
    lam_t_a = LAMBDA_T + D_t * log_ratio
    lam_s_a = LAMBDA_S + D_s * log_ratio

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
    R_time = g_00 + lam_t_a - t00
    R_diag = (np.diagonal(g_rot, axis1=1, axis2=2)
              + lam_s_a[:, None] - eigvals_T)

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
    print("Joint (Lambda_t, Lambda_s) core correction sweep (scheme E)")
    print("Lambda_mu^core(a) = Lambda_mu + D_mu * log(T_00(a)/<T_00>)")
    print("=" * 110)
    print()

    regime_preps = []
    for reg, n_lat in REGIMES:
        ps = gather_preps(reg, n_lat)
        if ps is None:
            continue
        regime_preps.append({"regime": reg, "N": n_lat, "preps": ps})

    print(f"{'D_t':>6} {'D_s':>6} | avg mean (N>=60) | strict pass | relaxed pass")
    print("-" * 70)

    sweep = {}
    best = {"avg_mean": float("inf"), "D_t": None, "D_s": None,
            "n_strict": 0, "n_relaxed": 0}

    for D_t in D_T_VALUES:
        for D_s in D_S_VALUES:
            ms_pool = []
            n_strict = 0
            n_relax = 0
            n_total = 0
            per_reg = []
            for rp in regime_preps:
                ds = []
                for prep in rp["preps"]:
                    ds.append(per_node_delta_joint(prep, D_t, D_s))
                d_all = np.concatenate(ds)
                med = float(np.median(d_all))
                mean = float(np.mean(d_all))
                per_reg.append({"regime": rp["regime"], "N": rp["N"],
                                "median": med, "mean": mean})
                if rp["N"] >= 60:
                    n_total += 1
                    ms_pool.append(mean)
                    if med <= 0.05 and mean <= 0.05:
                        n_strict += 1
                    if med <= 0.05 and mean <= 0.10:
                        n_relax += 1

            avg = float(np.mean(ms_pool)) if ms_pool else float("inf")
            sweep[f"Dt={D_t}_Ds={D_s}"] = {
                "D_t": D_t, "D_s": D_s,
                "per_regime": per_reg,
                "avg_mean_N_geq_60": avg,
                "n_strict": n_strict,
                "n_relaxed": n_relax,
                "n_total": n_total,
            }
            print(f"{D_t:>+6.3f} {D_s:>+6.3f} | {avg:.4f}            | "
                  f"{n_strict}/{n_total}         | {n_relax}/{n_total}")
            if avg < best["avg_mean"]:
                best = {"avg_mean": avg, "D_t": D_t, "D_s": D_s,
                        "n_strict": n_strict, "n_relaxed": n_relax,
                        "n_total": n_total}

    print()
    print(f"BEST joint (D_t, D_s): ({best['D_t']:+.4f}, {best['D_s']:+.4f})")
    print(f"  avg mean (N>=60) = {best['avg_mean']:.4f}")
    print(f"  STRICT  {best['n_strict']}/{best['n_total']}")
    print(f"  RELAXED {best['n_relaxed']}/{best['n_total']}")

    if best["n_strict"] == best["n_total"]:
        verdict = "JOINT_CORE_TENSOR_CLOSURE_HOLDS_STRICT"
    elif best["n_relaxed"] == best["n_total"]:
        verdict = "JOINT_CORE_TENSOR_CLOSURE_HOLDS_RELAXED"
    else:
        verdict = f"JOINT_CORE_TENSOR_CLOSURE_PARTIAL_{best['n_relaxed']}_OF_{best['n_total']}"
    print(f"VERDICT: {verdict}")

    out_path = REPO / "outputs" / "joint_lambda_correction_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "joint_lambda_t_lambda_s_core_correction_sweep_scheme_E",
            "schema_version": "1.0.0",
            "lambda_t_struct": LAMBDA_T,
            "lambda_s_struct": LAMBDA_S,
            "correction_form": (
                "Lambda_t(a) = Lambda_t + D_t * log(T_00(a)/<T_00>); "
                "Lambda_s(a) = Lambda_s + D_s * log(T_00(a)/<T_00>)"),
            "D_t_grid": D_T_VALUES, "D_s_grid": D_S_VALUES,
            "per_grid_point": sweep,
            "best": best,
            "verdict": verdict,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
