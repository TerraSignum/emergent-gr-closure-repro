"""Test scheme F: anisotropic Lambda correction along the largest
T-eigval direction.

Motivated by tail_eigendirection_alignment_audit.json:
  94.5% of tail nodes have max |R_diag| in the largest-T-eigval
  direction, with mean signed projection -0.597 (highly negative).

This means the residual G_ij - 8 pi G T_ij is systematically
negative along the largest T-eigval direction at tail nodes.
A tensorial correction of the form

  Lambda_munu^aniso(a) = Lambda_iso * g_munu
                       + alpha * f(T_00(a)) * v_a^i v_a^j

with v_a = the largest-T-eigval eigenvector at node a, projects
into the residual exactly where the gap lives.

Sweep alpha for two function choices:
  f1 = log(T_00/<T_00>)        (universal law form)
  f2 = (T_00/<T_00>) - 1       (linear)

Output: outputs/anisotropic_lambda_correction_audit.json
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
ALPHA_VALUES = [-0.5, -0.25, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.25, 0.5]
F_CHOICES = ["log_ratio", "linear"]


def per_node_delta_anisotropic(prep, alpha, f_choice):
    """Compute per-direction relative residual with anisotropic
    Lambda along largest T-eigval direction:
      Lambda_ij^aniso(a) = Lambda_s * delta_ij
                         + alpha * f(T_00(a)/<T_00>) * v^i v^j
    where v = largest-T-eigvec.
    """
    g_00 = np.asarray(prep["g_00_h"])
    g_ij = np.asarray(prep["g_ij_h"])
    t00 = np.asarray(prep["t00"])
    t_ij = np.asarray(prep["t_ij"])
    n_lat = g_00.shape[0]

    t00_mean = float(np.mean(t00))
    if f_choice == "log_ratio":
        f_a = np.log(np.maximum(t00 / max(t00_mean, 1e-12), 1e-10))
    else:
        f_a = t00 / max(t00_mean, 1e-12) - 1.0

    t_clean = np.where(np.isfinite(t_ij), t_ij, 0.0)
    g_clean = np.where(np.isfinite(g_ij), g_ij, 0.0)

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

    # Rotate G into T-eigenframe
    g_rot = np.einsum("aki,akl,alj->aij", eigvecs_T, g_clean, eigvecs_T)
    g_diag = np.diagonal(g_rot, axis1=1, axis2=2)        # (n,3)

    # Build Lambda diagonal in T-eigenframe:
    #   isotropic spatial Lambda_s on i=0,1,2
    #   plus alpha * f_a on the largest-eigval direction (i=2)
    lam_diag = np.full((n_lat, 3), LAMBDA_S)
    lam_diag[:, 2] = LAMBDA_S + alpha * f_a    # largest T-eigval direction

    R_time = g_00 + LAMBDA_T - t00
    R_diag = g_diag + lam_diag - eigvals_T

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
    print("Anisotropic Lambda correction along LARGEST T-eigval direction (scheme F)")
    print("Lambda_ij(a) = Lambda_s * delta_ij + alpha * f(T_00(a)/<T_00>) * v^i v^j")
    print("=" * 110)
    print()

    regime_preps = []
    for reg, n_lat in REGIMES:
        ps = gather_preps(reg, n_lat)
        if ps is None:
            continue
        regime_preps.append({"regime": reg, "N": n_lat, "preps": ps})

    sweeps = {}
    best_overall = {"mean": float("inf")}

    for f_choice in F_CHOICES:
        print(f"--- f = {f_choice} ---")
        print(f"{'alpha':>7} | " + " | ".join([
            f"{rp['regime']:<6} med/mean" for rp in regime_preps]))
        print("-" * (10 + 17 * len(regime_preps)))

        for alpha in ALPHA_VALUES:
            per_reg = []
            line = f"{alpha:>+7.4f} |"
            ms_pool = []
            for rp in regime_preps:
                ds = []
                for prep in rp["preps"]:
                    ds.append(per_node_delta_anisotropic(prep, alpha, f_choice))
                d_all = np.concatenate(ds)
                med = float(np.median(d_all))
                mean = float(np.mean(d_all))
                per_reg.append({"regime": rp["regime"], "N": rp["N"],
                                "median": med, "mean": mean})
                line += f" {med:.3f}/{mean:.3f} |"
                if rp["N"] >= 60:
                    ms_pool.append(mean)
            avg = float(np.mean(ms_pool)) if ms_pool else float("inf")
            n_strict = sum(1 for r in per_reg if r["N"] >= 60
                           and r["median"] <= 0.05 and r["mean"] <= 0.05)
            n_relax = sum(1 for r in per_reg if r["N"] >= 60
                          and r["median"] <= 0.05 and r["mean"] <= 0.10)
            n_total = sum(1 for r in per_reg if r["N"] >= 60)
            print(line + f"  avg_mean={avg:.4f}  strict {n_strict}/{n_total}  relax {n_relax}/{n_total}")

            sweeps[f"f={f_choice}_alpha={alpha}"] = {
                "f_choice": f_choice, "alpha": alpha,
                "per_regime": per_reg,
                "avg_mean_N_geq_60": avg,
                "n_strict": n_strict, "n_relaxed": n_relax, "n_total": n_total,
            }
            if avg < best_overall["mean"]:
                best_overall = {
                    "mean": avg, "f_choice": f_choice, "alpha": alpha,
                    "n_strict": n_strict, "n_relaxed": n_relax, "n_total": n_total,
                }
        print()

    print(f"BEST: f={best_overall['f_choice']}, alpha={best_overall['alpha']:+.4f}")
    print(f"  avg mean (N>=60) = {best_overall['mean']:.4f}")
    print(f"  STRICT  {best_overall['n_strict']}/{best_overall['n_total']}")
    print(f"  RELAXED {best_overall['n_relaxed']}/{best_overall['n_total']}")

    if best_overall["n_strict"] == best_overall["n_total"]:
        verdict = "ANISO_LAMBDA_CLOSURE_HOLDS_STRICT"
    elif best_overall["n_relaxed"] == best_overall["n_total"]:
        verdict = "ANISO_LAMBDA_CLOSURE_HOLDS_RELAXED"
    else:
        verdict = (f"ANISO_LAMBDA_CLOSURE_PARTIAL_"
                   f"{best_overall['n_relaxed']}_OF_{best_overall['n_total']}")
    print(f"VERDICT: {verdict}")

    out_path = REPO / "outputs" / "anisotropic_lambda_correction_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "anisotropic_lambda_largest_T_eigval_correction",
            "schema_version": "1.0.0",
            "lambda_t_struct": LAMBDA_T,
            "lambda_s_struct": LAMBDA_S,
            "form": "Lambda_ij(a) = Lambda_s * delta_ij + alpha * f(T_00/<T_00>) * v^i v^j (largest T-eigvec)",
            "f_choices": F_CHOICES,
            "alpha_values": ALPHA_VALUES,
            "sweep": sweeps,
            "best": best_overall,
            "verdict": verdict,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
