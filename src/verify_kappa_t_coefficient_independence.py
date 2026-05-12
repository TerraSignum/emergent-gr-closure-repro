"""(3) Test if kappa_t depends on the T_00 coefficient choice.

In per_seed_galerkin, T_00 is built as:
  T_00 = 0.5*Z_XI*var(Xi) + KAPPA_XI*var(|psi|) + ZETA_1*OMEGA*<|grad psi|^2> + ZETA_3*OMEGA*K_rec

Default values: Z_XI=KAPPA_XI=ZETA_1=OMEGA=1, ZETA_3=0.5, A_K=1, A_Q=0.5.

Test: if we vary these coefficients within plausible ranges and
recompute kappa_t, does the value 0.987 hold or shift?

If kappa_t is INVARIANT under coefficient choice -> fundamental
If kappa_t SHIFTS proportionally -> coefficient-induced

Output: outputs/kappa_t_coefficient_independence_audit.json
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
    edge_to_matrix, hessian_ricci_per_node, ELL_0, D_MIN, EPS_D, XI_THRESH)


# Use a representative subset
TRAIN_REGIMES = [
    ("P1", 28), ("P3", 36), ("P4", 42), ("P5", 50), ("P5N64", 64),
    ("P6", 60), ("P7", 72), ("P8", 84), ("P5N100", 100),
]

# Coefficient variants
VARIANTS = {
    "default":         {"Z": 1.0, "K": 1.0, "Z1": 1.0, "Z3": 0.5, "AK": 1.0, "AQ": 0.5, "OM": 1.0},
    "all_unit":        {"Z": 1.0, "K": 1.0, "Z1": 1.0, "Z3": 1.0, "AK": 1.0, "AQ": 1.0, "OM": 1.0},
    "Z_xi_doubled":    {"Z": 2.0, "K": 1.0, "Z1": 1.0, "Z3": 0.5, "AK": 1.0, "AQ": 0.5, "OM": 1.0},
    "K_rec_off":       {"Z": 1.0, "K": 1.0, "Z1": 1.0, "Z3": 0.0, "AK": 1.0, "AQ": 0.5, "OM": 1.0},
    "kinetic_only":    {"Z": 0.0, "K": 0.0, "Z1": 1.0, "Z3": 0.0, "AK": 1.0, "AQ": 0.5, "OM": 1.0},
    "potential_only":  {"Z": 1.0, "K": 1.0, "Z1": 0.0, "Z3": 0.5, "AK": 1.0, "AQ": 0.5, "OM": 1.0},
    "system_R_alpha":  {"Z": 0.81, "K": 0.99, "Z1": 1.0, "Z3": 0.005, "AK": 1.0, "AQ": 0.5, "OM": 1.0},  # alpha_xi^2, 1-gamma^2, gamma^2/2
}


def compute_t00_alt(xi_off, adj, weight_grad, d_mat, spatial, omega_a,
                     psi, k_field, q_field, params):
    """Re-implement t_munu_spectral.t00 with custom coefficients."""
    spatial_diff = spatial[None, :, :] - spatial[:, None, :]
    inv_d = np.where(adj > 0, 1.0 / d_mat, 0.0)
    psi_diff = psi[None, :] - psi[:, None]
    weight_term = weight_grad[:, :, None] * inv_d[:, :, None]
    grad_psi = (psi_diff[:, :, None] * spatial_diff * weight_term).sum(
        axis=1) / (omega_a[:, None] + 1e-12)
    norm_sq = (np.abs(grad_psi) ** 2).sum(axis=1)
    weight_adj = xi_off * adj
    xi_row_mean = (weight_adj.sum(axis=1) / (adj.sum(axis=1) + 1e-12))
    var_xi = (((weight_adj - xi_row_mean[:, None]) ** 2 * adj).sum(axis=1)
              / (adj.sum(axis=1) + 1e-12))
    amp_a = np.abs(psi)
    var_amp = (amp_a - amp_a.mean()) ** 2
    grad_psi_sq = norm_sq
    k_per = (k_field * adj).sum(axis=1) / (adj.sum(axis=1) + 1e-12)
    q_per = (q_field * adj).sum(axis=1) / (adj.sum(axis=1) + 1e-12)
    k_rec = params["AK"] * k_per + params["AQ"] * (1.0 - q_per)
    t00 = (0.5 * params["Z"] * var_xi
           + params["K"] * var_amp
           + params["Z1"] * params["OM"] * grad_psi_sq
           + params["Z3"] * params["OM"] * k_rec)
    return t00


def gather_kappa_t_for_variant(reg, n_lat, params):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists(): return None
    d = np.load(p, allow_pickle=True)
    e = d["dense_cell_edge_xi_values"]
    a = d["dense_cell_node_amplitude_values"]
    ph = d["dense_cell_node_phase_values"]
    g_pool, t_pool = [], []
    for s in range(min(e.shape[0], 32)):
        xi_mat = edge_to_matrix(e[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = a[s] * np.exp(1j*ph[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))

        # Compute G_00 (depends only on Hessian-Ricci, not T_00 coefficients)
        xi_off = xi_mat.copy()
        np.fill_diagonal(xi_off, 0.0)
        adj = (xi_off > XI_THRESH).astype(np.float64)
        weight_adj = xi_off * adj
        deg = weight_adj.sum(axis=1) + 1e-12
        deg_inv_sqrt = 1.0 / np.sqrt(deg)
        l_norm = (np.eye(n_lat) - deg_inv_sqrt[:, None]
                  * weight_adj * deg_inv_sqrt[None, :])
        eigvals_l, eigvecs_l = np.linalg.eigh(l_norm)
        spatial = eigvecs_l[:, 1:4]
        d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
        d_mat = np.maximum(d_mat, D_MIN)
        d_sq_safe = np.where(adj > 0, d_mat*d_mat, np.inf)
        weight_grad = np.where(adj > 0, weight_adj / (d_sq_safe + EPS_D), 0.0)
        omega_a = weight_grad.sum(axis=1)
        r_ij = hessian_ricci_per_node(xi_off, adj, d_mat, spatial, np)
        r_bar = np.trace(r_ij, axis1=1, axis2=2)
        g_00 = r_bar / 2.0

        # Compute T_00 with VARIANT coefficients
        t00 = compute_t00_alt(xi_off, adj, weight_grad, d_mat, spatial,
                                omega_a, psi, k_field, q_field, params)

        g_pool.append(g_00); t_pool.append(t00)
    g00 = np.concatenate(g_pool); t00 = np.concatenate(t_pool)
    # eps_leak = G_00 / T_00 per-node (median)
    mask = (np.abs(t00) > 0.01) & np.isfinite(t00) & np.isfinite(g00)
    if not np.any(mask): return None
    g00 = g00[mask]; t00 = t00[mask]
    eps_pool = g00 / t00
    return {"regime": reg, "N": n_lat,
            "T_00_med": float(np.median(t00)),
            "G_00_med": float(np.median(g00)),
            "eps_leak_med": float(np.median(eps_pool)),
            "kappa_t": 1.0 - float(np.median(eps_pool))}


def main() -> int:
    print("=" * 100)
    print("(3) Coefficient-independence test for kappa_t")
    print("=" * 100)
    print()
    print(f"{'variant':<20} | " + " | ".join([f"{r[0]:<9}" for r in TRAIN_REGIMES]))
    print(f"{'kappa_t per regime':>20} |")
    print("-" * 110)

    out = {}
    for vname, params in VARIANTS.items():
        kappas = []
        line = f"{vname:<20} |"
        for reg, n_lat in TRAIN_REGIMES:
            r = gather_kappa_t_for_variant(reg, n_lat, params)
            if r is None:
                line += f" {'--':>9} |"
                continue
            kappas.append(r["kappa_t"])
            line += f" {r['kappa_t']:>9.4f} |"
        if kappas:
            mean_k = float(np.mean(kappas))
            std_k = float(np.std(kappas))
            cv = std_k / abs(mean_k) * 100 if abs(mean_k) > 1e-12 else float("nan")
            line += f"  mean={mean_k:.4f}, CV={cv:.1f}%"
            out[vname] = {"params": params, "kappas": kappas,
                          "mean_kappa": mean_k, "std": std_k, "CV_percent": cv}
        print(line)

    # Stability across variants: how much does mean_kappa shift?
    print()
    means = [v["mean_kappa"] for v in out.values()]
    if means:
        spread = max(means) - min(means)
        print(f"Spread of mean_kappa across {len(means)} variants: {spread:.4f}")
        print(f"  min = {min(means):.4f}")
        print(f"  max = {max(means):.4f}")
        print(f"  range = {spread:.4f}")
        if spread < 0.02:
            verdict = "KAPPA_T_INVARIANT_UNDER_COEFFICIENT_CHOICE"
        elif spread < 0.10:
            verdict = "KAPPA_T_WEAKLY_DEPENDENT"
        else:
            verdict = "KAPPA_T_STRONGLY_COEFFICIENT_DEPENDENT"
        print(f"\nVERDICT: {verdict}")

    out_path = REPO / "outputs" / "kappa_t_coefficient_independence_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
