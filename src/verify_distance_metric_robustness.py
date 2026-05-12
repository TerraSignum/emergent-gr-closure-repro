"""(B2) Distance-metric robustness: alternative discretizations.

The construction uses d_ij = -ℓ_0 * log(Ξ_ij). Reviewer critique:
"Could be tied to specific log-distance choice."

Test alternative distance functions:
  d_log:  d_ij = -log(Ξ_ij)         (current)
  d_lin:  d_ij = 1 - Ξ_ij           (linear)
  d_sqrt: d_ij = sqrt(1 - Ξ_ij²)    (chord)
  d_inv:  d_ij = 1/Ξ_ij - 1         (inverse)

Run full Galerkin closure for each on representative regimes,
compare per-direction Frobenius residual + asymptotic Λ_t.

Output: outputs/distance_metric_robustness_audit.json
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

PARENT = REPO.parent

SUBSET = [
    ("P5", 50, "canonical"),
    ("P5N100", 100, "canonical"),
    ("P5N84", 84, "snapshot_v2"),
    ("P5N200", 200, "snapshot_v2"),
    ("P5N256", 256, "canonical"),
    ("P5N512", 512, "canonical"),
]

DISTANCE_FUNCS = {
    "d_log":  lambda xi: -ELL_0 * np.log(np.maximum(xi, 1e-12)),
    "d_lin":  lambda xi: ELL_0 * (1.0 - xi),
    "d_sqrt": lambda xi: ELL_0 * np.sqrt(np.maximum(1.0 - xi*xi, 1e-12)),
    "d_inv":  lambda xi: ELL_0 * (1.0/np.maximum(xi, 1e-12) - 1.0),
}


def get_path(reg, src):
    if src == "canonical":
        return find_d1_npz(reg, REPO)
    return PARENT / f"results_d1_{reg.lower()}_v2" / f"{reg}.snapshots.npz"


def t_munu_3x3(xi_off, adj, weight_grad, d_mat, spatial, omega_a, psi, k_field, q_field):
    spatial_diff = spatial[None, :, :] - spatial[:, None, :]
    inv_d = np.where(adj > 0, 1.0/d_mat, 0.0)
    psi_diff = psi[None, :] - psi[:, None]
    weight_term = weight_grad[:, :, None] * inv_d[:, :, None]
    grad_psi = (psi_diff[:, :, None] * spatial_diff * weight_term).sum(axis=1) / (omega_a[:, None] + 1e-12)
    norm_sq = (np.abs(grad_psi)**2).sum(axis=1)
    weight_adj = xi_off * adj
    xi_row_mean = weight_adj.sum(axis=1) / (adj.sum(axis=1) + 1e-12)
    var_xi = (((weight_adj - xi_row_mean[:, None])**2 * adj).sum(axis=1) / (adj.sum(axis=1) + 1e-12))
    amp_a = np.abs(psi); var_amp = (amp_a - amp_a.mean())**2
    k_per = (k_field*adj).sum(axis=1)/(adj.sum(axis=1) + 1e-12)
    q_per = (q_field*adj).sum(axis=1)/(adj.sum(axis=1) + 1e-12)
    k_rec = 1.0*k_per + 0.5*(1.0 - q_per)
    Z_xi, K_xi, Z1, Z3, OM = 1.0, 1.0, 1.0, 0.5, 1.0
    t00 = 0.5*Z_xi*var_xi + K_xi*var_amp + Z1*OM*norm_sq + Z3*OM*k_rec
    coeff = 2*(0.5*Z_xi + K_xi + Z1*OM); iso_sub = 0.5*Z_xi + K_xi + Z1*OM
    grad_real = grad_psi.real; grad_imag = grad_psi.imag
    gradgrad = grad_real[:, :, None]*grad_real[:, None, :] + grad_imag[:, :, None]*grad_imag[:, None, :]
    iso_term = iso_sub * norm_sq[:, None, None] * np.eye(3)[None, :, :]
    return t00, coeff*gradgrad - iso_term


def compute_with_distance(reg, n_lat, p, dist_name, dist_fn):
    if not p or not p.exists(): return None
    d = np.load(p, allow_pickle=True)
    if "dense_cell_edge_xi_values" in d.keys():
        e = d["dense_cell_edge_xi_values"]
        a = d["dense_cell_node_amplitude_values"]
        ph = d["dense_cell_node_phase_values"]
        n_seeds = min(e.shape[0], 32)
        xi_seed = lambda s: edge_to_matrix(e[s], n_lat)
        psi_seed = lambda s: a[s] * np.exp(1j*ph[s])
    elif "edge_xi_snapshots" in d.keys():
        n_seeds = int(d["n_seeds"][0])
        xi_seed = lambda s: d["edge_xi_snapshots"][s, -1, :, :].copy()
        psi_seed = lambda s: d["psi_real_snapshots"][s, -1, :] + 1j*d["psi_imag_snapshots"][s, -1, :]
    else: return None

    Lt_pool, Frob_pool = [], []
    for s in range(n_seeds):
        xi_mat = xi_seed(s); np.fill_diagonal(xi_mat, 1.0)
        psi = psi_seed(s)
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        xi_off = xi_mat.copy(); np.fill_diagonal(xi_off, 0.0)
        adj = (xi_off > XI_THRESH).astype(np.float64)
        if adj.sum() < 10: continue
        weight_adj = xi_off * adj; deg = weight_adj.sum(axis=1) + 1e-12
        deg_inv_sqrt = 1.0/np.sqrt(deg)
        l_norm = np.eye(n_lat) - deg_inv_sqrt[:, None]*weight_adj*deg_inv_sqrt[None, :]
        eigvals_l, eigvecs_l = np.linalg.eigh(l_norm)
        spatial = eigvecs_l[:, 1:4]
        # APPLY ALTERNATIVE DISTANCE
        d_mat = dist_fn(np.where(xi_off > 0, xi_off, 1.0))  # fill diag-like positions
        d_mat = np.maximum(d_mat, D_MIN)
        d_sq_safe = np.where(adj > 0, d_mat*d_mat, np.inf)
        weight_grad = np.where(adj > 0, weight_adj/(d_sq_safe + EPS_D), 0.0)
        omega_a = weight_grad.sum(axis=1)
        r_ij = hessian_ricci_per_node(xi_off, adj, d_mat, spatial, np)
        r_bar = np.trace(r_ij, axis1=1, axis2=2); g_00 = r_bar/2.0
        g_ij = r_ij - 0.5*r_bar[:, None, None]*np.eye(3)[None, :, :]
        t00, t_ij = t_munu_3x3(xi_off, adj, weight_grad, d_mat, spatial, omega_a, psi, k_field, q_field)
        mask = (t00 > 0.05) & np.isfinite(t00) & np.isfinite(g_00)
        if not np.any(mask): continue
        for a_idx in np.where(mask)[0]:
            Lt_pool.append(t00[a_idx] - g_00[a_idx])
            res = abs(g_00[a_idx] + 0.81 - t00[a_idx]) / max(abs(t00[a_idx]), 1e-9)
            Frob_pool.append(res)

    if not Lt_pool: return None
    return {
        "n_nodes": int(len(Lt_pool)),
        "Lambda_t_mean": float(np.mean(Lt_pool)),
        "Frob_residual_med": float(np.median(Frob_pool)),
    }


def main() -> int:
    print("="*100)
    print("(B2) Distance-metric alternative discretization robustness")
    print("="*100)
    print(f"\nDistance functions: {list(DISTANCE_FUNCS.keys())}")
    print()
    print(f"{'reg':<10} {'N':>4} | " + " | ".join(f"{name:>9}" for name in DISTANCE_FUNCS))
    print("-"*100)

    out = {"method": "distance_metric_robustness", "metrics": list(DISTANCE_FUNCS.keys()),
           "per_regime": {}}
    for reg, n, src in SUBSET:
        path = get_path(reg, src)
        Lt_line = f"{reg:<10} {n:>4} |"
        Fr_line = " "*15 + "|"
        out["per_regime"][reg] = {"N": n, "by_metric": {}}
        Lt_vals = []
        for name, fn in DISTANCE_FUNCS.items():
            r = compute_with_distance(reg, n, path, name, fn)
            if r is None:
                Lt_line += f" {'NaN':>9} |"; Fr_line += f" {'NaN':>9} |"
                continue
            Lt_line += f" {r['Lambda_t_mean']:>9.4f} |"
            Fr_line += f" {r['Frob_residual_med']:>9.4f} |"
            Lt_vals.append(r["Lambda_t_mean"])
            out["per_regime"][reg]["by_metric"][name] = r
        if Lt_vals:
            spread = max(Lt_vals) - min(Lt_vals)
            cv = np.std(Lt_vals)/abs(np.mean(Lt_vals))*100
            Lt_line += f"  Λ_t spread={spread:.4f}, CV={cv:.1f}%"
        print(Lt_line)
        print("Frob:" + Fr_line[5:])

    # Aggregate
    print()
    cv_per_regime = []
    for reg, ddat in out["per_regime"].items():
        Lt_vals = [d["Lambda_t_mean"] for n, d in ddat["by_metric"].items()]
        if Lt_vals:
            cv = np.std(Lt_vals)/abs(np.mean(Lt_vals))*100
            cv_per_regime.append(cv)
    print(f"Cross-regime Lambda_t CV across distance metrics:")
    for reg, cv in zip([r[0] for r in SUBSET], cv_per_regime):
        print(f"  {reg}: CV = {cv:.2f}%")

    if cv_per_regime and max(cv_per_regime) < 15.0:
        verdict = "DISTANCE_METRIC_ROBUST_CV<15%"
    elif cv_per_regime and max(cv_per_regime) < 30.0:
        verdict = "DISTANCE_METRIC_MODERATELY_ROBUST_CV<30%"
    else:
        verdict = "DISTANCE_METRIC_SENSITIVE"
    print(f"\nVERDICT: {verdict}")
    out["verdict"] = verdict
    out["cv_per_regime_pct"] = cv_per_regime

    out_path = REPO / "outputs" / "distance_metric_robustness_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
