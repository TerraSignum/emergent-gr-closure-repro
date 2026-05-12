"""(Solution-5) Characterize what drives finite-N 2+1 enhancement.

If the 2+1 sign asymmetry is a finite-N effect, what microscopic
property correlates with it? Test correlations between per-node
2+1-strength and:
  - local connectivity (degree)
  - local T_00 magnitude (heavy-tail indicator)
  - local Xi-row-mean (correlation density)
  - local |psi|^2 (amplitude density)

If 2+1 strength correlates strongly with one of these → physical
explanation. If random → discretization artifact independent of
physics.

Output: outputs/2plus1_finite_N_drivers_audit.json
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
    ("P1", 28, "canonical"),
    ("P5", 50, "canonical"),
    ("P5N64", 64, "canonical"),
    ("P5N72", 72, "canonical"),
    ("P5N84", 84, "canonical"),
    ("P5N100", 100, "canonical"),
    ("P5N128", 128, "canonical"),
    ("P5N200", 200, "canonical"),
    ("P5N256", 256, "canonical"),
    ("P5N300", 300, "canonical"),
    ("P5N512", 512, "canonical"),
]


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
    return t00, coeff*gradgrad - iso_term, var_xi, xi_row_mean, k_per


def per_seed_correlate(reg, n_lat, p):
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

    aniso_pool, deg_pool, t00_pool, xi_mean_pool, psi2_pool = [], [], [], [], []
    for s in range(n_seeds):
        xi_mat = xi_seed(s); np.fill_diagonal(xi_mat, 1.0)
        psi = psi_seed(s)
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        xi_off = xi_mat.copy(); np.fill_diagonal(xi_off, 0.0)
        adj = (xi_off > XI_THRESH).astype(np.float64)
        weight_adj = xi_off * adj; deg = adj.sum(axis=1)
        deg_safe = deg + 1e-12
        deg_inv_sqrt = 1.0/np.sqrt(weight_adj.sum(axis=1) + 1e-12)
        l_norm = np.eye(n_lat) - deg_inv_sqrt[:, None]*weight_adj*deg_inv_sqrt[None, :]
        eigvals_l, eigvecs_l = np.linalg.eigh(l_norm)
        spatial = eigvecs_l[:, 1:4]
        d_mat = -ELL_0*np.log(np.maximum(xi_off, 1e-12)); d_mat = np.maximum(d_mat, D_MIN)
        d_sq_safe = np.where(adj > 0, d_mat*d_mat, np.inf)
        weight_grad = np.where(adj > 0, weight_adj/(d_sq_safe + EPS_D), 0.0)
        omega_a = weight_grad.sum(axis=1)
        r_ij = hessian_ricci_per_node(xi_off, adj, d_mat, spatial, np)
        r_bar = np.trace(r_ij, axis1=1, axis2=2); g_00 = r_bar/2.0
        g_ij = r_ij - 0.5*r_bar[:, None, None]*np.eye(3)[None, :, :]
        t00, t_ij, var_xi, xi_row_mean, k_per = t_munu_3x3(
            xi_off, adj, weight_grad, d_mat, spatial, omega_a, psi, k_field, q_field)
        mask = (t00 > 0.05) & np.isfinite(t00) & np.isfinite(g_00)
        psi2_per = np.abs(psi)**2

        for a_idx in np.where(mask)[0]:
            T_loc = 0.5*(t_ij[a_idx] + t_ij[a_idx].T)
            G_loc = 0.5*(g_ij[a_idx] + g_ij[a_idx].T)
            try: ew, ev = np.linalg.eigh(T_loc)
            except np.linalg.LinAlgError: continue
            G_eig = ev.T @ G_loc @ ev
            G_diag = np.diag(G_eig)
            Ls = ew - G_diag  # ascending
            # Anisotropy strength: max(Ls) - min(Ls), normalized by mean magnitude
            aniso = float(np.max(Ls) - np.min(Ls))
            aniso_pool.append(aniso)
            deg_pool.append(float(deg[a_idx]))
            t00_pool.append(float(t00[a_idx]))
            xi_mean_pool.append(float(xi_row_mean[a_idx]))
            psi2_pool.append(float(psi2_per[a_idx]))

    if not aniso_pool: return None
    aniso = np.array(aniso_pool); deg = np.array(deg_pool)
    t00s = np.array(t00_pool); xis = np.array(xi_mean_pool); psi2s = np.array(psi2_pool)

    # Compute Spearman correlations
    from scipy.stats import spearmanr
    corr_deg = spearmanr(aniso, deg).correlation if len(aniso) > 5 else float('nan')
    corr_t00 = spearmanr(aniso, t00s).correlation if len(aniso) > 5 else float('nan')
    corr_xi = spearmanr(aniso, xis).correlation if len(aniso) > 5 else float('nan')
    corr_psi = spearmanr(aniso, psi2s).correlation if len(aniso) > 5 else float('nan')
    return {
        "regime": reg, "N": int(n_lat),
        "n_nodes": int(len(aniso)),
        "aniso_mean": float(aniso.mean()),
        "spearman_corr_with_degree": float(corr_deg),
        "spearman_corr_with_t00": float(corr_t00),
        "spearman_corr_with_xi_rowmean": float(corr_xi),
        "spearman_corr_with_psi_squared": float(corr_psi),
    }


def main() -> int:
    print("="*100)
    print("(Solution-5) What drives finite-N 2+1 anisotropy?")
    print("="*100)
    print()
    rows = []
    for reg, n, src in SUBSET:
        p = get_path(reg, src)
        r = per_seed_correlate(reg, n, p)
        if r is None: continue
        rows.append(r)

    print(f"{'reg':<10} {'N':>4} {'n_nodes':>7} | {'aniso_mean':>10} | {'corr w/ degree':>14} {'corr w/ T_00':>13} {'corr w/ Xi':>10} {'corr w/ psi²':>12}")
    print("-"*110)
    for r in rows:
        print(f"{r['regime']:<10} {r['N']:>4} {r['n_nodes']:>7} | {r['aniso_mean']:>10.4f} | "
              f"{r['spearman_corr_with_degree']:>+14.3f} {r['spearman_corr_with_t00']:>+13.3f} "
              f"{r['spearman_corr_with_xi_rowmean']:>+10.3f} {r['spearman_corr_with_psi_squared']:>+12.3f}")

    if rows:
        print()
        print("Cross-regime mean Spearman correlations:")
        for key in ["spearman_corr_with_degree", "spearman_corr_with_t00",
                   "spearman_corr_with_xi_rowmean", "spearman_corr_with_psi_squared"]:
            vals = [r[key] for r in rows if np.isfinite(r[key])]
            print(f"  {key:<35}: mean = {np.mean(vals):+.3f}, max abs = {max(abs(v) for v in vals):.3f}")
        print()
        print("Interpretation: |correlation| > 0.3 = meaningful driver; |corr| < 0.1 = no relation")

    out = {"method": "2plus1_finite_N_drivers", "per_regime": rows}
    out_path = REPO / "outputs" / "2plus1_finite_N_drivers_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
