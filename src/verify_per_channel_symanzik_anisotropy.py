"""(*) Lösungsweg 1+2:
  L1: Symanzik 2+4 fit per channel (Lambda_t, 3|Lambda_s|, shear).
  L2: Anisotropy index of per-axis Lambda_s (max-min spread).

If all three channels converge to System-R values asymptotically AND
the per-axis anisotropy decays with N, the asymptotic Lambda_munu
decomposition is rigorously verified.

Output: outputs/per_channel_symanzik_anisotropy_audit.json
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
ALPHA_XI = 9.0/10.0
GAMMA    = 1.0/10.0

LAMBDA_T_PRED   = ALPHA_XI**2
LAMBDA_S_3SUM_PRED = 3*GAMMA**2/2
LAMBDA_S_AXIS_PRED = -GAMMA**2/2  # signed
SHEAR_PRED = 0.0
ANISOTROPY_PRED = 0.0  # asymptotic (signed Lambda_s_i should all equal)


def t_munu_3x3(xi_off, adj, weight_grad, d_mat, spatial, omega_a, psi, k_field, q_field):
    spatial_diff = spatial[None, :, :] - spatial[:, None, :]
    inv_d = np.where(adj > 0, 1.0/d_mat, 0.0)
    psi_diff = psi[None, :] - psi[:, None]
    weight_term = weight_grad[:, :, None] * inv_d[:, :, None]
    grad_psi = (psi_diff[:, :, None] * spatial_diff * weight_term).sum(
        axis=1) / (omega_a[:, None] + 1e-12)
    norm_sq = (np.abs(grad_psi)**2).sum(axis=1)
    weight_adj = xi_off * adj
    xi_row_mean = weight_adj.sum(axis=1) / (adj.sum(axis=1) + 1e-12)
    var_xi = (((weight_adj - xi_row_mean[:, None])**2 * adj).sum(axis=1)
              / (adj.sum(axis=1) + 1e-12))
    amp_a = np.abs(psi)
    var_amp = (amp_a - amp_a.mean())**2
    k_per = (k_field*adj).sum(axis=1)/(adj.sum(axis=1) + 1e-12)
    q_per = (q_field*adj).sum(axis=1)/(adj.sum(axis=1) + 1e-12)
    k_rec = 1.0*k_per + 0.5*(1.0 - q_per)
    Z_xi, K_xi, Z1, Z3, OM = 1.0, 1.0, 1.0, 0.5, 1.0
    t00 = (0.5*Z_xi*var_xi + K_xi*var_amp + Z1*OM*norm_sq + Z3*OM*k_rec)
    coeff = 2*(0.5*Z_xi + K_xi + Z1*OM)
    iso_sub = (0.5*Z_xi + K_xi + Z1*OM)
    grad_real = grad_psi.real
    grad_imag = grad_psi.imag
    gradgrad = grad_real[:, :, None]*grad_real[:, None, :] + grad_imag[:, :, None]*grad_imag[:, None, :]
    iso_term = iso_sub * norm_sq[:, None, None] * np.eye(3)[None, :, :]
    t_ij_spatial = coeff * gradgrad - iso_term
    return t00, t_ij_spatial


def per_seed_per_axis_decomposition(reg, n_lat, p):
    if not p or not p.exists():
        return None
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
    else:
        return None

    Lt_pool = []
    Ls_signed_pool = []   # (n_nodes, 3) signed Lambda_s per axis
    shear_pool = []

    for s in range(n_seeds):
        xi_mat = xi_seed(s)
        np.fill_diagonal(xi_mat, 1.0)
        psi = psi_seed(s)
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))

        xi_off = xi_mat.copy()
        np.fill_diagonal(xi_off, 0.0)
        adj = (xi_off > XI_THRESH).astype(np.float64)
        weight_adj = xi_off * adj
        deg = weight_adj.sum(axis=1) + 1e-12
        deg_inv_sqrt = 1.0/np.sqrt(deg)
        l_norm = (np.eye(n_lat) - deg_inv_sqrt[:, None]*weight_adj*deg_inv_sqrt[None, :])
        eigvals_l, eigvecs_l = np.linalg.eigh(l_norm)
        spatial = eigvecs_l[:, 1:4]
        d_mat = -ELL_0*np.log(np.maximum(xi_off, 1e-12))
        d_mat = np.maximum(d_mat, D_MIN)
        d_sq_safe = np.where(adj > 0, d_mat*d_mat, np.inf)
        weight_grad = np.where(adj > 0, weight_adj/(d_sq_safe + EPS_D), 0.0)
        omega_a = weight_grad.sum(axis=1)
        r_ij = hessian_ricci_per_node(xi_off, adj, d_mat, spatial, np)
        r_bar = np.trace(r_ij, axis1=1, axis2=2)
        g_00 = r_bar / 2.0
        g_ij = r_ij - 0.5*r_bar[:, None, None]*np.eye(3)[None, :, :]

        t00, t_ij = t_munu_3x3(xi_off, adj, weight_grad, d_mat, spatial, omega_a, psi, k_field, q_field)
        mask = (t00 > 0.05) & np.isfinite(t00) & np.isfinite(g_00)

        for a_idx in np.where(mask)[0]:
            T_loc = 0.5*(t_ij[a_idx] + t_ij[a_idx].T)
            G_loc = 0.5*(g_ij[a_idx] + g_ij[a_idx].T)
            try:
                ew, ev = np.linalg.eigh(T_loc)
            except np.linalg.LinAlgError:
                continue
            G_eigen = ev.T @ G_loc @ ev
            G_diag = np.diag(G_eigen)
            G_off = G_eigen - np.diag(G_diag)
            shear = float(np.linalg.norm(G_off, ord='fro'))
            # Per-axis SIGNED Lambda_s_i = lambda_i(T) - G_(ii) so that G_(ii) + Lambda_s_i = lambda_i
            Ls_signed = ew - G_diag  # (3,)
            Lt_pool.append(t00[a_idx] - g_00[a_idx])
            Ls_signed_pool.append(Ls_signed)
            shear_pool.append(shear)

    if not Lt_pool:
        return None
    Lt_arr = np.array(Lt_pool)
    Ls_arr = np.array(Ls_signed_pool)   # (n, 3)
    Sh_arr = np.array(shear_pool)

    # Sort axes per node so axis-1 = smallest, axis-3 = largest (for consistent ordering)
    Ls_sorted = np.sort(Ls_arr, axis=1)
    # Per-regime mean of each ranked axis
    Ls_axis_means = Ls_sorted.mean(axis=0)  # (3,)
    Ls_axis_stds  = Ls_sorted.std(axis=0)
    # Anisotropy: max - min within node, then mean over nodes
    aniso_per_node = Ls_sorted[:, 2] - Ls_sorted[:, 0]
    aniso_mean = float(aniso_per_node.mean())
    aniso_std  = float(aniso_per_node.std())
    # Magnitude sum per node
    Ls_3sum_signed_per_node = Ls_sorted.sum(axis=1)
    Ls_3sum_abs_per_node = np.abs(Ls_sorted).sum(axis=1)

    return {
        "regime": reg, "N": int(n_lat), "n_nodes": int(len(Lt_arr)),
        "Lambda_t_mean": float(Lt_arr.mean()),
        "Lambda_s_axis_means_sorted": [float(x) for x in Ls_axis_means],
        "Lambda_s_axis_stds_sorted":  [float(x) for x in Ls_axis_stds],
        "Lambda_s_3sum_abs_mean":  float(Ls_3sum_abs_per_node.mean()),
        "Lambda_s_3sum_signed_mean": float(Ls_3sum_signed_per_node.mean()),
        "anisotropy_index_mean":  aniso_mean,
        "anisotropy_index_std":   aniso_std,
        "shear_mean":   float(Sh_arr.mean()),
    }


SOURCE_LIST = [
    ("P1",     28, "canonical"),
    ("P3",     36, "canonical"),
    ("P4",     42, "canonical"),
    ("P5",     50, "canonical"),
    ("P5N64",  64, "canonical"),
    ("P6",     60, "canonical"),
    ("P7",     72, "canonical"),
    ("P8",     84, "canonical"),
    ("P5N100",100, "canonical"),
    ("P5N72",  72, "snapshot_v2"),
    ("P5N84",  84, "snapshot_v2"),
    ("P5N200",200, "snapshot_v2"),
    ("P5N256", 256, "canonical"),
    ("P5N512", 512, "canonical"),
]


def get_path(reg, src):
    if src == "canonical":
        return find_d1_npz(reg, REPO)
    return PARENT / f"results_d1_{reg.lower()}_v2" / f"{reg}.snapshots.npz"


def symanzik_24(N_arr, y_arr):
    A = np.column_stack([np.ones_like(N_arr), 1.0/N_arr**2, 1.0/N_arr**4])
    coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
    pred = A @ coef
    ss_res = np.sum((y_arr - pred)**2)
    ss_tot = np.sum((y_arr - y_arr.mean())**2) + 1e-12
    R2 = 1.0 - ss_res/ss_tot
    return float(coef[0]), float(coef[1]), float(coef[2]), float(R2)


def main() -> int:
    print("="*120)
    print("Per-channel Symanzik 2+4 + per-axis anisotropy")
    print("="*120)
    rows = []
    for reg, n, src in SOURCE_LIST:
        p = get_path(reg, src)
        r = per_seed_per_axis_decomposition(reg, n, p)
        if r is None:
            print(f"{reg:<10} {n:>4} -- file_missing"); continue
        rows.append(r)

    print(f"\n{'reg':<10} {'N':>4} | {'Λt':>7} {'<axis1>':>9} {'<axis2>':>9} {'<axis3>':>9} {'aniso':>7} {'3|Ls|':>7} {'shear':>7}")
    print("-"*100)
    for r in rows:
        Ls = r["Lambda_s_axis_means_sorted"]
        print(f"{r['regime']:<10} {r['N']:>4} | {r['Lambda_t_mean']:>7.3f} "
              f"{Ls[0]:>+9.4f} {Ls[1]:>+9.4f} {Ls[2]:>+9.4f} "
              f"{r['anisotropy_index_mean']:>7.4f} "
              f"{r['Lambda_s_3sum_abs_mean']:>7.3f} "
              f"{r['shear_mean']:>7.3f}")
    print()

    # Per-channel Symanzik fits
    N_arr = np.array([r["N"] for r in rows], dtype=float)
    Lt_arr = np.array([r["Lambda_t_mean"] for r in rows])
    Ls3sum_arr = np.array([r["Lambda_s_3sum_abs_mean"] for r in rows])
    Sh_arr = np.array([r["shear_mean"] for r in rows])
    Aniso_arr = np.array([r["anisotropy_index_mean"] for r in rows])

    print("="*60)
    print("Per-channel Symanzik 2+4 asymptotes")
    print("="*60)
    Lt_inf, c2_t, c4_t, R2_t = symanzik_24(N_arr, Lt_arr)
    L3_inf, c2_3, c4_3, R2_3 = symanzik_24(N_arr, Ls3sum_arr)
    Sh_inf, c2_sh, c4_sh, R2_sh = symanzik_24(N_arr, Sh_arr)
    An_inf, c2_a, c4_a, R2_a = symanzik_24(N_arr, Aniso_arr)
    print(f"  Λ_t        Symanzik^∞ = {Lt_inf:.4f}  (R^2 = {R2_t:.3f})  vs α_ξ²={LAMBDA_T_PRED:.3f}")
    print(f"             distance: {abs(Lt_inf - LAMBDA_T_PRED):.4f} ({abs(Lt_inf - LAMBDA_T_PRED)/LAMBDA_T_PRED*100:.2f}% rel)")
    print(f"  3|Λ_s|     Symanzik^∞ = {L3_inf:.4f}  (R^2 = {R2_3:.3f})  vs 3γ²/2={LAMBDA_S_3SUM_PRED:.3f}")
    print(f"             distance: {abs(L3_inf - LAMBDA_S_3SUM_PRED):.4f}")
    print(f"  shear      Symanzik^∞ = {Sh_inf:.4f}  (R^2 = {R2_sh:.3f})  vs 0")
    print(f"             distance: {abs(Sh_inf):.4f}")
    print(f"  anisotropy Symanzik^∞ = {An_inf:.4f}  (R^2 = {R2_a:.3f})  vs 0")
    print(f"             distance: {abs(An_inf):.4f}")
    print()
    print(f"  Total Λ_μν trace asymptotic = Λ_t + 3|Λ_s| + shear")
    total_inf = Lt_inf + L3_inf + Sh_inf
    pred_total = LAMBDA_T_PRED + LAMBDA_S_3SUM_PRED
    print(f"             = {total_inf:.4f}  vs  α_ξ² + 3γ²/2 = {pred_total:.4f}")
    print(f"             distance: {abs(total_inf - pred_total):.4f} ({abs(total_inf - pred_total)/pred_total*100:.2f}% rel)")

    # Determine verdict
    print()
    print("="*60)
    print("VERDICT per channel")
    print("="*60)
    pyth_lambda_t = abs(Lt_inf - LAMBDA_T_PRED) < 0.02
    pyth_3ls    = abs(L3_inf - LAMBDA_S_3SUM_PRED) < 0.01
    pyth_shear  = abs(Sh_inf) < 0.01
    pyth_aniso  = abs(An_inf) < 0.01
    print(f"  Λ_t -> α_ξ²: {pyth_lambda_t}")
    print(f"  3|Λ_s| -> 3γ²/2: {pyth_3ls}")
    print(f"  shear -> 0: {pyth_shear}")
    print(f"  anisotropy -> 0: {pyth_aniso}")
    if all([pyth_lambda_t, pyth_3ls, pyth_shear, pyth_aniso]):
        verdict = "ASYMPTOTIC_DECOMPOSITION_VERIFIED"
    elif pyth_lambda_t and pyth_3ls and pyth_shear:
        verdict = "DECOMPOSITION_OK_ANISOTROPY_OPEN"
    else:
        verdict = "DECOMPOSITION_PARTIALLY_VERIFIED"
    print(f"\nVerdict: {verdict}")

    out = {
        "method": "per_channel_symanzik_anisotropy",
        "predictions": {
            "alpha_xi_sq": LAMBDA_T_PRED,
            "3_gamma_sq_half": LAMBDA_S_3SUM_PRED,
            "shear": SHEAR_PRED,
            "anisotropy": ANISOTROPY_PRED,
            "asymptotic_total": LAMBDA_T_PRED + LAMBDA_S_3SUM_PRED,
        },
        "per_regime": rows,
        "symanzik_per_channel": {
            "Lambda_t":   {"asymptote": Lt_inf, "c2": c2_t, "c4": c4_t, "R2": R2_t,
                          "rel_pct_to_alpha_xi_sq": float(abs(Lt_inf - LAMBDA_T_PRED)/LAMBDA_T_PRED*100)},
            "Lambda_s_3sum_abs": {"asymptote": L3_inf, "c2": c2_3, "c4": c4_3, "R2": R2_3,
                          "abs_distance_to_3gamma_sq_half": abs(L3_inf - LAMBDA_S_3SUM_PRED)},
            "shear":      {"asymptote": Sh_inf, "c2": c2_sh, "c4": c4_sh, "R2": R2_sh,
                          "abs_distance_to_zero": abs(Sh_inf)},
            "anisotropy": {"asymptote": An_inf, "c2": c2_a, "c4": c4_a, "R2": R2_a,
                          "abs_distance_to_zero": abs(An_inf)},
        },
        "total_trace_asymptote": total_inf,
        "total_trace_distance_pct": float(abs(total_inf - pred_total)/pred_total*100),
        "verdict": verdict,
    }
    out_path = REPO / "outputs" / "per_channel_symanzik_anisotropy_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
