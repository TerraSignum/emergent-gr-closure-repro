"""(*) Unit-budget redistribution test:
At finite N the C1^2 unit (alpha_xi^2 + 2*alpha_xi*gamma + gamma^2 = 1)
should partition Lambda_munu as
  - Lambda_t       -> alpha_xi^2 (asymptotic)
  - 3*|Lambda_s|   -> ~3*gamma^2/2
  - shear ||T_off|| -> ~2*alpha_xi*gamma (decaying Symanzik with N)

We measure directly:
  Lambda_t*(N)         = mean(T_00 - G_00)
  3|Lambda_s|*(N)      = mean over 3 axes of |lambda_i - G_(ii)|
  shear_off(N)         = mean ||(U^T G U)_off||_F  (per-node)

and the budget closure
  budget(N) = Lambda_t* + 3*|Lambda_s|* + shear_off

Asymptotic prediction:
  Lambda_t* -> 0.810 (alpha_xi^2)
  3*|Lambda_s|* -> 0.015 (3*gamma^2/2)
  shear_off -> 0 (Symanzik-decay)
  budget -> 0.825 (asymptotic; cross-term has migrated to shear at finite N
                  but decays in continuum, leaving net 0.825)

OR (alternative reading, full unit closure):
  At all N, Lambda_t* + 3*|Lambda_s|* + shear_off ~ 1 (cross-term
  is in shear and contributes 2*alpha_xi*gamma = 0.18 at finite N,
  decaying to 0 at infinite N).

Output: outputs/unit_budget_redistribution_audit.json
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

# Predictions
LAMBDA_T_PRED   = ALPHA_XI**2     # 0.810
LAMBDA_S_PRED   = -GAMMA**2/2     # -0.005 per axis
SHEAR_PRED_FINITE = 2*ALPHA_XI*GAMMA  # 0.180 expected at small N
TRACE_PRED_BUDGET = LAMBDA_T_PRED + 3*abs(LAMBDA_S_PRED)  # 0.825


def t_munu_3x3_per_node(xi_off, adj, weight_grad, d_mat, spatial,
                       omega_a, psi, k_field, q_field):
    """Build the per-node 3x3 spatial T_ij tensor and time T_00.

    Replicate t_munu_spectral logic:
    """
    n = xi_off.shape[0]
    spatial_diff = spatial[None, :, :] - spatial[:, None, :]
    inv_d = np.where(adj > 0, 1.0/d_mat, 0.0)
    psi_diff = psi[None, :] - psi[:, None]
    weight_term = weight_grad[:, :, None] * inv_d[:, :, None]
    grad_psi = (psi_diff[:, :, None] * spatial_diff * weight_term).sum(
        axis=1) / (omega_a[:, None] + 1e-12)
    norm_sq = (np.abs(grad_psi)**2).sum(axis=1)

    weight_adj = xi_off * adj
    xi_row_mean = (weight_adj.sum(axis=1) / (adj.sum(axis=1) + 1e-12))
    var_xi = (((weight_adj - xi_row_mean[:, None])**2 * adj).sum(axis=1)
              / (adj.sum(axis=1) + 1e-12))
    amp_a = np.abs(psi)
    var_amp = (amp_a - amp_a.mean())**2

    k_per = (k_field*adj).sum(axis=1) / (adj.sum(axis=1) + 1e-12)
    q_per = (q_field*adj).sum(axis=1) / (adj.sum(axis=1) + 1e-12)
    k_rec = 1.0*k_per + 0.5*(1.0 - q_per)
    Z_xi, K_xi, Z1, Z3, OM = 1.0, 1.0, 1.0, 0.5, 1.0

    t00 = (0.5*Z_xi*var_xi + K_xi*var_amp + Z1*OM*norm_sq + Z3*OM*k_rec)
    coeff = 2*(0.5*Z_xi + K_xi + Z1*OM)
    iso_sub = (0.5*Z_xi + K_xi + Z1*OM)

    # Per-node 3x3 spatial tensor: t_ij^a
    grad_real = grad_psi.real  # (n, 3)
    grad_imag = grad_psi.imag
    # outer product per node: gradgrad[a, i, j] = grad[a,i]*grad[a,j] (real part)
    gradgrad = grad_real[:, :, None]*grad_real[:, None, :] + grad_imag[:, :, None]*grad_imag[:, None, :]
    iso_term = iso_sub * norm_sq[:, None, None] * np.eye(3)[None, :, :]
    t_ij_spatial = coeff * gradgrad - iso_term  # (n, 3, 3)
    return t00, t_ij_spatial


def per_seed_full_munu(reg, n_lat, p):
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

    Lt_pool, Ls_pool, shear_pool, T00_pool, G00_pool = [], [], [], [], []
    Trace_T_pool = []
    for s in range(n_seeds):
        xi_mat = xi_seed(s)
        np.fill_diagonal(xi_mat, 1.0)
        psi = psi_seed(s)
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))

        # Compute lattice geometry: same as runner A
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
        r_ij = hessian_ricci_per_node(xi_off, adj, d_mat, spatial, np)  # (n, 3, 3)
        r_bar = np.trace(r_ij, axis1=1, axis2=2)
        g_00 = r_bar / 2.0
        g_ij = r_ij - 0.5*r_bar[:, None, None]*np.eye(3)[None, :, :]  # spatial 3x3 Einstein

        t00, t_ij = t_munu_3x3_per_node(xi_off, adj, weight_grad, d_mat,
                                        spatial, omega_a, psi, k_field, q_field)

        # Per-node mask
        mask = (t00 > 0.05) & np.isfinite(t00) & np.isfinite(g_00)
        if not np.any(mask):
            continue
        # T_ij eigendecomposition per node -> eigenframe
        # For each node, project G_ij into T_ij eigenframe, get diagonal lambda_i (T)
        # and off-diagonal shear of G in T-eigenframe.
        for a_idx in np.where(mask)[0]:
            T_loc = t_ij[a_idx]
            G_loc = g_ij[a_idx]
            # symmetrize
            T_loc = 0.5*(T_loc + T_loc.T)
            G_loc = 0.5*(G_loc + G_loc.T)
            try:
                ew, ev = np.linalg.eigh(T_loc)
            except np.linalg.LinAlgError:
                continue
            # Project G into T-eigenframe
            G_eigen = ev.T @ G_loc @ ev
            G_diag = np.diag(G_eigen)
            G_off = G_eigen - np.diag(G_diag)
            shear = float(np.linalg.norm(G_off, ord='fro'))
            # Per-axis Lambda_s_i = lambda_i - G_(ii) (so that G_(ii) + Lambda_s = lambda_i)
            Ls_per_axis = ew - G_diag
            T00_pool.append(t00[a_idx])
            G00_pool.append(g_00[a_idx])
            Lt_pool.append(t00[a_idx] - g_00[a_idx])
            Ls_pool.append(Ls_per_axis)        # (3,) per node
            shear_pool.append(shear)
            Trace_T_pool.append(float(np.trace(T_loc)))

    if not T00_pool:
        return None
    Lt_arr = np.array(Lt_pool)
    Ls_arr = np.array(Ls_pool)            # (n_nodes, 3)
    shear_arr = np.array(shear_pool)
    Trace_T_arr = np.array(Trace_T_pool)

    # Per-axis means and total
    Ls_mean_per_axis = Ls_arr.mean(axis=0)
    Ls_abs_total = float(np.mean(np.abs(Ls_arr).sum(axis=1)))  # mean over nodes of sum_i |Lambda_s_i|

    return {
        "regime": reg, "N": int(n_lat), "n_nodes": int(len(Lt_arr)),
        "Lambda_t_mean":  float(np.mean(Lt_arr)),
        "Lambda_s_axis_means": [float(x) for x in Ls_mean_per_axis],
        "Lambda_s_abs_sum": Ls_abs_total,
        "shear_off_mean":  float(np.mean(shear_arr)),
        "T_trace_mean":    float(np.mean(Trace_T_arr)),
        "T00_mean":        float(np.mean(T00_pool)),
        "G00_mean":        float(np.mean(G00_pool)),
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


def main() -> int:
    print("="*120)
    print("Unit-budget redistribution: Lambda_t + 3|Lambda_s| + shear_off vs C1^2 unit")
    print("="*120)
    print(f"  Predictions: alpha_xi^2 = {LAMBDA_T_PRED:.3f}, gamma^2 = {GAMMA**2:.3f}, 2*alpha_xi*gamma = {SHEAR_PRED_FINITE:.3f}")
    print(f"  Lambda_s = -gamma^2/2 = {LAMBDA_S_PRED:.4f} per axis -> 3*|Lambda_s| = {3*abs(LAMBDA_S_PRED):.3f}")
    print()
    print(f"{'reg':<10} {'N':>4} | {'Λ_t':>7} {'3|Λ_s|':>8} {'shear':>7} | {'budget':>8} {'<T_tr>':>8} {'<T_00>':>7}")
    print("-"*80)
    rows = []
    for reg, n, src in SOURCE_LIST:
        p = get_path(reg, src)
        r = per_seed_full_munu(reg, n, p)
        if r is None:
            print(f"{reg:<10} {n:>4}   FILE_MISSING_OR_EMPTY")
            continue
        budget = r["Lambda_t_mean"] + r["Lambda_s_abs_sum"] + r["shear_off_mean"]
        rows.append({**r, "budget": budget})
        print(f"{reg:<10} {n:>4} | {r['Lambda_t_mean']:>7.3f} {r['Lambda_s_abs_sum']:>8.3f} {r['shear_off_mean']:>7.3f} | {budget:>8.3f} {r['T_trace_mean']:>8.3f} {r['T00_mean']:>7.3f}")
    print()
    if rows:
        N_arr = np.array([r["N"] for r in rows], dtype=float)
        Lt_arr = np.array([r["Lambda_t_mean"] for r in rows])
        Ls_arr = np.array([r["Lambda_s_abs_sum"] for r in rows])
        Sh_arr = np.array([r["shear_off_mean"] for r in rows])

        print(f"=== N-trend (mean over seeds) ===")
        # Sort by N for trend
        idx = np.argsort(N_arr)
        for i in idx:
            print(f"  N={int(N_arr[i]):>4}  Λ_t={Lt_arr[i]:.3f}  3|Λ_s|={Ls_arr[i]:.3f}  shear={Sh_arr[i]:.3f}  budget={Lt_arr[i]+Ls_arr[i]+Sh_arr[i]:.3f}")

        print()
        print(f"=== Asymptotic predictions (continuum N->infty) ===")
        print(f"  Lambda_t -> alpha_xi^2 = 0.810")
        print(f"  3*|Lambda_s| -> 3*gamma^2/2 = 0.015 (asymptotic if Lambda_s = -gamma^2/2)")
        print(f"  shear_off -> 0 (Symanzik decay)")
        print(f"  budget -> 0.825 (asymptotic, no cross-term)")
        print()
        print(f"=== At smallest N ===")
        i_min = np.argmin(N_arr)
        print(f"  N={int(N_arr[i_min])}: Λ_t={Lt_arr[i_min]:.3f}, 3|Λ_s|={Ls_arr[i_min]:.3f}, shear={Sh_arr[i_min]:.3f}")
        print(f"  Excess over alpha_xi^2: {Lt_arr[i_min] - LAMBDA_T_PRED:.3f}")
        print(f"  shear vs 2*alpha_xi*gamma = 0.180: ratio = {Sh_arr[i_min]/SHEAR_PRED_FINITE:.2f}")
        print(f"=== At largest N ===")
        i_max = np.argmax(N_arr)
        print(f"  N={int(N_arr[i_max])}: Λ_t={Lt_arr[i_max]:.3f}, 3|Λ_s|={Ls_arr[i_max]:.3f}, shear={Sh_arr[i_max]:.3f}")
        print(f"  Excess over alpha_xi^2: {Lt_arr[i_max] - LAMBDA_T_PRED:.3f}")

        # Decay test of shear with N
        if len(N_arr) >= 5:
            log_N = np.log(N_arr)
            mask_pos_shear = Sh_arr > 1e-9
            if mask_pos_shear.sum() >= 3:
                slope_shear = np.polyfit(log_N[mask_pos_shear], np.log(Sh_arr[mask_pos_shear]), 1)
                print(f"  Shear vs N: log-log slope = {slope_shear[0]:.2f} (-2.54 expected from off-diagonal Symanzik)")

    out = {
        "method": "unit_budget_redistribution",
        "predictions": {
            "Lambda_t": LAMBDA_T_PRED, "Lambda_s_per_axis": LAMBDA_S_PRED,
            "shear_finite_N": SHEAR_PRED_FINITE,
            "asymptotic_budget": TRACE_PRED_BUDGET,
        },
        "per_regime": rows,
    }
    out_path = REPO / "outputs" / "unit_budget_redistribution_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
