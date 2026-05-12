"""(*) Achsen-Identifikation des 2+1 Lambda_s Signals:

Drei Frames simultan:
  Frame A: Per-Knoten T-Eigenframe sortiert (intrinsisch, was wir bisher hatten)
  Frame B: Global Fiedler-Frame (Laplacian-Eigenvektoren, Reihenfolge Fiedler1, 2, 3)
  Frame C: Per-Knoten T-Eigenframe NICHT sortiert, sondern nach Vorzeichen
           (negativste, mittlere, positivste -> ranked by VALUE not magnitude)

Wenn in Frame B der expandierende Axis konsistent bei Fiedler-2 oder Fiedler-3 sitzt,
dann hat die ausgezeichnete Achse eine konkrete physikalische Identifikation
(z.B. Fiedler-2 ist der Spektral-Cluster-Connectivity-Mode).

Output: outputs/axis_identification_audit.json
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
    ("P6N128",128, "snapshot_v2"),
    ("P8N128",128, "snapshot_v2"),
]


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


def per_seed_decompose(reg, n_lat, p):
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

    # Frame B: global Fiedler frame — diagonal of G in this frame
    Ls_fiedler1_pool = []
    Ls_fiedler2_pool = []
    Ls_fiedler3_pool = []
    # Frame A: per-node T-eigenframe sorted ASCENDING (sign-ranked)
    Ls_sortedA_pool = []
    # Diagnostic: for each node, what is the index of the LARGEST T-eigenvalue
    # in terms of original Fiedler axis (0, 1, 2)?
    largest_T_axis_pool = []
    smallest_T_axis_pool = []

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
        spatial = eigvecs_l[:, 1:4]   # Fiedler1, Fiedler2, Fiedler3
        d_mat = -ELL_0*np.log(np.maximum(xi_off, 1e-12))
        d_mat = np.maximum(d_mat, D_MIN)
        d_sq_safe = np.where(adj > 0, d_mat*d_mat, np.inf)
        weight_grad = np.where(adj > 0, weight_adj/(d_sq_safe + EPS_D), 0.0)
        omega_a = weight_grad.sum(axis=1)
        r_ij = hessian_ricci_per_node(xi_off, adj, d_mat, spatial, np)
        r_bar = np.trace(r_ij, axis1=1, axis2=2)
        g_00 = r_bar/2.0
        g_ij = r_ij - 0.5*r_bar[:, None, None]*np.eye(3)[None, :, :]

        t00, t_ij = t_munu_3x3(xi_off, adj, weight_grad, d_mat, spatial, omega_a, psi, k_field, q_field)
        mask = (t00 > 0.05) & np.isfinite(t00) & np.isfinite(g_00)

        for a_idx in np.where(mask)[0]:
            T_loc = 0.5*(t_ij[a_idx] + t_ij[a_idx].T)
            G_loc = 0.5*(g_ij[a_idx] + g_ij[a_idx].T)
            try:
                ew, ev = np.linalg.eigh(T_loc)  # ascending eigenvalues
            except np.linalg.LinAlgError:
                continue
            # Frame A: per-node T-eigenframe, axes sorted ascending (already ascending from eigh)
            G_eigen = ev.T @ G_loc @ ev
            G_diag_eigen = np.diag(G_eigen)
            Ls_sortedA = ew - G_diag_eigen   # (3,) sorted ascending in T-eigenvalue
            Ls_sortedA_pool.append(Ls_sortedA)

            # Frame B: global Fiedler-frame diagonal
            T_diag_F = np.diag(T_loc)  # raw diagonal in Fiedler frame
            G_diag_F = np.diag(G_loc)
            # Per-Fiedler-axis Lambda_s_k = T_kk - G_kk (so G_kk + Lambda_s_k = T_kk)
            Ls_F = T_diag_F - G_diag_F
            Ls_fiedler1_pool.append(Ls_F[0])
            Ls_fiedler2_pool.append(Ls_F[1])
            Ls_fiedler3_pool.append(Ls_F[2])

            # Diagnostic: which Fiedler axis is the LARGEST T eigenvalue closest to?
            # Project T-eigenvectors onto Fiedler basis; the absmax overlap tells us
            # which Fiedler axis the largest-T direction is closest to.
            # Largest T eigenvalue is ew[2] (last after ascending sort)
            largest_T_dir = ev[:, 2]   # eigenvector for largest eigenvalue
            smallest_T_dir = ev[:, 0]  # eigenvector for smallest eigenvalue
            largest_T_axis_pool.append(int(np.argmax(np.abs(largest_T_dir))))
            smallest_T_axis_pool.append(int(np.argmax(np.abs(smallest_T_dir))))

    if not Ls_sortedA_pool:
        return None

    Ls_sortedA = np.array(Ls_sortedA_pool)
    Ls_F1 = np.array(Ls_fiedler1_pool)
    Ls_F2 = np.array(Ls_fiedler2_pool)
    Ls_F3 = np.array(Ls_fiedler3_pool)

    largest_T_dist = {f"F{k}": int(sum(1 for x in largest_T_axis_pool if x == k))
                      for k in range(3)}
    smallest_T_dist = {f"F{k}": int(sum(1 for x in smallest_T_axis_pool if x == k))
                       for k in range(3)}

    return {
        "regime": reg, "N": int(n_lat), "n_nodes": int(len(Ls_sortedA)),
        # Frame A: T-eigenframe sorted ascending
        "Ls_sortedA_means": [float(x) for x in Ls_sortedA.mean(axis=0)],
        # Frame B: global Fiedler-frame
        "Ls_fiedler1_mean": float(Ls_F1.mean()),
        "Ls_fiedler2_mean": float(Ls_F2.mean()),
        "Ls_fiedler3_mean": float(Ls_F3.mean()),
        "Ls_fiedler1_std":  float(Ls_F1.std()),
        "Ls_fiedler2_std":  float(Ls_F2.std()),
        "Ls_fiedler3_std":  float(Ls_F3.std()),
        # Diagnostic: which Fiedler axis dominates the largest/smallest T direction
        "largest_T_axis_distribution": largest_T_dist,
        "smallest_T_axis_distribution": smallest_T_dist,
    }


def get_path(reg, src):
    if src == "canonical":
        return find_d1_npz(reg, REPO)
    return PARENT / f"results_d1_{reg.lower()}_v2" / f"{reg}.snapshots.npz"


def main() -> int:
    print("="*120)
    print("Axis identification: 2+1 in T-eigenframe vs global Fiedler frame")
    print("="*120)
    rows = []
    for reg, n, src in SOURCE_LIST:
        p = get_path(reg, src)
        r = per_seed_decompose(reg, n, p)
        if r is None:
            print(f"{reg:<10} {n:>4} -- file_missing"); continue
        rows.append(r)

    print()
    print("=== Frame A: per-node T-eigenframe sorted ascending ===")
    print(f"{'reg':<10} {'N':>4} | {'Lt-axis1':>10} {'Lt-axis2':>10} {'Lt-axis3':>10}")
    print("-"*55)
    for r in rows:
        L = r["Ls_sortedA_means"]
        print(f"{r['regime']:<10} {r['N']:>4} | {L[0]:>+10.4f} {L[1]:>+10.4f} {L[2]:>+10.4f}")

    print()
    print("=== Frame B: global Fiedler frame (NOT sorted, axes = Fiedler1, 2, 3) ===")
    print(f"{'reg':<10} {'N':>4} | {'L_F1':>9} {'L_F2':>9} {'L_F3':>9} | {'F1 std':>8} {'F2 std':>8} {'F3 std':>8}")
    print("-"*80)
    for r in rows:
        print(f"{r['regime']:<10} {r['N']:>4} | "
              f"{r['Ls_fiedler1_mean']:>+9.4f} {r['Ls_fiedler2_mean']:>+9.4f} {r['Ls_fiedler3_mean']:>+9.4f} | "
              f"{r['Ls_fiedler1_std']:>8.3f} {r['Ls_fiedler2_std']:>8.3f} {r['Ls_fiedler3_std']:>8.3f}")

    print()
    print("=== Diagnostic: which Fiedler axis dominates the LARGEST T-direction ===")
    print(f"{'reg':<10} {'N':>4} | {'L1=F0':>5} {'L1=F1':>5} {'L1=F2':>5} | {'S1=F0':>5} {'S1=F1':>5} {'S1=F2':>5}")
    print("-"*70)
    for r in rows:
        l = r["largest_T_axis_distribution"]
        sm = r["smallest_T_axis_distribution"]
        n = r["n_nodes"]
        print(f"{r['regime']:<10} {r['N']:>4} | "
              f"{l['F0']/n*100:>4.0f}% {l['F1']/n*100:>4.0f}% {l['F2']/n*100:>4.0f}% | "
              f"{sm['F0']/n*100:>4.0f}% {sm['F1']/n*100:>4.0f}% {sm['F2']/n*100:>4.0f}%")

    # Aggregate Frame B over regimes
    F1_means = np.array([r["Ls_fiedler1_mean"] for r in rows])
    F2_means = np.array([r["Ls_fiedler2_mean"] for r in rows])
    F3_means = np.array([r["Ls_fiedler3_mean"] for r in rows])
    print()
    print("=== Cross-regime mean of global Fiedler-frame diagonal Lambda_s ===")
    print(f"  Fiedler1: mean = {F1_means.mean():+.4f}, std = {F1_means.std():.4f}")
    print(f"  Fiedler2: mean = {F2_means.mean():+.4f}, std = {F2_means.std():.4f}")
    print(f"  Fiedler3: mean = {F3_means.mean():+.4f}, std = {F3_means.std():.4f}")
    print()
    print("If 2+1 anisotropy is a global Fiedler-frame effect:")
    print("  -> two of (F1, F2, F3) should consistently negative, one positive.")
    print("If 2+1 is purely per-node (rotates with each node):")
    print("  -> Fiedler-frame diagonals should average to ~0 (rotational averaging)")

    out = {
        "method": "axis_identification_2plus1",
        "n_regimes": len(rows),
        "per_regime": rows,
        "cross_regime_fiedler_means": {
            "F1": float(F1_means.mean()), "F2": float(F2_means.mean()), "F3": float(F3_means.mean())
        },
    }
    out_path = REPO / "outputs" / "axis_identification_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
