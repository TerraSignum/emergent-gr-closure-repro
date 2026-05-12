"""(*) Test User-Hypothese: 2+1 Anisotropie als Partikel-Mode-Emergence.

Hypothese: Per-Knoten 2+1 (2 transversal + 1 longitudinal) entspricht
einer lokalen Massive-Spin-1-Boson-Mode-Struktur.

Falsifikations-Tests:
  T1: Mode-Symmetry: |Lambda_s,long| ~ |Lambda_s,trans| (already partially confirmed)
  T2: Spatial Coherence: sind principal axes benachbarter Knoten korreliert?
      - Korreliert (cos(theta) >> 0) -> propagating-mode/wavepacket Bild
      - Random (cos(theta) ~ 0) -> stochastic local fluctuation
      - Anti-correlated (cos(theta) << 0) -> domain-wall structure
  T3: Mass-Scale: |Lambda_s,long| / |Lambda_s,trans| Verhältnis
      - = 1 -> reine Mode-Symmetrie (massiv)
      - >> 1 -> longitudinal dominiert (Stokes-mode? virtual?)
      - << 1 -> transversal dominiert (radiation-like, masseless?)
  T4: Domain-Größe: charakteristische Längenskala der axis-coherence
      - Vergleich gegen Lattice-Skala
  T5: Cross-N-Skalierung: Mode-Struktur regime-invariant oder N-abhängig?

Output: outputs/particle_mode_hypothesis_audit.json
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
    ("P5",     50, "canonical"),
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


def per_seed_principal_analysis(reg, n_lat, p):
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

    long_to_trans_ratios = []
    aligned_neighbor_cosines = []
    aniso_mag = []   # axis3 - axis1 magnitude per node

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
        g_00 = r_bar/2.0
        g_ij = r_ij - 0.5*r_bar[:, None, None]*np.eye(3)[None, :, :]

        t00, t_ij = t_munu_3x3(xi_off, adj, weight_grad, d_mat, spatial, omega_a, psi, k_field, q_field)
        mask = (t00 > 0.05) & np.isfinite(t00) & np.isfinite(g_00)

        # Per-node principal eigenvector storage for spatial coherence
        principal_axes = np.zeros((n_lat, 3))
        valid_node = np.zeros(n_lat, dtype=bool)
        for a_idx in np.where(mask)[0]:
            T_loc = 0.5*(t_ij[a_idx] + t_ij[a_idx].T)
            G_loc = 0.5*(g_ij[a_idx] + g_ij[a_idx].T)
            try:
                ew, ev = np.linalg.eigh(T_loc)
            except np.linalg.LinAlgError:
                continue
            G_eigen = ev.T @ G_loc @ ev
            G_diag = np.diag(G_eigen)
            Ls = ew - G_diag  # ascending in T-eigenvalue, so axis3 = largest -> longitudinal-like
            Lt_long = abs(Ls[2])  # largest T axis (positive Lambda_s_3 -> longitudinal)
            Lt_trans = (abs(Ls[0]) + abs(Ls[1]))/2  # average of two transverse magnitudes
            if Lt_trans > 1e-9:
                long_to_trans_ratios.append(Lt_long/Lt_trans)
            aniso_mag.append(Ls[2] - Ls[0])

            # Principal axis = eigenvector for largest T eigenvalue
            principal_axes[a_idx] = ev[:, 2]
            valid_node[a_idx] = True

        # T2: Spatial Coherence: cosines of neighbor principal axes
        # For each connected node pair (xi > XI_THRESH), compute angle between principal axes
        valid_idx = np.where(valid_node)[0]
        for i_idx in valid_idx:
            for j_idx in valid_idx:
                if i_idx >= j_idx:
                    continue
                if adj[i_idx, j_idx] > 0:
                    cos_theta = abs(principal_axes[i_idx] @ principal_axes[j_idx])
                    aligned_neighbor_cosines.append(cos_theta)

    if not long_to_trans_ratios:
        return None
    long_to_trans = np.array(long_to_trans_ratios)
    cosines = np.array(aligned_neighbor_cosines)
    aniso_arr = np.array(aniso_mag)

    return {
        "regime": reg, "N": int(n_lat),
        "n_nodes": int(len(long_to_trans)),
        "n_neighbor_pairs": int(len(cosines)),
        "long_to_trans_ratio_median": float(np.median(long_to_trans)),
        "long_to_trans_ratio_mean":   float(np.mean(long_to_trans)),
        "long_to_trans_ratio_std":    float(np.std(long_to_trans)),
        # Spatial coherence: <|cos(theta)|>
        # 0.5 = uniform random in 3D
        # 1.0 = perfectly aligned
        # Higher than 0.5 = significant alignment
        "neighbor_axis_alignment_mean": float(np.mean(cosines)),
        "neighbor_axis_alignment_med":  float(np.median(cosines)),
        "neighbor_axis_alignment_std":  float(np.std(cosines)),
        # Random-3D-baseline expected: <|cos(theta)|> = 0.5
        "alignment_excess_over_random": float(np.mean(cosines) - 0.5),
        "anisotropy_magnitude_mean":   float(np.mean(aniso_arr)),
    }


def get_path(reg, src):
    if src == "canonical":
        return find_d1_npz(reg, REPO)
    return PARENT / f"results_d1_{reg.lower()}_v2" / f"{reg}.snapshots.npz"


def main() -> int:
    print("="*120)
    print("Test User-Hypothese: 2+1 als emergente Spin-1-Mode-Struktur")
    print("="*120)
    print()
    print("T1: Mode-symmetrie (long/trans ratio): ~1 -> echte Mode-Symmetrie")
    print("T2: Spatial coherence (neighbor axis alignment): ")
    print("    - 0.5 = random (baseline 3D uniform sphere)")
    print("    - >0.6 = significant alignment (propagating mode / coherent domain)")
    print("    - >0.8 = strong wavepacket/domain")
    print()
    rows = []
    for reg, n, src in SOURCE_LIST:
        p = get_path(reg, src)
        r = per_seed_principal_analysis(reg, n, p)
        if r is None:
            print(f"{reg:<10} {n:>4} -- file_missing"); continue
        rows.append(r)

    print(f"{'reg':<10} {'N':>4} | {'n_pairs':>7} {'long/trans':>11} {'aniso':>7} | {'<|cos|>':>9} {'excess':>8}")
    print("-"*80)
    for r in rows:
        print(f"{r['regime']:<10} {r['N']:>4} | "
              f"{r['n_neighbor_pairs']:>7} "
              f"{r['long_to_trans_ratio_median']:>11.3f} "
              f"{r['anisotropy_magnitude_mean']:>7.4f} | "
              f"{r['neighbor_axis_alignment_mean']:>9.4f} "
              f"{r['alignment_excess_over_random']:>+8.4f}")

    if rows:
        print()
        print("=== Verdict ===")
        ratio_means = np.array([r["long_to_trans_ratio_median"] for r in rows])
        align_means = np.array([r["neighbor_axis_alignment_mean"] for r in rows])
        print(f"  T1 cross-regime long/trans ratio: median={np.median(ratio_means):.3f}, std={np.std(ratio_means):.3f}")
        print(f"     prediction (mode-symmetric massive spin-1): ratio = 1.000")
        print(f"     {'Mode-symmetric: SUPPORTED' if 0.7 <= np.median(ratio_means) <= 1.3 else 'Mode-symmetric: NOT SUPPORTED'}")
        print(f"  T2 cross-regime neighbor alignment: median={np.median(align_means):.3f}")
        print(f"     random-3D baseline: <|cos|> = 0.500")
        print(f"     wavepacket signature: <|cos|> > 0.600")
        print(f"     mean excess: {np.median(align_means) - 0.5:+.3f}")
        if np.median(align_means) > 0.6:
            verdict = "WAVEPACKET_LIKE_COHERENCE"
        elif np.median(align_means) > 0.55:
            verdict = "WEAK_COHERENCE_ABOVE_RANDOM"
        elif 0.45 <= np.median(align_means) <= 0.55:
            verdict = "RANDOM_LOCAL_FLUCTUATION"
        else:
            verdict = "ANTI_ALIGNED_DOMAIN_STRUCTURE"
        print(f"  T2 verdict: {verdict}")
        print()
        print(f"  Combined assessment of particle-emergence hypothesis:")
        if 0.7 <= np.median(ratio_means) <= 1.3 and np.median(align_means) > 0.55:
            print(f"  -> CONSISTENT with massive-spin-1-mode interpretation")
        elif 0.7 <= np.median(ratio_means) <= 1.3:
            print(f"  -> Mode-symmetric BUT NOT spatially coherent (purely local stochastic)")
        else:
            print(f"  -> Mode-asymmetric -> NOT supporting massive-spin-1 interpretation")

    out = {
        "method": "particle_mode_hypothesis_test",
        "predictions": {
            "long_trans_ratio_for_massive_spin1": 1.0,
            "spatial_coherence_for_propagating_mode": ">0.6",
            "random_3D_baseline": 0.5,
        },
        "per_regime": rows,
    }
    out_path = REPO / "outputs" / "particle_mode_hypothesis_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
