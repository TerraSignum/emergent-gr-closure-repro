"""(B1) Frame-invariance test for 2+1 anisotropy pattern.

Reviewer critique: "The 2+1 sign pattern in Λ_s only appears after
sorting by T-eigenvalue. Is it sorting-induced?"

Tests:
  T1: Sign-distribution per node — count #negative axes per node.
      Under random isotropic: binomial p=0.5 → P(2 neg) = 3/8 = 0.375
      Under forced 2+1: P(2 neg) >> 0.375 — measure how strong.
  T2: Random orthogonal rotation per node — apply random SO(3)
      rotation to T-eigenframe before extracting Λ_s. If 2+1 is
      sorting-artefact, rotation should destroy it; if it's
      intrinsic algebraic structure, rotation preserves it.
  T3: Pre-sort by sign instead of by T-eigenvalue magnitude.
      Compare distribution.

Output: outputs/2plus1_frame_invariance_audit.json
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
    ("P1", 28, "canonical"), ("P3", 36, "canonical"), ("P4", 42, "canonical"),
    ("P5", 50, "canonical"), ("P5N64", 64, "canonical"), ("P6", 60, "canonical"),
    ("P7", 72, "canonical"), ("P8", 84, "canonical"), ("P5N100", 100, "canonical"),
    ("P5N72", 72, "snapshot_v2"), ("P5N84", 84, "snapshot_v2"),
    ("P5N200", 200, "snapshot_v2"), ("P6N128", 128, "snapshot_v2"),
    ("P8N128", 128, "snapshot_v2"),
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
    t_ij_spatial = coeff * gradgrad - iso_term
    return t00, t_ij_spatial


def random_so3(rng):
    """Sample uniformly random SO(3) rotation."""
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return R


def per_seed_frame_test(reg, n_lat, p, rng):
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

    # Counters
    sign_dist_sorted = np.zeros(4, dtype=int)  # 0..3 negative
    sign_dist_random = np.zeros(4, dtype=int)
    aniso_sorted, aniso_random = [], []

    for s in range(n_seeds):
        xi_mat = xi_seed(s); np.fill_diagonal(xi_mat, 1.0)
        psi = psi_seed(s)
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        xi_off = xi_mat.copy(); np.fill_diagonal(xi_off, 0.0)
        adj = (xi_off > XI_THRESH).astype(np.float64)
        weight_adj = xi_off * adj; deg = weight_adj.sum(axis=1) + 1e-12
        deg_inv_sqrt = 1.0/np.sqrt(deg)
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
        t00, t_ij = t_munu_3x3(xi_off, adj, weight_grad, d_mat, spatial, omega_a, psi, k_field, q_field)
        mask = (t00 > 0.05) & np.isfinite(t00) & np.isfinite(g_00)

        for a_idx in np.where(mask)[0]:
            T_loc = 0.5*(t_ij[a_idx] + t_ij[a_idx].T)
            G_loc = 0.5*(g_ij[a_idx] + g_ij[a_idx].T)
            try:
                ew, ev = np.linalg.eigh(T_loc)
            except np.linalg.LinAlgError: continue

            # SORTED frame (by T eigenvalue, ascending → axis 1=smallest, 3=largest)
            G_eig = ev.T @ G_loc @ ev
            G_diag = np.diag(G_eig)
            Ls_sorted = ew - G_diag
            n_neg_sorted = int(np.sum(Ls_sorted < 0))
            sign_dist_sorted[n_neg_sorted] += 1
            aniso_sorted.append(float(np.max(Ls_sorted) - np.min(Ls_sorted)))

            # RANDOM ROTATION applied to T-eigenframe — destroys sorting if sorting-induced
            R = random_so3(rng)
            ev_rotated = ev @ R
            G_eig_rot = ev_rotated.T @ G_loc @ ev_rotated
            T_eig_rot = ev_rotated.T @ T_loc @ ev_rotated
            G_diag_rot = np.diag(G_eig_rot)
            T_diag_rot = np.diag(T_eig_rot)
            # Λ_s in random frame: T_eigenvalue (rotated) - G_diag (rotated)
            # Note: T_eigenvalue under rotation is no longer eigenvalue per se, but the diagonal
            # in rotated frame; for true frame-invariance test we use this.
            Ls_random = T_diag_rot - G_diag_rot
            n_neg_random = int(np.sum(Ls_random < 0))
            sign_dist_random[n_neg_random] += 1
            aniso_random.append(float(np.max(Ls_random) - np.min(Ls_random)))

    n_total_sorted = sign_dist_sorted.sum()
    n_total_random = sign_dist_random.sum()
    if n_total_sorted == 0: return None

    return {
        "regime": reg, "N": int(n_lat),
        "n_nodes": int(n_total_sorted),
        "sorted_sign_dist": [int(x) for x in sign_dist_sorted],
        "sorted_proportions": [float(x/n_total_sorted) for x in sign_dist_sorted],
        "random_sign_dist": [int(x) for x in sign_dist_random],
        "random_proportions": [float(x/n_total_random) for x in sign_dist_random] if n_total_random else None,
        "sorted_aniso_mean": float(np.mean(aniso_sorted)) if aniso_sorted else 0.0,
        "random_aniso_mean": float(np.mean(aniso_random)) if aniso_random else 0.0,
    }


def main() -> int:
    print("="*100)
    print("(B1) Frame-invariance test for 2+1 anisotropy")
    print("="*100)
    print()
    print("If 2+1 is sorting-induced: random-rotation should destroy 2-negative concentration")
    print("If 2+1 is intrinsic: random-rotation preserves 2-negative concentration")
    print()
    print("Random isotropic baseline: P(2 negative) = C(3,2) * (1/2)^3 = 3/8 = 0.375")
    print("Forced 2+1 concentration: P(2 negative) >> 0.375")
    print()
    rng = np.random.default_rng(42)
    rows = []
    for reg, n, src in SOURCE_LIST:
        p = get_path(reg, src)
        r = per_seed_frame_test(reg, n, p, rng)
        if r is None: print(f"{reg:<10} {n:>4}  -- file missing"); continue
        rows.append(r)

    print(f"{'reg':<10} {'N':>4} | {'sorted P(0,1,2,3 neg)':>32} | {'rotated P(0,1,2,3 neg)':>32}")
    print("-"*100)
    for r in sorted(rows, key=lambda x: x["N"]):
        sp = r["sorted_proportions"]; rp = r["random_proportions"]
        sp_s = f"{sp[0]:.2f},{sp[1]:.2f},{sp[2]:.2f},{sp[3]:.2f}"
        rp_s = f"{rp[0]:.2f},{rp[1]:.2f},{rp[2]:.2f},{rp[3]:.2f}" if rp else "—"
        print(f"{r['regime']:<10} {r['N']:>4} | {sp_s:>32} | {rp_s:>32}")

    # Aggregate
    print()
    if rows:
        sorted_p2 = np.array([r["sorted_proportions"][2] for r in rows])
        random_p2 = np.array([r["random_proportions"][2] for r in rows if r["random_proportions"]])
        print(f"Cross-regime P(2 negative axes):")
        print(f"  Sorted T-eigenframe:  mean = {sorted_p2.mean():.3f}, std = {sorted_p2.std():.3f}")
        print(f"  Random rotation:       mean = {random_p2.mean():.3f}, std = {random_p2.std():.3f}")
        print(f"  Random baseline:       0.375 (binomial p=0.5)")
        print()
        print(f"Aniso index (max-min Lambda_s):")
        sa = np.array([r["sorted_aniso_mean"] for r in rows])
        ra = np.array([r["random_aniso_mean"] for r in rows])
        print(f"  Sorted:  mean = {sa.mean():.4f}")
        print(f"  Rotated: mean = {ra.mean():.4f}")
        # Verdict
        if sorted_p2.mean() > 0.55 and random_p2.mean() > 0.45:
            verdict = "2PLUS1_PATTERN_FRAME_INVARIANT"
        elif sorted_p2.mean() > 0.55:
            verdict = "2PLUS1_PATTERN_FRAME_DEPENDENT_LIKELY_SORTING_INDUCED"
        else:
            verdict = "NO_STRONG_2PLUS1_PATTERN"
        print(f"\nVERDICT: {verdict}")

    out = {"method": "2plus1_frame_invariance", "per_regime": rows}
    if rows:
        out["cross_regime_summary"] = {
            "P_2_neg_sorted_mean":  float(sorted_p2.mean()),
            "P_2_neg_sorted_std":   float(sorted_p2.std()),
            "P_2_neg_random_mean":  float(random_p2.mean()),
            "P_2_neg_random_std":   float(random_p2.std()),
            "P_2_neg_isotropic_baseline": 0.375,
            "verdict": verdict,
        }
    out_path = REPO / "outputs" / "2plus1_frame_invariance_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
