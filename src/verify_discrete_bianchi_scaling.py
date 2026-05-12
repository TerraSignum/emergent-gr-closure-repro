"""(*) Empirical Discrete Bianchi check: does ||∇_μ G^μν||_F → 0 with N?

If yes → discrete Bianchi recovers in continuum limit, closing the
Open-2 conservation gap empirically.
If no → genuine Bianchi violation requiring Own2 correction.

Discrete divergence on adj-graph:
  (∇_μ G^μν)_a := Σ_{b in nbrs(a)} (G^μν_b - G^μν_a) · (e^μ_{ab} / d_ab)
where e^μ_{ab} is the unit edge direction in spatial frame.

We test the time-component ∇_μ G^μ0 (most physically relevant).

Output: outputs/discrete_bianchi_scaling_audit.json
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


def get_path(reg, src):
    if src == "canonical":
        return find_d1_npz(reg, REPO)
    return PARENT / f"results_d1_{reg.lower()}_v2" / f"{reg}.snapshots.npz"


def discrete_bianchi_per_seed(xi_mat, n_lat, weight_grad, d_mat, spatial,
                               g_00_per_node, g_ij_per_node):
    """Compute ||∇_μ G^μ0||_F as discrete divergence of G_0μ across adj-graph.

    Per-node B^0_a := Σ_{b: adj(a,b)} (G^00_b - G^00_a) / d_ab
                    + Σ_{b: adj(a,b),i} (G^i0_b - G^i0_a) · e^i_ab / d_ab

    Since we have Hessian-Ricci in 3-eigenframe (spatial), the cross-component
    G^i0 isn't directly computed. We estimate B^0 via the time-time divergence:
      B^0 ~ Σ_b (G^00_b - G^00_a) / d_ab over neighbors b.

    Also test the spatial divergence:
      B^i ~ Σ_b (G^ij_b - G^ij_a) e^j_ab / d_ab.

    For full Bianchi we want ||B^μ|| = ||(B^0, B^1, B^2, B^3)||
    """
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(np.float64)
    inv_d = np.where(adj > 0, 1.0/d_mat, 0.0)
    spatial_diff = spatial[None, :, :] - spatial[:, None, :]  # (n, n, 3) edges
    deg = adj.sum(axis=1) + 1e-12

    # Per-edge gradient of G^00: (G_b - G_a) / d_ab
    g00_diff = (g_00_per_node[None, :] - g_00_per_node[:, None]) * inv_d * adj  # (n, n)
    # B^0(a) = Σ_b g00_diff(a,b) / deg(a)  (averaged divergence)
    B_time = (g00_diff).sum(axis=1) / deg

    # Per-edge gradient of G^ij: (G^ij_b - G^ij_a) / d_ab projected onto e^i_ab
    # G^ij_per_node has shape (n, 3, 3). We compute B^i = Σ_b Σ_j (G^ij_b - G^ij_a) * e^j_ab / d_ab / deg
    # spatial_diff[a,b,j] is e^j_ab (unnormalized; we scale by inv_d to make it unit)
    edge_dir = spatial_diff * inv_d[:, :, None]  # (n, n, 3) approx unit dir
    # B_spatial[a, i] = Σ_b Σ_j (G^ij_b - G^ij_a) * edge_dir[a, b, j] * adj[a,b]
    # diff_g_ij[a, b, i, j] = G^ij_b - G^ij_a
    diff_g_ij = g_ij_per_node[None, :, :, :] - g_ij_per_node[:, None, :, :]  # (n, n, 3, 3)
    # Contract j with edge_dir, mask with adj
    B_spat = np.einsum('abij,abj,ab->ai', diff_g_ij, edge_dir, adj) / deg[:, None]
    # Norm per node
    B_full_norm = np.sqrt(B_time**2 + (B_spat**2).sum(axis=1))
    return B_time, B_spat, B_full_norm


def compute_for_regime(reg, n_lat, p):
    if not p or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    if "dense_cell_edge_xi_values" in d.keys():
        e = d["dense_cell_edge_xi_values"]
        a = d["dense_cell_node_amplitude_values"]
        ph = d["dense_cell_node_phase_values"]
        n_seeds = min(e.shape[0], 32)
        xi_seed = lambda s: edge_to_matrix(e[s], n_lat)
    elif "edge_xi_snapshots" in d.keys():
        n_seeds = int(d["n_seeds"][0])
        xi_seed = lambda s: d["edge_xi_snapshots"][s, -1, :, :].copy()
    else:
        return None

    B0_pool, Bsp_pool, Bfull_pool = [], [], []
    for s in range(n_seeds):
        xi_mat = xi_seed(s)
        np.fill_diagonal(xi_mat, 1.0)
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

        B_time, B_spat, B_full = discrete_bianchi_per_seed(
            xi_mat, n_lat, weight_grad, d_mat, spatial, g_00, g_ij)
        # Filter NaN/Inf
        mask = np.isfinite(B_full) & np.isfinite(B_time) & np.all(np.isfinite(B_spat), axis=1)
        if mask.sum() == 0: continue
        B0_pool.append(np.abs(B_time[mask]))
        Bsp_pool.append(np.linalg.norm(B_spat[mask], axis=1))
        Bfull_pool.append(B_full[mask])

    if not B0_pool: return None
    B0 = np.concatenate(B0_pool)
    Bsp = np.concatenate(Bsp_pool)
    Bfull = np.concatenate(Bfull_pool)
    return {
        "regime": reg, "N": int(n_lat), "n_nodes": int(len(B0)),
        "B_time_mean": float(B0.mean()),
        "B_time_med":  float(np.median(B0)),
        "B_spat_mean": float(Bsp.mean()),
        "B_spat_med":  float(np.median(Bsp)),
        "B_full_mean": float(Bfull.mean()),
        "B_full_med":  float(np.median(Bfull)),
    }


def symanzik_24(N_arr, y_arr):
    A = np.column_stack([np.ones_like(N_arr), 1.0/N_arr**2, 1.0/N_arr**4])
    coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
    pred = A @ coef
    ss_res = np.sum((y_arr - pred)**2)
    ss_tot = np.sum((y_arr - y_arr.mean())**2) + 1e-12
    R2 = 1.0 - ss_res/ss_tot
    return float(coef[0]), float(R2)


def power_law(N_arr, y_arr):
    """Fit y(N) = c · N^(-α). Return α."""
    mask = (y_arr > 1e-12) & (N_arr > 0)
    if mask.sum() < 3: return float('nan'), float('nan')
    coefs = np.polyfit(np.log(N_arr[mask]), np.log(y_arr[mask]), 1)
    slope, intercept = coefs[0], coefs[1]
    pred = slope*np.log(N_arr[mask]) + intercept
    ss_res = np.sum((np.log(y_arr[mask]) - pred)**2)
    ss_tot = np.sum((np.log(y_arr[mask]) - np.log(y_arr[mask]).mean())**2) + 1e-12
    R2 = 1.0 - ss_res/ss_tot
    return -float(slope), float(R2)  # decay exponent positive


def main() -> int:
    print("="*100)
    print("Discrete Bianchi check: ||∇_μ G^μν||_F → 0 with N? Empirical Symanzik test")
    print("="*100)
    rows = []
    for reg, n, src in SOURCE_LIST:
        p = get_path(reg, src)
        r = compute_for_regime(reg, n, p)
        if r is None:
            print(f"{reg:<10} {n:>4}  -- file missing or empty"); continue
        rows.append(r)

    print()
    print(f"{'reg':<10} {'N':>4} | {'B_time_med':>11} {'B_spat_med':>11} {'B_full_med':>11}")
    print("-"*60)
    for r in sorted(rows, key=lambda x: x["N"]):
        print(f"{r['regime']:<10} {r['N']:>4} | {r['B_time_med']:>11.4f} {r['B_spat_med']:>11.4f} {r['B_full_med']:>11.4f}")

    print()
    if rows:
        N_arr = np.array([r["N"] for r in rows], dtype=float)
        Bt = np.array([r["B_time_med"] for r in rows])
        Bs = np.array([r["B_spat_med"] for r in rows])
        Bf = np.array([r["B_full_med"] for r in rows])

        print("=== Symanzik 2+4 fits ===")
        Bt_inf, R2_t = symanzik_24(N_arr, Bt)
        Bs_inf, R2_s = symanzik_24(N_arr, Bs)
        Bf_inf, R2_f = symanzik_24(N_arr, Bf)
        print(f"  B_time_med^∞ = {Bt_inf:.4f} (R²={R2_t:.3f})")
        print(f"  B_spat_med^∞ = {Bs_inf:.4f} (R²={R2_s:.3f})")
        print(f"  B_full_med^∞ = {Bf_inf:.4f} (R²={R2_f:.3f})")

        print()
        print("=== Power-law fits (y ~ N^-α) ===")
        alpha_t, R2pt = power_law(N_arr, Bt)
        alpha_s, R2ps = power_law(N_arr, Bs)
        alpha_f, R2pf = power_law(N_arr, Bf)
        print(f"  α_time = {alpha_t:.2f} (R²={R2pt:.3f})")
        print(f"  α_spat = {alpha_s:.2f} (R²={R2ps:.3f})")
        print(f"  α_full = {alpha_f:.2f} (R²={R2pf:.3f})")
        print()
        print("Interpretation:")
        if Bt_inf < 0.01 and Bs_inf < 0.01:
            verdict = "DISCRETE_BIANCHI_RECOVERED_IN_CONTINUUM"
        elif alpha_t > 0.5 and alpha_s > 0.5:
            verdict = "BIANCHI_VIOLATION_DECAYS_BUT_NOT_TO_ZERO"
        else:
            verdict = "BIANCHI_VIOLATION_NOT_CLEANLY_DECAYING"
        print(f"  VERDICT: {verdict}")

    out = {
        "method": "discrete_bianchi_scaling",
        "per_regime": rows,
        "symanzik": {
            "B_time_inf": Bt_inf if rows else None,
            "B_spat_inf": Bs_inf if rows else None,
            "B_full_inf": Bf_inf if rows else None,
            "R2_time": R2_t if rows else None,
            "R2_spat": R2_s if rows else None,
        },
        "power_law": {
            "alpha_time": alpha_t if rows else None,
            "alpha_spat": alpha_s if rows else None,
            "alpha_full": alpha_f if rows else None,
        },
        "verdict": verdict if rows else "INSUFFICIENT_DATA",
    }
    out_path = REPO / "outputs" / "discrete_bianchi_scaling_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
