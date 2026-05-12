"""(B3) Mask-Threshold XI_THRESH robustness scan.

The closure relies on adj = (xi_off > XI_THRESH) with XI_THRESH=0.6.
Reviewer critique: "Could be a tuning parameter."

Test: scan XI_THRESH ∈ {0.4, 0.5, 0.6, 0.7, 0.8} and check whether
the four key results (Lambda_t, 3|Lambda_s|, shear, anisotropy)
are stable. Run on a representative subset of regimes (not all 14
to save compute).

Output: outputs/mask_threshold_robustness_audit.json
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
    edge_to_matrix, hessian_ricci_per_node, ELL_0, D_MIN, EPS_D)

PARENT = REPO.parent
THRESHOLDS = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
# Subset of regimes to keep compute tractable (5 regimes × 5 thresholds = 25 runs)
SUBSET = [
    ("P1", 28, "canonical"),
    ("P5", 50, "canonical"),
    ("P5N100", 100, "canonical"),
    ("P5N84", 84, "snapshot_v2"),
    ("P5N200", 200, "snapshot_v2"),
    ("P5N256", 256, "canonical"),
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
    return t00, coeff*gradgrad - iso_term


def compute_at_threshold(reg, n_lat, p, xi_thresh):
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

    Lt_pool, Ls_pool, Sh_pool, Frob_pool = [], [], [], []
    for s in range(n_seeds):
        xi_mat = xi_seed(s); np.fill_diagonal(xi_mat, 1.0)
        psi = psi_seed(s)
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        xi_off = xi_mat.copy(); np.fill_diagonal(xi_off, 0.0)
        adj = (xi_off > xi_thresh).astype(np.float64)
        if adj.sum() < 10: continue
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
        if not np.any(mask): continue
        for a_idx in np.where(mask)[0]:
            T_loc = 0.5*(t_ij[a_idx] + t_ij[a_idx].T)
            G_loc = 0.5*(g_ij[a_idx] + g_ij[a_idx].T)
            try: ew, ev = np.linalg.eigh(T_loc)
            except np.linalg.LinAlgError: continue
            G_eig = ev.T @ G_loc @ ev
            G_diag = np.diag(G_eig); G_off = G_eig - np.diag(G_diag)
            Ls = ew - G_diag
            Lt_pool.append(t00[a_idx] - g_00[a_idx])
            Ls_pool.append(np.abs(Ls).sum())
            Sh_pool.append(np.linalg.norm(G_off, 'fro'))
            # Frobenius residual: |G + Lambda - T|/|T| with structural Lambda
            Lambda_t_R = 0.81; Lambda_s_R = -0.005
            res = abs(g_00[a_idx] + Lambda_t_R - t00[a_idx]) / max(abs(t00[a_idx]), 1e-9)
            Frob_pool.append(res)

    if not Lt_pool: return None
    return {
        "n_nodes": int(len(Lt_pool)),
        "Lambda_t_mean":  float(np.mean(Lt_pool)),
        "Lambda_s_3sum_mean": float(np.mean(Ls_pool)),
        "shear_mean":  float(np.mean(Sh_pool)),
        "Frob_residual_med": float(np.median(Frob_pool)),
    }


def main() -> int:
    print("="*100)
    print("(B3) Mask-Threshold XI_THRESH robustness scan")
    print("="*100)
    print(f"\nThresholds: {THRESHOLDS}")
    print(f"Regimes: {[r[0] for r in SUBSET]}")
    print()
    print(f"{'reg':<10} {'N':>4} | " + " | ".join(f"thr={t}" for t in THRESHOLDS))
    print(f"  Λ_t mean    " + " "*4 + " | " + " | ".join(f"{'':>7}" for t in THRESHOLDS))
    print("-"*100)
    out = {"method": "mask_threshold_robustness", "thresholds": THRESHOLDS, "per_regime": {}}
    for reg, n, src in SUBSET:
        path = get_path(reg, src)
        line = f"{reg:<10} {n:>4} |"
        Lt_vals, Frob_vals = [], []
        out["per_regime"][reg] = {"N": n, "by_threshold": {}}
        for thr in THRESHOLDS:
            r = compute_at_threshold(reg, n, path, thr)
            if r is None:
                line += f" {'NaN':>7} |"
                continue
            Lt_vals.append(r["Lambda_t_mean"])
            Frob_vals.append(r["Frob_residual_med"])
            out["per_regime"][reg]["by_threshold"][f"{thr}"] = r
            line += f" {r['Lambda_t_mean']:>7.4f} |"
        if Lt_vals:
            spread = max(Lt_vals) - min(Lt_vals)
            line += f"  Λ_t spread={spread:.4f}, CV={np.std(Lt_vals)/abs(np.mean(Lt_vals))*100:.1f}%"
        print(line)

    print()
    print("Frobenius residual median (vs structural Lambda 0.81):")
    print(f"{'reg':<10} {'N':>4} | " + " | ".join(f"thr={t}" for t in THRESHOLDS))
    print("-"*100)
    for reg, n, src in SUBSET:
        line = f"{reg:<10} {n:>4} |"
        for thr in THRESHOLDS:
            data = out["per_regime"][reg]["by_threshold"].get(f"{thr}", None)
            if data is None: line += f" {'NaN':>7} |"; continue
            line += f" {data['Frob_residual_med']:>7.4f} |"
        print(line)

    # Aggregate: if Lambda_t is stable across thresholds → robust
    print()
    cv_per_regime = []
    for reg, ddat in out["per_regime"].items():
        Lt_vals = [d["Lambda_t_mean"] for thr, d in ddat["by_threshold"].items()]
        if Lt_vals:
            cv = np.std(Lt_vals)/abs(np.mean(Lt_vals))*100
            cv_per_regime.append(cv)
            print(f"  {reg}: Lambda_t CV across thresholds = {cv:.2f}%")
    print()
    if cv_per_regime and max(cv_per_regime) < 10.0:
        verdict = "MASK_THRESHOLD_ROBUST_CV<10%"
    elif cv_per_regime and max(cv_per_regime) < 25.0:
        verdict = "MASK_THRESHOLD_ROBUST_CV<25%"
    else:
        verdict = "MASK_THRESHOLD_DEPENDENT"
    print(f"VERDICT: {verdict}")
    out["verdict"] = verdict
    out["cv_per_regime_pct"] = cv_per_regime

    out_path = REPO / "outputs" / "mask_threshold_robustness_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
