"""Reviewer follow-up C: direct effective-Bianchi audit on the
source side of the Einstein-class equation.

The geometric Bianchi audit reports
    B^nu_geom(a) := nabla^disc_mu G^{mu nu}(a)
and finds Symanzik-2 asymptote ~1.6e-3 with R^2=0.96 on the
cleaned twelve-regime ladder. By the field equation
    G_{mu nu} + Lambda^{back}_{mu nu} = 8 pi G T^Xi_{mu nu}
the source-side divergence
    B^nu_eff(a) := nabla^disc_mu [8 pi G T^{Xi,mu nu}(a)
                                   - Lambda^{back, mu nu}(a)]
should equal B^nu_geom up to lattice quadrature noise -- BUT only
if T^Xi_{mu nu} and Lambda^back_{mu nu} are computed independently
of G_{mu nu}, NOT through the field equation as a back-substitution.

Here we compute B^nu_eff directly:
- T^Xi_{mu nu}(a) is taken from the spectral-Hilbert source-tensor
  pipeline (t_munu_spectral on the per-seed K, Q reconstruction),
- Lambda^back_{mu nu}(a) is taken in its local form
  Lambda_t^back(a) = T_{00}^{rec}(a) on the time diagonal and
  Lambda_s^back = -gamma^2/2 = -0.005 on the spatial diagonal
  (the canonical structural form documented in the parent paper).

The discrete divergence is then computed exactly as in
verify_discrete_bianchi_scaling.py for the geometric audit, and
the Symanzik-2 continuum extrapolation is reported alongside.

Output: outputs/stage6f_effective_bianchi_source_side.json
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

from _d1_npz_discovery import find_d1_npz  # noqa: E402
from verify_discrete_bianchi_scaling import (  # noqa: E402
    discrete_bianchi_per_seed, symanzik_24, power_law)
from verify_galerkin_runner_A_hessian_ricci import (  # noqa: E402
    edge_to_matrix, ELL_0, D_MIN, EPS_D, XI_THRESH, t_munu_spectral)

OUT = REPO / "outputs" / "stage6f_effective_bianchi_source_side.json"

LAMBDA_S_STRUCT = -0.005   # -gamma^2/2 = -1/200 (parent paper)
EIGHTPI_G = 8.0 * np.pi    # 8 pi G with G = 1 in lattice units

LADDER = [
    ("P1",     28, "canonical"),
    ("P3",     36, "canonical"),
    ("P4",     42, "canonical"),
    ("P5",     50, "canonical"),
    ("P5N64",  64, "canonical"),
    ("P6",     60, "canonical"),
    ("P7",     72, "canonical"),
    ("P8",     84, "canonical"),
    ("P5N100", 100, "canonical"),
    ("P5N200", 200, "snapshot_v2"),
    ("P5N256", 256, "canonical"),
    ("P5N512", 512, "canonical"),
]


def get_path(reg, src):
    if src == "canonical":
        return find_d1_npz(reg, REPO)
    return REPO.parent / f"results_d1_{reg.lower()}_v2" / f"{reg}.snapshots.npz"


def _compute_S_components(xi_mat, psi, k_field, q_field, n_lat):
    xi_mat = np.where(np.isfinite(xi_mat), xi_mat, 0.0)
    if np.any(~np.isfinite(psi)):
        psi = np.where(np.isfinite(psi.real) & np.isfinite(psi.imag),
                        psi, 0.0 + 0.0j)
    k_field = np.where(np.isfinite(k_field), k_field, 0.55)
    q_field = np.where(np.isfinite(q_field), q_field, 0.45)
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(np.float64)
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1) + 1e-12
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    l_norm = (np.eye(n_lat) - deg_inv_sqrt[:, None] * weight_adj
              * deg_inv_sqrt[None, :])
    eigvals_l, eigvecs_l = np.linalg.eigh(l_norm)
    spatial = eigvecs_l[:, 1:4]
    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat = np.maximum(d_mat, D_MIN)
    d_sq = d_mat * d_mat
    d_sq_safe = np.where(adj > 0, d_sq, np.inf)
    weight_grad = np.where(adj > 0, weight_adj / (d_sq_safe + EPS_D), 0.0)
    omega_a = weight_grad.sum(axis=1)

    t00, t_ij = t_munu_spectral(xi_off, adj, weight_grad, d_mat,
                                  spatial, omega_a, psi, k_field, q_field, np)

    # Local Lambda^back: time-component = T_00^rec proxy via row-mean K;
    # spatial isotropic -gamma^2/2 = -0.005.
    lam_t_local = k_field.mean(axis=1)
    lam_s_local = LAMBDA_S_STRUCT * np.ones(n_lat)

    # S^{mu nu}_munu = 8 pi G T^Xi_munu - Lambda^back_munu
    S_00 = EIGHTPI_G * t00 - lam_t_local
    eye3 = np.eye(3)
    S_ij = EIGHTPI_G * t_ij - (lam_s_local[:, None, None]
                                * eye3[None, :, :])
    return S_00, S_ij, spatial, d_mat, weight_grad, xi_mat


def compute_for_regime(reg, n_lat, p):
    if not p or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    if "dense_cell_edge_xi_values" in d.keys():
        e = d["dense_cell_edge_xi_values"]
        a_amp = d["dense_cell_node_amplitude_values"]
        ph = d["dense_cell_node_phase_values"]
        n_seeds = min(e.shape[0], 8)

        def xi_seed(s):
            return edge_to_matrix(e[s], n_lat)

        def psi_seed(s):
            return a_amp[s] * np.exp(1j * ph[s])

        def kq_seed(s):
            n = n_lat
            k_default = 0.55 * np.ones((n, n))
            q_default = 0.45 * np.ones((n, n))
            for key, default in (("ff_K_seed", k_default),
                                  ("ff_Q_seed", q_default)):
                full_key = f"{key}{s}"
                if full_key in d.keys():
                    return d[f"ff_K_seed{s}"], d[f"ff_Q_seed{s}"]
            return k_default, q_default
    elif "edge_xi_snapshots" in d.keys():
        n_seeds = min(int(d["n_seeds"][0]), 6)

        def xi_seed(s):
            return d["edge_xi_snapshots"][s, -1, :, :].copy()

        def psi_seed(s):
            amp = d["amp_snapshots"][s, -1, :]
            ph_ = d["phase_snapshots"][s, -1, :]
            return amp * np.exp(1j * ph_)

        def kq_seed(s):
            if "ff_K_snapshots" in d.keys():
                return (d["ff_K_snapshots"][s, -1, :, :],
                        d["ff_Q_snapshots"][s, -1, :, :])
            return (0.55 * np.ones((n_lat, n_lat)),
                    0.45 * np.ones((n_lat, n_lat)))
    else:
        return None

    B0_pool = []
    Bsp_pool = []
    Bfull_pool = []
    n_used = 0
    for s in range(n_seeds):
        try:
            xi_mat = xi_seed(s)
            psi = psi_seed(s)
            kfld, qfld = kq_seed(s)
            np.fill_diagonal(xi_mat, 1.0)
            S_00, S_ij, spatial, d_mat, weight_grad, xi_full = (
                _compute_S_components(xi_mat, psi, kfld, qfld, n_lat))

            # Use the same divergence operator as the geometric audit:
            # plug S_00, S_ij in place of G_00, G_ij.
            B_time, B_spat, B_full = discrete_bianchi_per_seed(
                xi_full, n_lat, weight_grad, d_mat, spatial,
                S_00, S_ij)
            mask = (np.isfinite(B_full)
                    & np.isfinite(B_time)
                    & np.all(np.isfinite(B_spat), axis=1))
            if mask.sum() == 0:
                continue
            B0_pool.append(np.abs(B_time[mask]))
            Bsp_pool.append(np.linalg.norm(B_spat[mask], axis=1))
            Bfull_pool.append(B_full[mask])
            n_used += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  warn: seed {s} skipped: {exc}")
            continue

    if not B0_pool:
        return None
    B0 = np.concatenate(B0_pool)
    Bsp = np.concatenate(Bsp_pool)
    Bfull = np.concatenate(Bfull_pool)
    return {
        "regime": reg, "N": int(n_lat), "n_seeds": int(n_used),
        "n_nodes": int(len(B0)),
        "B_eff_time_med": float(np.median(B0)),
        "B_eff_time_mean": float(B0.mean()),
        "B_eff_spat_med": float(np.median(Bsp)),
        "B_eff_full_med": float(np.median(Bfull)),
        "B_eff_full_p90": float(np.percentile(Bfull, 90)),
        "B_eff_full_p95": float(np.percentile(Bfull, 95)),
    }


def main():
    rows = []
    for reg, n_lat, src in LADDER:
        p = get_path(reg, src)
        print(f"  {reg} N={n_lat} src={src} path={p}")
        r = compute_for_regime(reg, n_lat, p)
        if r is None:
            print(f"    SKIP (no valid seeds)")
            continue
        rows.append(r)
        print(f"    B_eff_full_med = {r['B_eff_full_med']:.4e}, "
              f"p90 = {r['B_eff_full_p90']:.4e}, "
              f"n_nodes = {r['n_nodes']}")

    rows.sort(key=lambda r: r["N"])
    n_arr = np.array([r["N"] for r in rows], dtype=float)
    out = {
        "method": "direct effective-Bianchi audit on source side",
        "ladder": [{"regime": r["regime"], "N": r["N"],
                     "n_seeds": r["n_seeds"]} for r in rows],
        "per_regime": rows,
    }
    if len(rows) >= 3:
        for stat in ("B_eff_time_med", "B_eff_spat_med",
                     "B_eff_full_med", "B_eff_full_p90"):
            y = np.array([r[stat] for r in rows], dtype=float)
            y_inf, r2 = symanzik_24(n_arr, y)
            alpha, alpha_r2 = power_law(n_arr, y)
            out[f"{stat}_symanzik_24"] = {
                "y_inf": y_inf, "R2": r2,
                "alpha_power_law": alpha,
                "alpha_R2": alpha_r2,
            }

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")
    print()
    for stat in ("B_eff_full_med", "B_eff_full_p90"):
        if f"{stat}_symanzik_24" in out:
            f = out[f"{stat}_symanzik_24"]
            print(f"  {stat}: y_inf = {f['y_inf']:.4e}, "
                  f"R^2 = {f['R2']:.2f}, "
                  f"alpha = {f['alpha_power_law']:.2f}, "
                  f"R^2_pl = {f['alpha_R2']:.2f}")


if __name__ == "__main__":
    main()
