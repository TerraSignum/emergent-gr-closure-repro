r"""Calibrated per-node Galerkin Frobenius residual on D1 ladder.

Calibrates the Forman-Ricci-based per-node Ricci tensor R_ij^Forman(a) to
the bundled lattice scalar curvature R_bar_bundled (which uses the
lattice's own Ricci routine; positive convention, regime-mean per seed)
via a per-(regime, seed) scaling factor

  f = R_bar_bundled / mean_a R_bar_forman(a)

Then the calibrated Ricci tensor is

  R_ij^cal(a) = f * R_ij^Forman(a)
  R_bar^cal(a) = f * R_bar_forman(a)
  G_munu^cal(a) = R_munu^cal(a) - 0.5 * g_munu * R_bar^cal(a)

This puts our per-node Galerkin in the same numerical scale as the
manuscript's bundled R_bar diagnostics. The Frobenius residual
||G + Lambda g - 8 pi G T||_F per node is then directly comparable to
the manuscript's |G_00 + Lambda - T_00| pointwise diagnostic
(< 0.05 on N >= 30, max 0.019 at N = 84) and the Frobenius-decomposed
proxy (sub-0.05 floor on the regime ensemble).

The calibration factor f is reported per (regime, seed) so its
distribution and N-dependence are explicit. A constant f across N
indicates that Forman-Ricci and bundled-R_bar measure the same
physical quantity up to a sign+magnitude convention. A drifting f(N)
indicates a discretisation-scale mismatch.

Usage:
    python ./src/verify_galerkin_calibrated_gpu.py
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

LAMBDA_T = 4.0 / 3.0
LAMBDA_S = -0.37

Z_XI = KAPPA_XI = ZETA_1 = OMEGA = 1.0
ZETA_3 = 0.5
A_K = 1.0
A_Q = 0.5

ELL_0 = 1.0
D_MIN = 0.1
XI_THRESH = 0.1
EPS_D = D_MIN ** 2

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _d1_npz_discovery import find_d1_npz

LADDER_REGIMES = [
    ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
    ("P4", 42), ("P5", 50),
    ("P6", 60), ("P7", 72), ("P8", 84),
]
LADDER = [(r, n, find_d1_npz(r, REPO)) for r, n in LADDER_REGIMES]
N_SEEDS = 4


def forman_ricci(xi_off, adj, xp):
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1)
    sqrt_w = xp.sqrt(deg[:, None] * weight_adj)
    total_a = sqrt_w.sum(axis=1)
    s_a = total_a[:, None] - sqrt_w
    sqrt_w_t = xp.sqrt(deg[None, :] * weight_adj.T)
    total_b = sqrt_w_t.sum(axis=0)
    sqrt_w_ba = xp.sqrt(deg[None, :] * weight_adj)
    s_b = total_b[None, :] - sqrt_w_ba
    forman = (deg[:, None] + deg[None, :]) - s_a - s_b
    return forman * adj


def per_node_ricci(kappa, weight_adj, adj, d_mat, spatial, xp):
    spatial_diff = spatial[None, :, :] - spatial[:, None, :]
    inv_d = xp.where(adj > 0, 1.0 / d_mat, 0.0)
    e_alpha = spatial_diff * inv_d[:, :, None]
    weight_kappa = weight_adj * kappa
    r_ij = xp.einsum("ab,abi,abj->aij", weight_kappa, e_alpha, e_alpha)
    norm = (weight_adj.sum(axis=1) + 1e-12)
    return r_ij / norm[:, None, None]


def edge_to_matrix(edges, n):
    import numpy as np
    if hasattr(edges, "shape") and edges.shape == (n, n):
        return edges.copy()
    edges = list(edges)
    mat = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if idx < len(edges):
                mat[i, j] = edges[idx]
                mat[j, i] = edges[idx]
                idx += 1
    return mat


def calibrated_per_seed(xi_mat_np, psi_np, k_field_np, q_field_np,
                          n_lat, r_bar_bundled, xp):
    """Compute calibrated per-node Galerkin Frobenius residual.

    Returns dict including:
      - calibration_factor f
      - Forman-derived R_bar_per_node (uncalibrated)
      - Calibrated R_bar_per_node (matches bundled scale)
      - Per-node Frobenius residual under imported Lambda
      - Per-node Frobenius residual under blind Lambda fit
    """
    xi_mat = xp.asarray(xi_mat_np)
    psi = xp.asarray(psi_np)
    k_field = xp.asarray(k_field_np)
    q_field = xp.asarray(q_field_np)

    xi_off = xi_mat.copy()
    xp.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(xp.float64)
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1) + 1e-12
    deg_inv_sqrt = 1.0 / xp.sqrt(deg)
    l_norm = (xp.eye(n_lat, dtype=xp.float64)
              - (deg_inv_sqrt[:, None] * weight_adj
                 * deg_inv_sqrt[None, :]))
    eigvals_l, eigvecs_l = xp.linalg.eigh(l_norm)
    spatial = eigvecs_l[:, 1:4]

    d_mat = -ELL_0 * xp.log(xp.maximum(xi_off, 1e-12))
    d_mat = xp.maximum(d_mat, D_MIN)
    d_sq_safe = xp.where(adj > 0, d_mat * d_mat, xp.inf)
    weight_grad = xp.where(adj > 0, weight_adj / (d_sq_safe + EPS_D),
                            0.0)
    omega_a = weight_grad.sum(axis=1)

    # Forman-Ricci per edge.
    kappa_F = forman_ricci(xi_off, adj, xp)

    # Forman-derived R_ij and R_bar (uncalibrated).
    r_ij_F = per_node_ricci(kappa_F, weight_adj, adj, d_mat, spatial, xp)
    r_bar_F = xp.trace(r_ij_F, axis1=1, axis2=2)
    r_bar_F_mean = float(r_bar_F.mean())

    # Calibration factor: bring R_bar_F_mean to R_bar_bundled.
    if abs(r_bar_F_mean) > 1e-12:
        f_cal = r_bar_bundled / r_bar_F_mean
    else:
        f_cal = 0.0

    # Apply calibration to R_ij and derived G_munu.
    r_ij_cal = f_cal * r_ij_F
    r_bar_cal = xp.trace(r_ij_cal, axis1=1, axis2=2)
    eye3 = xp.eye(3, dtype=xp.float64)
    g_ij_cal = r_ij_cal - 0.5 * r_bar_cal[:, None, None] * eye3[None, :, :]
    g_00_cal = r_bar_cal / 2.0

    # Per-node T_munu using bundled K, Q row-mean.
    spatial_diff = spatial[None, :, :] - spatial[:, None, :]
    inv_d = xp.where(adj > 0, 1.0 / d_mat, 0.0)
    psi_diff = psi[None, :] - psi[:, None]
    weight_term = weight_grad[:, :, None] * inv_d[:, :, None]
    grad_psi = (psi_diff[:, :, None] * spatial_diff * weight_term).sum(
        axis=1) / (omega_a[:, None] + 1e-12)
    coeff = 2 * (0.5 * Z_XI + KAPPA_XI + ZETA_1 * OMEGA)
    iso_subtract = (0.5 * Z_XI + KAPPA_XI + ZETA_1 * OMEGA)
    g_real = xp.real(grad_psi)
    g_imag = xp.imag(grad_psi)
    outer = (g_real[:, :, None] * g_real[:, None, :]
             + g_imag[:, :, None] * g_imag[:, None, :])
    norm_sq = (xp.abs(grad_psi) ** 2).sum(axis=1)
    t_ij = (coeff * outer
            - iso_subtract * norm_sq[:, None, None]
              * eye3[None, :, :])

    xi_row_mean = (weight_adj.sum(axis=1) / (adj.sum(axis=1) + 1e-12))
    var_xi = (((weight_adj - xi_row_mean[:, None]) ** 2 * adj).sum(
        axis=1) / (adj.sum(axis=1) + 1e-12))
    amp_a = xp.abs(psi)
    var_amp = (amp_a - amp_a.mean()) ** 2
    grad_psi_sq = norm_sq

    k_per = (k_field * adj).sum(axis=1) / (adj.sum(axis=1) + 1e-12)
    q_per = (q_field * adj).sum(axis=1) / (adj.sum(axis=1) + 1e-12)
    k_rec = A_K * k_per + A_Q * (1.0 - q_per)

    t00 = (0.5 * Z_XI * var_xi
           + KAPPA_XI * var_amp
           + ZETA_1 * OMEGA * grad_psi_sq
           + ZETA_3 * OMEGA * k_rec)

    # Imported Lambda residual.
    res00_imp = g_00_cal + LAMBDA_T - t00
    spatial_res_imp = (g_ij_cal + LAMBDA_S * eye3[None, :, :]) - t_ij
    frob_imp_sq = (res00_imp ** 2
                   + (spatial_res_imp ** 2).sum(axis=(1, 2)))
    frob_imp = xp.sqrt(frob_imp_sq)

    # Blind Lambda fit (per seed).
    lam_t_blind = float(xp.mean(t00 - g_00_cal))
    g_diag = xp.stack([g_ij_cal[:, 0, 0], g_ij_cal[:, 1, 1],
                       g_ij_cal[:, 2, 2]], axis=1)
    t_diag = xp.stack([t_ij[:, 0, 0], t_ij[:, 1, 1], t_ij[:, 2, 2]],
                      axis=1)
    lam_s_blind = float(xp.mean(t_diag - g_diag))

    res00_blind = g_00_cal + lam_t_blind - t00
    spatial_res_blind = ((g_ij_cal + lam_s_blind * eye3[None, :, :])
                         - t_ij)
    frob_blind_sq = (res00_blind ** 2
                     + (spatial_res_blind ** 2).sum(axis=(1, 2)))
    frob_blind = xp.sqrt(frob_blind_sq)

    return {
        "r_bar_bundled": r_bar_bundled,
        "r_bar_forman_mean": r_bar_F_mean,
        "calibration_factor_f": f_cal,
        "r_bar_cal_per_node_mean": float(r_bar_cal.mean()),
        "r_bar_cal_per_node_std": float(r_bar_cal.std()),
        "t00_per_node_mean": float(t00.mean()),
        "g_00_cal_per_node_mean": float(g_00_cal.mean()),
        "lam_t_blind": lam_t_blind,
        "lam_s_blind": lam_s_blind,
        "frob_imp_per_node_mean": float(frob_imp.mean()),
        "frob_imp_per_node_median": float(xp.median(frob_imp)),
        "frob_imp_per_node_max": float(frob_imp.max()),
        "frob_blind_per_node_mean": float(frob_blind.mean()),
        "frob_blind_per_node_median": float(xp.median(frob_blind)),
    }


def main():
    print("=" * 78)
    print("Calibrated per-node Galerkin Frobenius residual")
    print("(Forman-Ricci -> bundled R_bar scale)")
    print("=" * 78)
    print()

    try:
        import cupy as xp
        backend = "cupy"
        print(f"GPU: {xp.cuda.Device(0).mem_info[1]/1e9:.1f} GB VRAM")
    except Exception as e:
        import numpy as xp
        backend = "numpy"
        print(f"CuPy unavailable; using NumPy. ({e})")
    print(f"Backend: {backend}")
    print(f"Imported Lambda: t={LAMBDA_T:.3f}, s={LAMBDA_S:.3f}")
    print()

    import numpy as np
    aggregate = []
    print(f"{'Reg':<8} {'N':>3} {'<R_F>':>10} {'<R_bun>':>10} "
          f"{'f':>10} {'<Frob_imp>':>12} {'med_imp':>10} "
          f"{'<Frob_blind>':>14} {'lam_t*':>8} {'lam_s*':>8}")
    print("-" * 110)
    for regime, n_lat, npz_path in LADDER:
        if npz_path is None or not npz_path.exists():
            continue
        d = np.load(npz_path, allow_pickle=True)
        edge_arr = d["dense_cell_edge_xi_values"]
        amp_arr = d["dense_cell_node_amplitude_values"]
        phase_arr = d["dense_cell_node_phase_values"]
        # Bundled R_bar at deepest cg level, per seed.
        r_bar_bundled_per_seed = d["R_bar_by_level"][:, -1]
        n_seeds_avail = min(edge_arr.shape[0], N_SEEDS)

        per_seed_results = []
        for s in range(n_seeds_avail):
            xi_mat = edge_to_matrix(edge_arr[s], n_lat)
            np.fill_diagonal(xi_mat, 1.0)
            psi = amp_arr[s] * np.exp(1j * phase_arr[s])
            k_field = (d.get(f"ff_K_seed{s}",
                              np.full((n_lat, n_lat), 0.55)))
            q_field = (d.get(f"ff_Q_seed{s}",
                              np.full((n_lat, n_lat), 0.45)))
            r_bar_b = float(r_bar_bundled_per_seed[s])
            result = calibrated_per_seed(xi_mat, psi, k_field, q_field,
                                          n_lat, r_bar_b, xp)
            result["seed"] = s
            per_seed_results.append(result)

        means = lambda key: (sum(r[key] for r in per_seed_results)
                              / len(per_seed_results))
        agg = {
            "regime": regime,
            "N": n_lat,
            "n_seeds": n_seeds_avail,
            "per_seed": per_seed_results,
            "mean_r_bar_forman": means("r_bar_forman_mean"),
            "mean_r_bar_bundled": means("r_bar_bundled"),
            "mean_calibration_f": means("calibration_factor_f"),
            "mean_frob_imp": means("frob_imp_per_node_mean"),
            "median_frob_imp": means("frob_imp_per_node_median"),
            "mean_frob_blind": means("frob_blind_per_node_mean"),
            "median_frob_blind": means("frob_blind_per_node_median"),
            "mean_lam_t_blind": means("lam_t_blind"),
            "mean_lam_s_blind": means("lam_s_blind"),
        }
        aggregate.append(agg)
        print(f"{regime:<8} {n_lat:>3} "
              f"{agg['mean_r_bar_forman']:>10.3f} "
              f"{agg['mean_r_bar_bundled']:>10.3f} "
              f"{agg['mean_calibration_f']:>10.4f} "
              f"{agg['mean_frob_imp']:>12.4f} "
              f"{agg['median_frob_imp']:>10.4f} "
              f"{agg['mean_frob_blind']:>14.4f} "
              f"{agg['mean_lam_t_blind']:>8.3f} "
              f"{agg['mean_lam_s_blind']:>8.3f}")

    print()
    print("Calibration factor f drift (should be regime/N-dependent if")
    print("Forman and bundled R_bar are different physical quantities).")
    print()
    print("Reference: bundled |G_00 + Lambda - T_00| < 0.05 pointwise")
    print("           on N >= 30 (manuscript principal closure claim).")

    out = {
        "schema_version": "1.0.0",
        "title": ("Calibrated per-node Galerkin Frobenius "
                  "residual on D1 ladder"),
        "backend": backend,
        "calibration_method": ("f = R_bar_bundled / mean_a "
                                "R_bar_forman_spectral_trace"),
        "imported_lambda": {"t": LAMBDA_T, "s": LAMBDA_S},
        "trend": [
            {"regime": a["regime"], "N": a["N"],
             "f": a["mean_calibration_f"],
             "frob_imp_mean": a["mean_frob_imp"],
             "frob_imp_median": a["median_frob_imp"],
             "frob_blind_mean": a["mean_frob_blind"],
             "frob_blind_median": a["median_frob_blind"],
             "lam_t_blind": a["mean_lam_t_blind"],
             "lam_s_blind": a["mean_lam_s_blind"]}
            for a in aggregate
        ],
    }
    out_path = OUTPUTS / "galerkin_calibrated_gpu.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
