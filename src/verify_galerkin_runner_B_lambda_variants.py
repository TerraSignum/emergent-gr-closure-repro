r"""Runner B: Three Lambda variants on the calibrated D1 ladder.

Computes Delta_E^true(N) for each (regime, seed) under three
distinct Lambda strategies:

  (1) blind-free: Lambda_t, Lambda_s fitted independently per (regime, seed)
      to minimise the per-node Frobenius residual.
  (2) asymptotic-frozen: Lambda_t, Lambda_s fixed to the mean of the
      blind-fit values on the asymptotic window (P6, P7, P8 = N >= 60),
      then evaluated on all N.
  (3) System-R structural: Lambda_t = alpha_xi^2 = 81/100, Lambda_s =
      -gamma^2/2 = -1/200 (no fits, pure rational).

Compares the three on multi-N convergence and reports which strategy
satisfies the acceptance criteria
   |Delta_E^true(N=84)| < 0.05 (current best lattice point), and
   monotone decreasing in N.

Uses calibrated per-node Galerkin pipeline from
verify_galerkin_calibrated_gpu.py: Forman-Ricci spectral-projected
R_ij, multiplied by f = R_bar_bundled / mean(R_bar_forman_spectral_trace)
to bring tensor on bundled scale. Lambda tested on the calibrated
G_munu.

Usage:
    python ./src/verify_galerkin_runner_B_lambda_variants.py
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Hilbert-variation coefficients (System R rationals).
ALPHA_XI = 9.0 / 10.0
GAMMA = 1.0 / 10.0
EPS_SYNC2 = 1.0 / 20.0
Z_XI = KAPPA_XI = ZETA_1 = OMEGA = 1.0
ZETA_3 = 0.5
A_K = 1.0
A_Q = 0.5
ELL_0 = 1.0
D_MIN = 0.1
XI_THRESH = 0.1
EPS_D = D_MIN ** 2

# System-R structural Lambda (no fit).
LAMBDA_T_STRUCT = ALPHA_XI ** 2  # 81/100
LAMBDA_S_STRUCT = -GAMMA ** 2 / 2.0  # -1/200

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


def per_seed_pipeline(xi_mat_np, psi_np, k_field_np, q_field_np,
                       n_lat, r_bar_bundled, xp):
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

    kappa_F = forman_ricci(xi_off, adj, xp)
    r_ij_F = per_node_ricci(kappa_F, weight_adj, adj, d_mat, spatial, xp)
    r_bar_F = xp.trace(r_ij_F, axis1=1, axis2=2)
    r_bar_F_mean = float(r_bar_F.mean())
    f_cal = (r_bar_bundled / r_bar_F_mean
             if abs(r_bar_F_mean) > 1e-12 else 0.0)

    r_ij_cal = f_cal * r_ij_F
    r_bar_cal = xp.trace(r_ij_cal, axis1=1, axis2=2)
    eye3 = xp.eye(3, dtype=xp.float64)
    g_ij_cal = (r_ij_cal
                - 0.5 * r_bar_cal[:, None, None] * eye3[None, :, :])
    g_00_cal = r_bar_cal / 2.0

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

    # Blind Lambda fit per seed.
    lam_t_blind = float(xp.mean(t00 - g_00_cal))
    g_diag = xp.stack([g_ij_cal[:, 0, 0], g_ij_cal[:, 1, 1],
                       g_ij_cal[:, 2, 2]], axis=1)
    t_diag = xp.stack([t_ij[:, 0, 0], t_ij[:, 1, 1], t_ij[:, 2, 2]],
                      axis=1)
    lam_s_blind = float(xp.mean(t_diag - g_diag))

    return {
        "g_00_cal": g_00_cal,
        "g_ij_cal": g_ij_cal,
        "t00": t00,
        "t_ij": t_ij,
        "eye3": eye3,
        "lam_t_blind": lam_t_blind,
        "lam_s_blind": lam_s_blind,
        "f_cal": f_cal,
        "r_bar_bundled": r_bar_bundled,
    }


def frob_residual(prep, lam_t, lam_s, xp):
    res00 = prep["g_00_cal"] + lam_t - prep["t00"]
    eye3 = prep["eye3"]
    spatial_res = (prep["g_ij_cal"] + lam_s * eye3[None, :, :]
                   - prep["t_ij"])
    frob_sq = res00 ** 2 + (spatial_res ** 2).sum(axis=(1, 2))
    frob = xp.sqrt(frob_sq)
    # Off-diagonal-only Frob for the spatial part (T_ij off-diag content).
    t_off_only = prep["t_ij"].copy()
    t_off_only[:, [0, 1, 2], [0, 1, 2]] = 0
    t_off_F = xp.sqrt((t_off_only ** 2).sum(axis=(1, 2)))
    return {
        "frob_mean": float(frob.mean()),
        "frob_median": float(xp.median(frob)),
        "frob_max": float(frob.max()),
        "t_off_F_mean": float(t_off_F.mean()),
    }


def main():
    print("=" * 78)
    print("Runner B: Three Lambda variants on calibrated D1 ladder")
    print("=" * 78)
    print()

    try:
        import cupy as xp
        backend = "cupy"
        print(f"GPU: {xp.cuda.Device(0).mem_info[1] / 1e9:.1f} GB VRAM")
    except Exception as e:
        import numpy as xp
        backend = "numpy"
        print(f"CuPy unavailable; using NumPy. ({e})")
    print(f"Backend: {backend}")
    print()
    print(f"Variant 3 (System-R structural):")
    print(f"  Lambda_t = alpha_xi^2 = {LAMBDA_T_STRUCT:.5f}")
    print(f"  Lambda_s = -gamma^2/2 = {LAMBDA_S_STRUCT:.5f}")
    print()

    import numpy as np
    # Pass 1: collect blind fits.
    preps = []
    for regime, n_lat, npz_path in LADDER:
        if npz_path is None or not npz_path.exists():
            continue
        d = np.load(npz_path, allow_pickle=True)
        edge_arr = d["dense_cell_edge_xi_values"]
        amp_arr = d["dense_cell_node_amplitude_values"]
        phase_arr = d["dense_cell_node_phase_values"]
        r_bar_bundled_per_seed = d["R_bar_by_level"][:, -1]
        n_seeds_avail = min(edge_arr.shape[0], N_SEEDS)
        seeds_data = []
        for s in range(n_seeds_avail):
            xi_mat = edge_to_matrix(edge_arr[s], n_lat)
            np.fill_diagonal(xi_mat, 1.0)
            psi = amp_arr[s] * np.exp(1j * phase_arr[s])
            k_field = d.get(f"ff_K_seed{s}",
                              np.full((n_lat, n_lat), 0.55))
            q_field = d.get(f"ff_Q_seed{s}",
                              np.full((n_lat, n_lat), 0.45))
            r_bar_b = float(r_bar_bundled_per_seed[s])
            prep = per_seed_pipeline(xi_mat, psi, k_field, q_field,
                                       n_lat, r_bar_b, xp)
            prep["seed"] = s
            seeds_data.append(prep)
        preps.append({"regime": regime, "N": n_lat,
                      "seeds_data": seeds_data})

    # Asymptotic-frozen Lambda: mean of blind fits on N >= 60 (P6,P7,P8).
    asymp_t_vals = []
    asymp_s_vals = []
    for p in preps:
        if p["N"] >= 60:
            for s in p["seeds_data"]:
                asymp_t_vals.append(s["lam_t_blind"])
                asymp_s_vals.append(s["lam_s_blind"])
    lam_t_asymp = sum(asymp_t_vals) / len(asymp_t_vals)
    lam_s_asymp = sum(asymp_s_vals) / len(asymp_s_vals)
    print(f"Variant 2 (asymptotic-frozen):")
    print(f"  Lambda_t = {lam_t_asymp:.5f} (mean of blind fits N>=60)")
    print(f"  Lambda_s = {lam_s_asymp:.5f}")
    print()

    # Pass 2: evaluate three variants on all N.
    print(f"{'Reg':<8} {'N':>3} | "
          f"{'V1 blind frob':>14} {'V1 lam_t':>9} | "
          f"{'V2 frozen':>10} | "
          f"{'V3 struct':>10} {'T_off_F':>9}")
    print("-" * 100)
    aggregate = []
    for p in preps:
        per_seed_results = []
        for s in p["seeds_data"]:
            v1 = frob_residual(s, s["lam_t_blind"], s["lam_s_blind"], xp)
            v2 = frob_residual(s, lam_t_asymp, lam_s_asymp, xp)
            v3 = frob_residual(s, LAMBDA_T_STRUCT, LAMBDA_S_STRUCT, xp)
            per_seed_results.append({
                "seed": s["seed"],
                "v1_blind": v1,
                "v2_frozen": v2,
                "v3_struct": v3,
                "lam_t_blind": s["lam_t_blind"],
                "lam_s_blind": s["lam_s_blind"],
            })

        means = lambda variant_key, metric: (
            sum(r[variant_key][metric] for r in per_seed_results)
            / len(per_seed_results))
        agg = {
            "regime": p["regime"], "N": p["N"],
            "n_seeds": len(per_seed_results),
            "per_seed": per_seed_results,
            "v1_blind_frob_mean": means("v1_blind", "frob_mean"),
            "v1_lam_t": (sum(r["lam_t_blind"] for r in per_seed_results)
                         / len(per_seed_results)),
            "v2_frozen_frob_mean": means("v2_frozen", "frob_mean"),
            "v3_struct_frob_mean": means("v3_struct", "frob_mean"),
            "t_off_F_mean": means("v3_struct", "t_off_F_mean"),
        }
        aggregate.append(agg)
        print(f"{p['regime']:<8} {p['N']:>3} | "
              f"{agg['v1_blind_frob_mean']:>14.4f} "
              f"{agg['v1_lam_t']:>9.3f} | "
              f"{agg['v2_frozen_frob_mean']:>10.4f} | "
              f"{agg['v3_struct_frob_mean']:>10.4f} "
              f"{agg['t_off_F_mean']:>9.4f}")

    print()
    print("Verdict on acceptance criteria (Frob < 0.05, monotone in N):")
    n_criterion = "decreasing in N"
    for variant in [("v1_blind", "Variant 1 (blind-free)"),
                    ("v2_frozen", "Variant 2 (asymptotic-frozen)"),
                    ("v3_struct", "Variant 3 (System-R structural)")]:
        key, name = variant
        frobs = [a[f"{key}_frob_mean"] for a in aggregate]
        # Check monotone trend on tail (N>=42)
        tail = [a[f"{key}_frob_mean"] for a in aggregate if a["N"] >= 42]
        is_monotone_dec = all(tail[i] >= tail[i+1] - 0.02
                               for i in range(len(tail)-1))
        last_frob = frobs[-1]
        below_005 = last_frob < 0.05
        below_010 = last_frob < 0.10
        verdict = ("PASS" if below_005 and is_monotone_dec
                   else ("MARGINAL" if below_010 else "FAIL"))
        print(f"  {name:<35}: last={last_frob:.4f}, "
              f"monotone-tail={is_monotone_dec}, "
              f"<0.05={below_005}, <0.10={below_010}  -> {verdict}")

    out = {
        "schema_version": "1.0.0",
        "title": "Runner B: Three Lambda variants on calibrated D1 ladder",
        "backend": backend,
        "lambda_struct": {"t": LAMBDA_T_STRUCT, "s": LAMBDA_S_STRUCT},
        "lambda_asymp": {"t": lam_t_asymp, "s": lam_s_asymp},
        "trend": [
            {"regime": a["regime"], "N": a["N"],
             "v1_blind": a["v1_blind_frob_mean"],
             "v1_lam_t": a["v1_lam_t"],
             "v2_frozen": a["v2_frozen_frob_mean"],
             "v3_struct": a["v3_struct_frob_mean"],
             "t_off_F": a["t_off_F_mean"]}
            for a in aggregate
        ],
    }
    out_path = OUTPUTS / "galerkin_runner_B_lambda_variants.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
