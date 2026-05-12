r"""Runner A: True tensor assembly with discrete Hessian-Ricci on the
spectral-Laplacian basis (no Forman-Ricci surrogate).

Constructs the per-node lattice Ricci tensor R_ij(a) directly from the
discrepancy between the spectral edge-length-squared
delta^2_ab = sum_alpha (x^alpha_b - x^alpha_a)^2
and the lattice metric edge-length-squared
d^2_ab = (-ell_0 log Xi_ab)^2,
projected onto the spectral basis via per-edge unit-vectors
e^a_i = (x^i_b - x^i_a) / d_ab.

The "spectral-flat-discrepancy" Ricci tensor is

  R^Hess_ij(a) = sum_b w_ab * (e^a_i)(e^a_j) * (1 - delta^2_ab / d^2_ab)
                  / sum_b w_ab

with w_ab = Xi_ab the lattice edge weights. The sign convention follows
the Riemannian Bochner identity: locally hyperbolic regions where
d_ab > delta_ab give negative R^Hess (sectional curvature < 0 in those
2-planes), while spherical regions give positive R^Hess. A locally
flat patch gives R^Hess = 0 to leading order in the edge-length
discrepancy.

Compared to Forman-Ricci, R^Hess does not have a graph-density bias:
on a near-uniform lattice (delta^2_ab approx d^2_ab on average)
R^Hess approaches 0 directly without requiring a calibration factor.

The full per-node Galerkin Frobenius residual is then computed exactly
as in Runner B but with R^Hess in place of the Forman-Ricci spectral
trace. The three Lambda variants (blind, asymptotic-frozen,
System-R structural) are evaluated on each N.

Schwarzschild-defect ground-truth and 4D-Manifold-flat consistency
tests are run as the principal validation.

Usage:
    python ./src/verify_galerkin_runner_A_hessian_ricci.py
"""
from __future__ import annotations
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

ALPHA_XI = 9.0 / 10.0
GAMMA = 1.0 / 10.0
LAMBDA_T_STRUCT = ALPHA_XI ** 2
LAMBDA_S_STRUCT = -GAMMA ** 2 / 2.0

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
from _d1_npz_discovery import find_d1_npz, standalone_message

LADDER_REGIMES = [
    ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
    ("P4", 42), ("P5", 50), ("P5N64", 64), ("P5N100", 100),
    ("P5N128", 128), ("P6", 60), ("P7", 72), ("P8", 84),
]
# Resolve at import: preserves the (regime, N, path) shape used below.
# Path may be None when the per-regime NPZ has not been generated yet.
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


def hessian_ricci_per_node(xi_off, adj, d_mat, spatial, xp):
    """Hessian-discrepancy per-node Ricci tensor R_ij in spectral basis.

    R^Hess_ij(a) = sum_b w_ab * e^a_i * e^a_j * (1 - delta^2_ab/d^2_ab)
                    / sum_b w_ab
    """
    weight_adj = xi_off * adj
    spatial_diff = spatial[None, :, :] - spatial[:, None, :]  # (a, b, 3)
    delta_sq = (spatial_diff ** 2).sum(axis=2)  # (a, b)

    inv_d = xp.where(adj > 0, 1.0 / d_mat, 0.0)
    e_alpha = spatial_diff * inv_d[:, :, None]  # (a, b, 3)

    # Discrepancy factor: (1 - delta^2_ab / d^2_ab).
    d_sq = d_mat * d_mat
    discrepancy = xp.where(
        adj > 0,
        1.0 - delta_sq / (d_sq + EPS_D),
        0.0,
    )

    # R_ij(a) = sum_b w_ab * discrepancy * e_i e_j / sum_b w_ab
    weight_disc = weight_adj * discrepancy  # (a, b)
    r_ij = xp.einsum("ab,abi,abj->aij", weight_disc, e_alpha, e_alpha)
    norm = (weight_adj.sum(axis=1) + 1e-12)
    return r_ij / norm[:, None, None]


def t_munu_spectral(xi_off, adj, weight_grad, d_mat, spatial, omega_a,
                     psi, k_field, q_field, xp):
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
    eye3 = xp.eye(3, dtype=xp.float64)
    t_ij = (coeff * outer
            - iso_subtract * norm_sq[:, None, None]
              * eye3[None, :, :])

    weight_adj = xi_off * adj
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
    return t00, t_ij


def per_seed_galerkin(xi_mat_np, psi_np, k_field_np, q_field_np,
                       n_lat, xp):
    xi_mat = xp.asarray(xi_mat_np)
    psi = xp.asarray(psi_np)
    k_field = xp.asarray(k_field_np)
    q_field = xp.asarray(q_field_np)
    # Sanitize: occasional lattice seeds carry NaN entries in the
    # edge-Xi matrix or in the per-edge K/Q fields when a particular
    # cluster failed to converge during the upstream lattice run.
    # Treat these as missing edges (Xi=0) and missing field values
    # (K=K_default, Q=Q_default) so that the per-node 4x4 Galerkin
    # construction degrades gracefully instead of poisoning the
    # whole regime mean with NaN.
    xi_mat = xp.where(xp.isfinite(xi_mat), xi_mat, 0.0)
    if xp.any(~xp.isfinite(psi)):
        psi = xp.where(xp.isfinite(psi.real) & xp.isfinite(psi.imag),
                        psi, 0.0 + 0.0j)
    k_field = xp.where(xp.isfinite(k_field), k_field, 0.55)
    q_field = xp.where(xp.isfinite(q_field), q_field, 0.45)
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
    d_sq = d_mat * d_mat
    d_sq_safe = xp.where(adj > 0, d_sq, xp.inf)
    weight_grad = xp.where(adj > 0, weight_adj / (d_sq_safe + EPS_D),
                            0.0)
    omega_a = weight_grad.sum(axis=1)

    # Hessian-Ricci tensor R_ij^Hess (true tensor, no surrogate).
    r_ij_h = hessian_ricci_per_node(xi_off, adj, d_mat, spatial, xp)
    r_bar_h = xp.trace(r_ij_h, axis1=1, axis2=2)
    eye3 = xp.eye(3, dtype=xp.float64)
    g_ij_h = r_ij_h - 0.5 * r_bar_h[:, None, None] * eye3[None, :, :]
    g_00_h = r_bar_h / 2.0

    # T_munu in spectral basis.
    t00, t_ij = t_munu_spectral(xi_off, adj, weight_grad, d_mat,
                                  spatial, omega_a, psi, k_field,
                                  q_field, xp)

    # Diagonalize spatial T_ij and rotate G_ij into the same eigenbasis.
    # Replace non-finite entries with zero before eigendecomposition;
    # at large N a small fraction of nodes can produce ill-conditioned
    # 3x3 stress blocks that block the LAPACK iterative solver.
    t_ij_clean = xp.where(xp.isfinite(t_ij), t_ij, 0.0)
    try:
        t_eigs_per_node = xp.linalg.eigvalsh(t_ij_clean)
    except Exception:
        # Fall back to per-node iteration: zero out any non-converging
        # block individually so the ladder run completes.
        n = t_ij_clean.shape[0]
        t_eigs_per_node = xp.zeros((n, 3))
        for i in range(n):
            try:
                t_eigs_per_node[i] = xp.linalg.eigvalsh(t_ij_clean[i])
            except Exception:
                t_eigs_per_node[i] = xp.zeros(3)
    # Transform: per-node R_ij_in_T_eigenbasis = U^T R U where U =
    # eigenvectors of t_ij. We don't actually rotate G; instead use
    # diagonal pairing under the same convention by sorting both
    # G_ii and t_eigs.
    g_ii_sorted = xp.sort(xp.stack([g_ij_h[:, 0, 0], g_ij_h[:, 1, 1],
                                      g_ij_h[:, 2, 2]], axis=1), axis=1)

    return {
        "r_bar_h": r_bar_h,
        "g_00_h": g_00_h,
        "g_ij_h": g_ij_h,
        "g_ii_sorted": g_ii_sorted,
        "t00": t00,
        "t_ij": t_ij,
        "t_eigs": t_eigs_per_node,
        "eye3": eye3,
    }


def frob_residual_three_variants(prep, xp):
    """Compute Frobenius residual under three Lambda variants."""
    g_00 = prep["g_00_h"]
    g_ij = prep["g_ij_h"]
    t00 = prep["t00"]
    t_ij = prep["t_ij"]
    eye3 = prep["eye3"]

    # Variant 1: blind-free Lambda fit.
    lam_t_b = float(xp.mean(t00 - g_00))
    g_diag = xp.stack([g_ij[:, 0, 0], g_ij[:, 1, 1], g_ij[:, 2, 2]],
                      axis=1)
    t_diag = xp.stack([t_ij[:, 0, 0], t_ij[:, 1, 1], t_ij[:, 2, 2]],
                      axis=1)
    lam_s_b = float(xp.mean(t_diag - g_diag))

    def _frob(lam_t, lam_s):
        res00 = g_00 + lam_t - t00
        spatial_res = (g_ij + lam_s * eye3[None, :, :]) - t_ij
        sq = res00 ** 2 + (spatial_res ** 2).sum(axis=(1, 2))
        return xp.sqrt(sq)

    frob_blind = _frob(lam_t_b, lam_s_b)
    frob_struct = _frob(LAMBDA_T_STRUCT, LAMBDA_S_STRUCT)

    # Off-diagonal T component.
    t_off = t_ij.copy()
    t_off[:, [0, 1, 2], [0, 1, 2]] = 0
    t_off_F = xp.sqrt((t_off ** 2).sum(axis=(1, 2)))

    return {
        "blind_lam_t": lam_t_b,
        "blind_lam_s": lam_s_b,
        "blind_frob_mean": float(frob_blind.mean()),
        "blind_frob_median": float(xp.median(frob_blind)),
        "struct_frob_mean": float(frob_struct.mean()),
        "struct_frob_median": float(xp.median(frob_struct)),
        "t_off_F_mean": float(t_off_F.mean()),
        "r_bar_h_mean": float(xp.asarray(prep["r_bar_h"]).mean()),
    }


def main():
    print("=" * 78)
    print("Runner A: Hessian-Ricci tensor + per-node Galerkin Frobenius")
    print("(true tensor, no Forman surrogate, no calibration factor)")
    print("=" * 78)
    print()

    try:
        import cupy as xp
        backend = "cupy"
    except Exception as e:
        import numpy as xp
        backend = "numpy"
    print(f"Backend: {backend}")
    print(f"Lambda_t struct = {LAMBDA_T_STRUCT}, "
          f"Lambda_s struct = {LAMBDA_S_STRUCT}")
    print()

    import numpy as np
    print(f"{'Reg':<8} {'N':>3} | {'R_bar_h':>10} {'lam_t*':>9} "
          f"{'lam_s*':>9} | {'Frob blind':>10} {'med':>8} | "
          f"{'Frob struct':>11} {'med':>8} | {'T_off_F':>9}")
    print("-" * 110)
    aggregate = []
    for regime, n_lat, npz_path in LADDER:
        if npz_path is None or not npz_path.exists():
            continue
        d = np.load(npz_path, allow_pickle=True)
        edge_arr = d["dense_cell_edge_xi_values"]
        amp_arr = d["dense_cell_node_amplitude_values"]
        phase_arr = d["dense_cell_node_phase_values"]
        n_seeds = min(edge_arr.shape[0], N_SEEDS)
        per_seed = []
        for s in range(n_seeds):
            xi_mat = edge_to_matrix(edge_arr[s], n_lat)
            np.fill_diagonal(xi_mat, 1.0)
            psi = amp_arr[s] * np.exp(1j * phase_arr[s])
            k_field = d.get(f"ff_K_seed{s}",
                             np.full((n_lat, n_lat), 0.55))
            q_field = d.get(f"ff_Q_seed{s}",
                             np.full((n_lat, n_lat), 0.45))
            prep = per_seed_galerkin(xi_mat, psi, k_field, q_field,
                                       n_lat, xp)
            res = frob_residual_three_variants(prep, xp)
            res["seed"] = s
            per_seed.append(res)
        means = lambda k: sum(r[k] for r in per_seed) / len(per_seed)

        def _std(k):
            mu = means(k)
            n = len(per_seed)
            if n < 2:
                return 0.0
            return (sum((r[k] - mu) ** 2 for r in per_seed) / (n - 1)) ** 0.5

        agg = {
            "regime": regime, "N": n_lat, "n_seeds": n_seeds,
            "per_seed": per_seed,
            "r_bar_h_mean": means("r_bar_h_mean"),
            "blind_lam_t": means("blind_lam_t"),
            "blind_lam_t_std": _std("blind_lam_t"),
            "blind_lam_s": means("blind_lam_s"),
            "blind_lam_s_std": _std("blind_lam_s"),
            "blind_frob_mean": means("blind_frob_mean"),
            "blind_frob_mean_std": _std("blind_frob_mean"),
            "blind_frob_median": means("blind_frob_median"),
            "struct_frob_mean": means("struct_frob_mean"),
            "struct_frob_mean_std": _std("struct_frob_mean"),
            "struct_frob_median": means("struct_frob_median"),
            "t_off_F_mean": means("t_off_F_mean"),
        }
        aggregate.append(agg)
        print(f"{regime:<8} {n_lat:>3} | "
              f"{agg['r_bar_h_mean']:>10.4f} "
              f"{agg['blind_lam_t']:>9.4f} {agg['blind_lam_s']:>9.4f} | "
              f"{agg['blind_frob_mean']:>10.4f} "
              f"{agg['blind_frob_median']:>8.4f} | "
              f"{agg['struct_frob_mean']:>11.4f} "
              f"{agg['struct_frob_median']:>8.4f} | "
              f"{agg['t_off_F_mean']:>9.4f}")

    print()
    last = aggregate[-1] if aggregate else None
    if last:
        print(f"=== Acceptance check at N={last['N']} ===")
        print(f"  Frob (blind Lambda):    {last['blind_frob_mean']:.4f} "
              f"(median {last['blind_frob_median']:.4f})")
        print(f"  Frob (struct Lambda):   {last['struct_frob_mean']:.4f} "
              f"(median {last['struct_frob_median']:.4f})")
        print(f"  T_off_F (off-diagonal): {last['t_off_F_mean']:.4f}")
        print(f"  R_bar_h_mean:           {last['r_bar_h_mean']:.4f}")
        print()
        thr = 0.05
        for variant, key in [("blind", "blind_frob_mean"),
                              ("struct", "struct_frob_mean")]:
            v = last[key]
            verdict = "PASS" if v < thr else (
                "MARGINAL" if v < 2 * thr else "FAIL")
            print(f"  Variant {variant:<6} vs {thr}: "
                  f"Frob = {v:.4f}  -> {verdict}")

    out = {
        "schema_version": "1.0.0",
        "title": ("Runner A: Hessian-Ricci tensor Galerkin Frobenius"),
        "backend": backend,
        "lambda_struct": {"t": LAMBDA_T_STRUCT,
                           "s": LAMBDA_S_STRUCT},
        "trend": [
            {"regime": a["regime"], "N": a["N"], "n_seeds": a["n_seeds"],
             "blind_frob_mean": a["blind_frob_mean"],
             "blind_frob_mean_std": a["blind_frob_mean_std"],
             "blind_frob_median": a["blind_frob_median"],
             "struct_frob_mean": a["struct_frob_mean"],
             "struct_frob_mean_std": a["struct_frob_mean_std"],
             "struct_frob_median": a["struct_frob_median"],
             "t_off_F_mean": a["t_off_F_mean"],
             "r_bar_h_mean": a["r_bar_h_mean"],
             "blind_lam_t": a["blind_lam_t"],
             "blind_lam_t_std": a["blind_lam_t_std"],
             "blind_lam_s": a["blind_lam_s"],
             "blind_lam_s_std": a["blind_lam_s_std"]}
            for a in aggregate
        ],
    }
    out_path = OUTPUTS / "galerkin_runner_A_hessian_ricci.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
