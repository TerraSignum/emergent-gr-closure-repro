r"""Runner D: Basis-invariance test for the per-node Galerkin Frobenius
residual.

Verifies that the Frobenius norm ||G + Lambda g - 8 pi G T||_F per node is
basis-invariant: i.e. it does not depend on the choice of frame (spectral
graph-Laplacian, Ricci principal, stress principal). The Frobenius norm
of any tensor is invariant under orthogonal change of basis, so this is
a SANITY CHECK of the implementation rather than a physics test. If
implemented correctly, all three bases must give identical residuals to
machine precision.

The three bases tested:
  (1) spectral Laplacian (top-3 eigenvectors of L_norm) — default
  (2) Ricci principal axes (eigenvectors of R_ij)
  (3) Stress principal axes (eigenvectors of T_ij)

For each (regime, seed): construct G_ij and T_ij in the spectral basis,
then rotate both into each of the three bases and compute Frobenius
residual. Report max relative deviation across bases.

Usage:
    python ./src/verify_galerkin_runner_D_basis_invariance.py
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
from _d1_npz_discovery import find_d1_npz

LADDER_REGIMES = [("P5", 50), ("P8", 84)]
LADDER = [(r, n, find_d1_npz(r, REPO)) for r, n in LADDER_REGIMES]
N_SEEDS = 2  # only need a few to verify invariance


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


def build_tensors(xi_mat_np, psi_np, k_field_np, q_field_np, n_lat,
                   xp):
    """Return per-node R_ij, R_bar, G_ij, T_ij, T_00 in spectral basis."""
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
    d_sq = d_mat * d_mat
    d_sq_safe = xp.where(adj > 0, d_sq, xp.inf)
    weight_grad = xp.where(adj > 0, weight_adj / (d_sq_safe + EPS_D),
                            0.0)
    omega_a = weight_grad.sum(axis=1)

    # Hessian-Ricci tensor R_ij in spectral basis.
    spatial_diff = spatial[None, :, :] - spatial[:, None, :]
    delta_sq = (spatial_diff ** 2).sum(axis=2)
    inv_d = xp.where(adj > 0, 1.0 / d_mat, 0.0)
    e_alpha = spatial_diff * inv_d[:, :, None]
    discrepancy = xp.where(adj > 0, 1.0 - delta_sq / (d_sq + EPS_D), 0.0)
    weight_disc = weight_adj * discrepancy
    r_ij = xp.einsum("ab,abi,abj->aij", weight_disc, e_alpha, e_alpha)
    norm = (weight_adj.sum(axis=1) + 1e-12)
    r_ij = r_ij / norm[:, None, None]
    r_bar = xp.trace(r_ij, axis1=1, axis2=2)
    eye3 = xp.eye(3, dtype=xp.float64)
    g_ij = r_ij - 0.5 * r_bar[:, None, None] * eye3[None, :, :]
    g_00 = r_bar / 2.0

    # T_munu in spectral basis.
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

    return {
        "r_ij": r_ij, "r_bar": r_bar, "g_ij": g_ij, "g_00": g_00,
        "t_ij": t_ij, "t00": t00, "eye3": eye3, "n": n_lat,
    }


def frob_in_basis(prep, basis_kind, lam_t, lam_s, xp):
    """Compute Frobenius residual after rotating G_ij, T_ij into the
    chosen basis. Result is basis-invariant for a correctly-implemented
    Frobenius norm.

    basis_kind:
      'spectral' — keep spectral basis (no rotation)
      'ricci'    — rotate into eigenbasis of r_ij(a)
      'stress'   — rotate into eigenbasis of t_ij(a)
    """
    g_00 = prep["g_00"]
    g_ij = prep["g_ij"]
    t_ij = prep["t_ij"]
    t00 = prep["t00"]
    eye3 = prep["eye3"]
    n = prep["n"]

    if basis_kind == "spectral":
        g_ij_b = g_ij
        t_ij_b = t_ij
    elif basis_kind == "ricci":
        # Eigendecomposition of r_ij per node, use eigenvectors as basis.
        r_ij = prep["r_ij"]
        _, U = xp.linalg.eigh(r_ij)
        g_ij_b = xp.einsum("aki,akl,alj->aij", U, g_ij, U)
        t_ij_b = xp.einsum("aki,akl,alj->aij", U, t_ij, U)
    elif basis_kind == "stress":
        _, U = xp.linalg.eigh(t_ij)
        g_ij_b = xp.einsum("aki,akl,alj->aij", U, g_ij, U)
        t_ij_b = xp.einsum("aki,akl,alj->aij", U, t_ij, U)
    else:
        raise ValueError(basis_kind)

    res00 = g_00 + lam_t - t00
    spatial_res = (g_ij_b + lam_s * eye3[None, :, :]) - t_ij_b
    sq = res00 ** 2 + (spatial_res ** 2).sum(axis=(1, 2))
    frob = xp.sqrt(sq)
    return {
        "frob_mean": float(frob.mean()),
        "frob_median": float(xp.median(frob)),
        "frob_max": float(frob.max()),
    }


def main():
    print("=" * 78)
    print("Runner D: Basis-invariance test (sanity check)")
    print("Frobenius norm should be identical in 3 bases (machine precision)")
    print("=" * 78)
    print()

    try:
        import cupy as xp
        backend = "cupy"
    except Exception:
        import numpy as xp
        backend = "numpy"
    print(f"Backend: {backend}")
    print()

    import numpy as np
    aggregate = []
    print(f"{'Reg':<6} {'N':>3} {'seed':>4} | "
          f"{'spectral':>12} {'ricci':>12} {'stress':>12} | "
          f"{'max rel dev':>12}")
    print("-" * 80)
    for regime, n_lat, npz_path in LADDER:
        if npz_path is None or not npz_path.exists():
            continue
        d = np.load(npz_path, allow_pickle=True)
        edge_arr = d["dense_cell_edge_xi_values"]
        amp_arr = d["dense_cell_node_amplitude_values"]
        phase_arr = d["dense_cell_node_phase_values"]
        n_seeds = min(edge_arr.shape[0], N_SEEDS)
        for s in range(n_seeds):
            xi_mat = edge_to_matrix(edge_arr[s], n_lat)
            np.fill_diagonal(xi_mat, 1.0)
            psi = amp_arr[s] * np.exp(1j * phase_arr[s])
            k_field = d.get(f"ff_K_seed{s}",
                              np.full((n_lat, n_lat), 0.55))
            q_field = d.get(f"ff_Q_seed{s}",
                              np.full((n_lat, n_lat), 0.45))
            prep = build_tensors(xi_mat, psi, k_field, q_field, n_lat,
                                   xp)

            results = {}
            for basis in ("spectral", "ricci", "stress"):
                results[basis] = frob_in_basis(
                    prep, basis,
                    LAMBDA_T_STRUCT, LAMBDA_S_STRUCT, xp)

            spec_mean = results["spectral"]["frob_mean"]
            ricci_mean = results["ricci"]["frob_mean"]
            stress_mean = results["stress"]["frob_mean"]
            max_rel_dev = max(
                abs(ricci_mean - spec_mean) / max(spec_mean, 1e-12),
                abs(stress_mean - spec_mean) / max(spec_mean, 1e-12),
            )
            verdict = ("INVARIANT" if max_rel_dev < 1e-10
                       else ("OK" if max_rel_dev < 0.05 else "FAIL"))
            aggregate.append({
                "regime": regime, "N": n_lat, "seed": s,
                "spectral_mean": spec_mean,
                "ricci_mean": ricci_mean,
                "stress_mean": stress_mean,
                "max_rel_dev": max_rel_dev,
                "verdict": verdict,
            })
            print(f"{regime:<6} {n_lat:>3} {s:>4} | "
                  f"{spec_mean:>12.6f} {ricci_mean:>12.6f} "
                  f"{stress_mean:>12.6f} | {max_rel_dev:>12.2e}  "
                  f"{verdict}")

    print()
    n_invariant = sum(1 for a in aggregate if a["verdict"] == "INVARIANT")
    n_ok = sum(1 for a in aggregate if a["verdict"] == "OK")
    n_fail = sum(1 for a in aggregate if a["verdict"] == "FAIL")
    print(f"Verdict summary: INVARIANT={n_invariant}, "
          f"OK={n_ok}, FAIL={n_fail}, total={len(aggregate)}")

    out = {
        "schema_version": "1.0.0",
        "title": "Runner D: Basis-invariance test",
        "backend": backend,
        "results": aggregate,
        "summary": {
            "n_invariant": n_invariant,
            "n_ok": n_ok,
            "n_fail": n_fail,
            "n_total": len(aggregate),
        },
    }
    out_path = OUTPUTS / "galerkin_runner_D_basis_invariance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
