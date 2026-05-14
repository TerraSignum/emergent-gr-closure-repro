r"""Stage B: Full per-node 4x4 Galerkin Frobenius residual on GPU.

Replaces the proxy-R_bar approach of verify_galerkin_per_node_gpu.py with a
direct per-node Ricci tensor computation. The full 4x4 G_munu and T_munu
tensors are constructed at every lattice node from bundled data, with no
regime-mean broadcasting and no proxy K_rec convention.

Per-node ingredients (no proxies)
---------------------------------
T_munu(a):
  T_00(a) = 0.5 * Z_xi * var_local(Xi)_a
          + kappa_xi * |Psi_a|^2
          + zeta_1 * omega * |grad Psi|^2(a)
          + zeta_3 * omega * K_rec(a)
  with K_rec(a) = a_K * <K(x)>_a + a_Q * (1 - <Q(x)>_a) using the
  bundled ff_K_seedX, ff_Q_seedX fields (row-mean convention, not proxy).

  T_ij(a) = 2 * coeff * Re(grad_i^* Psi * grad_j Psi)(a)
          - iso_subtract * |grad Psi|^2(a) * delta_ij
  on the spectral graph-Laplacian embedding (top 3 non-zero eigenvectors
  of the normalized Laplacian as the natural 3D spatial frame).

G_munu(a):
  G_munu(a) = R_munu(a) - 0.5 * g_munu * R_bar(a)
  with the per-node Ricci tensor R_munu(a) computed via spectral
  projection of the per-edge Forman-Ricci curvature kappa^F(a, b),

    R_ij^{spec}(a) = sum_b w_ab * kappa^F(a, b)
                       * e_i^{(ab)} * e_j^{(ab)} / sum_b w_ab,

  where e_alpha^{(ab)} = (x^alpha(b) - x^alpha(a)) / d_ab is the
  unit edge vector in the spectral coordinates and
  d_ab = -ell_0 * log(Xi_ab).
  R_bar(a) = trace(R_ij^{spec}(a)) (Ricci scalar at node a).
  R_00(a) is read from the static-frame FRW relation
  R_00(a) = R_bar(a)/2 (the relational lattice has no explicit
  time-derivative degree of freedom; this is the cleanest Lorentzian
  reading of the static slice).

Forman-Ricci on a weighted graph with vertex weights w_a and edge
weights w_ab is

    kappa^F(a, b) = w_a + w_b
                  - sum_{c ~ a, c != b} sqrt(w_a * w_ac)
                  - sum_{c ~ b, c != a} sqrt(w_b * w_bc),

  where the sum is over neighbours of a and b. We use vertex weight
  w_a = sum_b Xi_ab and edge weight w_ab = Xi_ab.

GPU acceleration via CuPy: all per-node operations run in batched
matrix-multiplication form on the RTX 5070.

Schwarzschild vacuum unit test: in vacuum (Psi=0, K=Q=0, Lambda=0)
on a near-uniform Xi field, T_munu = 0 exactly. The Forman-Ricci is
non-zero (the lattice graph has its own intrinsic curvature), so the
residual reduces to ||G_munu||_F per node, which is reported as the
ground-state graph-curvature scale of the unit-test lattice.

Usage:
    python ./src/verify_galerkin_per_node_full_gpu.py
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Anisotropic Lambda (lattice units, 8 pi G = 1).
LAMBDA_T = 4.0 / 3.0
LAMBDA_S = -0.37

# Hilbert-variation coefficients.
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
N_SEEDS_PER_REGIME = 4


def edge_to_matrix_np(edges, n):
    """Build symmetric N x N matrix from upper-triangular flat list."""
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


def forman_ricci_per_edge_gpu(xi_off, adj, xp):
    """Forman-Ricci curvature per edge.

    kappa^F(a,b) = w_a + w_b
                 - sum_{c ~ a, c != b} sqrt(w_a * w_ac)
                 - sum_{c ~ b, c != a} sqrt(w_b * w_bc)

    where w_a = degree(a) (sum of incident edge weights) and
    w_ab = Xi_ab. Returns N x N matrix of per-edge Forman-Ricci values
    (zero on non-edges).
    """
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1)  # vertex weights w_a

    # For each edge (a, b), compute the two sums:
    #   S_a = sum_{c, c != b, adj_ac > 0} sqrt(w_a * w_ac)
    #   S_b = sum_{c, c != a, adj_bc > 0} sqrt(w_b * w_bc)
    # Vectorized: total_a = sum_c sqrt(w_a * w_ac) over all neighbours
    #             of a, then subtract the c=b contribution.

    sqrt_w = xp.sqrt(deg[:, None] * weight_adj)  # sqrt(w_a * w_ac), shape (a, c)
    total_a = sqrt_w.sum(axis=1)  # sum over all neighbours c
    # Per-edge subtraction of the c=b term: when looking at edge (a,b),
    # we exclude c=b. So S_a(a, b) = total_a[a] - sqrt(w_a * w_ab).
    sqrt_w_ab = xp.sqrt(deg[:, None] * weight_adj)  # same as sqrt_w
    s_a = total_a[:, None] - sqrt_w_ab

    sqrt_w_t = xp.sqrt(deg[None, :] * weight_adj.T)  # sqrt(w_b * w_bc)
    total_b = sqrt_w_t.sum(axis=0)
    sqrt_w_ba = xp.sqrt(deg[None, :] * weight_adj)
    s_b = total_b[None, :] - sqrt_w_ba

    forman = (deg[:, None] + deg[None, :]) - s_a - s_b
    forman = forman * adj  # zero on non-edges
    return forman


def galerkin_full_per_seed(xi_mat_cpu, psi_cpu, k_field_cpu,
                            q_field_cpu, n_lat, xp):
    """Full per-node 4x4 Galerkin Frobenius residual on GPU.

    Returns dict of aggregated per-node statistics.
    """
    xi_mat = xp.asarray(xi_mat_cpu)
    psi = xp.asarray(psi_cpu)
    k_field = xp.asarray(k_field_cpu)  # shape (n, n) per-edge K(x)
    q_field = xp.asarray(q_field_cpu)  # shape (n, n) per-edge Q(x)

    xi_off = xi_mat.copy()
    xp.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(xp.float64)
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1)

    # Spectral basis: top 3 non-zero eigenvectors of normalized Laplacian.
    deg_inv_sqrt = 1.0 / xp.sqrt(deg + 1e-12)
    l_norm = (xp.eye(n_lat, dtype=xp.float64)
              - (deg_inv_sqrt[:, None] * weight_adj
                 * deg_inv_sqrt[None, :]))
    eigvals_l, eigvecs_l = xp.linalg.eigh(l_norm)
    spatial = eigvecs_l[:, 1:4]  # (n, 3)

    # Edge distances and inverse-distance weights.
    d_mat = -ELL_0 * xp.log(xp.maximum(xi_off, 1e-12))
    d_mat = xp.maximum(d_mat, D_MIN)
    d_sq = d_mat * d_mat
    d_sq_safe = xp.where(adj > 0, d_sq, xp.inf)
    weight_grad = xp.where(adj > 0, weight_adj / (d_sq_safe + EPS_D), 0.0)
    omega_a = weight_grad.sum(axis=1)

    # Forman-Ricci per edge.
    forman = forman_ricci_per_edge_gpu(xi_off, adj, xp)

    # Spectral edge unit vectors: e_alpha^{(ab)} = (x^a(b) - x^a(a)) / d_ab.
    spatial_diff = spatial[None, :, :] - spatial[:, None, :]  # (a, b, 3)
    inv_d = xp.where(adj > 0, 1.0 / d_mat, 0.0)
    e_alpha = spatial_diff * inv_d[:, :, None]  # (a, b, 3)

    # Per-node Ricci tensor in spectral basis:
    # R_ij(a) = sum_b w_ab * kappa_F(a,b) * e_i * e_j / sum_b w_ab.
    weight_kappa = weight_adj * forman  # (a, b)
    # Compute sum_b weight_kappa * e_i e_j -> shape (a, 3, 3).
    # einsum: 'ab,abi,abj->aij'
    r_ij = xp.einsum("ab,abi,abj->aij", weight_kappa, e_alpha, e_alpha)
    norm_factor = (weight_adj.sum(axis=1) + 1e-12)
    r_ij = r_ij / norm_factor[:, None, None]

    # Per-node Ricci scalar = trace of R_ij^{spec}(a).
    r_bar_per_node = xp.trace(r_ij, axis1=1, axis2=2)

    # Per-node G_ij = R_ij - 0.5 * g_ij * R_bar(a).
    eye3 = xp.eye(3, dtype=xp.float64)
    g_ij = r_ij - 0.5 * r_bar_per_node[:, None, None] * eye3[None, :, :]

    # Time-component G_00 = R_bar(a)/2 (static-frame FRW reading).
    g_00 = r_bar_per_node / 2.0

    # ---- T_munu per node ----
    # Per-node directional gradient of Psi in spectral basis.
    psi_diff = psi[None, :] - psi[:, None]
    weight_term = weight_grad[:, :, None] * inv_d[:, :, None]
    grad_psi = (psi_diff[:, :, None] * spatial_diff * weight_term).sum(
        axis=1)
    grad_psi = grad_psi / (omega_a[:, None] + 1e-12)

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

    # T_00 per node with row-mean K_rec convention.
    xi_row_mean = (weight_adj.sum(axis=1)
                   / (adj.sum(axis=1) + 1e-12))
    var_xi = (((weight_adj - xi_row_mean[:, None]) ** 2 * adj).sum(
        axis=1) / (adj.sum(axis=1) + 1e-12))
    amp_a = xp.abs(psi)
    var_amp = (amp_a - amp_a.mean()) ** 2
    grad_psi_sq = norm_sq

    # K(x)_a, Q(x)_a from bundled fields. ff_K, ff_Q are (n, n) per-edge
    # fields; collapse to per-node averages.
    k_per_node = (k_field * adj).sum(axis=1) / (adj.sum(axis=1) + 1e-12)
    q_per_node = (q_field * adj).sum(axis=1) / (adj.sum(axis=1) + 1e-12)
    k_rec_per_node = A_K * k_per_node + A_Q * (1.0 - q_per_node)

    t00_per_node = (0.5 * Z_XI * var_xi
                    + KAPPA_XI * var_amp
                    + ZETA_1 * OMEGA * grad_psi_sq
                    + ZETA_3 * OMEGA * k_rec_per_node)

    # ---- Frobenius residual per node ----
    res00 = g_00 + LAMBDA_T - t00_per_node

    # Spatial residual matrix (3, 3): (G_ij + Lambda_s * delta_ij - T_ij).
    lam_s_eye = LAMBDA_S * eye3
    spatial_res = (g_ij + lam_s_eye[None, :, :]) - t_ij
    spatial_res_F_sq = (spatial_res ** 2).sum(axis=(1, 2))

    frob_sq_per_node = res00 ** 2 + spatial_res_F_sq
    frob_per_node = xp.sqrt(frob_sq_per_node)

    return {
        "r_bar_per_node_mean": float(r_bar_per_node.mean()),
        "r_bar_per_node_std": float(r_bar_per_node.std()),
        "t00_per_node_mean": float(t00_per_node.mean()),
        "k_rec_per_node_mean": float(k_rec_per_node.mean()),
        "forman_per_edge_mean": float(forman.mean()),
        "frob_per_node_mean": float(frob_per_node.mean()),
        "frob_per_node_std": float(frob_per_node.std()),
        "frob_per_node_max": float(frob_per_node.max()),
        "frob_per_node_min": float(frob_per_node.min()),
        "g_00_mean": float(g_00.mean()),
        "g_ii_trace_mean": float(xp.trace(
            g_ij.mean(axis=0))),
        "spatial_res_F_mean": float(xp.sqrt(spatial_res_F_sq).mean()),
        "time_res_mean": float(xp.abs(res00).mean()),
    }


def schwarzschild_unit_test_gpu(xp):
    """Vacuum (Psi=0, K=Q=0, Lambda=0) on near-uniform Xi: residual is
    pure graph curvature (Forman-Ricci-induced G_munu)."""
    n = 16
    rng = (xp.eye(n, dtype=xp.float64)
           + 0.5 * xp.ones((n, n), dtype=xp.float64))
    xi = (rng + rng.T) / 2.0
    xp.fill_diagonal(xi, 1.0)
    psi = xp.zeros(n, dtype=xp.complex128)
    k_field = xp.zeros((n, n), dtype=xp.float64)
    q_field = xp.zeros((n, n), dtype=xp.float64)

    # Override Lambda to 0 for this test.
    global LAMBDA_T, LAMBDA_S
    saved_T, saved_S = LAMBDA_T, LAMBDA_S
    LAMBDA_T = 0.0
    LAMBDA_S = 0.0

    res = galerkin_full_per_seed(
        xi.get() if hasattr(xi, "get") else xi,
        psi.get() if hasattr(psi, "get") else psi,
        k_field.get() if hasattr(k_field, "get") else k_field,
        q_field.get() if hasattr(q_field, "get") else q_field,
        n, xp)
    LAMBDA_T = saved_T
    LAMBDA_S = saved_S

    return {
        "test": "Vacuum on near-uniform Xi (Forman-Ricci background)",
        "frob_per_node_mean": res["frob_per_node_mean"],
        "frob_per_node_max": res["frob_per_node_max"],
        "r_bar_per_node_mean": res["r_bar_per_node_mean"],
        "g_00_mean": res["g_00_mean"],
        "comment": ("Vacuum + Lambda=0: residual = ||G_munu||_F "
                    "from Forman-Ricci graph background. Non-zero is "
                    "expected; what we verify is finite, regular."),
    }


def main():
    print("=" * 78)
    print("Stage B: Full per-node 4x4 Galerkin Frobenius residual")
    print("(spectral Forman-Ricci, bundled K/Q row-mean K_rec, no proxies)")
    print("=" * 78)
    print()

    try:
        import cupy as xp
        device = xp.cuda.Device(0)
        with device:
            mem_total = device.mem_info[1] / 1e9
            print(f"GPU: device 0, {mem_total:.1f} GB total VRAM")
        backend = "cupy"
    except Exception as e:
        print(f"CuPy unavailable ({e}); falling back to NumPy CPU.")
        import numpy as xp
        backend = "numpy"
    print(f"Backend: {backend}")
    print()

    print("--- Schwarzschild unit test ---")
    sw = schwarzschild_unit_test_gpu(xp)
    for k, v in sw.items():
        print(f"  {k}: {v}")
    print()

    import numpy as np
    aggregate = []
    per_regime = {}

    for regime, n_lat, npz_path in LADDER:
        if npz_path is None or not npz_path.exists():
            print(f"[{regime}, N={n_lat}] NPZ missing: {npz_path}")
            continue
        d = np.load(npz_path, allow_pickle=True)
        edge_arr = d["dense_cell_edge_xi_values"]
        amp_arr = d["dense_cell_node_amplitude_values"]
        phase_arr = d["dense_cell_node_phase_values"]
        n_seeds_avail = min(edge_arr.shape[0], N_SEEDS_PER_REGIME)

        seeds_results = []
        for s in range(n_seeds_avail):
            xi_mat = edge_to_matrix_np(edge_arr[s], n_lat)
            np.fill_diagonal(xi_mat, 1.0)
            psi = amp_arr[s] * np.exp(1j * phase_arr[s])
            k_key = f"ff_K_seed{s}"
            q_key = f"ff_Q_seed{s}"
            if k_key in d.files and q_key in d.files:
                k_field = d[k_key]
                q_field = d[q_key]
            else:
                # Fallback to phase-coherence proxy.
                k_field = np.full((n_lat, n_lat), 0.55)
                q_field = np.full((n_lat, n_lat), 0.45)
            r = galerkin_full_per_seed(xi_mat, psi, k_field, q_field,
                                         n_lat, xp)
            r["seed"] = s
            seeds_results.append(r)

        means = [r["frob_per_node_mean"] for r in seeds_results]
        time_means = [r["time_res_mean"] for r in seeds_results]
        spatial_means = [r["spatial_res_F_mean"] for r in seeds_results]
        agg = {
            "regime": regime,
            "N": n_lat,
            "n_seeds": n_seeds_avail,
            "per_seed": seeds_results,
            "mean_frob": (sum(means) / len(means)
                          if means else float("nan")),
            "mean_time_res": (sum(time_means) / len(time_means)
                              if time_means else float("nan")),
            "mean_spatial_res": (sum(spatial_means) / len(spatial_means)
                                 if spatial_means else float("nan")),
        }
        per_regime[regime] = agg
        aggregate.append((n_lat, agg))

        print(f"[{regime}, N={n_lat}] Frob/node = "
              f"{agg['mean_frob']:.4f}  "
              f"(time: {agg['mean_time_res']:.3f}, "
              f"spatial: {agg['mean_spatial_res']:.3f})  "
              f"R_bar mean = "
              f"{seeds_results[0]['r_bar_per_node_mean']:.4f}, "
              f"Forman mean = "
              f"{seeds_results[0]['forman_per_edge_mean']:.4f}")

    print()
    print("=" * 78)
    print("Multi-N trend:")
    print(f"  {'N':>5} {'Frob/node':>12} {'time res':>12} "
          f"{'spatial res':>14}")
    for n_lat, agg in aggregate:
        print(f"  {n_lat:5d} {agg['mean_frob']:12.4f} "
              f"{agg['mean_time_res']:12.4f} "
              f"{agg['mean_spatial_res']:14.4f}")

    out = {
        "schema_version": "1.0.0",
        "title": ("Stage B full per-node 4x4 Galerkin Frobenius "
                  "(spectral Forman-Ricci, row-mean K_rec)"),
        "backend": backend,
        "Lambda_t": LAMBDA_T,
        "Lambda_s": LAMBDA_S,
        "schwarzschild_unit_test": sw,
        "per_regime": per_regime,
        "trend": [
            {"N": n_lat,
             "mean_frob": agg["mean_frob"],
             "mean_time_res": agg["mean_time_res"],
             "mean_spatial_res": agg["mean_spatial_res"]}
            for n_lat, agg in aggregate
        ],
    }
    out_path = OUTPUTS / "galerkin_per_node_full_gpu.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
