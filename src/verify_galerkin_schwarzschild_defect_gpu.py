r"""Schwarzschild-defect ground-truth test for the per-node Galerkin pipeline.

Builds a synthetic lattice with a single point-mass defect at the centre,
producing the Newtonian Xi perturbation
  delta_Xi(r) = -M_core / r
on a finite N-node geometry. The framework's existing schwarzschild_defect.py
machinery (SCD-01..05) confirms this defect reproduces the analytical
Schwarzschild metric to g_00 error 0.028% on the bundled lattice.

This script applies the per-node Galerkin Forman-Ricci pipeline to the
defect lattice and checks whether the per-node Frobenius residual against
the analytical Schwarzschild T_munu (point-mass at origin, vacuum elsewhere)
converges to 0 in N. If yes, the per-node Galerkin pipeline is consistent
with the framework's already-closed Schwarzschild claim.

Construction
------------
On an N-point lattice with positions x_a in R^3 (random sampled in a ball),
plus a 4th time-direction (Wick-rotated to Euclidean for static analysis):
  Xi_ab = exp(-d_ab / ell_0) * Xi_correction(r_a, r_b)
where the defect modifies pairwise weights by a Schwarzschild-like factor.

For each lattice point at radius r_a from origin:
  Phi(r_a) = -M_core / r_a   (Newtonian potential)
  g_00(r_a) = -(1 + 2 Phi(r_a))
  g_rr(r_a) = +(1 - 2 Phi(r_a))

Analytical T_munu for Schwarzschild vacuum: T_munu = 0 (vacuum solution).
Analytical G_munu for Schwarzschild: G_munu = 0 (Einstein vacuum equations).

So the per-node residual ||G + Lambda g - 8 pi G T||_F should converge to
||Lambda g||_F per node, which is small if Lambda is appropriately chosen.

Multi-N test
------------
Run on N in {30, 50, 80, 120, 150} with 4 random seeds each.
Check whether the per-node residual decreases as N -> infinity.

Usage:
    python ./src/verify_galerkin_schwarzschild_defect_gpu.py
"""
from __future__ import annotations
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Schwarzschild test parameters (in lattice units, 8 pi G = 1).
M_CORE = 2.96  # from bundled scd01.core_mass_lu_p1
ELL_0 = 1.0
D_MIN = 0.1
XI_THRESH = 0.1
EPS_D = D_MIN ** 2

# We use a very small Lambda for the vacuum test.
LAMBDA_T = 0.0
LAMBDA_S = 0.0


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


def per_node_R_ij(kappa, xi_off, adj, d_mat, spatial, xp):
    weight_adj = xi_off * adj
    spatial_diff = spatial[None, :, :] - spatial[:, None, :]
    inv_d = xp.where(adj > 0, 1.0 / d_mat, 0.0)
    e_alpha = spatial_diff * inv_d[:, :, None]
    weight_kappa = weight_adj * kappa
    r_ij = xp.einsum("ab,abi,abj->aij", weight_kappa, e_alpha, e_alpha)
    norm = (weight_adj.sum(axis=1) + 1e-12)
    return r_ij / norm[:, None, None]


def schwarzschild_defect_lattice(n_points, ell, m_core, xp,
                                   seed=0, r_min=1.5, r_max=8.0):
    """Construct a lattice with a Schwarzschild defect.

    Sample n_points uniformly in a 3D shell [r_min, r_max].
    Pairwise distance d_ab = |x_a - x_b|.
    Xi_ab = exp(-d_ab / ell) * exp(-(Phi(r_a) + Phi(r_b)) * d_ab / 2)
    where Phi(r) = -m_core / r is the Newtonian potential.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    # Random points in shell: r uniform in [r_min, r_max], angles uniform.
    r = rng.uniform(r_min, r_max, size=n_points)
    theta = xp.arccos(xp.asarray(rng.uniform(-1, 1, size=n_points)))
    phi = xp.asarray(rng.uniform(0, 2 * np.pi, size=n_points))
    r_xp = xp.asarray(r)
    x = r_xp * xp.sin(theta) * xp.cos(phi)
    y = r_xp * xp.sin(theta) * xp.sin(phi)
    z = r_xp * xp.cos(theta)
    coords = xp.stack([x, y, z], axis=1)

    # Pairwise distances.
    d_ab = xp.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(
        axis=2))

    # Newtonian potential at each point.
    phi_r = -m_core / r_xp

    # Defect-modified Xi: weighted by potential along the edge.
    # Approximation: Xi_ab = exp(-d_ab / ell - 0.5 * (Phi_a + Phi_b))
    phi_pair = 0.5 * (phi_r[:, None] + phi_r[None, :])
    log_xi = -d_ab / ell - phi_pair
    xi_mat = xp.exp(log_xi)
    xp.fill_diagonal(xi_mat, 1.0)

    # Schwarzschild-analytical g_00, g_rr at each r_a (lattice units).
    g_00_analytical = -(1.0 + 2.0 * phi_r)  # for small Phi
    g_rr_analytical = +(1.0 - 2.0 * phi_r)

    return {
        "xi_mat": xi_mat,
        "coords": coords,
        "r": r_xp,
        "phi_r": phi_r,
        "g_00_analytical": g_00_analytical,
        "g_rr_analytical": g_rr_analytical,
        "n_points": n_points,
    }


def galerkin_on_defect_lattice(defect_data, xp):
    """Run per-node Galerkin pipeline on the Schwarzschild-defect lattice.

    Returns per-node R_ij(a), R_bar(a), G_munu(a), comparison to
    analytical Schwarzschild g_munu, and Frobenius residual against
    vacuum T_munu = 0.
    """
    xi_mat = defect_data["xi_mat"]
    n = xi_mat.shape[0]
    xi_off = xi_mat.copy()
    xp.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(xp.float64)
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1) + 1e-12

    deg_inv_sqrt = 1.0 / xp.sqrt(deg)
    l_norm = (xp.eye(n, dtype=xp.float64)
              - (deg_inv_sqrt[:, None] * weight_adj
                 * deg_inv_sqrt[None, :]))
    eigvals_l, eigvecs_l = xp.linalg.eigh(l_norm)
    spatial = eigvecs_l[:, 1:4]

    d_mat = -ELL_0 * xp.log(xp.maximum(xi_off, 1e-12))
    d_mat = xp.maximum(d_mat, D_MIN)

    # Forman-Ricci.
    kappa = forman_ricci(xi_off, adj, xp)

    # Per-node Ricci tensor and scalar.
    r_ij = per_node_R_ij(kappa, xi_off, adj, d_mat, spatial, xp)
    r_bar = xp.trace(r_ij, axis1=1, axis2=2)

    # Einstein tensor: G_ij = R_ij - 0.5 g_ij R_bar; G_00 = R_bar/2.
    eye3 = xp.eye(3, dtype=xp.float64)
    g_ij = r_ij - 0.5 * r_bar[:, None, None] * eye3[None, :, :]
    g_00 = r_bar / 2.0

    # Vacuum residual: T_munu = 0, so residual = G_munu + Lambda g.
    res00 = g_00 + LAMBDA_T  # T_00 = 0
    spatial_res = g_ij + LAMBDA_S * eye3[None, :, :]  # T_ij = 0
    spatial_res_F_sq = (spatial_res ** 2).sum(axis=(1, 2))
    frob = xp.sqrt(res00 ** 2 + spatial_res_F_sq)

    # The principal Schwarzschild-vacuum test:
    #   For Schwarzschild vacuum, T_munu = 0 and G_munu = 0,
    #   so ||G + Lambda g||_F (with Lambda = 0) should be 0 per node.
    # If our pipeline correctly identifies the vacuum solution, the
    # per-node Frobenius residual approaches 0 in the asymptotic limit.
    #
    # NOTE: G_00 = R_bar/2 here is the EINSTEIN tensor component, not
    # the metric component. The metric g_00 ~ -(1 + 2 Phi) is a
    # different quantity and lives in the metric tensor space, while
    # our Galerkin pipeline operates on the Einstein/curvature side.
    return {
        "n": n,
        "r_bar_per_node_mean": float(r_bar.mean()),
        "r_bar_per_node_std": float(r_bar.std()),
        "g_einstein_00_per_node_mean": float(g_00.mean()),
        "g_einstein_00_per_node_std": float(g_00.std()),
        "frob_per_node_mean": float(frob.mean()),
        "frob_per_node_max": float(frob.max()),
        "frob_per_node_min": float(frob.min()),
        "frob_per_node_median": float(xp.median(frob)),
    }


def main():
    print("=" * 78)
    print("Schwarzschild-defect ground-truth test for per-node Galerkin")
    print("=" * 78)
    print()

    try:
        import cupy as xp
        backend = "cupy"
        print(f"GPU: {xp.cuda.Device(0).mem_info[1] / 1e9:.1f} GB VRAM")
    except Exception as e:
        print(f"CuPy unavailable; using NumPy. ({e})")
        import numpy as xp
        backend = "numpy"
    print(f"Backend: {backend}")
    print(f"M_core = {M_CORE} (from bundled scd01.core_mass_lu_p1)")
    print(f"r-shell: [1.5, 8.0] lattice units")
    print(f"Lambda: t={LAMBDA_T}, s={LAMBDA_S} (vacuum test)")
    print()

    Ns = [30, 50, 80, 120, 150]
    n_seeds = 4
    aggregate = []

    print("--- Multi-N Schwarzschild-vacuum test ---")
    print("    For Schwarzschild vacuum (T_munu = 0): ||G_munu||_F should -> 0")
    print()
    print(f"{'N':>4} {'<Frob>':>10} {'median Frob':>14} "
          f"{'<R_bar>':>10} {'<G_einstein_00>':>16}")
    print("-" * 78)
    for N in Ns:
        per_seed = []
        for s in range(n_seeds):
            defect = schwarzschild_defect_lattice(N, ELL_0, M_CORE,
                                                    xp, seed=42 + s)
            r = galerkin_on_defect_lattice(defect, xp)
            per_seed.append(r)
        means = lambda key: sum(r[key] for r in per_seed) / len(per_seed)
        agg = {
            "N": N,
            "n_seeds": n_seeds,
            "per_seed": per_seed,
            "mean_frob": means("frob_per_node_mean"),
            "median_frob": means("frob_per_node_median"),
            "mean_r_bar": means("r_bar_per_node_mean"),
            "mean_g_einstein_00": means("g_einstein_00_per_node_mean"),
        }
        aggregate.append(agg)
        print(f"{N:>4} {agg['mean_frob']:>10.4f} "
              f"{agg['median_frob']:>14.4f} "
              f"{agg['mean_r_bar']:>10.4f} "
              f"{agg['mean_g_einstein_00']:>16.4f}")

    print()
    print("Comparison: D1 lattice physics at N=84 -> Frob ~ 0.79")
    print("            Schwarzschild vacuum should give Frob ~ 0")
    if aggregate:
        last = aggregate[-1]
        print(f"            Schwarzschild test at N={last['N']}: "
              f"<Frob> = {last['mean_frob']:.4f}, "
              f"median = {last['median_frob']:.4f}")
        print(f"            Ratio physics/vacuum: "
              f"{0.79/last['mean_frob'] if last['mean_frob']>1e-9 else float('inf'):.1f}x")

    out = {
        "schema_version": "1.0.0",
        "title": ("Schwarzschild-defect ground-truth test for "
                  "per-node Galerkin"),
        "backend": backend,
        "m_core": M_CORE,
        "ell_0": ELL_0,
        "lambda_t": LAMBDA_T,
        "lambda_s": LAMBDA_S,
        "comparison_bundled_scd02": {
            "g_00_test_p1": 2.83273036,
            "g_00_exact_p1": 2.83192948,
            "g_00_relative_error": 0.000283,
        },
        "trend": [
            {"N": a["N"],
             "mean_frob": a["mean_frob"],
             "median_frob": a["median_frob"],
             "mean_r_bar": a["mean_r_bar"],
             "mean_g_einstein_00": a["mean_g_einstein_00"]}
            for a in aggregate
        ],
    }
    out_path = OUTPUTS / "galerkin_schwarzschild_defect_gpu.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
