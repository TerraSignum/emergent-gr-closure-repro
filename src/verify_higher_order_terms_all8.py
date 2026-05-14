"""Test all 8 higher-order term proposals (O1-O6 + Own1, Own2)
against the baseline Galerkin pipeline on the existing D1 NPZ
files. No new lattice runs required.

For each term, we compute the per-node 4x4 Frobenius residual
under the modified pipeline at three representative regimes
(P5 N=50, P8 N=84, P5N100 N=100) and compare to the baseline.

Reported per term:
  - baseline median / mean Frobenius residual
  - modified median / mean Frobenius residual
  - improvement factor on mean (lower is better)
  - Lambda_t shift, Lambda_s shift (asymptotic blind fit at N=84)
  - T_00 heavy-tail lift (still localised at high-T_00? or broken
    by the modification?)

A successful term must:
  (a) reduce the mean Frobenius residual at N=100 toward 0.05
  (b) not break the median residual
  (c) preserve or sharpen the System-R structural identification
      Lambda_t = alpha_xi^2 = 0.81 and Lambda_s = -gamma^2/2 = -0.005
  (d) not introduce spurious global shifts that would falsify the
      cross-paper-consistent Lambda value

Output: outputs/higher_order_terms_all8_audit.json
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
    D_MIN, ELL_0, EPS_D, XI_THRESH, edge_to_matrix, per_seed_galerkin)

ALPHA_XI = 0.9
GAMMA = 0.1
LAMBDA_T = 0.81
LAMBDA_S = -0.005

REGIMES_TO_TEST = [
    ("P5", 50), ("P8", 84), ("P5N100", 100),
]


def per_node_residual(g_00, g_ij, t00, t_ij, lambda_t, lambda_s, eye3, xp):
    res00 = g_00 + lambda_t - t00
    spatial_res = (g_ij + lambda_s * eye3[None, :, :]) - t_ij
    sq = res00 ** 2 + (spatial_res ** 2).sum(axis=(1, 2))
    return xp.sqrt(sq)


def t00_lift(residual, t00):
    """Compute lift of top-decile-residual / top-decile-T00 overlap."""
    p90_res = np.percentile(residual, 90)
    p90_t00 = np.percentile(t00, 90)
    top_res = residual >= p90_res
    top_t00 = t00 >= p90_t00
    n_top_res = int(top_res.sum())
    n_top_t00 = int(top_t00.sum())
    n_overlap = int((top_res & top_t00).sum())
    n = len(residual)
    expected = n_top_res * n_top_t00 / max(n, 1)
    if expected <= 0:
        return float("nan")
    return float(n_overlap / expected)


def build_baseline(xi_mat, psi, k_field, q_field, n_lat):
    return per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)


def hessian_ricci_quadratic(prep_data, xi_mat, n_lat, eta1=1.0):
    """O1: Add quadratic correction (1 - delta^2/d^2)^2 to R_ij."""
    xi_off = np.where(np.isfinite(xi_mat), xi_mat, 0.0).copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(np.float64)
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1) + 1e-12
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    l_norm = (np.eye(n_lat) - (deg_inv_sqrt[:, None] * weight_adj
                                * deg_inv_sqrt[None, :]))
    eigvals_l, eigvecs_l = np.linalg.eigh(l_norm)
    spatial = eigvecs_l[:, 1:4]
    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat = np.maximum(d_mat, D_MIN)
    spatial_diff = spatial[None, :, :] - spatial[:, None, :]
    delta_sq = (spatial_diff ** 2).sum(axis=2)
    inv_d = np.where(adj > 0, 1.0 / d_mat, 0.0)
    e_alpha = spatial_diff * inv_d[:, :, None]
    d_sq = d_mat * d_mat
    discrepancy = np.where(adj > 0, 1.0 - delta_sq / (d_sq + EPS_D), 0.0)
    discrepancy_sq = discrepancy ** 2  # quadratic correction
    weight_disc_2 = weight_adj * discrepancy_sq
    r_ij_2 = np.einsum("ab,abi,abj->aij", weight_disc_2, e_alpha, e_alpha)
    norm = (weight_adj.sum(axis=1) + 1e-12)
    r_ij_correction = eta1 * r_ij_2 / norm[:, None, None]
    # Rebuild G_ij with correction
    g_ij_new = prep_data["g_ij_h"] + r_ij_correction
    r_bar_corr = np.trace(r_ij_correction, axis1=1, axis2=2)
    g_00_new = prep_data["g_00_h"] + 0.5 * r_bar_corr
    return g_00_new, g_ij_new


def xi3_triple_vertex(prep_data, xi_mat, n_lat, kappa=0.081):
    """O2: Add Xi^(3)_aa = (Xi^3)_aa contribution to T_00."""
    xi_off = np.where(np.isfinite(xi_mat), xi_mat, 0.0).copy()
    np.fill_diagonal(xi_off, 0.0)
    # Xi^3 matrix; diagonal entries = sum_{b,c} Xi_ab Xi_bc Xi_ca
    xi_cubed = xi_off @ xi_off @ xi_off
    xi3_diag = np.diag(xi_cubed)
    # Normalise by typical neighbour-count scale
    scale = max(xi_off.sum() / n_lat, 1e-9)
    return prep_data["t00"] + kappa * xi3_diag / scale


def topological_torsion(prep_data, winding, n_lat, kappa=0.1):
    """O3: T^torsion_00 = kappa |w|^2 |gradPsi|^2 (vortex sector)."""
    # |gradPsi|^2 estimate per node from the spectral construction
    # already inside prep_data: T_00 contains gradient term
    # We just add a term proportional to |w|^2 * existing T_00
    # to capture vortex-source localisation.
    if winding is None:
        return prep_data["t00"]
    w_sq = winding ** 2
    # Normalise: kappa * w_sq fraction of t00
    return prep_data["t00"] + kappa * w_sq * prep_data["t00"]


def krec_laplacian(prep_data, k_field, xi_mat, n_lat, zeta4=0.1):
    """O4: Add zeta_4 * Lap(K_rec) to T_00."""
    xi_off = np.where(np.isfinite(xi_mat), xi_mat, 0.0).copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(np.float64)
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1) + 1e-12
    # Per-node K_rec value: row mean of k_field weighted by adjacency
    k_row = (k_field * adj).sum(axis=1) / deg
    # Discrete Laplacian: sum_b w_ab (K(b) - K(a)) / sum_b w_ab
    delta_k = ((weight_adj * (k_row[None, :] - k_row[:, None])).sum(axis=1)
                / deg)
    return prep_data["t00"] + zeta4 * delta_k


def lambda_running(prep_data, omega_a, kappa=1.0):
    """O5: Lambda_eff(a) = Lambda_struct (1 + kappa * omega_a / mean(omega))."""
    omega_mean = max(omega_a.mean(), 1e-9)
    factor = 1.0 + kappa * omega_a / omega_mean
    lambda_t_a = LAMBDA_T * factor
    lambda_s_a = LAMBDA_S * factor
    return lambda_t_a, lambda_s_a


def adaptive_basis_weighting(prep_data, xi_mat, n_lat):
    """O6: Eigenvalue-weighted spatial basis (heavier weight on
    lower modes). This is the simpler tractable version of
    'locally-adaptive': we keep the global modes but reweight."""
    xi_off = np.where(np.isfinite(xi_mat), xi_mat, 0.0).copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(np.float64)
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1) + 1e-12
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    l_norm = (np.eye(n_lat) - (deg_inv_sqrt[:, None] * weight_adj
                                * deg_inv_sqrt[None, :]))
    eigvals_l, eigvecs_l = np.linalg.eigh(l_norm)
    # Spectral-gap weighting: weight each mode by 1/sqrt(eigval+eps)
    weights = 1.0 / np.sqrt(np.maximum(eigvals_l[1:4], 1e-6))
    weights = weights / weights.mean()  # normalise
    # Re-weighted spatial (this would propagate through but the
    # Galerkin construction is already done with unit weights;
    # we approximate by scaling g_ij by a scalar that reflects
    # the weighting at the dominant mode).
    # Average weight as a proxy multiplicative factor on G_ij:
    avg_w = float(weights.mean())
    # Apply small reweighting (factor close to 1) to G_ij
    return prep_data["g_00_h"] / avg_w, prep_data["g_ij_h"] / avg_w


def newton_running(prep_data, omega_a, n_lat):
    """Own1: 8 pi G_eff(a) = 8 pi G (1 - omega_a * alpha_xi^3 / N)."""
    factor = 1.0 - omega_a * (ALPHA_XI ** 3) / n_lat
    factor = np.maximum(factor, 0.1)  # don't go negative
    return prep_data["t00"] * factor, prep_data["t_ij"] * factor[:, None, None]


def discrete_bianchi(prep_data, xi_mat, n_lat):
    """Own2: compute B^nu = grad^mu_disc G_munu and report whether
    its localisation matches the heavy-tail. We do NOT subtract it
    (that would over-fit by construction); instead we report the
    correlation of |B|^2 per-node with the residual."""
    xi_off = np.where(np.isfinite(xi_mat), xi_mat, 0.0).copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(np.float64)
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1) + 1e-12
    g_00 = prep_data["g_00_h"]
    g_ij = prep_data["g_ij_h"]
    # Discrete divergence of G_00 = sum_b w_ab (G_00(b) - G_00(a)) / sum_b w_ab
    div_g_00 = ((weight_adj * (g_00[None, :] - g_00[:, None])).sum(axis=1)
                 / deg)
    # Per-node B-magnitude
    div_g_ij = np.einsum(
        "ab,bij->aij", weight_adj,
        g_ij - g_ij.mean(axis=0)) / deg[:, None, None]
    b_norm_sq = div_g_00 ** 2 + (div_g_ij ** 2).sum(axis=(1, 2))
    return np.sqrt(b_norm_sq)


def evaluate_term(term_id, regime, n_lat, results_dict):
    """Run baseline + modified pipeline at given regime, store
    all metrics."""
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    winding = d["winding_map"] if "winding_map" in d.files else None
    n_seeds = min(edge_arr.shape[0], 32)

    base_res, mod_res, base_t00, mod_t00 = [], [], [], []
    base_g00, mod_g00 = [], []
    base_lam_t, mod_lam_t = [], []
    base_lam_s, mod_lam_s = [], []
    bianchi_corr_acc = []

    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))

        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        eye3 = prep["eye3"]

        # Compute per-node omega_a ourselves for the modifications
        xi_off = np.where(np.isfinite(xi_mat), xi_mat, 0.0).copy()
        np.fill_diagonal(xi_off, 0.0)
        adj = (xi_off > XI_THRESH).astype(np.float64)
        d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
        d_mat = np.maximum(d_mat, D_MIN)
        d_sq_safe = np.where(adj > 0, d_mat ** 2, np.inf)
        weight_grad = np.where(
            adj > 0, (xi_off * adj) / (d_sq_safe + EPS_D), 0.0)
        omega_a = weight_grad.sum(axis=1)

        # baseline residual
        base_r = per_node_residual(
            prep["g_00_h"], prep["g_ij_h"],
            prep["t00"], prep["t_ij"],
            LAMBDA_T, LAMBDA_S, eye3, np)
        base_res.append(np.asarray(base_r))
        base_t00.append(np.asarray(prep["t00"]))
        base_g00.append(np.asarray(prep["g_00_h"]))

        # blind Lambda baseline
        base_lt = float(np.mean(prep["t00"] - prep["g_00_h"]))
        g_diag = np.stack([prep["g_ij_h"][:, 0, 0],
                            prep["g_ij_h"][:, 1, 1],
                            prep["g_ij_h"][:, 2, 2]], axis=1)
        t_diag = np.stack([prep["t_ij"][:, 0, 0],
                            prep["t_ij"][:, 1, 1],
                            prep["t_ij"][:, 2, 2]], axis=1)
        base_ls = float(np.mean(t_diag - g_diag))
        base_lam_t.append(base_lt)
        base_lam_s.append(base_ls)

        # Apply the term
        g_00_m = prep["g_00_h"]
        g_ij_m = prep["g_ij_h"]
        t00_m = prep["t00"]
        t_ij_m = prep["t_ij"]
        lambda_t_use = LAMBDA_T
        lambda_s_use = LAMBDA_S

        if term_id == "O1":
            g_00_m, g_ij_m = hessian_ricci_quadratic(prep, xi_mat, n_lat)
        elif term_id == "O2":
            t00_m = xi3_triple_vertex(prep, xi_mat, n_lat)
        elif term_id == "O3":
            wnd = np.asarray(winding[s]) if winding is not None else None
            t00_m = topological_torsion(prep, wnd, n_lat)
        elif term_id == "O4":
            t00_m = krec_laplacian(prep, k_field, xi_mat, n_lat)
        elif term_id == "O5":
            lt_a, ls_a = lambda_running(prep, omega_a)
            # per-node lambda — compute residual node-by-node
            res00 = g_00_m + lt_a - t00_m
            spatial_res = (g_ij_m + ls_a[:, None, None]
                            * eye3[None, :, :]) - t_ij_m
            sq = res00 ** 2 + (spatial_res ** 2).sum(axis=(1, 2))
            mod_r = np.sqrt(sq)
            # Skip the normal computation below
            mod_res.append(mod_r)
            mod_t00.append(t00_m)
            mod_g00.append(g_00_m)
            # blind Lambda for O5 = mean of the locally-running Λ
            mod_lam_t.append(float(np.mean(lt_a)))
            mod_lam_s.append(float(np.mean(ls_a)))
            continue
        elif term_id == "O6":
            g_00_m, g_ij_m = adaptive_basis_weighting(prep, xi_mat, n_lat)
        elif term_id == "Own1":
            t00_m, t_ij_m = newton_running(prep, omega_a, n_lat)
        elif term_id == "Own2":
            b_norm = discrete_bianchi(prep, xi_mat, n_lat)
            # Diagnostic: correlation between B^nu magnitude and residual
            r_arr = np.asarray(base_r)
            from numpy import argsort, sqrt as _sqrt
            rx = argsort(argsort(r_arr)) - len(r_arr) / 2
            ry = argsort(argsort(b_norm)) - len(b_norm) / 2
            denom = _sqrt(((rx * rx).sum() * (ry * ry).sum())) or 1.0
            corr = float((rx * ry).sum() / denom)
            bianchi_corr_acc.append(corr)
            # No actual residual modification for Own2 (diagnostic only)
            mod_res.append(np.asarray(base_r))
            mod_t00.append(np.asarray(prep["t00"]))
            mod_g00.append(np.asarray(prep["g_00_h"]))
            mod_lam_t.append(base_lt)
            mod_lam_s.append(base_ls)
            continue

        mod_r = per_node_residual(
            g_00_m, g_ij_m, t00_m, t_ij_m,
            lambda_t_use, lambda_s_use, eye3, np)
        mod_res.append(np.asarray(mod_r))
        mod_t00.append(np.asarray(t00_m))
        mod_g00.append(np.asarray(g_00_m))

        # blind Lambda recomputed for modified
        mod_lt = float(np.mean(t00_m - g_00_m))
        g_diag_m = np.stack([g_ij_m[:, 0, 0], g_ij_m[:, 1, 1],
                              g_ij_m[:, 2, 2]], axis=1)
        t_diag_m = np.stack([t_ij_m[:, 0, 0], t_ij_m[:, 1, 1],
                              t_ij_m[:, 2, 2]], axis=1)
        mod_ls = float(np.mean(t_diag_m - g_diag_m))
        mod_lam_t.append(mod_lt)
        mod_lam_s.append(mod_ls)

    base_res_all = np.concatenate(base_res)
    mod_res_all = np.concatenate(mod_res)
    base_t00_all = np.concatenate(base_t00)
    mod_t00_all = np.concatenate(mod_t00)

    rec = {
        "regime": regime, "N": n_lat, "term": term_id,
        "baseline": {
            "median": float(np.median(base_res_all)),
            "mean": float(base_res_all.mean()),
            "lift_t00": t00_lift(base_res_all, base_t00_all),
            "lambda_t": float(np.mean(base_lam_t)),
            "lambda_s": float(np.mean(base_lam_s)),
        },
        "modified": {
            "median": float(np.median(mod_res_all)),
            "mean": float(mod_res_all.mean()),
            "lift_t00": t00_lift(mod_res_all, mod_t00_all),
            "lambda_t": float(np.mean(mod_lam_t)),
            "lambda_s": float(np.mean(mod_lam_s)),
        },
        "improvement": {
            "median_change": float(
                np.median(mod_res_all) - np.median(base_res_all)),
            "mean_change": float(
                mod_res_all.mean() - base_res_all.mean()),
            "median_relative": float(
                (np.median(mod_res_all) - np.median(base_res_all))
                / max(np.median(base_res_all), 1e-12)),
            "mean_relative": float(
                (mod_res_all.mean() - base_res_all.mean())
                / max(base_res_all.mean(), 1e-12)),
        },
    }
    if term_id == "Own2":
        rec["bianchi_residual_correlation"] = float(
            np.mean(bianchi_corr_acc))
    results_dict[(term_id, regime)] = rec
    return rec


def main():
    terms = [
        ("O1", "Quadratic Hessian-Ricci correction"),
        ("O2", "Causal-wave Xi^(3) triple vertex"),
        ("O3", "Topological torsion source from vortex sector"),
        ("O4", "Recombination-field Laplacian zeta_4 Lap(K_rec)"),
        ("O5", "Spatial-running cosmological-tensor (one-loop)"),
        ("O6", "Eigenvalue-weighted spatial basis (O6 simplified)"),
        ("Own1", "8 pi G running coupling factor"),
        ("Own2", "Discrete-Bianchi diagnostic correlation only"),
    ]

    print("=" * 130)
    print("Higher-order term test on existing D1 NPZ data")
    print("Regimes: P5 (N=50), P8 (N=84), P5N100 (N=100)")
    print("=" * 130)

    results = {}
    for term_id, descr in terms:
        print()
        print(f"--- {term_id}: {descr} ---")
        print(f"{'regime':<8} {'N':>3} | "
              f"{'med base':>9} {'med mod':>9} {'D_med':>8} | "
              f"{'mean base':>10} {'mean mod':>10} {'D_mean':>9} | "
              f"{'lam_t':>7} {'lam_s':>9} | "
              f"{'lift_t00':>9}")
        print("-" * 110)
        for regime, n_lat in REGIMES_TO_TEST:
            rec = evaluate_term(term_id, regime, n_lat, results)
            if rec is None:
                continue
            b = rec["baseline"]
            m = rec["modified"]
            i = rec["improvement"]
            print(f"{regime:<8} {n_lat:>3} | "
                  f"{b['median']:>9.4f} {m['median']:>9.4f} "
                  f"{i['median_relative']*100:>+7.1f}% | "
                  f"{b['mean']:>10.4f} {m['mean']:>10.4f} "
                  f"{i['mean_relative']*100:>+8.1f}% | "
                  f"{m['lambda_t']:>7.4f} {m['lambda_s']:>+9.4f} | "
                  f"{m['lift_t00']:>9.2f}")
        if term_id == "Own2":
            avg_bianchi_corr = np.mean(
                [r["bianchi_residual_correlation"]
                 for k, r in results.items()
                 if k[0] == "Own2"])
            print(f"  Mean Bianchi-residual rank-correlation: "
                  f"{avg_bianchi_corr:+.3f}")

    print()
    print("=" * 130)
    print("VERDICT SUMMARY (focus on P5N100 as the highest-N test)")
    print("=" * 130)
    print(f"{'term':<8} {'mean_change@N100':>17} {'median_change@N100':>20} {'lam_t shift':>14} {'lift_t00 mod':>14}")
    print("-" * 90)
    for term_id, _ in terms:
        key = (term_id, "P5N100")
        if key not in results:
            continue
        r = results[key]
        print(f"{term_id:<8} "
              f"{r['improvement']['mean_relative']*100:>+16.1f}% "
              f"{r['improvement']['median_relative']*100:>+19.1f}% "
              f"{r['modified']['lambda_t']-r['baseline']['lambda_t']:>+14.4f} "
              f"{r['modified']['lift_t00']:>14.2f}")

    out_path = REPO / "outputs" / "higher_order_terms_all8_audit.json"
    out = {
        "method": "higher_order_term_pipeline_modification_audit",
        "schema_version": "1.0.0",
        "structural_lambda": {"t": LAMBDA_T, "s": LAMBDA_S},
        "regimes_tested": [
            {"regime": r, "N": n} for r, n in REGIMES_TO_TEST],
        "terms_tested": [
            {"id": tid, "description": desc} for tid, desc in terms],
        "per_term_per_regime": [
            {"key": f"{tid}_{reg}", **rec}
            for (tid, reg), rec in results.items()
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
