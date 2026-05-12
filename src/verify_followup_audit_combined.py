"""Follow-up audit combining three tests:

  (A) Combined O1+Own1 test: quadratic Hessian-Ricci correction
      AND 8 pi G-running coupling applied jointly. The two single
      tests gave -5.3% and -11.0% mean reductions individually
      with mostly orthogonal mechanisms; jointly they should
      stack additively.

  (B) O5 coefficient sweep: kappa in {0.001, 0.01, 0.05, 0.1, 0.5,
      1.0} for the spatial-running cosmological-tensor
      Lambda_eff(a) = Lambda_struct (1 + kappa * omega_a / mean(omega)).
      The kappa=1.0 baseline overshot massively; the right value
      should be much smaller and is presumably an algebraic
      System-R quantity like alpha_xi^4 or gamma^2.

  (C) Gravity-nucleation indirect tests, using what existing
      data permits:
        (C1) global time-series lead-lag analysis between the
             energy-proxy (eta_p_ts) and the cluster-activation
             proxy (active_fraction_ts) and the frustration
             relaxation (negative_triangle_share_ts). If
             eta_p leads active_fraction in time, that is
             consistent with the nucleation picture.
        (C2) spatial-clustering of the top-decile-residual nodes
             measured via their internal pairwise Xi-density: if
             they are graph-connected dense subclusters rather
             than randomly distributed, this supports the
             nucleation picture (nucleation seeds organise into
             coherent matter-cores).
        (C3) per-seed reproducibility of the top-decile-residual
             positions: if the same lattice positions are
             top-decile across seeds, the localisation is a real
             geometric structure rather than a random snapshot.

Output: outputs/followup_audit_combined.json
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
from verify_higher_order_terms_all8 import (
    LAMBDA_T, LAMBDA_S, ALPHA_XI, GAMMA,
    per_node_residual, t00_lift,
    hessian_ricci_quadratic, lambda_running)


REGIMES_TO_TEST = [
    ("P5", 50), ("P8", 84), ("P5N100", 100),
]


# ---- Part A: combined O1+Own1 ----

def evaluate_combined_O1_Own1(regime, n_lat):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)

    base_res, mod_res, base_t00 = [], [], []
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        eye3 = prep["eye3"]

        # Compute omega_a
        xi_off = np.where(np.isfinite(xi_mat), xi_mat, 0.0).copy()
        np.fill_diagonal(xi_off, 0.0)
        adj = (xi_off > XI_THRESH).astype(np.float64)
        d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
        d_mat = np.maximum(d_mat, D_MIN)
        d_sq_safe = np.where(adj > 0, d_mat ** 2, np.inf)
        weight_grad = np.where(
            adj > 0, (xi_off * adj) / (d_sq_safe + EPS_D), 0.0)
        omega_a = weight_grad.sum(axis=1)

        # Baseline
        base_r = per_node_residual(
            prep["g_00_h"], prep["g_ij_h"],
            prep["t00"], prep["t_ij"],
            LAMBDA_T, LAMBDA_S, eye3, np)
        base_res.append(np.asarray(base_r))
        base_t00.append(np.asarray(prep["t00"]))

        # Apply O1 (quadratic Hessian) + Own1 (8 pi G running)
        g_00_O1, g_ij_O1 = hessian_ricci_quadratic(prep, xi_mat, n_lat)
        # Own1: rescale T side
        factor = 1.0 - omega_a * (ALPHA_XI ** 3) / n_lat
        factor = np.maximum(factor, 0.1)
        t00_m = prep["t00"] * factor
        t_ij_m = prep["t_ij"] * factor[:, None, None]
        # Combined residual
        mod_r = per_node_residual(
            g_00_O1, g_ij_O1, t00_m, t_ij_m,
            LAMBDA_T, LAMBDA_S, eye3, np)
        mod_res.append(np.asarray(mod_r))

    base_r = np.concatenate(base_res)
    mod_r = np.concatenate(mod_res)
    base_t00_a = np.concatenate(base_t00)

    return {
        "regime": regime, "N": n_lat,
        "baseline": {
            "median": float(np.median(base_r)),
            "mean": float(base_r.mean()),
            "lift_t00": t00_lift(base_r, base_t00_a),
        },
        "O1_plus_Own1": {
            "median": float(np.median(mod_r)),
            "mean": float(mod_r.mean()),
            "lift_t00": t00_lift(mod_r, base_t00_a),
        },
        "improvement": {
            "median_relative_pct": float(
                (np.median(mod_r) - np.median(base_r))
                / max(np.median(base_r), 1e-12) * 100),
            "mean_relative_pct": float(
                (mod_r.mean() - base_r.mean())
                / max(base_r.mean(), 1e-12) * 100),
        },
    }


# ---- Part B: O5 kappa sweep ----

def evaluate_O5_at_kappa(regime, n_lat, kappa):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)

    base_res, mod_res, base_t00 = [], [], []
    lam_t_arr = []
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        eye3 = prep["eye3"]

        xi_off = np.where(np.isfinite(xi_mat), xi_mat, 0.0).copy()
        np.fill_diagonal(xi_off, 0.0)
        adj = (xi_off > XI_THRESH).astype(np.float64)
        d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
        d_mat = np.maximum(d_mat, D_MIN)
        d_sq_safe = np.where(adj > 0, d_mat ** 2, np.inf)
        weight_grad = np.where(
            adj > 0, (xi_off * adj) / (d_sq_safe + EPS_D), 0.0)
        omega_a = weight_grad.sum(axis=1)

        base_r = per_node_residual(
            prep["g_00_h"], prep["g_ij_h"],
            prep["t00"], prep["t_ij"],
            LAMBDA_T, LAMBDA_S, eye3, np)
        base_res.append(np.asarray(base_r))
        base_t00.append(np.asarray(prep["t00"]))

        # O5: position-dependent Lambda
        omega_mean = max(omega_a.mean(), 1e-9)
        factor = 1.0 + kappa * omega_a / omega_mean
        lt_a = LAMBDA_T * factor
        ls_a = LAMBDA_S * factor

        res00 = prep["g_00_h"] + lt_a - prep["t00"]
        spatial_res = (prep["g_ij_h"] + ls_a[:, None, None]
                        * eye3[None, :, :]) - prep["t_ij"]
        sq = res00 ** 2 + (spatial_res ** 2).sum(axis=(1, 2))
        mod_r = np.sqrt(sq)
        mod_res.append(np.asarray(mod_r))
        lam_t_arr.append(float(np.mean(lt_a)))

    base_r = np.concatenate(base_res)
    mod_r = np.concatenate(mod_res)
    base_t00_a = np.concatenate(base_t00)

    return {
        "regime": regime, "N": n_lat, "kappa": kappa,
        "baseline_mean": float(base_r.mean()),
        "baseline_median": float(np.median(base_r)),
        "modified_mean": float(mod_r.mean()),
        "modified_median": float(np.median(mod_r)),
        "lift_t00_modified": t00_lift(mod_r, base_t00_a),
        "lambda_t_effective_mean": float(np.mean(lam_t_arr)),
        "mean_relative_pct": float(
            (mod_r.mean() - base_r.mean()) / max(base_r.mean(), 1e-12) * 100),
        "median_relative_pct": float(
            (np.median(mod_r) - np.median(base_r))
            / max(np.median(base_r), 1e-12) * 100),
    }


# ---- Part C: Gravity-nucleation indirect tests ----

def lead_lag_correlation(x, y, max_lag=10):
    """Compute cross-correlation between two time-series at lags
    -max_lag..max_lag. Positive lag means y leads x; negative
    lag means x leads y."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    n = len(x)
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            corr = float(np.mean(x[:n - lag] * y[lag:]))
        else:
            corr = float(np.mean(x[-lag:] * y[:n + lag]))
        out[lag] = corr
    best_lag = max(out, key=lambda k: abs(out[k]))
    return out, best_lag


def gravity_nucleation_test(regime, n_lat):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    if "eta_p_ts" not in d.files or "active_fraction_ts" not in d.files:
        return None

    eta_p_ts = np.asarray(d["eta_p_ts"])
    active_frac_ts = np.asarray(d["active_fraction_ts"])
    neg_tri_ts = np.asarray(d["negative_triangle_share_ts"]) \
        if "negative_triangle_share_ts" in d.files else None
    soft_action_ts = np.asarray(d["soft_action_ts"]) \
        if "soft_action_ts" in d.files else None

    n_seeds = min(eta_p_ts.shape[0], 32)
    n_steps = eta_p_ts.shape[1]

    # Per-seed lead-lag analysis
    leadlag = {"eta_p_vs_active_fraction": [],
               "eta_p_vs_negative_triangle": [],
               "active_fraction_vs_negative_triangle": []}
    for s in range(n_seeds):
        if not np.all(np.isfinite(eta_p_ts[s])) or \
                not np.all(np.isfinite(active_frac_ts[s])):
            continue
        cc, lag = lead_lag_correlation(
            eta_p_ts[s], active_frac_ts[s], max_lag=8)
        leadlag["eta_p_vs_active_fraction"].append(
            {"seed": s, "best_lag": lag, "corr_at_best_lag": cc[lag],
             "corr_lag_0": cc[0]})
        if neg_tri_ts is not None and np.all(np.isfinite(neg_tri_ts[s])):
            cc, lag = lead_lag_correlation(
                eta_p_ts[s], neg_tri_ts[s], max_lag=8)
            leadlag["eta_p_vs_negative_triangle"].append(
                {"seed": s, "best_lag": lag, "corr_at_best_lag": cc[lag],
                 "corr_lag_0": cc[0]})
            cc, lag = lead_lag_correlation(
                active_frac_ts[s], neg_tri_ts[s], max_lag=8)
            leadlag["active_fraction_vs_negative_triangle"].append(
                {"seed": s, "best_lag": lag, "corr_at_best_lag": cc[lag],
                 "corr_lag_0": cc[0]})

    return {
        "regime": regime, "N": n_lat,
        "n_time_steps": int(n_steps),
        "lead_lag_analysis": leadlag,
        "interpretation": (
            "best_lag > 0 means second series leads first. "
            "Nucleation hypothesis: eta_p (energy density proxy) "
            "should LEAD active_fraction (cluster activation) "
            "i.e. best_lag > 0 in 'eta_p_vs_active_fraction'."
        ),
    }


def spatial_cluster_test(regime, n_lat):
    """Test C2: are top-decile-residual nodes graph-connected
    (clustered) or random (scattered)?"""
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)

    cluster_results = []
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        eye3 = prep["eye3"]
        residual = np.asarray(per_node_residual(
            prep["g_00_h"], prep["g_ij_h"],
            prep["t00"], prep["t_ij"],
            LAMBDA_T, LAMBDA_S, eye3, np))
        p90 = np.percentile(residual, 90)
        top_mask = residual >= p90
        # Compute internal Xi-density between top-decile nodes
        xi_off = np.where(np.isfinite(xi_mat), xi_mat, 0.0).copy()
        np.fill_diagonal(xi_off, 0.0)
        # Submatrix among top-decile nodes
        idx = np.where(top_mask)[0]
        if len(idx) < 2:
            continue
        subm = xi_off[np.ix_(idx, idx)]
        # Mean Xi between top nodes
        mean_xi_inside = float(subm.mean())
        # Mean Xi globally (excluding diagonal)
        mean_xi_global = float(xi_off[xi_off > 0].mean()) \
            if (xi_off > 0).sum() > 0 else 0.0
        # Connected-component structure inside top-decile
        adj_inside = (subm > XI_THRESH).astype(int)
        np.fill_diagonal(adj_inside, 0)
        # BFS for connected components
        n_inside = len(idx)
        visited = np.zeros(n_inside, dtype=bool)
        sizes = []
        for start in range(n_inside):
            if visited[start]:
                continue
            stack = [start]
            comp = []
            while stack:
                v = stack.pop()
                if visited[v]:
                    continue
                visited[v] = True
                comp.append(v)
                for u in range(n_inside):
                    if not visited[u] and adj_inside[v, u]:
                        stack.append(u)
            sizes.append(len(comp))
        cluster_results.append({
            "seed": s,
            "n_top_decile": int(n_inside),
            "mean_xi_inside_top10": mean_xi_inside,
            "mean_xi_global": mean_xi_global,
            "ratio_inside_global": float(
                mean_xi_inside / max(mean_xi_global, 1e-12)),
            "n_connected_components": len(sizes),
            "largest_component_size": int(max(sizes)),
            "size_distribution": sorted(sizes, reverse=True),
        })
    return {"regime": regime, "N": n_lat, "per_seed": cluster_results}


def seed_reproducibility_test(regime, n_lat):
    """Test C3: are top-decile-residual node positions
    reproducible across seeds? Compute Jaccard overlap of top-10%
    masks between seed pairs."""
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)

    masks = []
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        eye3 = prep["eye3"]
        residual = np.asarray(per_node_residual(
            prep["g_00_h"], prep["g_ij_h"],
            prep["t00"], prep["t_ij"],
            LAMBDA_T, LAMBDA_S, eye3, np))
        p90 = np.percentile(residual, 90)
        masks.append(residual >= p90)

    pair_jaccards = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            inter = (masks[i] & masks[j]).sum()
            union = (masks[i] | masks[j]).sum()
            j_idx = float(inter / max(union, 1))
            pair_jaccards.append({"i": i, "j": j, "jaccard": j_idx})
    expected_random = (0.10) ** 2 / (2 * 0.10 - (0.10) ** 2)
    return {
        "regime": regime, "N": n_lat,
        "n_seeds": n_seeds,
        "pair_jaccards": pair_jaccards,
        "mean_jaccard": float(
            np.mean([pj["jaccard"] for pj in pair_jaccards]))
            if pair_jaccards else float("nan"),
        "expected_jaccard_random": expected_random,
        "lift_over_random": float(
            np.mean([pj["jaccard"] for pj in pair_jaccards]) /
            max(expected_random, 1e-12)) if pair_jaccards else float("nan"),
    }


def main():
    print("=" * 110)
    print("Follow-up audit: combined O1+Own1, O5 kappa sweep, "
          "gravity-nucleation indirect tests")
    print("=" * 110)

    results = {
        "A_combined_O1_Own1": [],
        "B_O5_kappa_sweep": [],
        "C1_gravity_nucleation_lead_lag": [],
        "C2_spatial_cluster_of_heavy_tail": [],
        "C3_seed_reproducibility": [],
    }

    # Part A
    print("\n--- (A) Combined O1+Own1 ---")
    print(f"{'reg':<8} {'N':>3} | {'med base':>9} {'med mod':>9} {'D_med':>8} | "
          f"{'mean base':>10} {'mean mod':>10} {'D_mean':>8} | {'lift_T00 mod':>12}")
    print("-" * 90)
    for regime, n_lat in REGIMES_TO_TEST:
        r = evaluate_combined_O1_Own1(regime, n_lat)
        if r is None:
            continue
        results["A_combined_O1_Own1"].append(r)
        print(f"{regime:<8} {n_lat:>3} | "
              f"{r['baseline']['median']:>9.4f} "
              f"{r['O1_plus_Own1']['median']:>9.4f} "
              f"{r['improvement']['median_relative_pct']:>+7.1f}% | "
              f"{r['baseline']['mean']:>10.4f} "
              f"{r['O1_plus_Own1']['mean']:>10.4f} "
              f"{r['improvement']['mean_relative_pct']:>+7.1f}% | "
              f"{r['O1_plus_Own1']['lift_t00']:>12.2f}")

    # Part B
    print("\n--- (B) O5 kappa sweep at P5N100 (within-regime extension) ---")
    print(f"{'kappa':>8} | {'mean base':>10} {'mean mod':>10} {'D_mean':>8} | "
          f"{'med base':>9} {'med mod':>9} {'D_med':>8} | {'lambda_t':>9} {'lift_T00':>9}")
    print("-" * 100)
    for kappa in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
        r = evaluate_O5_at_kappa("P5N100", 100, kappa)
        if r is None:
            continue
        results["B_O5_kappa_sweep"].append(r)
        print(f"{kappa:>8.3f} | "
              f"{r['baseline_mean']:>10.4f} "
              f"{r['modified_mean']:>10.4f} "
              f"{r['mean_relative_pct']:>+7.1f}% | "
              f"{r['baseline_median']:>9.4f} "
              f"{r['modified_median']:>9.4f} "
              f"{r['median_relative_pct']:>+7.1f}% | "
              f"{r['lambda_t_effective_mean']:>9.4f} "
              f"{r['lift_t00_modified']:>9.2f}")

    # Part C1: lead-lag analysis
    print("\n--- (C1) Gravity-nucleation lead-lag (eta_p vs cluster activation) ---")
    for regime, n_lat in [("P5N100", 100), ("P8", 84), ("P5", 50)]:
        r = gravity_nucleation_test(regime, n_lat)
        if r is None:
            continue
        results["C1_gravity_nucleation_lead_lag"].append(r)
        print(f"\n{regime:<8} N={n_lat:>3} (n_steps={r['n_time_steps']})")
        ll = r["lead_lag_analysis"]
        for s_data in ll.get("eta_p_vs_active_fraction", []):
            print(f"  seed {s_data['seed']}: best_lag={s_data['best_lag']:>+3d} "
                  f"corr={s_data['corr_at_best_lag']:>+.3f}  "
                  f"(lag-0 corr {s_data['corr_lag_0']:>+.3f})")

    # Part C2: spatial clustering
    print("\n--- (C2) Spatial clustering of heavy-tail nodes (top-decile) ---")
    print(f"{'regime':<8} {'seed':>4} {'n_top':>6} {'inside_xi':>10} "
          f"{'global_xi':>10} {'inside/global':>14} {'n_components':>13} {'largest':>8}")
    for regime, n_lat in [("P5", 50), ("P8", 84), ("P5N100", 100)]:
        r = spatial_cluster_test(regime, n_lat)
        if r is None:
            continue
        results["C2_spatial_cluster_of_heavy_tail"].append(r)
        for sd in r["per_seed"]:
            print(f"{regime:<8} {sd['seed']:>4} {sd['n_top_decile']:>6} "
                  f"{sd['mean_xi_inside_top10']:>10.4f} "
                  f"{sd['mean_xi_global']:>10.4f} "
                  f"{sd['ratio_inside_global']:>14.3f} "
                  f"{sd['n_connected_components']:>13} "
                  f"{sd['largest_component_size']:>8}")

    # Part C3: seed reproducibility
    print("\n--- (C3) Seed reproducibility of top-decile-residual mask ---")
    print(f"{'regime':<8} {'N':>3} {'mean Jaccard':>13} {'random expectation':>20} "
          f"{'lift':>8}")
    for regime, n_lat in REGIMES_TO_TEST:
        r = seed_reproducibility_test(regime, n_lat)
        if r is None:
            continue
        results["C3_seed_reproducibility"].append(r)
        print(f"{regime:<8} {n_lat:>3} {r['mean_jaccard']:>13.4f} "
              f"{r['expected_jaccard_random']:>20.4f} "
              f"{r['lift_over_random']:>8.2f}")

    # Verdict
    print("\n" + "=" * 110)
    print("VERDICT")
    print("=" * 110)
    a_n100 = next((r for r in results["A_combined_O1_Own1"]
                    if r["regime"] == "P5N100"), None)
    if a_n100:
        print(f"  (A) O1+Own1 combined at N=100: "
              f"mean {a_n100['baseline']['mean']:.4f} -> "
              f"{a_n100['O1_plus_Own1']['mean']:.4f} "
              f"({a_n100['improvement']['mean_relative_pct']:+.1f}%); "
              f"lift T00 {a_n100['O1_plus_Own1']['lift_t00']:.2f}")
    print()
    print("  (B) O5 kappa sweep optimum (lowest mean residual at P5N100):")
    if results["B_O5_kappa_sweep"]:
        best_kappa = min(results["B_O5_kappa_sweep"],
                          key=lambda r: r["modified_mean"])
        print(f"    kappa={best_kappa['kappa']:.3f}: "
              f"mean {best_kappa['modified_mean']:.4f} "
              f"({best_kappa['mean_relative_pct']:+.1f}% vs baseline); "
              f"lambda_t {best_kappa['lambda_t_effective_mean']:.4f}; "
              f"lift T00 {best_kappa['lift_t00_modified']:.2f}")
    print()
    print("  (C1) Gravity-nucleation: positive average best_lag in")
    print("       eta_p_vs_active_fraction across seeds means eta_p "
          "leads cluster activation -> nucleation supported.")
    print("  (C2) Spatial clustering: ratio_inside/global > 1 means top-decile")
    print("       residual nodes are denser-clustered than random.")
    print("  (C3) Seed reproducibility: lift > 1 means top-decile positions")
    print("       are seed-stable -> real geometric structure, not noise.")

    out_path = REPO / "outputs" / "followup_audit_combined.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
