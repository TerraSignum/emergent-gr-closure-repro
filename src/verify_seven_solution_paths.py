"""Run seven candidate solution paths in one consolidated audit.

  (1) Coarse-graining: average residuals over k-NN clusters
  (2) Riemann^2 higher-curvature correction
  (4) Nonlinear universal law (log^2 and saturating arctan)
  (5) Bianchi-identity discrete divergence test
  (6) Multi-snapshot averaging on P5N100 snapshot file

(3) P5N128-tuned runs in background separately.
(7) Formal derivation is analytical text, not a script.

Output: outputs/seven_solution_paths_audit.json
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
    edge_to_matrix, per_seed_galerkin, XI_THRESH, ELL_0, D_MIN, EPS_D)
from verify_per_eigendirection_residual import (
    per_node_eigendirection_residuals)


REGIMES = [
    ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]
LAMBDA_T = 0.81
LAMBDA_S = -0.005


# ---------------------------------------------------------------- helpers
def gather_basic(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)
    xi_seeds, deltas, t00s, preps = [], [], [], []
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        res = per_node_eigendirection_residuals(prep, LAMBDA_T, LAMBDA_S)
        R_t = res["R_time"]; R_d = res["R_diag"]; R_o = res["R_off"]
        t_e = res["T_eigvals"]
        t00 = np.asarray(prep["t00"])
        R_norm = np.sqrt(R_t ** 2 + (R_d ** 2).sum(axis=1) + R_o ** 2)
        T_norm = np.sqrt(t00 ** 2 + (t_e ** 2).sum(axis=1))
        delta = R_norm / np.maximum(T_norm, 1e-12)
        xi_seeds.append(xi_mat)
        deltas.append(delta)
        t00s.append(t00)
        preps.append(prep)
    return {
        "xi_seeds": xi_seeds,
        "deltas": deltas,
        "t00s": t00s,
        "preps": preps,
        "n_lat": n_lat,
    }


# ---------------------------------------------------------------- (1) coarse-graining
def path_1_coarse_graining(regime_data):
    """Average Delta over k-NN clusters of size k=4 in graph topology."""
    print("=" * 90)
    print("(1) COARSE-GRAINING via k-NN clusters")
    print("=" * 90)
    print(f"{'reg':<8} {'N':>3} | {'raw med':>9} {'cg med':>9} {'raw mean':>9} {'cg mean':>9}")
    print("-" * 60)

    out = []
    for reg, rd in regime_data.items():
        n = rd["n_lat"]
        cg_med_pool, cg_mean_pool = [], []
        for s, (xi_mat, delta) in enumerate(zip(rd["xi_seeds"], rd["deltas"])):
            xi_off = xi_mat.copy()
            np.fill_diagonal(xi_off, 0.0)
            adj = (xi_off > XI_THRESH).astype(np.float64)
            # For each node a, find its top-3 strongest edge neighbours
            cg_delta = np.zeros(n)
            for a in range(n):
                nbrs = np.where(adj[a] > 0)[0]
                if len(nbrs) == 0:
                    cg_delta[a] = delta[a]
                    continue
                # Weighted by edge xi
                w = xi_off[a, nbrs]
                top_k = np.argsort(-w)[:min(4, len(nbrs))]
                idx = np.concatenate(([a], nbrs[top_k]))
                w_full = np.concatenate(([1.0], w[top_k]))
                cg_delta[a] = float(np.average(delta[idx], weights=w_full))
            cg_med_pool.append(cg_delta)
            cg_mean_pool.append(cg_delta)
        raw_d = np.concatenate(rd["deltas"])
        cg_d = np.concatenate(cg_med_pool)
        med_raw = float(np.median(raw_d)); mean_raw = float(raw_d.mean())
        med_cg = float(np.median(cg_d)); mean_cg = float(cg_d.mean())
        print(f"{reg:<8} {n:>3} | {med_raw:>9.4f} {med_cg:>9.4f} {mean_raw:>9.4f} {mean_cg:>9.4f}")
        out.append({"regime": reg, "N": n,
                    "raw_med": med_raw, "cg_med": med_cg,
                    "raw_mean": mean_raw, "cg_mean": mean_cg})
    cg_means_60 = [r["cg_mean"] for r in out if r["N"] >= 60]
    avg_cg_mean = float(np.mean(cg_means_60))
    print(f"\n  Avg cg_mean (N>=60) = {avg_cg_mean:.4f}")
    if avg_cg_mean <= 0.05:
        verdict = "COARSE_GRAINING_CLOSES_STRICT"
    elif avg_cg_mean <= 0.10:
        verdict = "COARSE_GRAINING_RELAXED"
    else:
        verdict = "COARSE_GRAINING_NO_IMPROVEMENT"
    print(f"  VERDICT: {verdict}")
    return {"per_regime": out, "avg_cg_mean_N_geq_60": avg_cg_mean,
            "verdict": verdict}


# ---------------------------------------------------------------- (4) nonlinear law
def path_4_nonlinear_universal(regime_data):
    """Fit Delta_tail = a0 + a1*log(r) + a2*log^2(r), and saturating arctan(c*log(r))."""
    print()
    print("=" * 90)
    print("(4) NONLINEAR UNIVERSAL LAW (log + log^2 and saturating)")
    print("=" * 90)

    pool_log_r, pool_delta = [], []
    for reg, rd in regime_data.items():
        n = rd["n_lat"]
        if reg == "P5":
            continue   # exclude marginal outlier
        delta_all = np.concatenate(rd["deltas"])
        t00_all = np.concatenate(rd["t00s"])
        t00_mean = float(np.mean(t00_all))
        log_r = np.log(np.maximum(t00_all / max(t00_mean, 1e-12), 1e-10))
        order = np.argsort(-delta_all)
        n_tail = max(1, int(len(delta_all) * 0.10))
        tail_idx = order[:n_tail]
        pool_log_r.append(log_r[tail_idx])
        pool_delta.append(delta_all[tail_idx])
    log_r = np.concatenate(pool_log_r)
    delta = np.concatenate(pool_delta)

    def fit_with_R2(X, y):
        c, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
        pred = X @ c
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return c, 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Linear (baseline)
    X1 = np.column_stack([np.ones_like(log_r), log_r])
    c1, r2_1 = fit_with_R2(X1, delta)
    # Quadratic
    X2 = np.column_stack([np.ones_like(log_r), log_r, log_r ** 2])
    c2, r2_2 = fit_with_R2(X2, delta)
    # Saturating: y = a + b * arctan(c * log_r) — fix c=1 for simplicity
    sat = np.arctan(log_r)
    X3 = np.column_stack([np.ones_like(log_r), sat])
    c3, r2_3 = fit_with_R2(X3, delta)
    # Saturating with scale c=2
    sat2 = np.arctan(2 * log_r)
    X4 = np.column_stack([np.ones_like(log_r), sat2])
    c4, r2_4 = fit_with_R2(X4, delta)

    print(f"  Linear:        Delta = {c1[0]:+.4f} + {c1[1]:+.4f}*log(T/<T>),  R^2 = {r2_1:.4f}")
    print(f"  Quadratic:     Delta = {c2[0]:+.4f} + {c2[1]:+.4f}*log + {c2[2]:+.4f}*log^2,  R^2 = {r2_2:.4f}")
    print(f"  Sat(c=1):      Delta = {c3[0]:+.4f} + {c3[1]:+.4f}*arctan(log),  R^2 = {r2_3:.4f}")
    print(f"  Sat(c=2):      Delta = {c4[0]:+.4f} + {c4[1]:+.4f}*arctan(2*log),  R^2 = {r2_4:.4f}")

    best_label = max([("linear", r2_1), ("quadratic", r2_2),
                      ("sat_c1", r2_3), ("sat_c2", r2_4)],
                     key=lambda x: x[1])[0]
    if r2_2 - r2_1 > 0.05:
        verdict = "NONLINEAR_LAW_QUADRATIC_BETTER"
    elif max(r2_3, r2_4) - r2_1 > 0.05:
        verdict = "NONLINEAR_LAW_SATURATING_BETTER"
    else:
        verdict = "LINEAR_LAW_ALREADY_OPTIMAL"
    print(f"  Best: {best_label}, VERDICT: {verdict}")
    return {
        "linear": {"a0": float(c1[0]), "a1": float(c1[1]), "R_squared": r2_1},
        "quadratic": {"a0": float(c2[0]), "a1": float(c2[1]),
                      "a2": float(c2[2]), "R_squared": r2_2},
        "sat_c1": {"a0": float(c3[0]), "b": float(c3[1]), "R_squared": r2_3},
        "sat_c2": {"a0": float(c4[0]), "b": float(c4[1]), "R_squared": r2_4},
        "best_form": best_label,
        "verdict": verdict,
    }


# ---------------------------------------------------------------- (5) Bianchi
def path_5_bianchi(regime_data):
    """Discrete Bianchi-residual: nabla^mu G_munu vs 8*pi*G * nabla^mu T_munu.
    Approximate the divergence as graph Laplacian on the 4-trace G_munu*g^munu = R_bar.
    """
    print()
    print("=" * 90)
    print("(5) BIANCHI-IDENTITY discrete divergence (proxy: scalar curvature divergence)")
    print("=" * 90)
    print("Note: full vector-valued lattice Bianchi requires connection;")
    print("      we use the scalar proxy ||grad(R_bar - 8 pi G T_trace)||.")

    out = []
    for reg, rd in regime_data.items():
        n = rd["n_lat"]
        bianchi_pool = []
        for s, prep in enumerate(rd["preps"]):
            xi_mat = rd["xi_seeds"][s]
            r_bar = np.asarray(prep["r_bar_h"])  # scalar Ricci
            t00 = np.asarray(prep["t00"])
            t_ij = np.asarray(prep["t_ij"])
            t_trace = t00 + np.trace(t_ij, axis1=1, axis2=2)
            scalar_eq = r_bar - 1.0 * t_trace  # 8 pi G = 1
            xi_off = xi_mat.copy()
            np.fill_diagonal(xi_off, 0.0)
            adj = (xi_off > XI_THRESH).astype(np.float64)
            d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
            d_mat = np.maximum(d_mat, D_MIN)
            d_sq = d_mat * d_mat
            d_sq_safe = np.where(adj > 0, d_sq, np.inf)
            w = np.where(adj > 0, xi_off / (d_sq_safe + EPS_D), 0.0)
            # Discrete gradient magnitude
            diff_sq = (scalar_eq[None, :] - scalar_eq[:, None]) ** 2
            grad_sq = (w * diff_sq).sum(axis=1)
            grad_mag = np.sqrt(np.maximum(grad_sq, 0.0))
            bianchi_pool.append(grad_mag)
        b_all = np.concatenate(bianchi_pool)
        med = float(np.median(b_all)); mean = float(b_all.mean())
        out.append({"regime": reg, "N": n, "bianchi_med": med, "bianchi_mean": mean})
        print(f"  {reg:<8} N={n:>3}: bianchi_grad med={med:.5f}, mean={mean:.5f}")
    avg_med = float(np.mean([r["bianchi_med"] for r in out if r["N"] >= 60]))
    if avg_med <= 0.05:
        verdict = "BIANCHI_RESIDUAL_SMALL"
    else:
        verdict = "BIANCHI_RESIDUAL_NON_TRIVIAL"
    print(f"  VERDICT: {verdict} (avg med N>=60 = {avg_med:.5f})")
    return {"per_regime": out, "avg_med_N_geq_60": avg_med, "verdict": verdict}


# ---------------------------------------------------------------- (6) multi-snapshot
def path_6_multi_snapshot():
    """Average Delta over multiple snapshots of P5N100."""
    print()
    print("=" * 90)
    print("(6) MULTI-SNAPSHOT AVERAGING (P5N100 snapshot file)")
    print("=" * 90)

    snap_path = REPO.parent / "results_d1_p5n100_snapshot" / "P5N100.snapshots.npz"
    if not snap_path.exists():
        snap_path = REPO.parent / "results_d1_p5n100_snapshot_16seeds" / "P5N100.snapshots.npz"
    if not snap_path.exists():
        print(f"  Snapshot file not found at {snap_path}")
        return {"verdict": "SNAPSHOT_FILE_UNAVAILABLE"}

    d = np.load(snap_path, allow_pickle=True)
    edge_snaps = d["edge_xi_snapshots"]   # (n_seeds, n_snaps, N, N)
    psi_r = d["psi_real_snapshots"]
    psi_i = d["psi_imag_snapshots"]
    n_seeds = edge_snaps.shape[0]
    n_snaps = edge_snaps.shape[1]
    n_lat = edge_snaps.shape[2]
    print(f"  Snapshots: n_seeds={n_seeds}, n_snaps={n_snaps}, N={n_lat}")

    # Compute Delta for each snapshot, then average per node across snapshots
    delta_per_snapshot = []
    for snap_idx in range(n_snaps):
        deltas_seeds = []
        for s in range(min(n_seeds, 4)):
            xi_mat = edge_snaps[s, snap_idx].copy()
            np.fill_diagonal(xi_mat, 1.0)
            psi = psi_r[s, snap_idx] + 1j * psi_i[s, snap_idx]
            k_field = np.full((n_lat, n_lat), 0.55)
            q_field = np.full((n_lat, n_lat), 0.45)
            prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
            res = per_node_eigendirection_residuals(prep, LAMBDA_T, LAMBDA_S)
            R_t = res["R_time"]; R_d = res["R_diag"]; R_o = res["R_off"]
            t_e = res["T_eigvals"]
            t00 = np.asarray(prep["t00"])
            R_n = np.sqrt(R_t ** 2 + (R_d ** 2).sum(axis=1) + R_o ** 2)
            T_n = np.sqrt(t00 ** 2 + (t_e ** 2).sum(axis=1))
            d_v = R_n / np.maximum(T_n, 1e-12)
            deltas_seeds.append(d_v)
        delta_per_snapshot.append(np.concatenate(deltas_seeds))

    # Average across snapshots
    avg_delta = np.mean(delta_per_snapshot, axis=0)
    last_delta = delta_per_snapshot[-1]
    med_avg = float(np.median(avg_delta)); mean_avg = float(avg_delta.mean())
    med_last = float(np.median(last_delta)); mean_last = float(last_delta.mean())

    print(f"  Last snapshot only: med={med_last:.4f}, mean={mean_last:.4f}")
    print(f"  Multi-snapshot avg: med={med_avg:.4f}, mean={mean_avg:.4f}")

    if mean_avg < mean_last - 0.005:
        verdict = "MULTI_SNAPSHOT_REDUCES_RESIDUAL"
    elif abs(mean_avg - mean_last) < 0.005:
        verdict = "MULTI_SNAPSHOT_NO_CHANGE"
    else:
        verdict = "MULTI_SNAPSHOT_INCREASES_RESIDUAL"
    print(f"  VERDICT: {verdict}")
    return {
        "n_snaps": n_snaps,
        "last_snapshot": {"median": med_last, "mean": mean_last},
        "multi_snapshot_avg": {"median": med_avg, "mean": mean_avg},
        "verdict": verdict,
    }


# ---------------------------------------------------------------- (2) Riemann^2
def path_2_riemann_squared(regime_data):
    """Add alpha * R_bar^2 correction to G_00 on tail nodes; sweep alpha."""
    print()
    print("=" * 90)
    print("(2) RIEMANN^2 higher-curvature correction (proxy: R_bar^2)")
    print("=" * 90)

    alphas = [-0.5, -0.1, -0.05, 0.0, 0.05, 0.1, 0.5]
    sweep = {}
    print(f"{'alpha':>7} | " + " | ".join([f"{r:<6} mean" for r in regime_data]))
    print("-" * (10 + 14 * len(regime_data)))
    for alpha in alphas:
        line = f"{alpha:>+7.3f} |"
        per_reg = []
        for reg, rd in regime_data.items():
            n = rd["n_lat"]
            ds = []
            for prep in rd["preps"]:
                r_bar = np.asarray(prep["r_bar_h"])
                t00 = np.asarray(prep["t00"])
                # G_00_corrected = G_00 + alpha * R_bar^2
                prep_mod = dict(prep)
                prep_mod["g_00_h"] = np.asarray(prep["g_00_h"]) + alpha * r_bar ** 2
                res = per_node_eigendirection_residuals(prep_mod, LAMBDA_T, LAMBDA_S)
                R_t = res["R_time"]; R_d = res["R_diag"]; R_o = res["R_off"]
                t_e = res["T_eigvals"]
                R_n = np.sqrt(R_t ** 2 + (R_d ** 2).sum(axis=1) + R_o ** 2)
                T_n = np.sqrt(t00 ** 2 + (t_e ** 2).sum(axis=1))
                ds.append(R_n / np.maximum(T_n, 1e-12))
            d_all = np.concatenate(ds)
            mean = float(d_all.mean())
            per_reg.append({"regime": reg, "N": n, "mean": mean})
            line += f" {mean:.4f}     |"
        sweep[f"alpha={alpha}"] = {"alpha": alpha, "per_regime": per_reg}
        print(line)
    # Best alpha
    best = min(sweep.items(),
               key=lambda kv: np.mean([r["mean"] for r in kv[1]["per_regime"]
                                       if r["N"] >= 60]))
    avg = float(np.mean([r["mean"] for r in best[1]["per_regime"] if r["N"] >= 60]))
    print(f"  BEST alpha: {best[1]['alpha']:+.3f}, avg_mean(N>=60) = {avg:.4f}")
    if avg <= 0.05:
        verdict = "RIEMANN2_CLOSURE_HOLDS"
    elif avg < 0.07:
        verdict = "RIEMANN2_PARTIAL"
    else:
        verdict = "RIEMANN2_NO_IMPROVEMENT"
    print(f"  VERDICT: {verdict}")
    return {"sweep": sweep, "best_alpha": best[1]["alpha"],
            "best_avg_mean_N_geq_60": avg, "verdict": verdict}


# ---------------------------------------------------------------- main
def main() -> int:
    print("Pre-loading regime data...")
    regime_data = {}
    for reg, n_lat in REGIMES:
        rd = gather_basic(reg, n_lat)
        if rd is not None:
            regime_data[reg] = rd
    print(f"Loaded {len(regime_data)} regimes")
    print()

    out = {
        "method": "seven_solution_paths_consolidated",
        "schema_version": "1.0.0",
        "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
    }
    out["path_1_coarse_graining"] = path_1_coarse_graining(regime_data)
    out["path_2_riemann_squared"] = path_2_riemann_squared(regime_data)
    out["path_4_nonlinear_universal"] = path_4_nonlinear_universal(regime_data)
    out["path_5_bianchi"] = path_5_bianchi(regime_data)
    out["path_6_multi_snapshot"] = path_6_multi_snapshot()

    out_path = REPO / "outputs" / "seven_solution_paths_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
