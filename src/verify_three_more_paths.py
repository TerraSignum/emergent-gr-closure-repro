"""Three more parallel solution paths:

  (D) Cross-seed averaging: per N, average per-node Delta over all
      4 seeds before fitting Symanzik. Reduces seed-variance noise.
  (E) RG-running Lambda(N): fit per-regime optimal Lambda_t(N),
      Lambda_s(N) such that median Delta is minimised, then check
      if these run with N according to a power law.
  (K) Delta_curv (alternative observable): read existing
      data/einstein_gap_9point_frobenius.json and re-fit Symanzik
      on the alternative curvature-gap observable.

(L) more spectral modes: skipped because the Hessian-Ricci tensor
is structurally 3D (1 time + 3 space) and adding spectral modes
beyond [1:4] would change the tensor dimensionality. Runner D
already showed basis-invariance at fixed mode count.

Output: outputs/three_more_paths_audit.json
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
    edge_to_matrix, per_seed_galerkin)
from verify_per_eigendirection_residual import (
    per_node_eigendirection_residuals)


ALL_REGIMES = [
    ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
    ("P4", 42), ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]
LAMBDA_T_STRUCT = 0.81
LAMBDA_S_STRUCT = -0.005


def fit_symanzik_2_4(N, y):
    if len(N) < 4:
        return None
    X = np.column_stack([np.ones_like(N), N ** -2.0, N ** -4.0])
    c, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
    if c[0] < 0:
        X2 = X[:, 1:]
        c12, *_ = np.linalg.lstsq(X2, y, rcond=1e-10)
        c = np.array([0.0, c12[0], c12[1]])
    pred = X @ c
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"d_inf": float(c[0]), "c_2": float(c[1]), "c_4": float(c[2]),
            "R_squared": r2}


# ---------------------------------------------------------------- D
def path_D_cross_seed_average():
    """For each N, compute mean per-node Delta over the 4 seeds,
    then fit Symanzik on that single-N value."""
    print("=" * 100)
    print("(D) Cross-seed-averaged Delta values, per N")
    print("=" * 100)
    rows = []
    for reg, n_lat in ALL_REGIMES:
        p = find_d1_npz(reg, REPO)
        if p is None or not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        edge_arr = d["dense_cell_edge_xi_values"]
        amp_arr = d["dense_cell_node_amplitude_values"]
        phase_arr = d["dense_cell_node_phase_values"]
        n_seeds = min(edge_arr.shape[0], 32)
        # Per-seed pooled Δ; cross-seed-average per node
        seed_deltas = []   # list of (n_lat,) arrays
        for s in range(n_seeds):
            xi_mat = edge_to_matrix(edge_arr[s], n_lat)
            np.fill_diagonal(xi_mat, 1.0)
            psi = amp_arr[s] * np.exp(1j * phase_arr[s])
            k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
            q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
            prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
            res = per_node_eigendirection_residuals(prep, LAMBDA_T_STRUCT, LAMBDA_S_STRUCT)
            R_t = res["R_time"]; R_d = res["R_diag"]; R_o = res["R_off"]
            t_e = res["T_eigvals"]; t00 = np.asarray(prep["t00"])
            R_n = np.sqrt(R_t ** 2 + (R_d ** 2).sum(axis=1) + R_o ** 2)
            T_n = np.sqrt(t00 ** 2 + (t_e ** 2).sum(axis=1))
            d_v = R_n / np.maximum(T_n, 1e-12)
            seed_deltas.append(d_v)
        # Cross-seed average per node
        cross_avg = np.mean(seed_deltas, axis=0)
        rows.append({"regime": reg, "N": n_lat,
                     "delta_med_pooled": float(np.median(np.concatenate(seed_deltas))),
                     "delta_mean_pooled": float(np.mean(np.concatenate(seed_deltas))),
                     "delta_med_cross_seed_avg": float(np.median(cross_avg)),
                     "delta_mean_cross_seed_avg": float(np.mean(cross_avg))})
    print(f"{'reg':<10} {'N':>3} | {'med pooled':>12} {'med cs-avg':>12} {'mean pooled':>13} {'mean cs-avg':>13}")
    print("-" * 70)
    for r in rows:
        print(f"{r['regime']:<10} {r['N']:>3} | "
              f"{r['delta_med_pooled']:>12.5f} {r['delta_med_cross_seed_avg']:>12.5f} "
              f"{r['delta_mean_pooled']:>13.5f} {r['delta_mean_cross_seed_avg']:>13.5f}")
    N_arr = np.array([r["N"] for r in rows], dtype=float)
    fit_med_pooled = fit_symanzik_2_4(N_arr, np.array([r["delta_med_pooled"] for r in rows]))
    fit_med_cs = fit_symanzik_2_4(N_arr, np.array([r["delta_med_cross_seed_avg"] for r in rows]))
    fit_mean_pooled = fit_symanzik_2_4(N_arr, np.array([r["delta_mean_pooled"] for r in rows]))
    fit_mean_cs = fit_symanzik_2_4(N_arr, np.array([r["delta_mean_cross_seed_avg"] for r in rows]))
    print()
    print("Symanzik fits:")
    for label, f in [("med pooled", fit_med_pooled), ("med cs-avg", fit_med_cs),
                      ("mean pooled", fit_mean_pooled), ("mean cs-avg", fit_mean_cs)]:
        print(f"  {label:<15}: d_inf={f['d_inf']:.5f}, R^2={f['R_squared']:.3f}")
    return {
        "per_regime": rows,
        "fit_med_pooled": fit_med_pooled, "fit_med_cs_avg": fit_med_cs,
        "fit_mean_pooled": fit_mean_pooled, "fit_mean_cs_avg": fit_mean_cs,
    }


# ---------------------------------------------------------------- E
def path_E_rg_running_lambda():
    """Per-regime, find optimal Lambda_t and Lambda_s such that
    median(R_time + Lambda_t - T_00) = 0 and median(R_diag + Lambda_s) = 0.
    Then check if Lambda_t(N), Lambda_s(N) run with N as a power law."""
    print()
    print("=" * 100)
    print("(E) RG-running Lambda_t(N) and Lambda_s(N): per-regime optimum")
    print("=" * 100)
    rows = []
    for reg, n_lat in ALL_REGIMES:
        p = find_d1_npz(reg, REPO)
        if p is None or not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        edge_arr = d["dense_cell_edge_xi_values"]
        amp_arr = d["dense_cell_node_amplitude_values"]
        phase_arr = d["dense_cell_node_phase_values"]
        n_seeds = min(edge_arr.shape[0], 32)
        g00s, t00s, g_diag_s, t_eigs_s = [], [], [], []
        for s in range(n_seeds):
            xi_mat = edge_to_matrix(edge_arr[s], n_lat)
            np.fill_diagonal(xi_mat, 1.0)
            psi = amp_arr[s] * np.exp(1j * phase_arr[s])
            k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
            q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
            prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
            res = per_node_eigendirection_residuals(prep, 0.0, 0.0)
            g00s.append(np.asarray(prep["g_00_h"]))
            t00s.append(np.asarray(prep["t00"]))
            # res["R_time"] is g_00 - t00 (since lambda=0)
            # res["R_diag"] is g_diag - t_eigvals
            t_eigs_s.append(res["T_eigvals"])
            g_diag_s.append(res["R_diag"] + res["T_eigvals"])
        g00 = np.concatenate(g00s); t00 = np.concatenate(t00s)
        g_diag = np.concatenate(g_diag_s, axis=0)  # (n_total, 3)
        t_eigs = np.concatenate(t_eigs_s, axis=0)
        # Optimal Lambda_t such that median(g00 + L - t00) = 0:  L = median(t00 - g00)
        opt_lt = float(np.median(t00 - g00))
        # Optimal Lambda_s such that median diag residual = 0
        diag_res = g_diag - t_eigs   # (n,3)
        opt_ls = float(np.median(t_eigs.flatten() - g_diag.flatten()))
        rows.append({"regime": reg, "N": n_lat,
                     "opt_lambda_t_med": opt_lt,
                     "opt_lambda_s_med": opt_ls})
    print(f"{'regime':<10} {'N':>3} | {'opt Lambda_t':>13} {'opt Lambda_s':>13}")
    print("-" * 50)
    for r in rows:
        print(f"{r['regime']:<10} {r['N']:>3} | {r['opt_lambda_t_med']:>+13.5f} {r['opt_lambda_s_med']:>+13.5f}")
    # RG fit: Lambda_t(N) = Lambda_t_inf + B/N^p
    N_arr = np.array([r["N"] for r in rows], dtype=float)
    lt_arr = np.array([r["opt_lambda_t_med"] for r in rows])
    ls_arr = np.array([r["opt_lambda_s_med"] for r in rows])
    # Symanzik form on Lambda_t directly
    f_lt = fit_symanzik_2_4(N_arr, lt_arr)
    f_ls = fit_symanzik_2_4(N_arr, ls_arr)
    print()
    print("Symanzik fit on Lambda_t(N):")
    print(f"  d_inf = {f_lt['d_inf']:.5f}, c_2 = {f_lt['c_2']:.2f}, R^2 = {f_lt['R_squared']:.3f}")
    print(f"  Compare to System-R Lambda_t = 0.81")
    print(f"Symanzik fit on Lambda_s(N):")
    print(f"  d_inf = {f_ls['d_inf']:.5f}, c_2 = {f_ls['c_2']:.2f}, R^2 = {f_ls['R_squared']:.3f}")
    print(f"  Compare to System-R Lambda_s = -0.005")
    return {
        "per_regime": rows,
        "lambda_t_running_fit": f_lt,
        "lambda_s_running_fit": f_ls,
    }


# ---------------------------------------------------------------- K
def path_K_delta_curv():
    """Read existing einstein_gap_9point_frobenius.json and check
    Symanzik fit on the Delta_curv observable."""
    print()
    print("=" * 100)
    print("(K) Delta_curv alternative observable")
    print("=" * 100)
    p = REPO / "data" / "einstein_gap_9point_frobenius.json"
    if not p.exists():
        print(f"  {p} not found")
        return {"verdict": "DATA_NOT_AVAILABLE"}
    d = json.load(open(p, "r"))
    candidates = d.get("residual_candidates", {}) if isinstance(d, dict) else {}
    # Find the Delta_curv-related candidates
    keys = [k for k in candidates if "curv" in k.lower()
            or "Delta_curv" in k]
    print(f"  Candidates found: {keys}")
    out = {}
    for k in keys:
        info = candidates[k]
        ns = info.get("ns", []) if isinstance(info, dict) else []
        vals = info.get("values", []) if isinstance(info, dict) else []
        if not ns or not vals: continue
        Ns = np.array(ns, dtype=float)
        ys = np.abs(np.array(vals, dtype=float))
        if len(Ns) < 4: continue
        f = fit_symanzik_2_4(Ns, ys)
        print(f"  {k}: n_pts={len(Ns)}, d_inf={f['d_inf']:.5f}, R^2={f['R_squared']:.3f}")
        out[k] = {"n_pts": int(len(Ns)),
                  "d_inf": f["d_inf"], "R_squared": f["R_squared"],
                  "values": [float(v) for v in ys]}
    return out


def main() -> int:
    out = {"method": "three_more_solution_paths", "schema_version": "1.0.0"}
    out["path_D_cross_seed"] = path_D_cross_seed_average()
    out["path_E_rg_lambda"] = path_E_rg_running_lambda()
    out["path_K_delta_curv"] = path_K_delta_curv()
    out_path = REPO / "outputs" / "three_more_paths_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
