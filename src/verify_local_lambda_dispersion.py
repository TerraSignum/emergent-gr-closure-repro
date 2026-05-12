"""Local Lambda dispersion test for asymptotic-exact closure.

Per node a, the equation
  G_munu(a) + Lambda_munu - 8 pi G T_munu(a) = 0
is exactly closeable by a *node-local* Lambda_munu(a):
  Lambda_munu(a) := T_munu(a) - G_munu(a)
where we use 8 pi G = 1.

This is a tautology IF Lambda is allowed to vary freely per node.
The interesting question is whether the per-node Lambda_munu(a)
distribution is asymptotically PEAKED at a single uniform value:

  - If sigma(Lambda_munu(a)) -> 0 with N: the closure is asymptotically
    exact with a single global Lambda; the spread is a finite-N effect.

  - If sigma(Lambda_munu(a)) -> const > 0: closure has irreducible
    spatial-running content that no isotropic Lambda can absorb.

We measure both the SCALAR dispersion sigma(Lambda_t(a)) (time-time)
and the TENSOR dispersion sigma(Lambda_s(a)) (spatial trace) and
the FROBENIUS dispersion sigma_F = sqrt(<||Lambda_munu - <Lambda_munu>||^2>).

A power-law sigma(N) ~ N^(-alpha) with alpha > 0, R^2 >= 0.7, and
sigma(N=100) <= 0.05 establishes asymptotic-exact closure:

  Lambda_munu^lattice(N -> infty) -> Lambda_munu^uniform
  with Lambda_t^uniform = mean_a Lambda_t(a) -> 0.81 (System-R)
       Lambda_s^uniform = mean_a Lambda_s(a) -> -0.005 (System-R)

Output: outputs/local_lambda_dispersion_audit.json
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


REGIMES = [
    ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]
LAMBDA_T_STRUCT = 0.81
LAMBDA_S_STRUCT = -0.005


def per_node_local_lambdas(prep):
    """Compute per-node Lambda_t(a), Lambda_s(a) such that the
    diagonal residual is identically zero per node:

      Lambda_t(a)  := T_00(a) - G_00(a)
      Lambda_s(a)  := mean_i (lambda_i(a) - G_(ii)(a))   spatial trace
      Lambda_TF(a) := lambda_i(a) - G_(ii)(a) - Lambda_s(a)  (3-vector)
    """
    res = per_node_eigendirection_residuals(prep, lambda_t=0.0, lambda_s=0.0)
    # residuals at lambda=0:
    #   R_time = G_00 - T_00
    #   R_diag = G_(ii) - lambda_i      (3-vec)
    # so node-local Lambda values:
    #   Lambda_t(a) = T_00 - G_00 = -R_time
    #   Lambda_s(a) = mean_i (lambda_i - G_(ii)) = -mean(R_diag)
    #   Lambda_TF_i(a) = (lambda_i - G_(ii)) - Lambda_s(a)
    lam_t_a = -res["R_time"]
    lam_per_dir = -res["R_diag"]                   # (n,3)
    lam_s_a = lam_per_dir.mean(axis=1)             # (n,)
    lam_TF_per_dir = lam_per_dir - lam_s_a[:, None]  # (n,3)
    lam_TF_norm = np.sqrt((lam_TF_per_dir ** 2).sum(axis=1))
    return {
        "lambda_t": lam_t_a,
        "lambda_s": lam_s_a,
        "lambda_TF_norm": lam_TF_norm,
        "R_off": res["R_off"],            # off-diagonal cannot be absorbed by Lambda
    }


def gather_regime(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)
    pool = {"lambda_t": [], "lambda_s": [], "lambda_TF_norm": [], "R_off": []}
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        loc = per_node_local_lambdas(prep)
        for k in pool:
            pool[k].append(loc[k])
    return {k: np.concatenate(v) for k, v in pool.items()}


def power_law(N, y):
    if np.any(y <= 0) or len(N) < 3:
        return float("nan"), float("nan")
    log_N, log_y = np.log(N), np.log(y)
    slope, intercept = np.polyfit(log_N, log_y, 1)
    pred = slope * log_N + intercept
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(-slope), float(r2)


def main() -> int:
    print("=" * 110)
    print("Local Lambda dispersion test: does the per-node Lambda_munu(a) distribution")
    print("collapse to a single uniform value as N -> infty? If yes, closure is exact.")
    print("=" * 110)
    print()
    print(f"{'reg':<8} {'N':>3} | "
          f"{'<Lam_t>':>9} {'sigma_Lam_t':>12} | "
          f"{'<Lam_s>':>9} {'sigma_Lam_s':>12} | "
          f"{'<|Lam_TF|>':>11} {'sigma_TF':>10} | "
          f"{'<|R_off|>':>10}")
    print("-" * 110)

    rows = []
    for reg, n_lat in REGIMES:
        d = gather_regime(reg, n_lat)
        if d is None:
            continue
        mean_t = float(d["lambda_t"].mean())
        sig_t = float(d["lambda_t"].std())
        mean_s = float(d["lambda_s"].mean())
        sig_s = float(d["lambda_s"].std())
        mean_TF = float(d["lambda_TF_norm"].mean())
        sig_TF = float(d["lambda_TF_norm"].std())
        mean_R_off = float(d["R_off"].mean())
        rows.append({
            "regime": reg, "N": n_lat,
            "mean_lambda_t": mean_t, "sigma_lambda_t": sig_t,
            "mean_lambda_s": mean_s, "sigma_lambda_s": sig_s,
            "mean_lambda_TF_norm": mean_TF, "sigma_lambda_TF_norm": sig_TF,
            "mean_R_off": mean_R_off,
        })
        print(f"{reg:<8} {n_lat:>3} | "
              f"{mean_t:>+9.4f} {sig_t:>12.4f} | "
              f"{mean_s:>+9.4f} {sig_s:>12.4f} | "
              f"{mean_TF:>11.4f} {sig_TF:>10.4f} | "
              f"{mean_R_off:>10.4f}")

    # Power-law fits on dispersions
    N_arr = np.array([r["N"] for r in rows], dtype=float)
    fits = {}
    for key in ("sigma_lambda_t", "sigma_lambda_s",
                "sigma_lambda_TF_norm", "mean_R_off"):
        vals = np.array([r[key] for r in rows])
        a, r2 = power_law(N_arr, vals)
        fits[key] = {"alpha": a, "r_squared": r2}

    print()
    print("Power-law fits  sigma(N) ~ N^(-alpha):  (alpha > 0 means dispersion shrinks)")
    print(f"{'observable':<28} {'alpha':>10} {'R^2':>8}")
    print("-" * 50)
    for k, f in fits.items():
        print(f"{k:<28} {f['alpha']:>+10.3f} {f['r_squared']:>8.3f}")

    # Asymptotic mean values
    mean_lam_t_avg = float(np.mean([r["mean_lambda_t"] for r in rows]))
    mean_lam_s_avg = float(np.mean([r["mean_lambda_s"] for r in rows]))
    print()
    print("Cross-N mean per-node Lambdas:")
    print(f"  <Lambda_t> across N = {mean_lam_t_avg:+.5f}  (System-R: 0.81000)")
    print(f"  <Lambda_s> across N = {mean_lam_s_avg:+.5f}  (System-R: -0.00500)")
    print()

    # Asymptotic exact closure test:
    # sigma_lambda_t(N=100) <= 0.05, alpha > 0, R^2 >= 0.7
    sig_t_at_100 = next((r["sigma_lambda_t"] for r in rows if r["N"] == 100),
                          float("inf"))
    sig_s_at_100 = next((r["sigma_lambda_s"] for r in rows if r["N"] == 100),
                          float("inf"))
    sig_TF_at_100 = next((r["sigma_lambda_TF_norm"] for r in rows if r["N"] == 100),
                          float("inf"))
    asymp_lam_t = (fits["sigma_lambda_t"]["alpha"] > 0
                  and fits["sigma_lambda_t"]["r_squared"] >= 0.7
                  and sig_t_at_100 <= 0.05)
    asymp_lam_s = (fits["sigma_lambda_s"]["alpha"] > 0
                  and fits["sigma_lambda_s"]["r_squared"] >= 0.7
                  and sig_s_at_100 <= 0.05)
    asymp_TF = (fits["sigma_lambda_TF_norm"]["alpha"] > 0
                and fits["sigma_lambda_TF_norm"]["r_squared"] >= 0.7
                and sig_TF_at_100 <= 0.05)

    print(f"Asymptotic-exact closure tests at N=100:")
    print(f"  sigma(Lambda_t)  = {sig_t_at_100:.4f} (target <=0.05): "
          f"{'PASS' if asymp_lam_t else 'FAIL'}")
    print(f"  sigma(Lambda_s)  = {sig_s_at_100:.4f} (target <=0.05): "
          f"{'PASS' if asymp_lam_s else 'FAIL'}")
    print(f"  sigma(|Lambda_TF|) = {sig_TF_at_100:.4f} (target <=0.05): "
          f"{'PASS' if asymp_TF else 'FAIL'}")

    overall_asymp_exact = asymp_lam_t and asymp_lam_s and asymp_TF

    print()
    if overall_asymp_exact:
        print("VERDICT: ASYMPTOTIC_EXACT_CLOSURE_HOLDS")
    else:
        print("VERDICT: ASYMPTOTIC_EXACT_CLOSURE_FAILS")
        print("         per-node Lambda dispersion does NOT collapse to a single")
        print("         uniform value; spatial-running Lambda is structural, not noise.")

    out_path = REPO / "outputs" / "local_lambda_dispersion_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "per_node_local_lambda_dispersion_test",
            "schema_version": "1.0.0",
            "lambda_t_struct": LAMBDA_T_STRUCT,
            "lambda_s_struct": LAMBDA_S_STRUCT,
            "per_regime": rows,
            "power_law_fits": fits,
            "cross_N_mean_lambda_t": mean_lam_t_avg,
            "cross_N_mean_lambda_s": mean_lam_s_avg,
            "asymptotic_exact_closure": overall_asymp_exact,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
