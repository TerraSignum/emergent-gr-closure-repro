"""Test scheme C: tensorial T_00 correction.

Modify the matter stress-energy at the tensor level:

  T_00^core(a) = T_00(a) + C * log(T_00(a)/<T_00>)

then recompute the per-direction RELATIVE Frobenius residual with
the corrected T_00 in place of the raw T_00. Sweep C in a grid
to find the value that minimises the raw mean residual at all N>=60.

If the optimal C is approximately the same across all regimes,
the core correction is a fixed structural law (one parameter).

Output: outputs/T00_core_correction_audit.json
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
LAMBDA_T = 0.81
LAMBDA_S = -0.005
C_VALUES = [-1.5, -1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0, 1.5]


def per_node_delta_with_correction(prep, C_correction):
    """Recompute per-direction relative residual with T_00 corrected."""
    t00_raw = np.asarray(prep["t00"]).copy()
    t00_mean = float(np.mean(t00_raw))
    log_ratio = np.log(np.maximum(t00_raw / max(t00_mean, 1e-12), 1e-10))
    t00_corrected = t00_raw + C_correction * log_ratio

    # Build a modified prep with t00 replaced
    prep_mod = dict(prep)
    prep_mod["t00"] = t00_corrected

    res = per_node_eigendirection_residuals(prep_mod, LAMBDA_T, LAMBDA_S)
    R_t = res["R_time"]; R_d = res["R_diag"]; R_o = res["R_off"]
    t_e = res["T_eigvals"]
    R = np.sqrt(R_t ** 2 + (R_d ** 2).sum(axis=1) + R_o ** 2)
    T = np.sqrt(t00_corrected ** 2 + (t_e ** 2).sum(axis=1))
    return R / np.maximum(T, 1e-12)


def gather_preps(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)
    preps = []
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        preps.append(prep)
    return preps


def main() -> int:
    print("=" * 110)
    print("T_00 core-correction sweep (scheme C tensorial)")
    print("T_00^core(a) = T_00(a) + C * log(T_00(a)/<T_00>)")
    print("=" * 110)
    print()

    # Pre-load all preps
    regime_preps = []
    for reg, n_lat in REGIMES:
        ps = gather_preps(reg, n_lat)
        if ps is None:
            continue
        regime_preps.append({"regime": reg, "N": n_lat, "preps": ps})

    print(f"{'C':>6} | " + " | ".join([
        f"{rp['regime']:<6} med/mean" for rp in regime_preps
    ]))
    print("-" * (12 + 17 * len(regime_preps)))

    sweep_results = {}
    for C in C_VALUES:
        per_reg = []
        line = f"{C:>+6.2f} |"
        for rp in regime_preps:
            deltas = []
            for prep in rp["preps"]:
                d = per_node_delta_with_correction(prep, C)
                deltas.append(d)
            d_all = np.concatenate(deltas)
            med = float(np.median(d_all))
            mean = float(np.mean(d_all))
            per_reg.append({"regime": rp["regime"], "N": rp["N"],
                            "median": med, "mean": mean})
            line += f" {med:.3f}/{mean:.3f} |"
        sweep_results[f"C={C}"] = {"C": C, "per_regime": per_reg}
        print(line)

    # Identify best C by smallest pooled mean across N>=60
    best_C, best_mean_avg = None, float("inf")
    for tag, info in sweep_results.items():
        ms = [r["mean"] for r in info["per_regime"] if r["N"] >= 60]
        if not ms:
            continue
        avg = float(np.mean(ms))
        if avg < best_mean_avg:
            best_mean_avg = avg
            best_C = info["C"]

    print()
    print(f"BEST C (smallest mean averaged over N>=60): C = {best_C:+.4f}")
    print(f"  averaged mean delta (N>=60) = {best_mean_avg:.4f}")

    # Per-regime closure verdict at best C
    info_best = sweep_results[f"C={best_C}"]
    n_pass_strict = 0
    n_pass_relaxed = 0
    n_total = 0
    for r in info_best["per_regime"]:
        if r["N"] < 60:
            continue
        n_total += 1
        if r["median"] <= 0.05 and r["mean"] <= 0.05:
            n_pass_strict += 1
        if r["median"] <= 0.05 and r["mean"] <= 0.10:
            n_pass_relaxed += 1
    print(f"  STRICT (med<=0.05 AND mean<=0.05): {n_pass_strict}/{n_total} regimes pass")
    print(f"  RELAXED (med<=0.05 AND mean<=0.10): {n_pass_relaxed}/{n_total} regimes pass")

    if n_pass_strict == n_total:
        verdict = "CORE_CORRECTED_FULL_TENSOR_CLOSURE_HOLDS_STRICT"
    elif n_pass_relaxed == n_total:
        verdict = "CORE_CORRECTED_FULL_TENSOR_CLOSURE_HOLDS_RELAXED"
    else:
        verdict = "CORE_CORRECTED_PARTIAL_PASS"
    print(f"VERDICT: {verdict}")

    out = {
        "method": "T_00_core_correction_sweep_scheme_C",
        "schema_version": "1.0.0",
        "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
        "correction_form": "T_00^core(a) = T_00(a) + C * log(T_00(a)/<T_00>)",
        "C_values_swept": C_VALUES,
        "per_C": sweep_results,
        "best_C": best_C,
        "best_avg_mean_N_geq_60": best_mean_avg,
        "verdict": verdict,
    }
    out_path = REPO / "outputs" / "T00_core_correction_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
