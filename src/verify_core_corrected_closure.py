"""Test if subtracting the empirical universal tail law

  Delta_pred(a) = max(0, 0.234 + 1.304 * log(T_00(a)/<T_00>))

from the raw per-direction relative residual brings the
RAW MEAN below the closure threshold (0.05 / 0.10) at all N>=60.

Three correction schemes compared:

  (A) Subtract predicted Delta from ALL nodes (clipped >=0)
  (B) Subtract only from top-10% tail nodes (matter-core nodes)
  (C) Subtract only when log(T_00/<T_00>) > log(2) (high-density only)

If scheme (A) or (B) drives raw mean <= 0.05, the matter-core
residual is FULLY ABSORBED by the universal density-contrast law.
This converts the "raw-mean tail problem" into a closed-form
correction at the lattice level.

Output: outputs/core_corrected_closure_audit.json
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

# Universal density-contrast law from
# outputs/tail_universal_simplified_audit.json (excluding P5_N50)
A0 = 0.234
A1 = 1.304


def per_node_delta_and_t00(prep):
    res = per_node_eigendirection_residuals(prep, LAMBDA_T, LAMBDA_S)
    R_t = res["R_time"]; R_d = res["R_diag"]; R_o = res["R_off"]
    t_e = res["T_eigvals"]; t00 = np.asarray(prep["t00"])
    R = np.sqrt(R_t ** 2 + (R_d ** 2).sum(axis=1) + R_o ** 2)
    T = np.sqrt(t00 ** 2 + (t_e ** 2).sum(axis=1))
    return R / np.maximum(T, 1e-12), np.abs(t00)


def gather(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)
    deltas, t00s = [], []
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        delta, t00 = per_node_delta_and_t00(prep)
        deltas.append(delta)
        t00s.append(t00)
    return np.concatenate(deltas), np.concatenate(t00s)


def universal_law(t00, t00_mean):
    """Δ_pred = max(0, A0 + A1 * log(T_00/<T_00>))"""
    log_ratio = np.log(np.maximum(t00 / max(t00_mean, 1e-12), 1e-10))
    pred = A0 + A1 * log_ratio
    return np.maximum(pred, 0.0)


def main() -> int:
    print("=" * 110)
    print("Core-corrected closure test")
    print(f"Universal law: Delta_pred = max(0, {A0:+.4f} + {A1:+.4f}*log(T_00/<T_00>))")
    print("Threshold: median <= 0.05 AND mean <= 0.10  (closure)")
    print("            mean <= 0.05 -> strict closure")
    print("=" * 110)
    print()
    print(f"{'reg':<8} {'N':>3} | "
          f"{'med raw':>8} {'mean raw':>9} | "
          f"{'med (A)':>8} {'mean (A)':>9} | "
          f"{'med (B)':>8} {'mean (B)':>9} | "
          f"{'med (C)':>8} {'mean (C)':>9}")
    print("-" * 110)

    rows = []
    for reg, n_lat in REGIMES:
        gt = gather(reg, n_lat)
        if gt is None:
            continue
        delta, t00 = gt
        t00_mean = float(np.mean(t00))
        delta_pred = universal_law(t00, t00_mean)

        # Scheme (A): subtract from all nodes (residual after correction)
        delta_A = np.abs(delta - delta_pred)
        # Scheme (B): only subtract from top-10% tail
        n_tail = max(1, int(len(delta) * 0.10))
        order = np.argsort(-delta)
        tail_mask = np.zeros(len(delta), dtype=bool)
        tail_mask[order[:n_tail]] = True
        delta_B = delta.copy()
        delta_B[tail_mask] = np.abs(delta[tail_mask] - delta_pred[tail_mask])
        # Scheme (C): only subtract where log(T/<T>) > log(2) (high density)
        log_r = np.log(np.maximum(t00 / max(t00_mean, 1e-12), 1e-10))
        high_mask = log_r > np.log(2.0)
        delta_C = delta.copy()
        delta_C[high_mask] = np.abs(delta[high_mask] - delta_pred[high_mask])

        med_raw = float(np.median(delta))
        mean_raw = float(np.mean(delta))
        med_A = float(np.median(delta_A)); mean_A = float(np.mean(delta_A))
        med_B = float(np.median(delta_B)); mean_B = float(np.mean(delta_B))
        med_C = float(np.median(delta_C)); mean_C = float(np.mean(delta_C))

        rows.append({
            "regime": reg, "N": n_lat,
            "n_nodes": int(len(delta)),
            "n_high_density": int(high_mask.sum()),
            "raw":         {"median": med_raw, "mean": mean_raw},
            "scheme_A_subtract_all": {"median": med_A, "mean": mean_A},
            "scheme_B_subtract_tail_only": {"median": med_B, "mean": mean_B},
            "scheme_C_subtract_high_density": {"median": med_C, "mean": mean_C},
        })
        print(f"{reg:<8} {n_lat:>3} | "
              f"{med_raw:>8.4f} {mean_raw:>9.4f} | "
              f"{med_A:>8.4f} {mean_A:>9.4f} | "
              f"{med_B:>8.4f} {mean_B:>9.4f} | "
              f"{med_C:>8.4f} {mean_C:>9.4f}")

    # Verdicts
    print()
    print("Closure verdicts (median <= 0.05 AND mean <= 0.05 -> STRICT):")
    schemes = ["raw", "scheme_A_subtract_all",
               "scheme_B_subtract_tail_only", "scheme_C_subtract_high_density"]
    for sch in schemes:
        all_pass_strict = all(
            r[sch]["median"] <= 0.05 and r[sch]["mean"] <= 0.05
            for r in rows if r["N"] >= 60
        )
        all_pass_relaxed = all(
            r[sch]["median"] <= 0.05 and r[sch]["mean"] <= 0.10
            for r in rows if r["N"] >= 60
        )
        print(f"  {sch:<35} N>=60: STRICT={'PASS' if all_pass_strict else 'FAIL':<4}  "
              f"RELAXED={'PASS' if all_pass_relaxed else 'FAIL'}")

    # Identify best scheme (smallest mean across N>=60)
    means_by_scheme = {sch: float(np.mean([r[sch]["mean"] for r in rows if r["N"] >= 60]))
                       for sch in schemes}
    best = min(means_by_scheme.items(), key=lambda kv: kv[1])
    print()
    print(f"BEST scheme by smallest mean across N>=60: {best[0]} (mean={best[1]:.4f})")

    out_path = REPO / "outputs" / "core_corrected_closure_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "core_corrected_closure_test",
            "schema_version": "1.0.0",
            "universal_law": {
                "form": "Delta_pred = max(0, A0 + A1*log(T_00/<T_00>))",
                "A0": A0, "A1": A1,
            },
            "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
            "per_regime": rows,
            "best_scheme_by_mean": best[0],
            "best_scheme_mean": best[1],
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
