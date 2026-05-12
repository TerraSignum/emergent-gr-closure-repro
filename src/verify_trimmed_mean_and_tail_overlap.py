"""Three statistical tests to harden the matter-core-residual claim:

1. TRIMMED MEAN: per-direction relative residual Delta(a),
   drop the top-10% nodes by Delta and recompute mean.
   If trimmed_mean(Delta) <= 0.05 at all N, the heavy-tail
   IS the closure obstruction; the bulk closes cleanly.

2. TAIL-vs-T_00 OVERLAP: of the top-10% Delta(a) nodes,
   what fraction is also in the top-10% T_00(a) nodes?
   If overlap >= 0.5, the residual tail IS the matter core
   (geometric-condensation cluster identified separately).
   Quantify with Spearman rank correlation between Delta(a)
   and T_00(a).

3. N=128 (or whatever lattice run is next available):
   does median pass <= 0.05 AND mean pass <= 0.10?
   If yes -> claim is empirically extended.

Output: outputs/trimmed_mean_and_tail_overlap_audit.json
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
    ("P5N128", 128),  # runs only if NPZ exists
]
LAMBDA_T = 0.81
LAMBDA_S = -0.005
TAIL_FRAC = 0.10  # top 10%


def per_node_delta_and_t00(prep):
    res = per_node_eigendirection_residuals(prep, LAMBDA_T, LAMBDA_S)
    R_time = res["R_time"]
    R_diag = res["R_diag"]
    R_off = res["R_off"]
    t_eigs = res["T_eigvals"]
    t00 = np.asarray(prep["t00"])
    R_norm = np.sqrt(R_time ** 2 + (R_diag ** 2).sum(axis=1) + R_off ** 2)
    T_norm = np.sqrt(t00 ** 2 + (t_eigs ** 2).sum(axis=1))
    delta = R_norm / np.maximum(T_norm, 1e-12)
    return delta, np.abs(t00)


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


def spearman_corr(x, y):
    """Spearman rank correlation."""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    print("=" * 110)
    print("Trimmed-mean + tail-overlap audit (matter-core residual hypothesis)")
    print("=" * 110)
    print()
    header = (f"{'reg':<8} {'N':>3} | "
              f"{'med':>7} {'mean':>7} {'mean_t':>7} | "
              f"{'med t10':>8} {'mean t10':>9} | "
              f"{'overlap':>8} {'spearman':>9}")
    print(header)
    print("-" * len(header))

    rows = []
    for reg, n_lat in REGIMES:
        gt = gather(reg, n_lat)
        if gt is None:
            continue
        delta, t00 = gt

        # Sort by delta descending
        order_d = np.argsort(-delta)
        n_total = len(delta)
        n_tail = max(1, int(n_total * TAIL_FRAC))
        tail_delta_idx = set(order_d[:n_tail].tolist())

        # Sort by t00 descending
        order_t = np.argsort(-t00)
        tail_t00_idx = set(order_t[:n_tail].tolist())

        # Trimmed: drop top n_tail
        trimmed_mask = np.ones(n_total, dtype=bool)
        trimmed_mask[order_d[:n_tail]] = False
        delta_trim = delta[trimmed_mask]

        # Top-tail-T_00 stats
        delta_top_t00 = delta[order_t[:n_tail]]

        # Overlap
        overlap = len(tail_delta_idx & tail_t00_idx) / max(n_tail, 1)
        sp = spearman_corr(delta, t00)

        med = float(np.median(delta))
        mean = float(np.mean(delta))
        med_trim = float(np.median(delta_trim))
        mean_trim = float(np.mean(delta_trim))
        mean_top_t00 = float(np.mean(delta_top_t00))

        rows.append({
            "regime": reg, "N": n_lat,
            "n_nodes_total": int(n_total),
            "delta_median": med,
            "delta_mean": mean,
            "delta_mean_top_t00_decile": mean_top_t00,
            "delta_median_trimmed_top10": med_trim,
            "delta_mean_trimmed_top10": mean_trim,
            "tail_overlap_delta_with_T00": overlap,
            "spearman_delta_T00": sp,
            "median_pass_0p05": med <= 0.05,
            "mean_pass_0p10": mean <= 0.10,
            "median_trim_pass_0p05": med_trim <= 0.05,
            "mean_trim_pass_0p10": mean_trim <= 0.10,
        })
        print(f"{reg:<8} {n_lat:>3} | "
              f"{med:>7.4f} {mean:>7.4f} {mean_top_t00:>7.4f} | "
              f"{med_trim:>8.4f} {mean_trim:>9.4f} | "
              f"{overlap:>8.3f} {sp:>+9.4f}")

    # Aggregate verdicts
    print()
    print("Verdicts:")
    if all(r["median_trim_pass_0p05"] and r["mean_trim_pass_0p10"] for r in rows):
        verdict_trim = "TRIMMED_PASS_AT_ALL_N"
    elif all(r["median_trim_pass_0p05"] for r in rows):
        verdict_trim = "TRIMMED_MEDIAN_PASS_MEAN_PARTIAL"
    else:
        verdict_trim = "TRIMMED_PASS_PARTIAL"

    overlaps = [r["tail_overlap_delta_with_T00"] for r in rows]
    sps = [r["spearman_delta_T00"] for r in rows]
    overlap_mean = float(np.mean(overlaps))
    sp_mean = float(np.mean(sps))
    print(f"  Trimmed (drop top 10%): {verdict_trim}")
    print(f"  Mean tail overlap with top-T_00 decile: {overlap_mean:.3f}")
    print(f"  Mean Spearman(Delta, T_00): {sp_mean:+.3f}")

    if overlap_mean >= 0.5 and sp_mean >= 0.3:
        verdict_tail = "TAIL_IS_MATTER_CORE"
    elif overlap_mean >= 0.3:
        verdict_tail = "TAIL_PARTIALLY_OVERLAPS_MATTER_CORE"
    else:
        verdict_tail = "TAIL_NOT_DRIVEN_BY_T_00"
    print(f"  Tail-source verdict: {verdict_tail}")

    # N=128 specific verdict
    n128 = next((r for r in rows if r["N"] == 128), None)
    if n128:
        n128_pass = n128["median_pass_0p05"] and n128["mean_pass_0p10"]
        print(f"  N=128 closure (median<=0.05 AND mean<=0.10): "
              f"{'PASS' if n128_pass else 'FAIL'}  "
              f"(med={n128['delta_median']:.4f}, mean={n128['delta_mean']:.4f})")
    else:
        n128_pass = None
        print("  N=128: NOT YET RUN (lattice data missing)")

    out = {
        "method": "trimmed_mean_and_tail_overlap_audit",
        "schema_version": "1.0.0",
        "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
        "tail_fraction": TAIL_FRAC,
        "per_regime": rows,
        "trimmed_verdict": verdict_trim,
        "tail_source_verdict": verdict_tail,
        "mean_overlap_top_decile": overlap_mean,
        "mean_spearman_delta_T00": sp_mean,
        "N128_passes": n128_pass,
    }
    out_path = REPO / "outputs" / "trimmed_mean_and_tail_overlap_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
