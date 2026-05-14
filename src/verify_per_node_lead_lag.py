"""Per-node Granger-causality lead-lag test of the gravity-
nucleation working hypothesis (Paper 4, sec:limitations Outlook).

Consumes the snapshot NPZ produced by
``worldformula.experiments.run_d1_snapshot`` and tests the
temporal ordering at the heavy-tail residual nodes:

  T_00(a, t) leads Xi_loc(a, t) leads cluster_size(a, t).

Per-node Granger-causality test:
  for each top-decile-residual node a and each candidate lead-lag
  k in {1..max_lag}, compute the cross-correlation of T_00(a, t)
  with Xi_loc(a, t+k). Positive lag with significant correlation
  is evidence that the energy density LEADS the relational
  stiffening at that node, supporting the nucleation picture.

Output: outputs/per_node_lead_lag_audit.json
"""
from __future__ import annotations
import argparse
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

from verify_galerkin_runner_A_hessian_ricci import (
    XI_THRESH, per_seed_galerkin)
from verify_higher_order_terms_all8 import (
    LAMBDA_T, LAMBDA_S, per_node_residual)


def cross_correlation(x, y, max_lag=4):
    """For each lag k in [-max_lag, max_lag] compute corr(x[:T-k], y[k:T]).
    Positive lag -> y is shifted later, i.e. x at time t correlates
    with y at time t+k -> x leads y by k steps."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    T = len(x)
    out = {}
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            xs, ys = x[:T - k], y[k:]
        else:
            xs, ys = x[-k:], y[:T + k]
        if len(xs) < 3:
            out[k] = float("nan")
            continue
        xs = xs - xs.mean()
        ys = ys - ys.mean()
        denom = np.sqrt((xs * xs).sum() * (ys * ys).sum())
        out[k] = float((xs * ys).sum() / denom) if denom > 1e-12 else 0.0
    return out


def per_node_t00_xi_local(xi_mat, psi, k_field, q_field, n_lat):
    """Compute per-node T_00 and per-node local Xi-density at one
    snapshot.

    Local Xi-density at a is the mean Xi-edge-weight to its
    neighbours (weight-adjacency row sum / non-zero count)."""
    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
    t00 = np.asarray(prep["t00"])

    # Local Xi-density per node
    xi_off = np.where(np.isfinite(xi_mat), xi_mat, 0.0).copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(np.float64)
    deg = adj.sum(axis=1)
    weight_adj = xi_off * adj
    xi_loc = weight_adj.sum(axis=1) / np.maximum(deg, 1)
    return t00, xi_loc, prep


def per_node_residual_at_snapshot(prep, eye3=None):
    if eye3 is None:
        eye3 = prep["eye3"]
    return np.asarray(per_node_residual(
        prep["g_00_h"], prep["g_ij_h"],
        prep["t00"], prep["t_ij"],
        LAMBDA_T, LAMBDA_S, eye3, np))


def analyse(snapshot_path: Path, max_lag: int = 4):
    if not snapshot_path.exists():
        return {"error": f"snapshot file not found: {snapshot_path}"}
    d = np.load(snapshot_path, allow_pickle=True)
    if "edge_xi_snapshots" not in d.files:
        return {"error": "edge_xi_snapshots not in NPZ"}

    xi_snaps = d["edge_xi_snapshots"]   # (n_seeds, n_snap, n_lat, n_lat)
    psir_snaps = d["psi_real_snapshots"]
    psii_snaps = d["psi_imag_snapshots"]
    snapshot_steps = np.asarray(d["snapshot_steps"])
    n_seeds, n_snap, n_lat, _ = xi_snaps.shape

    print(f"Snapshot file: {snapshot_path}")
    print(f"  shape: {n_seeds} seeds, {n_snap} snapshots, N_lat={n_lat}")
    print(f"  snapshot steps: {snapshot_steps.tolist()}")

    # For each seed, compute per-node T_00(a,t), Xi_loc(a,t),
    # residual(a, t) per snapshot.
    seed_results = []
    for s in range(n_seeds):
        print(f"\n--- seed {s} ---", flush=True)
        t00_ts = np.zeros((n_snap, n_lat))
        xi_loc_ts = np.zeros((n_snap, n_lat))
        residual_ts = np.zeros((n_snap, n_lat))

        for t_idx in range(n_snap):
            xi_mat = np.asarray(xi_snaps[s, t_idx])
            psi = psir_snaps[s, t_idx] + 1j * psii_snaps[s, t_idx]
            # K, Q fields are not snapshotted; use defaults
            k_field = np.full((n_lat, n_lat), 0.55)
            q_field = np.full((n_lat, n_lat), 0.45)
            t00, xi_loc, prep = per_node_t00_xi_local(
                xi_mat, psi, k_field, q_field, n_lat)
            t00_ts[t_idx] = t00
            xi_loc_ts[t_idx] = xi_loc
            residual_ts[t_idx] = per_node_residual_at_snapshot(prep)

        # Identify nucleation candidate nodes from FINAL snapshot:
        # top-decile residual at the last available time.
        final_residual = residual_ts[-1]
        p90 = np.percentile(final_residual, 90)
        nucleation_candidates = np.where(final_residual >= p90)[0]
        non_candidates = np.where(final_residual < p90)[0]

        # Per-node lead-lag: T_00 vs Xi_loc with positive lag = T_00 leads
        candidate_lags = []
        for a in nucleation_candidates:
            cc = cross_correlation(
                t00_ts[:, a], xi_loc_ts[:, a], max_lag=max_lag)
            best_k = max(cc, key=lambda k: cc[k])
            candidate_lags.append({
                "node": int(a),
                "best_lag": int(best_k),
                "corr_at_best_lag": cc[best_k],
                "corr_at_lag_0": cc[0],
                "corr_at_lag_+1": cc.get(1, float("nan")),
                "corr_at_lag_-1": cc.get(-1, float("nan")),
            })
        non_lags = []
        for a in non_candidates[: len(nucleation_candidates) * 5]:
            cc = cross_correlation(
                t00_ts[:, a], xi_loc_ts[:, a], max_lag=max_lag)
            best_k = max(cc, key=lambda k: cc[k])
            non_lags.append({
                "node": int(a),
                "best_lag": int(best_k),
                "corr_at_best_lag": cc[best_k],
            })

        # Aggregate
        cand_best_lag_mean = float(np.mean(
            [r["best_lag"] for r in candidate_lags])) if candidate_lags else float("nan")
        cand_pos_lag_fraction = float(np.mean(
            [r["best_lag"] > 0 for r in candidate_lags])) if candidate_lags else float("nan")
        non_best_lag_mean = float(np.mean(
            [r["best_lag"] for r in non_lags])) if non_lags else float("nan")

        print(f"  nucleation candidates (top10 res): {len(nucleation_candidates)}")
        print(f"    mean best lag: {cand_best_lag_mean:+.2f} steps")
        print(f"    fraction with positive lag (T_00 leads): "
              f"{cand_pos_lag_fraction:.2%}")
        print(f"  non-candidate sample mean best lag: {non_best_lag_mean:+.2f}")

        seed_results.append({
            "seed": s,
            "n_nucleation_candidates": int(len(nucleation_candidates)),
            "candidate_best_lag_mean": cand_best_lag_mean,
            "candidate_positive_lag_fraction": cand_pos_lag_fraction,
            "non_candidate_best_lag_mean": non_best_lag_mean,
            "candidate_per_node": candidate_lags,
        })

    # Aggregate across seeds
    if seed_results:
        avg_cand_lag = float(np.mean(
            [r["candidate_best_lag_mean"] for r in seed_results
             if not np.isnan(r["candidate_best_lag_mean"])]))
        avg_pos_frac = float(np.mean(
            [r["candidate_positive_lag_fraction"] for r in seed_results
             if not np.isnan(r["candidate_positive_lag_fraction"])]))
        avg_non_lag = float(np.mean(
            [r["non_candidate_best_lag_mean"] for r in seed_results
             if not np.isnan(r["non_candidate_best_lag_mean"])]))
    else:
        avg_cand_lag = avg_pos_frac = avg_non_lag = float("nan")

    print()
    print("=" * 70)
    print("VERDICT (across seeds):")
    print("=" * 70)
    print(f"  candidate (top-decile-residual) mean best lag: "
          f"{avg_cand_lag:+.2f} steps")
    print(f"  candidate fraction with T_00-leads-Xi (lag > 0): "
          f"{avg_pos_frac:.2%}")
    print(f"  non-candidate mean best lag: {avg_non_lag:+.2f}")
    print()
    print("  Nucleation hypothesis support:")
    if avg_cand_lag > 0.5 and avg_pos_frac > 0.55:
        print("    POSITIVE: candidates show systematic T_00-leads-Xi pattern.")
    elif avg_cand_lag > 0 and avg_pos_frac > 0.5:
        print("    WEAK: marginal positive lag signal at candidates.")
    elif abs(avg_cand_lag) < 0.5:
        print("    NULL: no temporal lead-lag signal distinguishable.")
    else:
        print("    NEGATIVE: candidates show Xi-leads-T_00 (opposite of "
              "nucleation prediction).")

    return {
        "snapshot_file": str(snapshot_path),
        "n_seeds": int(n_seeds),
        "n_snapshots": int(n_snap),
        "n_lat": int(n_lat),
        "max_lag": int(max_lag),
        "snapshot_steps": snapshot_steps.tolist(),
        "per_seed": seed_results,
        "aggregate": {
            "candidate_mean_best_lag": avg_cand_lag,
            "candidate_positive_lag_fraction": avg_pos_frac,
            "non_candidate_mean_best_lag": avg_non_lag,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-file", type=str, required=True,
        help="Path to *.snapshots.npz produced by run_d1_snapshot")
    parser.add_argument("--max-lag", type=int, default=4)
    parser.add_argument(
        "--output", type=str,
        default=str(REPO / "outputs" / "per_node_lead_lag_audit.json"))
    args = parser.parse_args()

    rec = analyse(Path(args.snapshot_file), max_lag=args.max_lag)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, default=str)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
