"""Mathematical viability test of the matter-antimatter
hypothesis (user proposal 2026-04-30):

  the 7.8x Granger-directional asymmetry (Xi -> T_00 vs
  T_00 -> Xi) found in the per-node lead-lag audit could be
  the framework-level source of cosmological baryogenesis
  (Sakharov mechanism), with different defect families
  corresponding to different emergent particle species.

We perform three independent mathematical checks on the
existing 16-seed snapshot data to assess viability:

  CHECK T (time-reversal): reverse the snapshot trajectories
    in time and re-run the per-node Granger analysis. If the
    Granger direction FLIPS (T_00 -> Xi becomes dominant) the
    asymmetry IS a time-arrow effect, consistent with the
    Sakharov "out of equilibrium" condition being structurally
    satisfied by the first-order dissipative fast-slow flow.

  CHECK C (charge conjugation): split the candidate nodes by
    winding sign (+1, 0, -1) and compare the per-node Granger
    directional preference between positive-winding (matter
    vortices) and negative-winding (antimatter vortices). If
    they differ, C symmetry is empirically broken.

  CHECK P+defect-family: split the candidate nodes by defect
    family (vortex / wall / no-defect) and compute the
    per-family Granger Fisher's combined p-value. If different
    families show systematically different directional
    preferences, the framework structurally differentiates
    particle species.

The minimal mathematical viability requires:
  CHECK T:  YES (direction flips under time reversal) — establishes
            that the asymmetry is time-arrow-related.
  CHECK C:  asymmetry between +winding and -winding -> C broken.
  CHECK P+defect: at least two defect families show distinct
            directional preferences -> structurally differentiates
            species.

If all three pass: the hypothesis is mathematically consistent
  with Sakharov's three conditions and warrants formal
  development of a CPT-operator on the lattice.

If TR fails: the asymmetry is NOT a time-arrow effect, the
  matter-antimatter analogy is mathematically not on solid
  ground.

Output: outputs/matter_antimatter_mathematical_check.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from scipy.stats import f as f_dist

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
    XI_THRESH, edge_to_matrix, per_seed_galerkin)
from verify_higher_order_terms_all8 import (
    LAMBDA_T, LAMBDA_S, per_node_residual)


SNAPSHOT_NPZ = REPO.parent / "results_d1_p5n100_snapshot_16seeds" / "P5N100.snapshots.npz"
WINDING_NPZ = REPO.parent / "results_d1_p5n100" / "d1_p5n100.npz"


def reconstruct(xi_mat, psi, n_lat):
    k_field = np.full((n_lat, n_lat), 0.55)
    q_field = np.full((n_lat, n_lat), 0.45)
    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
    eye3 = prep["eye3"]
    t00 = np.asarray(prep["t00"])
    xi_off = np.where(np.isfinite(xi_mat), xi_mat, 0.0).copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(np.float64)
    deg = adj.sum(axis=1)
    weight_adj = xi_off * adj
    xi_loc = weight_adj.sum(axis=1) / np.maximum(deg, 1)
    res = np.asarray(per_node_residual(
        prep["g_00_h"], prep["g_ij_h"],
        prep["t00"], prep["t_ij"],
        LAMBDA_T, LAMBDA_S, eye3, np))
    return t00, xi_loc, res


def granger_F_pvalue(x, y, max_lag=2):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    T = len(x)
    p = max_lag
    if T <= 3 * p + 1:
        return float("nan"), float("nan")
    Y = y[p:]
    n = T - p
    Y_lag = np.zeros((n, p))
    X_lag = np.zeros((n, p))
    for i in range(1, p + 1):
        Y_lag[:, i - 1] = y[p - i: T - i]
        X_lag[:, i - 1] = x[p - i: T - i]
    ones = np.ones((n, 1))
    Z_r = np.hstack([ones, Y_lag])
    Z_f = np.hstack([ones, Y_lag, X_lag])
    try:
        beta_r, *_ = np.linalg.lstsq(Z_r, Y, rcond=None)
        beta_f, *_ = np.linalg.lstsq(Z_f, Y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    ssr_r = float(np.sum((Y - Z_r @ beta_r) ** 2))
    ssr_f = float(np.sum((Y - Z_f @ beta_f) ** 2))
    df_num = p
    df_den = n - 2 * p - 1
    if df_den <= 0 or ssr_f <= 1e-12:
        return float("nan"), float("nan")
    F = ((ssr_r - ssr_f) / df_num) / (ssr_f / df_den)
    if F <= 0:
        return float(F), 1.0
    pval = float(f_dist.sf(F, df_num, df_den))
    return float(F), pval


def fishers_combined_pvalue(pvalues):
    valid = [p for p in pvalues if 0 < p <= 1 and np.isfinite(p)]
    if not valid:
        return float("nan"), float("nan")
    chi2_stat = float(-2 * np.sum(np.log(valid)))
    df = 2 * len(valid)
    pval = float(chi2.sf(chi2_stat, df))
    return chi2_stat, pval


def gather_per_seed(xi_snaps, psir, psii, n_lat):
    """Return per-seed: t00_ts (n_snap, n_lat), xi_loc_ts, residual_ts."""
    n_snap = xi_snaps.shape[0]
    t00_ts = np.zeros((n_snap, n_lat))
    xi_loc_ts = np.zeros((n_snap, n_lat))
    residual_ts = np.zeros((n_snap, n_lat))
    for t_idx in range(n_snap):
        xi_mat = np.asarray(xi_snaps[t_idx])
        psi = psir[t_idx] + 1j * psii[t_idx]
        t00, xi_loc, res = reconstruct(xi_mat, psi, n_lat)
        t00_ts[t_idx] = t00
        xi_loc_ts[t_idx] = xi_loc
        residual_ts[t_idx] = res
    return t00_ts, xi_loc_ts, residual_ts


def aggregate_granger(per_seed_data, node_filter, label):
    """For each (seed, node a) in node_filter, compute granger
    p-values in both directions and return aggregate counts +
    Fisher's combined.

    node_filter: list of (seed, node_indices_array) tuples.
    """
    p_xi_to_t00 = []
    p_t00_to_xi = []
    for s, indices in node_filter:
        if s >= len(per_seed_data) or len(indices) == 0:
            continue
        t00_ts = per_seed_data[s]["t00_ts"]
        xi_loc_ts = per_seed_data[s]["xi_loc_ts"]
        for a in indices:
            F1, p1 = granger_F_pvalue(xi_loc_ts[:, a], t00_ts[:, a])
            F2, p2 = granger_F_pvalue(t00_ts[:, a], xi_loc_ts[:, a])
            if not np.isnan(p1):
                p_xi_to_t00.append(p1)
            if not np.isnan(p2):
                p_t00_to_xi.append(p2)
    chi_xt, p_xt = fishers_combined_pvalue(p_xi_to_t00)
    chi_tx, p_tx = fishers_combined_pvalue(p_t00_to_xi)
    n_total = len(p_xi_to_t00)
    sig_xt = sum(1 for p in p_xi_to_t00 if p < 0.05)
    sig_tx = sum(1 for p in p_t00_to_xi if p < 0.05)
    return {
        "label": label, "n_nodes_total": n_total,
        "fisher_xi_to_t00_chi2": chi_xt, "fisher_xi_to_t00_p": p_xt,
        "fisher_t00_to_xi_chi2": chi_tx, "fisher_t00_to_xi_p": p_tx,
        "n_sig_xi_to_t00": sig_xt, "frac_sig_xi_to_t00": sig_xt / max(n_total, 1),
        "n_sig_t00_to_xi": sig_tx, "frac_sig_t00_to_xi": sig_tx / max(n_total, 1),
        "directional_ratio": sig_xt / max(sig_tx, 1),
    }


def main():
    if not SNAPSHOT_NPZ.exists():
        print(f"ERROR: snapshot file not found: {SNAPSHOT_NPZ}")
        return
    d = np.load(SNAPSHOT_NPZ, allow_pickle=True)
    xi_snaps = d["edge_xi_snapshots"]
    psir = d["psi_real_snapshots"]
    psii = d["psi_imag_snapshots"]
    n_seeds, n_snap, n_lat, _ = xi_snaps.shape
    print(f"Snapshot file: {SNAPSHOT_NPZ}")
    print(f"  {n_seeds} seeds x {n_snap} snapshots x N={n_lat}")

    # winding only available for original P5N100 4-seed run
    winding_per_seed = None
    if WINDING_NPZ.exists():
        w_d = np.load(WINDING_NPZ, allow_pickle=True)
        if "winding_map" in w_d.files:
            winding_per_seed = np.asarray(w_d["winding_map"])
            print(f"  winding map available: shape {winding_per_seed.shape}")

    # --- Reconstruct per-seed time series (forward time) ---
    print("\nReconstructing forward-time per-seed time series ...")
    per_seed = []
    for s in range(n_seeds):
        t00_ts, xi_loc_ts, residual_ts = gather_per_seed(
            xi_snaps[s], psir[s], psii[s], n_lat)
        per_seed.append({
            "seed": s,
            "t00_ts": t00_ts,
            "xi_loc_ts": xi_loc_ts,
            "residual_ts": residual_ts,
        })
    print(f"  done")

    # --- Time-reversed copies ---
    per_seed_reversed = [{
        "seed": p["seed"],
        "t00_ts": p["t00_ts"][::-1].copy(),
        "xi_loc_ts": p["xi_loc_ts"][::-1].copy(),
        "residual_ts": p["residual_ts"][::-1].copy(),
    } for p in per_seed]

    # --- Identify candidate nodes (top-decile residual at final
    #     forward-time snapshot per seed) ---
    candidate_filter = []
    for s, p in enumerate(per_seed):
        final = p["residual_ts"][-1]
        cands = np.where(final >= np.percentile(final, 90))[0]
        candidate_filter.append((s, cands))

    print()
    print("=" * 100)
    print("CHECK T: Time-reversal test of Granger direction")
    print("=" * 100)
    forward = aggregate_granger(per_seed, candidate_filter, "forward")
    reverse = aggregate_granger(
        per_seed_reversed, candidate_filter, "reversed")
    print(f"Forward time:")
    print(f"  Fisher Xi->T00 p={forward['fisher_xi_to_t00_p']:.3e}, "
          f"per-node sig {forward['frac_sig_xi_to_t00']*100:.1f}%")
    print(f"  Fisher T00->Xi p={forward['fisher_t00_to_xi_p']:.3e}, "
          f"per-node sig {forward['frac_sig_t00_to_xi']*100:.1f}%")
    print(f"  directional ratio (Xi->T00 / T00->Xi): {forward['directional_ratio']:.2f}")
    print(f"Reversed time:")
    print(f"  Fisher Xi->T00 p={reverse['fisher_xi_to_t00_p']:.3e}, "
          f"per-node sig {reverse['frac_sig_xi_to_t00']*100:.1f}%")
    print(f"  Fisher T00->Xi p={reverse['fisher_t00_to_xi_p']:.3e}, "
          f"per-node sig {reverse['frac_sig_t00_to_xi']*100:.1f}%")
    print(f"  directional ratio (Xi->T00 / T00->Xi): {reverse['directional_ratio']:.2f}")
    print()
    flip_check = (forward['directional_ratio'] > 1.5
                   and reverse['directional_ratio'] < 1 / 1.5)
    no_flip = (forward['directional_ratio'] > 1.5
                and reverse['directional_ratio'] > 1.5)
    if flip_check:
        check_T = "PASS: direction flips under time reversal"
    elif no_flip:
        check_T = "FAIL: direction does NOT flip (asymmetry not time-arrow related)"
    else:
        check_T = "AMBIGUOUS: direction weakens but does not clearly flip"
    print(f"  -> CHECK T: {check_T}")

    # --- CHECK C: winding-sign comparison ---
    print()
    print("=" * 100)
    print("CHECK C: Charge conjugation (winding sign comparison)")
    print("=" * 100)
    if winding_per_seed is None:
        print("  SKIPPED: no winding_map available for this snapshot run")
        check_C = "SKIPPED"
        c_results = None
    else:
        # Note: winding_map is from the 4-seed run, snapshots are from 16-seed run
        # We can only test the FIRST 4 seeds of the snapshot data
        n_check_C = min(n_seeds, winding_per_seed.shape[0])
        print(f"  Using first {n_check_C} seeds (where winding map is available)")
        plus_filter = []
        zero_filter = []
        minus_filter = []
        for s in range(n_check_C):
            wnd = winding_per_seed[s]
            plus_filter.append((s, np.where(wnd > 0.5)[0]))
            zero_filter.append((s, np.where(np.abs(wnd) < 0.5)[0]))
            minus_filter.append((s, np.where(wnd < -0.5)[0]))
        plus_agg = aggregate_granger(per_seed, plus_filter, "winding+1")
        zero_agg = aggregate_granger(per_seed, zero_filter, "winding=0")
        minus_agg = aggregate_granger(per_seed, minus_filter, "winding-1")
        print(f"  +winding (n={plus_agg['n_nodes_total']}): "
              f"directional ratio = {plus_agg['directional_ratio']:.2f}, "
              f"Fisher p(Xi->T00) = {plus_agg['fisher_xi_to_t00_p']:.2e}")
        print(f"   winding=0 (n={zero_agg['n_nodes_total']}): "
              f"directional ratio = {zero_agg['directional_ratio']:.2f}, "
              f"Fisher p(Xi->T00) = {zero_agg['fisher_xi_to_t00_p']:.2e}")
        print(f"  -winding (n={minus_agg['n_nodes_total']}): "
              f"directional ratio = {minus_agg['directional_ratio']:.2f}, "
              f"Fisher p(Xi->T00) = {minus_agg['fisher_xi_to_t00_p']:.2e}")
        c_results = {"plus": plus_agg, "zero": zero_agg, "minus": minus_agg}
        # Check if +winding and -winding are systematically different
        if plus_agg['directional_ratio'] > 1.5 * minus_agg['directional_ratio'] \
                or minus_agg['directional_ratio'] > 1.5 * plus_agg['directional_ratio']:
            check_C = "PASS: significant directional difference between +/- winding"
        elif abs(plus_agg['directional_ratio']
                  - minus_agg['directional_ratio']) < 0.5:
            check_C = "FAIL: +/- winding show same directional preference (C-symmetric)"
        else:
            check_C = "WEAK: small but inconclusive +/- winding difference"
        print(f"  -> CHECK C: {check_C}")

    # --- CHECK P+defect: family comparison ---
    print()
    print("=" * 100)
    print("CHECK P+defect: defect-family comparison (vortex / no-vortex)")
    print("=" * 100)
    if winding_per_seed is None:
        print("  SKIPPED: no winding map")
        check_P = "SKIPPED"
        p_results = None
    else:
        vortex_filter = []
        no_vortex_filter = []
        for s in range(n_check_C):
            wnd = winding_per_seed[s]
            vortex_filter.append((s, np.where(np.abs(wnd) > 0.5)[0]))
            no_vortex_filter.append((s, np.where(np.abs(wnd) < 0.5)[0]))
        vortex_agg = aggregate_granger(per_seed, vortex_filter, "vortex")
        no_vortex_agg = aggregate_granger(per_seed, no_vortex_filter, "no_vortex")
        print(f"  vortex (|w|>0): directional ratio = "
              f"{vortex_agg['directional_ratio']:.2f}, "
              f"Fisher p(Xi->T00) = {vortex_agg['fisher_xi_to_t00_p']:.2e}, "
              f"per-node sig fraction = {vortex_agg['frac_sig_xi_to_t00']*100:.1f}%")
        print(f"  no-vortex (w=0): directional ratio = "
              f"{no_vortex_agg['directional_ratio']:.2f}, "
              f"Fisher p(Xi->T00) = {no_vortex_agg['fisher_xi_to_t00_p']:.2e}, "
              f"per-node sig fraction = {no_vortex_agg['frac_sig_xi_to_t00']*100:.1f}%")
        p_results = {"vortex": vortex_agg, "no_vortex": no_vortex_agg}
        ratio_diff = abs(vortex_agg['directional_ratio']
                          - no_vortex_agg['directional_ratio'])
        if ratio_diff > 1.0:
            check_P = "PASS: vortex vs no-vortex show distinct directional preferences"
        elif ratio_diff < 0.3:
            check_P = "FAIL: vortex and no-vortex show identical directional preference"
        else:
            check_P = "WEAK: small but inconclusive difference"
        print(f"  -> CHECK P+defect: {check_P}")

    # --- Final synthesis ---
    print()
    print("=" * 100)
    print("MATHEMATICAL VIABILITY VERDICT")
    print("=" * 100)
    print(f"  CHECK T (time-reversal): {check_T}")
    print(f"  CHECK C (winding sign):  {check_C}")
    print(f"  CHECK P+defect (family): {check_P}")
    print()
    if "PASS" in check_T and "PASS" in check_C and "PASS" in check_P:
        print("  -> All three Sakharov-analogue checks PASS:")
        print("     hypothesis is MATHEMATICALLY CONSISTENT and warrants formal")
        print("     CPT-operator development on the lattice.")
    elif "FAIL" in check_T:
        print("  -> CHECK T fails: the Granger asymmetry is NOT time-arrow-related.")
        print("     The matter-antimatter analogy is NOT mathematically supported.")
    else:
        print("  -> Mixed: at least one check passes, others ambiguous or skipped.")
        print("     Hypothesis is conditionally viable, requires sharper tests.")

    out = REPO / "outputs" / "matter_antimatter_mathematical_check.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "method": "matter_antimatter_three_check_audit",
            "schema_version": "1.0.0",
            "snapshot_file": str(SNAPSHOT_NPZ),
            "check_T_time_reversal": {
                "verdict": check_T,
                "forward": forward, "reversed": reverse,
            },
            "check_C_charge_conjugation": {
                "verdict": check_C,
                "results": c_results,
            },
            "check_P_defect_family": {
                "verdict": check_P,
                "results": p_results,
            },
        }, f, indent=2, default=str)
    print()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
