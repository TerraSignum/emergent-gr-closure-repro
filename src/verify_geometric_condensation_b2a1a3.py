"""B2+A1+A3 expansion of the geometric-condensation audit.

  B2: 16 seeds (vs 4) at P5N100 -> n=160 candidate nodes
  A1: hierarchical bootstrap (resample seeds, then within-seed
       resample nodes) for proper standard error
  A3: per-node Granger F-test with chi^2 p-value + Fisher's
       combined test across candidates

Inputs: results_d1_p5n100_snapshot_16seeds/P5N100.snapshots.npz

Output: outputs/geometric_condensation_b2a1a3_audit.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import f as f_dist
from scipy.stats import chi2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


class _BlockCupy:
    def find_module(self, name, path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self

    def load_module(self, name):
        raise ImportError("cupy disabled")

sys.meta_path.insert(0, _BlockCupy())

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

from verify_galerkin_runner_A_hessian_ricci import (
    XI_THRESH, edge_to_matrix, per_seed_galerkin)
from verify_higher_order_terms_all8 import (
    LAMBDA_T, LAMBDA_S, per_node_residual)


SNAPSHOT_NPZ = REPO.parent / "results_d1_p5n100_snapshot_16seeds" / "P5N100.snapshots.npz"


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


def cross_corr(x, y, max_lag=4):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    T = len(x)
    out = np.zeros(2 * max_lag + 1)
    for i, k in enumerate(range(-max_lag, max_lag + 1)):
        if k >= 0:
            xs, ys = x[:T - k], y[k:]
        else:
            xs, ys = x[-k:], y[:T + k]
        if len(xs) < 3:
            out[i] = np.nan
            continue
        xs = xs - xs.mean()
        ys = ys - ys.mean()
        denom = np.sqrt((xs * xs).sum() * (ys * ys).sum())
        out[i] = float((xs * ys).sum() / denom) if denom > 1e-12 else 0.0
    return out


def granger_F_pvalue(x, y, max_lag=2):
    """Returns (F-stat, p-value) for H0: past X does not improve Y prediction."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    T = len(x)
    p = max_lag
    if T <= 3 * p + 1:
        return float("nan"), float("nan")
    Y = y[p:]
    n = T - p
    YlagS = np.zeros((n, p))
    XlagS = np.zeros((n, p))
    for i in range(1, p + 1):
        YlagS[:, i - 1] = y[p - i: T - i]
        XlagS[:, i - 1] = x[p - i: T - i]
    ones = np.ones((n, 1))
    Z_r = np.hstack([ones, YlagS])
    Z_f = np.hstack([ones, YlagS, XlagS])
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
    """Fisher's method: combined chi^2 = -2 sum log(p_i), df = 2k."""
    valid = [p for p in pvalues if 0 < p <= 1 and np.isfinite(p)]
    if not valid:
        return float("nan"), float("nan")
    chi2_stat = float(-2 * np.sum(np.log(valid)))
    df = 2 * len(valid)
    pval = float(chi2.sf(chi2_stat, df))
    return chi2_stat, pval


def main():
    if not SNAPSHOT_NPZ.exists():
        print(f"ERROR: snapshot file not found: {SNAPSHOT_NPZ}")
        return
    d = np.load(SNAPSHOT_NPZ, allow_pickle=True)
    xi_snaps = d["edge_xi_snapshots"]
    psir_snaps = d["psi_real_snapshots"]
    psii_snaps = d["psi_imag_snapshots"]
    snapshot_steps = np.asarray(d["snapshot_steps"])
    n_seeds, n_snap, n_lat, _ = xi_snaps.shape
    print(f"Snapshot file: {SNAPSHOT_NPZ}")
    print(f"  {n_seeds} seeds x {n_snap} snapshots x N={n_lat}")
    print()

    # --- Reconstruct per-seed time-series ---
    print("Reconstructing per-seed, per-node time series ...")
    seed_data = []
    for s in range(n_seeds):
        t00_ts = np.zeros((n_snap, n_lat))
        xi_loc_ts = np.zeros((n_snap, n_lat))
        residual_ts = np.zeros((n_snap, n_lat))
        for t_idx in range(n_snap):
            xi_mat = np.asarray(xi_snaps[s, t_idx])
            psi = psir_snaps[s, t_idx] + 1j * psii_snaps[s, t_idx]
            t00, xi_loc, res = reconstruct(xi_mat, psi, n_lat)
            t00_ts[t_idx] = t00
            xi_loc_ts[t_idx] = xi_loc
            residual_ts[t_idx] = res
        final_residual = residual_ts[-1]
        p90 = np.percentile(final_residual, 90)
        cands = np.where(final_residual >= p90)[0]
        rng = np.random.default_rng(seed=s)
        non_cands = rng.choice(
            np.where(final_residual < p90)[0],
            size=len(cands), replace=False)
        seed_data.append({
            "seed": s,
            "t00_ts": t00_ts, "xi_loc_ts": xi_loc_ts,
            "residual_ts": residual_ts,
            "candidates": cands,
            "non_cands": non_cands,
        })
    print(f"  done: {sum(len(s['candidates']) for s in seed_data)} "
          f"candidate nodes pooled")

    # --- A3: per-node Granger F-test with chi^2 p-value ---
    print()
    print("=" * 70)
    print("(A3) Per-node Granger test with chi^2 p-values + Fisher's combined")
    print("=" * 70)
    p_xi_to_t00_cand = []
    p_t00_to_xi_cand = []
    F_xi_to_t00_cand = []
    F_t00_to_xi_cand = []
    p_xi_to_t00_non = []
    p_t00_to_xi_non = []
    for sd in seed_data:
        for a in sd["candidates"]:
            F1, p1 = granger_F_pvalue(
                sd["xi_loc_ts"][:, a], sd["t00_ts"][:, a], max_lag=2)
            F2, p2 = granger_F_pvalue(
                sd["t00_ts"][:, a], sd["xi_loc_ts"][:, a], max_lag=2)
            F_xi_to_t00_cand.append(F1)
            F_t00_to_xi_cand.append(F2)
            p_xi_to_t00_cand.append(p1)
            p_t00_to_xi_cand.append(p2)
        for a in sd["non_cands"]:
            F1, p1 = granger_F_pvalue(
                sd["xi_loc_ts"][:, a], sd["t00_ts"][:, a], max_lag=2)
            F2, p2 = granger_F_pvalue(
                sd["t00_ts"][:, a], sd["xi_loc_ts"][:, a], max_lag=2)
            p_xi_to_t00_non.append(p1)
            p_t00_to_xi_non.append(p2)

    chi_xi_t00, p_xi_t00 = fishers_combined_pvalue(p_xi_to_t00_cand)
    chi_t00_xi, p_t00_xi = fishers_combined_pvalue(p_t00_to_xi_cand)
    chi_xi_t00_n, p_xi_t00_n = fishers_combined_pvalue(p_xi_to_t00_non)
    chi_t00_xi_n, p_t00_xi_n = fishers_combined_pvalue(p_t00_to_xi_non)

    print(f"  Candidates pooled (n={len(p_xi_to_t00_cand)}):")
    print(f"    Fisher's chi^2 (Xi -> T_00): "
          f"chi^2={chi_xi_t00:.2f}, df={2*len(p_xi_to_t00_cand)}, "
          f"p={p_xi_t00:.4e}")
    print(f"    Fisher's chi^2 (T_00 -> Xi): "
          f"chi^2={chi_t00_xi:.2f}, df={2*len(p_t00_to_xi_cand)}, "
          f"p={p_t00_xi:.4e}")
    print(f"  Non-candidates pooled (n={len(p_xi_to_t00_non)}):")
    print(f"    Fisher's chi^2 (Xi -> T_00): p={p_xi_t00_n:.4e}")
    print(f"    Fisher's chi^2 (T_00 -> Xi): p={p_t00_xi_n:.4e}")

    sig_xi_t00 = sum(1 for p in p_xi_to_t00_cand
                      if not np.isnan(p) and p < 0.05)
    sig_t00_xi = sum(1 for p in p_t00_to_xi_cand
                      if not np.isnan(p) and p < 0.05)
    print(f"  Per-node p<0.05 (Xi->T_00): {sig_xi_t00}/{len(p_xi_to_t00_cand)} "
          f"({100*sig_xi_t00/max(len(p_xi_to_t00_cand),1):.1f}%)")
    print(f"  Per-node p<0.05 (T_00->Xi): {sig_t00_xi}/{len(p_t00_to_xi_cand)} "
          f"({100*sig_t00_xi/max(len(p_t00_to_xi_cand),1):.1f}%)")

    # --- A1: hierarchical bootstrap of best-lag ---
    print()
    print("=" * 70)
    print("(A1) Hierarchical bootstrap of best-lag mean")
    print("=" * 70)
    seed_lags = []
    for sd in seed_data:
        seed_lags_local = []
        for a in sd["candidates"]:
            cc = cross_corr(sd["xi_loc_ts"][:, a], sd["t00_ts"][:, a],
                              max_lag=4)
            seed_lags_local.append(int(np.argmax(cc) - 4))
        seed_lags.append(seed_lags_local)

    pooled = np.concatenate([np.asarray(s) for s in seed_lags])
    obs_mean = float(pooled.mean())
    print(f"  Total candidate nodes: {len(pooled)}")
    print(f"  Observed pooled mean lag: {obs_mean:+.3f}")

    # Per-seed mean
    per_seed_mean = np.array([np.mean(s) if len(s) else np.nan
                                 for s in seed_lags])
    print(f"  Per-seed mean lags: "
          f"min={per_seed_mean.min():+.2f}, max={per_seed_mean.max():+.2f}, "
          f"median={np.median(per_seed_mean):+.2f}")

    # Hierarchical bootstrap
    n_boot = 5000
    rng = np.random.default_rng(seed=42)
    boot_means_h = np.zeros(n_boot)
    for i in range(n_boot):
        chosen_seeds = rng.integers(0, n_seeds, size=n_seeds)
        sample_lags = []
        for cs in chosen_seeds:
            local = seed_lags[cs]
            if len(local) == 0:
                continue
            inner = rng.integers(0, len(local), size=len(local))
            sample_lags.extend(np.asarray(local)[inner])
        boot_means_h[i] = float(np.mean(sample_lags)) if sample_lags else 0.0

    ci_h_lo = float(np.percentile(boot_means_h, 2.5))
    ci_h_hi = float(np.percentile(boot_means_h, 97.5))
    p_h_geq_zero = float(np.mean(boot_means_h >= 0))
    print(f"  Hierarchical bootstrap (n_boot={n_boot}):")
    print(f"    95% CI: [{ci_h_lo:+.3f}, {ci_h_hi:+.3f}]")
    print(f"    P(boot mean >= 0): {p_h_geq_zero:.3f}")

    # Naive flat bootstrap for comparison
    boot_means_f = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(pooled), size=len(pooled))
        boot_means_f[i] = float(pooled[idx].mean())
    ci_f_lo = float(np.percentile(boot_means_f, 2.5))
    ci_f_hi = float(np.percentile(boot_means_f, 97.5))
    print(f"  Flat (non-hierarchical) bootstrap for comparison:")
    print(f"    95% CI: [{ci_f_lo:+.3f}, {ci_f_hi:+.3f}]")

    # --- Robustness across cuts ---
    print()
    print("=" * 70)
    print("(Robustness) cuts top-5%, top-10%, top-15%")
    print("=" * 70)
    cuts = []
    for cut_pct in (5, 10, 15):
        lags = []
        for sd in seed_data:
            p_cut = np.percentile(sd["residual_ts"][-1], 100 - cut_pct)
            cands = np.where(sd["residual_ts"][-1] >= p_cut)[0]
            for a in cands:
                cc = cross_corr(sd["xi_loc_ts"][:, a], sd["t00_ts"][:, a],
                                  max_lag=4)
                lags.append(int(np.argmax(cc) - 4))
        m = float(np.mean(lags))
        sem = float(np.std(lags) / np.sqrt(len(lags)))
        sigma_signif = m / max(sem, 1e-12)
        cuts.append({
            "cut_pct": cut_pct, "n_nodes": len(lags),
            "mean_lag": m, "sem": sem, "n_sigma": sigma_signif,
        })
        print(f"  top-{cut_pct}% (n={len(lags)}): "
              f"{m:+.3f} ± {sem:.3f} ({sigma_signif:+.2f}sigma)")

    # --- Visualisation: B2 result panel ---
    print()
    print("(viz) -> fig16")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    cand_lags_pool = pooled.tolist()
    non_lags_pool = []
    for sd in seed_data:
        for a in sd["non_cands"]:
            cc = cross_corr(sd["xi_loc_ts"][:, a], sd["t00_ts"][:, a],
                              max_lag=4)
            non_lags_pool.append(int(np.argmax(cc) - 4))
    bins = np.arange(-4.5, 5.5, 1)
    ax.hist(non_lags_pool, bins=bins, alpha=0.5,
             label=f"random (n={len(non_lags_pool)})",
             color="lightgrey", edgecolor="grey")
    ax.hist(cand_lags_pool, bins=bins, alpha=0.7,
             label=f"heavy-tail (n={len(cand_lags_pool)})",
             color="red", edgecolor="darkred")
    ax.axvline(0, color="black", linestyle=":", alpha=0.5)
    ax.axvline(np.mean(cand_lags_pool), color="red", linestyle="--",
                label=f"heavy-tail mean {np.mean(cand_lags_pool):+.2f}")
    ax.axvline(np.mean(non_lags_pool), color="grey", linestyle="--",
                label=f"random mean {np.mean(non_lags_pool):+.2f}")
    ax.set_xlabel("best lag $k$ in steps "
                   "(positive $\\equiv \\Xi$ leads $T_{00}$)")
    ax.set_ylabel("number of nodes")
    ax.set_title(f"B2 expansion: 16 seeds, n={len(cand_lags_pool)} candidates")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(boot_means_h, bins=40, color="steelblue", alpha=0.8,
             edgecolor="navy", label="hierarchical bootstrap")
    ax.hist(boot_means_f, bins=40, color="orange", alpha=0.4,
             edgecolor="darkorange", label="flat bootstrap")
    ax.axvline(obs_mean, color="red", linestyle="--",
                label=f"obs {obs_mean:+.3f}")
    ax.axvline(ci_h_lo, color="navy", linestyle=":",
                label=f"hier CI [{ci_h_lo:+.2f},{ci_h_hi:+.2f}]")
    ax.axvline(ci_h_hi, color="navy", linestyle=":")
    ax.axvline(0, color="green", linestyle="-", alpha=0.5, label="zero")
    ax.set_xlabel("bootstrap mean lag")
    ax.set_ylabel("count")
    ax.set_title("A1 hierarchical bootstrap (5000 resamples)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig16_path = REPO / "paper" / "figures" / "fig16_b2a1a3_significance"
    plt.savefig(f"{fig16_path}.pdf", bbox_inches="tight")
    plt.savefig(f"{fig16_path}.png", bbox_inches="tight", dpi=150)
    print(f"  saved {fig16_path}.{{pdf,png}}")

    # --- Save audit ---
    audit = {
        "method": "B2_A1_A3_geometric_condensation_audit",
        "n_seeds": int(n_seeds),
        "n_snapshots": int(n_snap),
        "n_lat": int(n_lat),
        "n_candidates_pooled": int(len(pooled)),
        "snapshot_steps": snapshot_steps.tolist(),
        "best_lag_pooled_mean": obs_mean,
        "per_seed_mean_lags": per_seed_mean.tolist(),
        "hierarchical_bootstrap": {
            "n_boot": n_boot,
            "ci_95_low": ci_h_lo, "ci_95_high": ci_h_hi,
            "p_geq_zero": p_h_geq_zero,
        },
        "flat_bootstrap": {
            "ci_95_low": ci_f_lo, "ci_95_high": ci_f_hi,
        },
        "fisher_combined": {
            "Xi_to_T00_chi2": chi_xi_t00,
            "Xi_to_T00_pvalue": p_xi_t00,
            "T00_to_Xi_chi2": chi_t00_xi,
            "T00_to_Xi_pvalue": p_t00_xi,
            "noncand_Xi_to_T00_pvalue": p_xi_t00_n,
            "noncand_T00_to_Xi_pvalue": p_t00_xi_n,
            "n_per_node_p_lt_005_xi_to_t00": int(sig_xi_t00),
            "n_per_node_p_lt_005_t00_to_xi": int(sig_t00_xi),
            "n_total_candidates": int(len(p_xi_to_t00_cand)),
        },
        "robustness_cuts": cuts,
    }

    out = REPO / "outputs" / "geometric_condensation_b2a1a3_audit.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)

    print()
    print("=" * 70)
    print("FINAL VERDICT (B2 + A1 + A3)")
    print("=" * 70)
    print(f"  n_candidates: {len(pooled)} (16 seeds × 10 each)")
    print(f"  observed mean lag: {obs_mean:+.3f} steps")
    print(f"  hierarchical 95% CI: [{ci_h_lo:+.3f}, {ci_h_hi:+.3f}]")
    print(f"  Fisher's combined p (Xi -> T_00): {p_xi_t00:.2e}")
    print(f"  Fisher's combined p (T_00 -> Xi): {p_t00_xi:.2e}")
    if ci_h_lo > 0:
        print("  -> STRONG: Xi leads T_00 statistically significantly")
    elif obs_mean > 0 and p_xi_t00 < 0.05 and p_xi_t00 < p_t00_xi:
        print("  -> POSITIVE: directional Granger evidence Xi -> T_00")
    elif obs_mean > 0:
        print("  -> WEAK: positive mean lag but CI includes 0")
    else:
        print("  -> NULL/REVERSE: not in Xi-leads direction")
    print(f"  Saved {out}")


if __name__ == "__main__":
    main()
