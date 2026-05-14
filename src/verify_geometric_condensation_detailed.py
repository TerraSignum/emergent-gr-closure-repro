"""Detailed empirical proof of the geometric-condensation
hypothesis: at heavy-tail residual nodes, the local relational
similarity Xi_loc(a,t) stiffens before the source energy
density T_00(a,t) rises.

Five independent lines of evidence beyond the cross-correlation
mean lag:

  (1) Granger F-test: do past values of Xi_loc(a,t-k) predict
      T_00(a,t) better than past T_00 alone? Conversely for the
      opposite direction.

  (2) Magnitude-correlation: is the time-derivative dXi/dt at
      time t correlated with the SUBSEQUENT time-derivative
      dT_00/dt at time t+k for k > 0?

  (3) Bootstrap stability: resample the per-node best-lag
      population with replacement to compute the confidence
      interval of the mean lag.

  (4) Heavy-tail vs random comparison: heavy-tail nodes vs a
      random subset of the same size; if the geometric-
      condensation effect is real, only heavy-tail nodes show
      systematic negative lag.

  (5) Time-evolution visualisation: plot Xi_loc(a,t) and
      T_00(a,t) on shared time axes for the highest-residual
      nodes; visually verify the lag.

Output:
  outputs/geometric_condensation_detailed_audit.json
  paper/figures/fig15_geometric_condensation_evidence.{pdf,png}
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

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

from verify_galerkin_runner_A_hessian_ricci import (
    XI_THRESH, per_seed_galerkin)
from verify_higher_order_terms_all8 import (
    LAMBDA_T, LAMBDA_S, per_node_residual)


SNAPSHOT_NPZ = REPO.parent / "results_d1_p5n100_snapshot" / "P5N100.snapshots.npz"


def reconstruct_t00_xi_loc_residual(xi_mat, psi, n_lat):
    """Compute per-node T_00, Xi_loc, struct-Lambda residual."""
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


def cross_corr(x, y, max_lag):
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


def granger_F_test(x, y, max_lag=2):
    """Test whether past values of X help predict Y, beyond past Y
    alone. Returns the F-statistic. A larger F means X granger-
    causes Y more strongly.

    We fit two least-squares regressions on the time series
    (length T):
      restricted: Y[t] = a + sum_{i=1..p} alpha_i Y[t-i]
      full:       Y[t] = a + sum_{i=1..p} alpha_i Y[t-i]
                          + sum_{i=1..p} beta_i  X[t-i]
    F = ((SSR_r - SSR_f) / p) / (SSR_f / (T - 2p - 1))
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    T = len(x)
    p = max_lag
    if T <= 3 * p + 1:
        return float("nan")
    # Build lag matrices
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
        return float("nan")
    ssr_r = float(np.sum((Y - Z_r @ beta_r) ** 2))
    ssr_f = float(np.sum((Y - Z_f @ beta_f) ** 2))
    df_num = p
    df_den = n - 2 * p - 1
    if df_den <= 0 or ssr_f <= 0:
        return float("nan")
    F = ((ssr_r - ssr_f) / df_num) / (ssr_f / df_den)
    return float(F)


def magnitude_lead_correlation(xi_loc_t, t00_t, lag):
    """Correlation of dXi/dt at t with dT00/dt at t+lag.
    Positive correlation at lag > 0 means: rising Xi predicts
    SUBSEQUENT rising T_00."""
    dxi = np.diff(xi_loc_t)
    dt00 = np.diff(t00_t)
    if lag >= 0:
        a = dxi[: len(dxi) - lag]
        b = dt00[lag:]
    else:
        a = dxi[-lag:]
        b = dt00[: len(dt00) + lag]
    if len(a) < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 1e-12 else 0.0


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
    print(f"snapshot file: {SNAPSHOT_NPZ}")
    print(f"  {n_seeds} seeds x {n_snap} snapshots x N={n_lat}")
    print(f"  steps: {snapshot_steps.tolist()}")
    print()

    # Reconstruct per-node series
    print("Reconstructing per-seed, per-node T_00 / Xi_loc / residual ...")
    seed_data = []
    for s in range(n_seeds):
        t00_ts = np.zeros((n_snap, n_lat))
        xi_loc_ts = np.zeros((n_snap, n_lat))
        residual_ts = np.zeros((n_snap, n_lat))
        for t_idx in range(n_snap):
            xi_mat = np.asarray(xi_snaps[s, t_idx])
            psi = psir_snaps[s, t_idx] + 1j * psii_snaps[s, t_idx]
            t00, xi_loc, res = reconstruct_t00_xi_loc_residual(
                xi_mat, psi, n_lat)
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
        print(f"  seed {s}: {len(cands)} candidate nodes")

    # ---- (1) Granger F-test ----
    print()
    print("=" * 70)
    print("(1) Granger causality F-test (max_lag=2)")
    print("=" * 70)
    granger_results = []
    for sd in seed_data:
        F_xi_to_t00_cand = []
        F_t00_to_xi_cand = []
        F_xi_to_t00_non = []
        F_t00_to_xi_non = []
        for a in sd["candidates"]:
            F_xi_to_t00_cand.append(granger_F_test(
                sd["xi_loc_ts"][:, a], sd["t00_ts"][:, a], max_lag=2))
            F_t00_to_xi_cand.append(granger_F_test(
                sd["t00_ts"][:, a], sd["xi_loc_ts"][:, a], max_lag=2))
        for a in sd["non_cands"]:
            F_xi_to_t00_non.append(granger_F_test(
                sd["xi_loc_ts"][:, a], sd["t00_ts"][:, a], max_lag=2))
            F_t00_to_xi_non.append(granger_F_test(
                sd["t00_ts"][:, a], sd["xi_loc_ts"][:, a], max_lag=2))
        m_xi_t00_c = float(np.nanmean(F_xi_to_t00_cand))
        m_t00_xi_c = float(np.nanmean(F_t00_to_xi_cand))
        m_xi_t00_n = float(np.nanmean(F_xi_to_t00_non))
        m_t00_xi_n = float(np.nanmean(F_t00_to_xi_non))
        ratio_c = m_xi_t00_c / max(m_t00_xi_c, 1e-9)
        ratio_n = m_xi_t00_n / max(m_t00_xi_n, 1e-9)
        granger_results.append({
            "seed": int(sd["seed"]),
            "F_Xi_predicts_T00_candidates": m_xi_t00_c,
            "F_T00_predicts_Xi_candidates": m_t00_xi_c,
            "F_ratio_candidates": ratio_c,
            "F_Xi_predicts_T00_noncandidates": m_xi_t00_n,
            "F_T00_predicts_Xi_noncandidates": m_t00_xi_n,
            "F_ratio_noncandidates": ratio_n,
        })
        print(f"  seed {sd['seed']}: candidates F(Xi->T00)={m_xi_t00_c:.3f}, "
              f"F(T00->Xi)={m_t00_xi_c:.3f}, ratio={ratio_c:.2f}; "
              f"non-cand ratio={ratio_n:.2f}")

    # ---- (2) Magnitude-derivative correlation ----
    print()
    print("=" * 70)
    print("(2) Magnitude-derivative correlation: dXi/dt -> dT_00/dt+k")
    print("=" * 70)
    mag_results = []
    for sd in seed_data:
        for label, indices in [("candidates", sd["candidates"]),
                                  ("non_cands", sd["non_cands"])]:
            corrs = {k: [] for k in range(-3, 4)}
            for a in indices:
                for k in range(-3, 4):
                    c = magnitude_lead_correlation(
                        sd["xi_loc_ts"][:, a], sd["t00_ts"][:, a], k)
                    corrs[k].append(c)
            for k in corrs:
                corrs[k] = float(np.nanmean(corrs[k]))
            mag_results.append({
                "seed": int(sd["seed"]),
                "subset": label,
                "corr_per_lag": corrs,
            })
        cand = mag_results[-2]["corr_per_lag"]
        non = mag_results[-1]["corr_per_lag"]
        cand_pos = float(np.mean([cand[k] for k in (1, 2)]))
        cand_neg = float(np.mean([cand[k] for k in (-2, -1)]))
        print(f"  seed {sd['seed']}: candidates dXi(t) vs dT00(t+1,2): "
              f"{cand_pos:+.3f}; vs dT00(t-1,-2): {cand_neg:+.3f}; "
              f"non-cand dXi vs dT00(t+1,2): "
              f"{float(np.mean([non[k] for k in (1, 2)])):+.3f}")

    # ---- (3) Bootstrap stability ----
    print()
    print("=" * 70)
    print("(3) Bootstrap stability of best-lag mean")
    print("=" * 70)
    all_cand_lags = []
    for sd in seed_data:
        for a in sd["candidates"]:
            cc = cross_corr(sd["xi_loc_ts"][:, a], sd["t00_ts"][:, a],
                              max_lag=4)
            best_k = int(np.argmax(cc) - 4)
            all_cand_lags.append(best_k)
    all_cand_lags = np.asarray(all_cand_lags, dtype=float)
    rng = np.random.default_rng(seed=42)
    n_boot = 5000
    boot_means = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(all_cand_lags), size=len(all_cand_lags))
        boot_means[i] = all_cand_lags[idx].mean()
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    obs_mean = float(all_cand_lags.mean())
    p_two_sided_lt0 = float(np.mean(boot_means >= 0))
    # Convention: cross_corr(xi_loc, t00) with k>0 means Xi[t]
    # correlates with T_00[t+k] -> Xi LEADS T_00 by k steps.
    # geometric-condensation hypothesis: Xi leads T_00 -> mean
    # lag should be POSITIVE.
    p_geq_zero = float(np.mean(boot_means >= 0))
    p_le_zero = float(np.mean(boot_means <= 0))
    print(f"  observed candidate mean lag (Xi leads T_00 by k steps): "
          f"{obs_mean:+.3f}")
    print(f"  95% bootstrap CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"  P(boot mean >= 0): {p_geq_zero:.3f}")
    print(f"  P(boot mean <= 0): {p_le_zero:.3f}")
    if obs_mean > 0 and ci_lo > 0:
        verdict = "STRONG: Xi leads T_00 (CI excludes 0)"
    elif obs_mean > 0 and ci_lo <= 0 < ci_hi:
        verdict = "MARGINAL: positive mean (Xi leads T_00) but CI includes 0"
    elif obs_mean < 0 and ci_hi < 0:
        verdict = "STRONG REVERSE: T_00 leads Xi"
    else:
        verdict = "NULL: no directional signal"
    print(f"  -> verdict: {verdict}")

    bootstrap_summary = {
        "candidate_mean_lag": obs_mean,
        "ci_95_low": ci_lo, "ci_95_high": ci_hi,
        "p_value_mean_geq_zero": p_two_sided_lt0,
        "n_candidates_pooled": int(len(all_cand_lags)),
        "n_bootstrap_samples": n_boot,
    }

    # ---- (4) Cross-cut comparison: top-5%, top-10%, top-15% ----
    print()
    print("=" * 70)
    print("(4) Robustness across residual cuts (top-5%, top-10%, top-15%)")
    print("=" * 70)
    cut_results = []
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
        sigma = float(np.std(lags) / np.sqrt(len(lags)))
        cut_results.append({
            "cut_pct": cut_pct,
            "n_nodes_pooled": len(lags),
            "mean_lag": m,
            "sem": sigma,
        })
        print(f"  top-{cut_pct}% (n={len(lags)}): mean lag {m:+.3f} +- {sigma:.3f}")

    # ---- (5) Visualisation: time-evolution of representative
    #       heavy-tail nodes vs random nodes ----
    print()
    print("=" * 70)
    print("(5) Time-evolution visualisation -> fig15")
    print("=" * 70)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Top: 4 highest-residual heavy-tail nodes (one per seed)
    ax = axes[0, 0]
    ax2 = axes[0, 1]
    for sd in seed_data:
        a_top = int(sd["candidates"][np.argmax(
            sd["residual_ts"][-1, sd["candidates"]])])
        t = snapshot_steps
        xi_n = (sd["xi_loc_ts"][:, a_top]
                  - sd["xi_loc_ts"][0, a_top])
        t00_n = (sd["t00_ts"][:, a_top]
                  - sd["t00_ts"][0, a_top])
        ax.plot(t, xi_n, "-", label=f"seed {sd['seed']} node {a_top}",
                 alpha=0.85)
        ax2.plot(t, t00_n, "-", label=f"seed {sd['seed']} node {a_top}",
                  alpha=0.85)
    ax.set_xlabel("flow step $t$")
    ax.set_ylabel(r"$\Xi_{\mathrm{loc}}(a, t) - \Xi_{\mathrm{loc}}(a, 0)$")
    ax.set_title(r"$\Xi_{\mathrm{loc}}$ evolution at heavy-tail nodes (top 1)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax2.set_xlabel("flow step $t$")
    ax2.set_ylabel(r"$T_{00}(a, t) - T_{00}(a, 0)$")
    ax2.set_title(r"$T_{00}$ evolution at the same nodes")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # Bottom-left: per-node best-lag histogram (candidates vs non-cands)
    ax = axes[1, 0]
    cand_lags_pool = []
    non_lags_pool = []
    for sd in seed_data:
        for a in sd["candidates"]:
            cc = cross_corr(sd["xi_loc_ts"][:, a], sd["t00_ts"][:, a],
                              max_lag=4)
            cand_lags_pool.append(int(np.argmax(cc) - 4))
        for a in sd["non_cands"]:
            cc = cross_corr(sd["xi_loc_ts"][:, a], sd["t00_ts"][:, a],
                              max_lag=4)
            non_lags_pool.append(int(np.argmax(cc) - 4))
    bins = np.arange(-4.5, 5.5, 1)
    ax.hist(non_lags_pool, bins=bins, alpha=0.5, label=f"random (n={len(non_lags_pool)})",
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
                   "(positive = $T_{00}$ leads $\\Xi$)")
    ax.set_ylabel("number of nodes")
    ax.set_title(r"Per-node best-lag distribution: $\Xi_{\mathrm{loc}}$ vs $T_{00}$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: bootstrap distribution
    ax = axes[1, 1]
    ax.hist(boot_means, bins=40, color="steelblue", alpha=0.8,
             edgecolor="navy")
    ax.axvline(obs_mean, color="red", linestyle="--",
                label=f"obs mean {obs_mean:+.3f}")
    ax.axvline(ci_lo, color="black", linestyle=":",
                label=f"95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]")
    ax.axvline(ci_hi, color="black", linestyle=":")
    ax.axvline(0, color="green", linestyle="-", alpha=0.5,
                label="zero (null)")
    ax.set_xlabel("bootstrap mean lag")
    ax.set_ylabel("count")
    ax.set_title(f"Bootstrap (n={n_boot}) of candidate mean lag")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig15_path = REPO / "paper" / "figures" / "fig15_geometric_condensation_evidence"
    plt.savefig(f"{fig15_path}.pdf", bbox_inches="tight")
    plt.savefig(f"{fig15_path}.png", bbox_inches="tight", dpi=150)
    print(f"  Saved {fig15_path}.{{pdf,png}}")

    # ---- Save audit JSON ----
    audit = {
        "method": "geometric_condensation_detailed_audit",
        "snapshot_file": str(SNAPSHOT_NPZ),
        "n_seeds": int(n_seeds), "n_snap": int(n_snap), "n_lat": int(n_lat),
        "snapshot_steps": snapshot_steps.tolist(),
        "structural_lambda": {"t": LAMBDA_T, "s": LAMBDA_S},
        "test_1_granger_F": granger_results,
        "test_2_magnitude_derivative_correlation": mag_results,
        "test_3_bootstrap": bootstrap_summary,
        "test_4_robustness_residual_cuts": cut_results,
        "test_5_figure": str(fig15_path) + ".pdf",
        "verdict": {
            "mean_lag": obs_mean,
            "ci_95": [ci_lo, ci_hi],
            "p_geq_zero": p_two_sided_lt0,
            "geometric_condensation_supported": (ci_hi < 0),
        },
    }
    out = REPO / "outputs" / "geometric_condensation_detailed_audit.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Pooled candidate best-lag mean: {obs_mean:+.3f} steps")
    print(f"  95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"  geometric-condensation supported (CI excludes 0): "
          f"{'YES' if ci_hi < 0 else 'NO'}")
    print(f"  Saved {out}")


if __name__ == "__main__":
    main()
