"""Per-node lead-lag test of the matter-nucleation hypothesis on
the P5N512 12-seed 13-snapshot time series.

Background. The matter-nucleation working hypothesis (P4 Outlook,
Item 15 of audit/caveats_open_items_2026_05_07.md) reads the
heavy-tail residual of the per-node Frobenius residual as
nucleation seeds of a nonlinear matter--gravity coupling: locally
elevated source-tensor density triggers a non-linear curvature
response, the relational similarity Xi_ij stiffens and densifies
in the same neighbourhood, and the pattern propagates.

The signature of nucleation: in the time series, the *Xi-edge
densification* should precede or coincide with the *T_00
elevation* at matter-core nodes. The earlier audit
(geometric_condensation_detailed_audit) reported a 5/5 lines of
evidence with bootstrap CI [-0.45, +1.58] for the mean lead lag,
just barely including zero. The N=512 twelve-seed thirteen-
snapshot data give substantially more time-resolution to tighten
this bound.

Per-node lead-lag, framework definitions:

  rho_lag(a) := corr_t( Xi_dens(a, t), T_00(a, t + lag) )

with rho_lag(a) the Pearson correlation across snapshots t,
Xi_dens(a, t) the local Xi-density (sum over neighbours weighted
by Xi), and T_00(a, t) the bundled per-node energy density at
time t. The hypothesis is rho_lag > 0 for lag in {-1, 0, +1}
(symmetric peak around lag=0 with positive correlation,
indicating co-localised matter-Xi structure).

Output: outputs/verify_heavy_tail_lead_lag_p5n512.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


class _BlockCupy:
    def find_module(self, name, _path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self

    def load_module(self, name):
        raise ImportError("cupy disabled")


sys.meta_path.insert(0, _BlockCupy())

REPO = Path(__file__).resolve().parents[1]
REPO_ROOT = REPO.parent

NPZ_PATH = (REPO_ROOT
            / "results_d1_p5n512_12seeds"
            / "P5N512.snapshots.npz")


def per_node_t00_series(xi_series, psi_series, k_series, q_series):
    """Per-node T_00 at each snapshot. Shape: (n_snap, n_lat)."""
    n_snap, n_lat, _ = xi_series.shape
    out = np.zeros((n_snap, n_lat), dtype=float)
    for t in range(n_snap):
        xi = xi_series[t].copy()
        np.fill_diagonal(xi, 0.0)
        amp_sq = np.abs(psi_series[t]) ** 2
        var_xi = np.var(xi, axis=1)
        kf = k_series[t]
        qf = q_series[t]
        k_diag = np.diag(kf) if kf.ndim == 2 else np.asarray(kf)
        q_diag = np.diag(qf) if qf.ndim == 2 else np.asarray(qf)
        out[t] = k_diag * amp_sq + q_diag * var_xi
    return np.maximum(out, 1e-12)


def per_node_xi_density(xi_series):
    """Xi-density per node = sum_j Xi_ij over off-diagonal j."""
    n_snap, n_lat, _ = xi_series.shape
    out = np.zeros((n_snap, n_lat), dtype=float)
    for t in range(n_snap):
        xi = xi_series[t].copy()
        np.fill_diagonal(xi, 0.0)
        out[t] = xi.sum(axis=1)
    return out


def lagged_corr(x, y, lag: int):
    """Pearson correlation across t of x[t] and y[t+lag]."""
    if lag == 0:
        a, b = x, y
    elif lag > 0:
        a, b = x[:-lag], y[lag:]
    else:
        a, b = x[-lag:], y[:lag]
    if a.size < 3:
        return float("nan")
    sa = a.std()
    sb = b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean()
                 / (sa * sb))


def main() -> int:
    if not NPZ_PATH.exists():
        print(f"  data not found: {NPZ_PATH}")
        return 1
    z = np.load(NPZ_PATH, allow_pickle=True)
    snaps = z["edge_xi_snapshots"]
    psi_re = z["psi_real_snapshots"]
    psi_im = z["psi_imag_snapshots"]
    k_snaps = z["k_snapshots"]
    q_snaps = z["q_snapshots"]
    n_seeds, n_snap, n_lat, _ = snaps.shape
    print(f"  loaded {n_seeds} seeds x {n_snap} snapshots x "
          f"{n_lat} nodes")

    rng = np.random.default_rng(2026)
    n_boot = 500
    p99_lag_zero_per_seed = []
    p99_lag_minus1_per_seed = []
    p99_lag_plus1_per_seed = []
    full_lag_zero_per_seed = []
    seed_summaries = []

    for s in range(n_seeds):
        xi_t = np.asarray(snaps[s], dtype=float)
        psi_t = (np.asarray(psi_re[s], dtype=float)
                  + 1j * np.asarray(psi_im[s], dtype=float))
        k_t = np.asarray(k_snaps[s], dtype=float)
        q_t = np.asarray(q_snaps[s], dtype=float)
        # Symmetrise xi at each snapshot
        for t in range(n_snap):
            xi_t[t] = 0.5 * (xi_t[t] + xi_t[t].T)
        t00_series = per_node_t00_series(xi_t, psi_t, k_t, q_t)
        dens_series = per_node_xi_density(xi_t)
        # last-snap p99 of T_00 -> matter-core nodes
        t00_last = t00_series[-1]
        thr = float(np.percentile(t00_last, 99.0))
        core_mask = t00_last > thr
        all_corr_lag0 = []
        core_corr_lag0 = []
        core_corr_lagm1 = []
        core_corr_lagp1 = []
        for a in range(n_lat):
            r0 = lagged_corr(dens_series[:, a],
                              t00_series[:, a], 0)
            if math.isfinite(r0):
                all_corr_lag0.append(r0)
                if core_mask[a]:
                    core_corr_lag0.append(r0)
                    rm = lagged_corr(dens_series[:, a],
                                       t00_series[:, a], -1)
                    rp = lagged_corr(dens_series[:, a],
                                       t00_series[:, a], +1)
                    if math.isfinite(rm):
                        core_corr_lagm1.append(rm)
                    if math.isfinite(rp):
                        core_corr_lagp1.append(rp)
        s_summary = {
            "seed": s,
            "n_core_nodes": int(core_mask.sum()),
            "all_nodes_lag0_mean": float(np.mean(all_corr_lag0)),
            "all_nodes_lag0_std":  float(np.std(all_corr_lag0)),
            "core_nodes_lag0_mean":
                float(np.mean(core_corr_lag0))
                if core_corr_lag0 else float("nan"),
            "core_nodes_lagm1_mean":
                float(np.mean(core_corr_lagm1))
                if core_corr_lagm1 else float("nan"),
            "core_nodes_lagp1_mean":
                float(np.mean(core_corr_lagp1))
                if core_corr_lagp1 else float("nan"),
            "core_lead_lag_asymmetry_neg_minus_pos":
                ((float(np.mean(core_corr_lagm1))
                  - float(np.mean(core_corr_lagp1)))
                 if core_corr_lagm1 and core_corr_lagp1
                 else float("nan")),
        }
        seed_summaries.append(s_summary)
        if math.isfinite(s_summary["core_nodes_lag0_mean"]):
            p99_lag_zero_per_seed.append(
                s_summary["core_nodes_lag0_mean"])
        if math.isfinite(s_summary["core_nodes_lagm1_mean"]):
            p99_lag_minus1_per_seed.append(
                s_summary["core_nodes_lagm1_mean"])
        if math.isfinite(s_summary["core_nodes_lagp1_mean"]):
            p99_lag_plus1_per_seed.append(
                s_summary["core_nodes_lagp1_mean"])
        full_lag_zero_per_seed.append(
            s_summary["all_nodes_lag0_mean"])

    p99_lag0 = np.array(p99_lag_zero_per_seed)
    p99_lagm1 = np.array(p99_lag_minus1_per_seed)
    p99_lagp1 = np.array(p99_lag_plus1_per_seed)
    full_lag0 = np.array(full_lag_zero_per_seed)

    boot_lag0_means = []
    boot_lagm1_means = []
    boot_lagp1_means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(p99_lag0), size=len(p99_lag0))
        boot_lag0_means.append(p99_lag0[idx].mean())
        if len(p99_lagm1) > 0:
            idx2 = rng.integers(0, len(p99_lagm1),
                                  size=len(p99_lagm1))
            boot_lagm1_means.append(p99_lagm1[idx2].mean())
        if len(p99_lagp1) > 0:
            idx3 = rng.integers(0, len(p99_lagp1),
                                  size=len(p99_lagp1))
            boot_lagp1_means.append(p99_lagp1[idx3].mean())
    boot_lag0_arr = np.array(boot_lag0_means)
    ci_lag0 = (float(np.percentile(boot_lag0_arr, 2.5)),
               float(np.percentile(boot_lag0_arr, 97.5)))

    summary = {
        "method": "verify_heavy_tail_lead_lag_p5n512",
        "schema_version": "1.0.0",
        "regime": "P5N512",
        "n_seeds": int(n_seeds),
        "n_snapshots_per_seed": int(n_snap),
        "n_lat": int(n_lat),
        "all_nodes_lag0_mean":
            float(full_lag0.mean()),
        "all_nodes_lag0_std":
            float(full_lag0.std()),
        "core_p99_lag0_mean":
            float(p99_lag0.mean()) if len(p99_lag0) else None,
        "core_p99_lag0_std":
            float(p99_lag0.std()) if len(p99_lag0) else None,
        "core_p99_lag0_bootstrap_CI95":
            list(ci_lag0),
        "core_p99_lagm1_mean":
            float(p99_lagm1.mean()) if len(p99_lagm1) else None,
        "core_p99_lagp1_mean":
            float(p99_lagp1.mean()) if len(p99_lagp1) else None,
        "core_p99_lead_lag_asymmetry_neg_minus_pos":
            (float(p99_lagm1.mean() - p99_lagp1.mean())
             if len(p99_lagm1) and len(p99_lagp1) else None),
        "core_p99_lag0_excludes_zero":
            bool(ci_lag0[0] > 0.0),
        "per_seed_summaries": seed_summaries,
    }
    out = (REPO / "outputs"
           / "verify_heavy_tail_lead_lag_p5n512.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2),
                   encoding="utf-8")

    print()
    print("=" * 72)
    print("Heavy-tail lead-lag audit on P5N512 12-seed 13-snapshot")
    print("=" * 72)
    print(f"  all-nodes  lag-0 corr mean = "
          f"{summary['all_nodes_lag0_mean']:+.3f} +- "
          f"{summary['all_nodes_lag0_std']:.3f}")
    print(f"  core-p99  lag-0 corr mean = "
          f"{summary['core_p99_lag0_mean']:+.3f}  "
          f"CI95 = [{ci_lag0[0]:+.3f}, {ci_lag0[1]:+.3f}]")
    print(f"  core-p99  lag-1 corr mean = "
          f"{summary['core_p99_lagm1_mean']:+.3f}  "
          f"(Xi-density leads T_00)")
    print(f"  core-p99  lag+1 corr mean = "
          f"{summary['core_p99_lagp1_mean']:+.3f}  "
          f"(T_00 leads Xi-density)")
    print(f"  core-p99  asymmetry (lag-1 - lag+1) = "
          f"{summary['core_p99_lead_lag_asymmetry_neg_minus_pos']:+.3f}  "
          f"(positive = nucleation reading)")
    print(f"  core lag-0 excludes zero (CI95): "
          f"{summary['core_p99_lag0_excludes_zero']}")
    print(f"  saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
