"""Generate fig13: heavy-tail residual localisation at high-T_00
nodes.

Two-panel figure:
  (a) Per-node scatter (residual vs T_00) for three representative
      regimes (P5 N=50, P5N100 N=100, P8 N=84), showing that the
      top-decile-residual nodes (red) cluster at the high-T_00 end
      of the scatter.
  (b) Cross-regime overlap-lift trace: lift=N(top10-residual AND
      top10-T_00) / E[random] for each regime, with a horizontal
      line at lift=1 (no enrichment). Shows the systematic
      enrichment with z >> 3 at every regime.
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

from _d1_npz_discovery import find_d1_npz
from verify_galerkin_runner_A_hessian_ricci import (
    edge_to_matrix, per_seed_galerkin)


def per_node_residual_struct(prep, xp):
    g_00 = prep["g_00_h"]
    g_ij = prep["g_ij_h"]
    t00 = prep["t00"]
    t_ij = prep["t_ij"]
    eye3 = prep["eye3"]
    LAMBDA_T = 0.81
    LAMBDA_S = -0.005
    res00 = g_00 + LAMBDA_T - t00
    spatial_res = (g_ij + LAMBDA_S * eye3[None, :, :]) - t_ij
    sq = res00 ** 2 + (spatial_res ** 2).sum(axis=(1, 2))
    return xp.sqrt(sq)


def gather_per_node(regime, n_lat):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    keys = set(d.files)
    if "dense_cell_edge_xi_values" in keys:
        edge_arr = d["dense_cell_edge_xi_values"]
        amp_arr = d["dense_cell_node_amplitude_values"]
        phase_arr = d["dense_cell_node_phase_values"]
        n_seeds = min(edge_arr.shape[0], 32)
        get_xi = lambda s: edge_to_matrix(edge_arr[s], n_lat)
        get_psi = lambda s: amp_arr[s] * np.exp(1j * phase_arr[s])
    elif "edge_xi_snapshots" in keys:
        snaps = d["edge_xi_snapshots"]
        psi_re = d["psi_real_snapshots"]
        psi_im = d["psi_imag_snapshots"]
        last = snaps.shape[1] - 1
        n_seeds = min(snaps.shape[0], 32)
        get_xi = lambda s: np.asarray(snaps[s, last], dtype=float).copy()
        get_psi = lambda s: (np.asarray(psi_re[s, last], dtype=float)
                              + 1j*np.asarray(psi_im[s, last], dtype=float))
    else:
        return None
    res_all = []
    t00_all = []
    for s in range(n_seeds):
        xi_mat = get_xi(s)
        np.fill_diagonal(xi_mat, 1.0)
        psi = get_psi(s)
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        res = np.asarray(per_node_residual_struct(prep, np))
        t00 = np.asarray(prep["t00"])
        res_all.append(res)
        t00_all.append(t00)
    return np.concatenate(res_all), np.concatenate(t00_all)


def main():
    LADDER_REGIMES = [
        ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
        ("P4", 42), ("P5", 50), ("P6", 60), ("P5N64", 64),
        ("P7", 72), ("P5N72", 72), ("P8", 84), ("P5N84", 84),
        ("P5N100", 100), ("P5N128", 128), ("P5N200", 200),
        ("P5N256", 256), ("P5N300", 300), ("P5N512", 512),
    ]
    SCATTER_REGIMES = [("P5", 50), ("P8", 84), ("P5N100", 100)]

    # Load lift/z data from the audit.
    with open(REPO / "outputs"
              / "defect_candidates_consolidated_audit.json") as f:
        audit = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                              gridspec_kw={"width_ratios": [3, 2]})

    # Panel (a): per-node scatter
    ax = axes[0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (regime, n_lat), color in zip(SCATTER_REGIMES, colors):
        data = gather_per_node(regime, n_lat)
        if data is None:
            continue
        res, t00 = data
        p90_res = np.percentile(res, 90)
        top_mask = res >= p90_res
        # Bottom 90% in lighter color
        ax.scatter(t00[~top_mask], res[~top_mask],
                    c=color, s=10, alpha=0.25,
                    label=f"{regime} ($N={n_lat}$) bot 90%")
        # Top 10% in saturated color with red edge
        ax.scatter(t00[top_mask], res[top_mask],
                    c=color, s=30, alpha=0.95,
                    edgecolors="red", linewidths=0.5,
                    label=f"{regime} ($N={n_lat}$) top 10% (red edge)")
    ax.axhline(0.05, color="gray", linestyle=":",
                label="closure threshold 0.05")
    ax.set_xlabel(r"per-node source energy density $T_{00}^{\Xi}(a)$")
    ax.set_ylabel(r"per-node 4$\times$4 Frobenius residual $\Delta_E^{\mathrm{true}}(a)$")
    ax.set_yscale("log")
    ax.set_title("Heavy-tail residual is matter-localised: scatter "
                  "at three representative regimes")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    # Panel (b): lift / z vs N
    ax = axes[1]
    by_regime = {r["regime"]: r for r in audit["per_regime"]}
    ns = []
    lifts = []
    zs = []
    for regime, n_lat in LADDER_REGIMES:
        if regime not in by_regime:
            continue
        rec = by_regime[regime]
        ov = rec["overlap_top10_residual_vs_top10_t00"]
        ns.append(n_lat)
        lifts.append(ov["lift"])
        zs.append(ov["z_score"])
    ns = np.array(ns)
    lifts = np.array(lifts)
    zs = np.array(zs)

    ax2 = ax.twinx()
    p1 = ax.bar(ns, lifts, width=4, color="#1f77b4", alpha=0.6,
                 label="lift (top10 res $\\cap$ top10 $T_{00}$ / random)")
    p2 = ax2.plot(ns, zs, "o-", color="#d62728",
                   label="z-score")
    ax.axhline(1.0, color="gray", linestyle=":",
                label="random null lift = 1")
    ax2.axhline(3.0, color="red", linestyle="--", alpha=0.5)
    ax2.text(105, 3.2, "$z=3$", color="red", fontsize=9)
    ax.set_xlabel(r"lattice size $N$")
    ax.set_ylabel("overlap lift")
    ax2.set_ylabel("z-score")
    ax.set_title("Cross-regime overlap statistic")
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
               loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_pdf = REPO / "paper" / "figures" / "fig13_heavy_tail_t00.pdf"
    out_png = REPO / "paper" / "figures" / "fig13_heavy_tail_t00.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    print(f"Cross-regime mean lift: {lifts.mean():.2f}")
    print(f"Cross-regime mean z-score: {zs.mean():+.1f}")


if __name__ == "__main__":
    main()
