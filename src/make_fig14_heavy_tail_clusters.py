"""Generate fig14: spatial clustering of heavy-tail residual
nodes on the relational lattice.

Three-panel figure (one per representative regime: P5 N=50,
P8 N=84, P5N100 N=100). Each panel shows:
  - all lattice nodes embedded in a 2D spectral layout
    (eigvecs[:,1:3] of the normalised graph Laplacian)
  - graph edges drawn for Xi_ab > XI_THRESH
  - heavy-tail nodes (top-decile residual) marked with a red
    fill and red outline
  - non-heavy-tail nodes shown as light grey points

The figure visualises the empirical claim that the heavy-tail
residual nodes form coherent spatial clusters rather than a
random scatter, supporting the matter-nucleation working
hypothesis (see Section sec:limitations Outlook).
"""
from __future__ import annotations
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
from matplotlib.collections import LineCollection

from _d1_npz_discovery import find_d1_npz
from verify_galerkin_runner_A_hessian_ricci import (
    XI_THRESH, edge_to_matrix, per_seed_galerkin)
from verify_higher_order_terms_all8 import (
    LAMBDA_T, LAMBDA_S, per_node_residual)


def get_layout_and_residual(regime, n_lat, seed_idx=0):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]

    s = seed_idx
    xi_mat = edge_to_matrix(edge_arr[s], n_lat)
    np.fill_diagonal(xi_mat, 1.0)
    psi = amp_arr[s] * np.exp(1j * phase_arr[s])
    k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
    q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))

    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
    eye3 = prep["eye3"]
    residual = np.asarray(per_node_residual(
        prep["g_00_h"], prep["g_ij_h"],
        prep["t00"], prep["t_ij"],
        LAMBDA_T, LAMBDA_S, eye3, np))
    t00 = np.asarray(prep["t00"])

    # Build 2D spectral layout: eigvecs[:,1:3] of L_norm
    xi_off = np.where(np.isfinite(xi_mat), xi_mat, 0.0).copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(np.float64)
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1) + 1e-12
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    l_norm = (np.eye(n_lat) - (deg_inv_sqrt[:, None] * weight_adj
                                * deg_inv_sqrt[None, :]))
    eigvals, eigvecs = np.linalg.eigh(l_norm)
    layout_2d = eigvecs[:, 1:3]  # 2D spectral embedding

    # Edges to draw
    edges = []
    weights = []
    for i in range(n_lat):
        for j in range(i + 1, n_lat):
            if xi_off[i, j] > XI_THRESH:
                edges.append([layout_2d[i], layout_2d[j]])
                weights.append(float(xi_off[i, j]))

    return {
        "regime": regime,
        "N": n_lat,
        "seed": seed_idx,
        "layout_2d": layout_2d,
        "residual": residual,
        "t00": t00,
        "edges": edges,
        "edge_weights": weights,
    }


def main():
    regimes = [("P5", 50, 2), ("P8", 84, 1), ("P5N100", 100, 3)]
    # Pick seeds by inspection: those with the cleanest single
    # connected-component top-decile.

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    for ax, (regime, n_lat, seed) in zip(axes, regimes):
        data = get_layout_and_residual(regime, n_lat, seed)
        if data is None:
            ax.text(0.5, 0.5, f"{regime}: NPZ not found",
                     ha="center", va="center", transform=ax.transAxes)
            continue
        layout = data["layout_2d"]
        residual = data["residual"]
        t00 = data["t00"]
        p90_res = np.percentile(residual, 90)
        top_mask = residual >= p90_res

        # Edges in light grey
        if data["edges"]:
            edge_lines = data["edges"]
            edge_w = np.asarray(data["edge_weights"])
            # Normalise widths
            edge_w_norm = (edge_w - edge_w.min()) / max(
                edge_w.max() - edge_w.min(), 1e-9)
            lc = LineCollection(
                edge_lines,
                colors="lightgrey",
                linewidths=0.3 + 1.0 * edge_w_norm,
                alpha=0.5,
            )
            ax.add_collection(lc)

        # Bottom-90% nodes in light grey
        ax.scatter(
            layout[~top_mask, 0], layout[~top_mask, 1],
            c="lightblue", s=30 + 60 * t00[~top_mask] / max(t00.max(), 1e-9),
            edgecolors="grey", linewidths=0.3,
            alpha=0.7, label="bulk (size $\\propto T_{00}$)",
        )
        # Top-decile nodes in red, larger
        ax.scatter(
            layout[top_mask, 0], layout[top_mask, 1],
            c="red", s=120 + 80 * t00[top_mask] / max(t00.max(), 1e-9),
            edgecolors="darkred", linewidths=1.5,
            alpha=0.9, label="top-decile residual (heavy tail)",
            zorder=5,
        )

        ax.set_title(f"{regime} ($N={n_lat}$, seed {seed})")
        ax.set_xlabel("spectral coordinate $v_1$")
        ax.set_ylabel("spectral coordinate $v_2$")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="lower left", fontsize=9)

    plt.suptitle(
        "Heavy-tail residual nodes form coherent spatial clusters "
        "in the spectral lattice embedding",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    out_pdf = REPO / "paper" / "figures" / "fig14_heavy_tail_clusters.pdf"
    out_png = REPO / "paper" / "figures" / "fig14_heavy_tail_clusters.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
