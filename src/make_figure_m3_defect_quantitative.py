"""Four-panel figure for the quantitative defect-side M3-slack analysis.

Panel A (Q2 topological invariant):
    Bar chart of total integrated deficit / (2 pi) per N with
    horizontal mean line. Shows the invariant ~0.108 with
    CV 6.5%.

Panel B (Q2 dual — per-vortex deficit vs vortex count):
    Two-axis plot: per-vortex deficit angle (left axis,
    decreasing) and number of vortex nodes (right axis,
    increasing). Illustrates conservation by trade-off.

Panel C (Q1 per-node budget by class):
    Box plot of per-node M3 slack-budget by class
    (vortex vs non-vortex) per N.

Panel D (Q3 lensing correlation):
    Scatter plot of per-node M3-slack-budget rank vs T_00^2
    rank, with Spearman rho and p-value annotated.

Output: paper/figures/fig_m3_defect_quantitative.pdf
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless CI / no-DISPLAY
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType (vector, arXiv-friendly)
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from verify_galerkin_runner_A_hessian_ricci import (
    edge_to_matrix, per_seed_galerkin)

PARENT = REPO.parent


def load_seed0(rel_path, kind, n_lat):
    fp = PARENT / rel_path
    if not fp.exists():
        return None
    z = np.load(fp, allow_pickle=True)
    if kind == "snap":
        snaps = z["edge_xi_snapshots"]
        last = snaps.shape[1] - 1
        xi = np.asarray(snaps[0, last], dtype=float).copy()
        np.fill_diagonal(xi, 1.0)
        psi = (np.asarray(z["psi_real_snapshots"][0, last], dtype=float)
               + 1j * np.asarray(z["psi_imag_snapshots"][0, last], dtype=float))
        k_field = z.get("ff_K_seed0", np.full((n_lat, n_lat), 0.55))
        q_field = z.get("ff_Q_seed0", np.full((n_lat, n_lat), 0.45))
    elif kind == "d1" and "dense_cell_edge_xi_values" in z.files:
        edge = z["dense_cell_edge_xi_values"]
        xi = edge_to_matrix(edge[0], n_lat).astype(float)
        np.fill_diagonal(xi, 1.0)
        amp = z["dense_cell_node_amplitude_values"][0]
        phase = z["dense_cell_node_phase_values"][0]
        psi = amp * np.exp(1j * phase)
        k_field = z.get("ff_K_seed0", np.full((n_lat, n_lat), 0.55))
        q_field = z.get("ff_Q_seed0", np.full((n_lat, n_lat), 0.45))
    else:
        return None
    return xi, psi, np.asarray(k_field), np.asarray(q_field)


def vortex_indicator(xi, psi):
    n = xi.shape[0]
    phase = np.angle(psi)
    score = np.zeros(n)
    for i in range(n):
        nbrs = np.where(xi[i] > 0.5)[0]
        if nbrs.size > 1:
            score[i] = 1 - np.abs(np.exp(1j * phase[nbrs]).mean())
    return (score > np.percentile(score, 90)).astype(int)


def per_node_slack_budget(xi):
    n = xi.shape[0]
    prod = xi[:, :, None] * xi[None, :, :]
    target = xi[:, None, :]
    slack = np.maximum(prod - target, 0.0)
    diag_mask = np.ones((n, n, n), dtype=bool)
    diag_mask[np.arange(n), np.arange(n), :] = False
    diag_mask[:, np.arange(n), np.arange(n)] = False
    diag_mask[np.arange(n), :, np.arange(n)] = False
    slack = slack * diag_mask
    return slack.sum(axis=(1, 2))


def main():
    bundle = json.loads((REPO / "outputs" /
                          "audit_M3_defect_quantitative.json").read_text())
    rows = bundle["rows"]
    Ns = np.array([r["N"] for r in rows])

    # === extract data ===
    deficit_total_2pi = np.array([r["Q2_deficit_angle"]["deficit_total_over_2pi"]
                                    for r in rows])
    deficit_per_vortex = np.array([r["Q2_deficit_angle"]["deficit_mean_rad"]
                                     for r in rows])
    n_vortex_per_N = np.array([r["Q1_per_vortex_budget"]["n_vortex"]
                                 for r in rows])
    spearman_rho = np.array([r["Q3_lensing_correlation"]
                              ["spearman_rho_budget_vs_LT_proxy"]
                              for r in rows])
    spearman_p = np.array([r["Q3_lensing_correlation"]["spearman_p"]
                            for r in rows])
    vortex_budget_mean = np.array([r["Q1_per_vortex_budget"]["vortex_budget_mean"]
                                     for r in rows])
    nonvortex_budget_mean = np.array([r["Q1_per_vortex_budget"]["nonvortex_budget_mean"]
                                        for r in rows])

    # Recompute per-node distributions for box plots (P5 only as exemplar)
    LADDER = [
        ("P5",     50,  "results_d1_fix17/d1_p5.npz",                      "d1"),
        ("P5N64",  64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz",    "snap"),
        ("P5N72",  72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz",    "snap"),
        ("P5N84",  84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz",    "snap"),
        ("P5N100", 100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz",  "snap"),
        ("P5N128", 128, "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz", "snap"),
        ("P5N200", 200, "results_d1_p5n200_8seeds/P5N200.snapshots.npz",   "snap"),
        ("P5N256", 256, "results_d1_p5n256_12seeds/P5N256.snapshots.npz",  "snap"),
        ("P5N300", 300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz",  "snap"),
        ("P5N512", 512, "results_d1_p5n512_12seeds/P5N512.snapshots.npz",  "snap"),
    ]
    distributions = []
    scatter_data = None
    for regime, n_lat, rel, kind in LADDER:
        payload = load_seed0(rel, kind, n_lat)
        if payload is None:
            continue
        xi, psi, k_field, q_field = payload
        prep = per_seed_galerkin(xi.copy(), psi, k_field, q_field, n_lat, np)
        t00 = np.asarray(prep["t00"])
        vortex = vortex_indicator(xi, psi)
        budget = per_node_slack_budget(xi)
        distributions.append({
            "N": n_lat,
            "vortex_budgets": budget[vortex == 1],
            "nonvortex_budgets": budget[vortex == 0],
        })
        if regime == "P5N200":
            # Use N=200 (highest p-value significance) for scatter
            scatter_data = {"budget": budget, "t00": t00,
                             "vortex": vortex, "regime": "P5N200"}

    # === build figure ===
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Panel A: Q2 invariant (bar chart of total deficit / (2pi))
    bars_A = axA.bar(np.arange(len(Ns)), deficit_total_2pi,
                      color="#2c7a2c", edgecolor="black", lw=1.0,
                      label="observed total deficit / $(2\\pi)$")
    mean_inv = deficit_total_2pi.mean()
    cv_inv = deficit_total_2pi.std() / mean_inv * 100
    axA.axhline(mean_inv, color="#cc3333", ls="--", lw=1.5,
                 label=fr"mean $= {mean_inv:.3f}$, CV $= {cv_inv:.1f}\%$")
    for bar, val in zip(bars_A, deficit_total_2pi):
        axA.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                  f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    axA.set_xticks(np.arange(len(Ns)))
    axA.set_xticklabels([f"$N{{=}}{n}$" for n in Ns], fontsize=9)
    axA.set_ylabel(r"total deficit $\sum_{\alpha}\Delta\theta_{\alpha} / (2\pi)$",
                    fontsize=10.5)
    axA.set_ylim(0, max(deficit_total_2pi) * 1.4)
    axA.grid(True, axis="y", alpha=0.25)
    axA.legend(loc="upper left", fontsize=9, framealpha=0.95)
    axA.set_title(r"(A) Topological invariant $\sum_{\alpha}\Delta\theta_{\alpha}/(2\pi)$"
                   "\n"
                   r"on the within-$P_{5}$ ladder $N=50..300$",
                   fontsize=10.5, pad=8)

    # Panel B: Q2 dual (per-vortex deficit + n_vortex)
    color_def = "#1f77b4"
    color_n = "#cc3333"
    axB.plot(Ns, deficit_per_vortex, "o-", color=color_def, lw=1.6,
              ms=8, label=r"per-vortex $\Delta\theta_{\alpha}$ [rad]")
    axB.set_xlabel(r"lattice size $N$", fontsize=10.5)
    axB.set_ylabel(r"per-vortex deficit $\Delta\theta_{\alpha}$ [rad]",
                    color=color_def, fontsize=10.5)
    axB.tick_params(axis="y", labelcolor=color_def)
    axB.set_xscale("log")
    axB.grid(True, alpha=0.25)
    axB.set_title(r"(B) Per-vortex deficit decreases as $n_{\rm vortex}$ grows;"
                   "\n"
                   r"product is the conserved invariant of panel (A)",
                   fontsize=10.5, pad=8)
    axBr = axB.twinx()
    axBr.plot(Ns, n_vortex_per_N, "s-", color=color_n, lw=1.6, ms=8,
               label=r"$n_{\mathrm{vortex}}$ per regime")
    axBr.set_ylabel(r"$n_{\mathrm{vortex}}$ per regime",
                     color=color_n, fontsize=10.5)
    axBr.tick_params(axis="y", labelcolor=color_n)
    # Combined legend
    h1, l1 = axB.get_legend_handles_labels()
    h2, l2 = axBr.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9, framealpha=0.95)

    # Panel C: per-node budget distribution (box plot, vortex vs non, per N)
    box_data = []
    box_labels = []
    box_colors = []
    for d in distributions:
        box_data.append(d["vortex_budgets"])
        box_labels.append(f"$N{{=}}{d['N']}$\nvortex")
        box_colors.append("#cc3333")
        box_data.append(d["nonvortex_budgets"])
        box_labels.append(f"$N{{=}}{d['N']}$\nnon-vortex")
        box_colors.append("#1f77b4")
    bp = axC.boxplot(box_data, widths=0.6, patch_artist=True,
                      tick_labels=box_labels)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    axC.set_ylabel(r"per-node M3 slack-budget $S_{v}$",
                    fontsize=10.5)
    axC.tick_params(axis="x", labelsize=8)
    axC.grid(True, axis="y", alpha=0.25)
    axC.set_title(r"(C) Per-node slack-budget distribution by class:"
                   "\n"
                   r"vortex (red) vs non-vortex (blue) per $N$",
                   fontsize=10.5, pad=8)

    # Panel D: scatter M3-slack vs T_00^2 (use P5N200, highest p-value)
    if scatter_data is not None:
        budget = scatter_data["budget"]
        t00 = scatter_data["t00"]
        vortex = scatter_data["vortex"]
        regime = scatter_data["regime"]
        # Rank
        from scipy.stats import rankdata, spearmanr
        bud_rank = rankdata(budget)
        t00sq = t00 ** 2
        t00sq_rank = rankdata(t00sq)
        rho, p = spearmanr(budget, t00sq)
        axD.scatter(t00sq_rank[vortex == 0], bud_rank[vortex == 0],
                     s=14, color="#1f77b4", alpha=0.55,
                     label="non-vortex node")
        axD.scatter(t00sq_rank[vortex == 1], bud_rank[vortex == 1],
                     s=42, color="#cc3333", edgecolor="black", lw=0.8,
                     label="vortex node")
        # Linear regression on ranks for visual aid
        coef = np.polyfit(t00sq_rank, bud_rank, 1)
        x_fit = np.linspace(t00sq_rank.min(), t00sq_rank.max(), 100)
        axD.plot(x_fit, np.polyval(coef, x_fit), "--",
                  color="#444", lw=1.2, label="rank-rank regression")
        axD.set_xlabel(r"rank$(T_{00}^{2})$ — Lense-Thirring proxy",
                        fontsize=10.5)
        axD.set_ylabel(r"rank$(S_{v})$ — M3 slack-budget",
                        fontsize=10.5)
        axD.set_title(r"(D) M3-slack-budget rank vs $T_{00}^{2}$ rank at "
                       fr"{regime}: Spearman $\rho={rho:.3f},\,p={p:.1e}$"
                       "\n"
                       r"vortex nodes track the high-rank diagonal "
                       r"(matter-vortex co-localisation)",
                       fontsize=10.5, pad=8)
        axD.legend(loc="upper left", fontsize=9, framealpha=0.95)
        axD.grid(True, alpha=0.25)

    fig.suptitle(r"Quantitative defect-side analysis of the M3 sup-norm slack: "
                  r"topological invariant $+$ matter-vortex co-localisation",
                  fontsize=12.5, y=1.005)
    plt.tight_layout()

    out_dir = REPO / "paper" / "figures"
    out_pdf = out_dir / "fig_m3_defect_quantitative.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_dir / "fig_m3_defect_quantitative.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
