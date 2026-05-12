"""Regenerate paper/figures/fig1_xi_to_metric.pdf.

Two design constraints addressed in this version:

1. Bubble text must fit inside the bubble (the previous version had
   "relational similarity" and "audit-positive" bleeding outside the
   box borders).
2. The bottom box must show the consequence that the caption claims:
   emergent-Einstein closure with gamma_PPN = beta_PPN = 1 in the
   closure domain, not only the bulk Einstein equation.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib
matplotlib.use("Agg")  # headless CI / no-DISPLAY
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType (vector, arXiv-friendly)
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt

OUT = (Path(__file__).resolve().parent.parent
       / "paper" / "figures" / "fig1_xi_to_metric.pdf")
OUT_PNG = OUT.with_suffix(".png")


def make_fig():
    fig, ax = plt.subplots(figsize=(11.0, 5.2), dpi=160)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Four top-row stage boxes. Widths chosen so the longest text fits
    # comfortably inside each box with margin on both sides.
    stages = [
        (0.30, 2.10, 2.40, 1.10,
         r"$\Xi_{ij}\in[0,1]$"
         "\n"
         r"relational input"),
        (3.00, 2.10, 2.70, 1.10,
         r"$d_{ij} = -\ell_{0}\,\log\Xi_{ij}$"
         "\n"
         r"$M_{0}$--$M_{3}$ metric"),
        (5.95, 2.10, 2.85, 1.10,
         "fast-slow gradient flow"
         "\n"
         "quasi-metric tube"),
        (9.10, 2.10, 2.60, 1.10,
         "CLP scores A--D"
         "\n"
         "continuum-limit audit"),
    ]
    for x, y, w, h, label in stages:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05", linewidth=1.2,
            edgecolor="black", facecolor="white"))
        ax.text(x + w/2, y + h/2, label,
                ha="center", va="center", fontsize=10)

    # Arrows between top-row stages (positioned at the midpoint between
    # adjacent boxes, at the same vertical centre y=2.65).
    gaps = [(2.70, 3.00), (5.70, 5.95), (8.80, 9.10)]
    for x0, x1 in gaps:
        ax.annotate("",
                    xy=(x1, 2.65), xytext=(x0, 2.65),
                    arrowprops=dict(arrowstyle="->", linewidth=1.5))

    # Bottom box: Einstein closure with PPN identification, matching
    # the caption text exactly.
    ax.add_patch(mpatches.FancyBboxPatch(
        (1.60, 0.30), 8.80, 1.30,
        boxstyle="round,pad=0.05", linewidth=1.2,
        edgecolor="black", facecolor="#f0f0f0"))
    ax.text(
        6.00, 1.10,
        r"$G_{\mu\nu} + \Lambda^{\mathrm{back}}_{\mu\nu}"
        r" = 8\pi G\,T^{\Xi}_{\mu\nu}$"
        r"   (bulk-percentile sense; $p_{99}/\!\sup$ matter-core support)",
        ha="center", va="center", fontsize=10)
    ax.text(
        6.00, 0.55,
        r"emergent-Einstein closure: "
        r"$\gamma_{\mathrm{PPN}} = \beta_{\mathrm{PPN}} = 1$"
        r" in the closure domain",
        ha="center", va="center", fontsize=10, style="italic")

    # Vertical arrow from the metric stage down into the bottom box.
    ax.annotate("",
                xy=(6.00, 1.65), xytext=(6.00, 2.10),
                arrowprops=dict(arrowstyle="->", linewidth=1.5))

    # Title at the top of the figure.
    ax.text(
        6.00, 4.50,
        "Four-stage closure: relational input "
        r"$\to$ metric $\to$ stabilised geometry "
        r"$\to$ continuum-limit audit "
        r"$\to$ emergent-Einstein residual",
        ha="center", va="center", fontsize=11, style="italic")

    fig.tight_layout()
    fig.savefig(OUT, format="pdf", bbox_inches="tight")
    fig.savefig(OUT_PNG, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    make_fig()
