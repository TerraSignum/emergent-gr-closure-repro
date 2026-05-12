"""Render the geometric identification figure of the Delta-residual
core hierarchy.

Reads
  outputs/stage6h_matter_core_phase_diagram.json
  outputs/stage6h_core_geometric_identification.json
and produces a three-panel figure:

  (A) Core-mass-fraction M(C^p)/M_total per regime, log-scale
      (shows hard-core 1/N scaling vs source-active saturation)
  (B) Layer-wise Jaccard overlap with five geometric markers
      (clear separation: M4 R_00 + M5 T_00-dev high, M1 M2 M3 low)
  (C) Layer-wise R_00 < 0 fraction across regimes
      (signed-halo signature)

Output: paper/figures/fig_core_geometric_identification.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 130,
})

REPO = Path(__file__).resolve().parent.parent
IN_PHASE = REPO / "outputs" / "stage6h_matter_core_phase_diagram.json"
IN_GEOM = REPO / "outputs" / "stage6h_core_geometric_identification.json"
OUT_PDF = REPO / "paper" / "figures" / "fig_core_geometric_identification.pdf"
OUT_PNG = REPO / "paper" / "figures" / "fig_core_geometric_identification.png"

LAYERS = ["C95", "C99", "C99_5", "C99_9", "Csup"]
LAYER_LABELS = {
    "C95": r"$\mathcal{C}^{(95)}$",
    "C99": r"$\mathcal{C}^{(99)}$",
    "C99_5": r"$\mathcal{C}^{(99.5)}$",
    "C99_9": r"$\mathcal{C}^{(99.9)}$",
    "Csup": r"$\mathcal{C}^{(\sup)}$",
}
LAYER_COLOURS = {
    "C95": "#2E86AB",
    "C99": "#3FA7D6",
    "C99_5": "#F18F01",
    "C99_9": "#C73E1D",
    "Csup": "#7F1D1D",
}

MARKERS = [
    ("M1", "M1_jacc_psi_low5", r"$|\psi|^{2}$ bot-$5\%$" + "\n(vortex)"),
    ("M2", "M2_jacc_grad_top5", "phase-grad top-$5\\%$\n(domain-walls)"),
    ("M3", "M3_jacc_coh_low5", "local-coh bot-$5\\%$\n(phase-incoh.)"),
    ("M4", "M4_jacc_R00_top5", r"$|R_{00}|$ top-$5\%$" + "\n(energy-resid.)"),
    ("M5", "M5_jacc_t00dev_top5", r"$|T_{00}\!-\!\overline{T}_{00}|$" + "\ntop-$5\\%$ (matter-dev.)"),
]


def main() -> int:
    phase = json.loads(IN_PHASE.read_text())
    geom = json.loads(IN_GEOM.read_text())

    fig = plt.figure(figsize=(13.0, 5.2))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.0, 1.15, 1.0],
        wspace=0.32, left=0.06, right=0.985,
        top=0.90, bottom=0.18,
    )
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])

    # ── Panel A: core-mass-fraction per regime ───────────────
    regimes = phase["regimes"]
    n_arr = np.array([r["N"] for r in regimes])
    for layer in LAYERS:
        m = np.array([r[layer]["core_mass_psi_sq_frac"] for r in regimes])
        ax_A.semilogy(n_arr, np.maximum(m, 1e-4), marker="o", lw=1.4,
                      ms=5, color=LAYER_COLOURS[layer],
                      label=LAYER_LABELS[layer])
    ax_A.set_xlabel(r"lattice extension $N$")
    ax_A.set_ylabel(r"$M(\mathcal{C}^{(p)})\,/\,M_{\mathrm{tot}}$")
    ax_A.set_title(r"(A) Layer mass-fraction (matter $|\psi|^{2}$ on layer)")
    ax_A.grid(True, which="both", alpha=0.25)
    ax_A.legend(loc="lower left", framealpha=0.9, ncol=2)

    # ── Panel B: Jaccard overlap with geometric markers ──────
    summary = geom["cross_regime_summary"]
    n_layers = len(LAYERS)
    n_markers = len(MARKERS)
    width = 0.16
    x_pos = np.arange(n_markers)
    for i, layer in enumerate(LAYERS):
        vals = [summary[layer][m_key] for _, m_key, _ in MARKERS]
        offset = (i - (n_layers - 1) / 2) * width
        ax_B.bar(x_pos + offset, vals, width=width,
                 color=LAYER_COLOURS[layer],
                 label=LAYER_LABELS[layer])
    ax_B.axhline(0.05, color="gray", ls=":", lw=1.0,
                  alpha=0.7, label=r"random-overlap level $0.05$")
    ax_B.set_xticks(x_pos)
    ax_B.set_xticklabels([m_lab for _, _, m_lab in MARKERS], fontsize=8)
    ax_B.set_ylabel("Jaccard overlap (cross-regime mean)")
    ax_B.set_title("(B) Geometric identification of the core layers")
    ax_B.set_ylim(0, 0.85)
    ax_B.grid(True, axis="y", alpha=0.25)
    ax_B.legend(loc="upper left", framealpha=0.9, fontsize=8, ncol=2)

    # ── Panel C: R_00 < 0 fraction (signed halo) ─────────────
    for r in regimes:
        # plot R_time<0 fraction across layers
        vals = [r[layer]["frac_R_time_negative"] for layer in LAYERS]
        ax_C.plot(np.arange(n_layers), vals, marker="o", ms=4, lw=1.0,
                   alpha=0.6, color="#666666")
    # cross-regime mean line
    means = []
    for layer in LAYERS:
        v = [r[layer]["frac_R_time_negative"] for r in regimes]
        means.append(float(np.mean(v)))
    ax_C.plot(np.arange(n_layers), means, marker="s", ms=8, lw=2.5,
              color="#C73E1D", label="cross-regime mean")
    ax_C.axhline(0.5, color="black", ls="--", lw=0.8, alpha=0.6,
                  label=r"random sign $1/2$")
    ax_C.set_xticks(np.arange(n_layers))
    ax_C.set_xticklabels([LAYER_LABELS[l] for l in LAYERS])
    ax_C.set_ylabel(r"fraction $R_{00}\!<\!0$ on layer")
    ax_C.set_title("(C) Signed-halo signature on the layers")
    ax_C.set_ylim(0.3, 1.0)
    ax_C.grid(True, axis="y", alpha=0.25)
    ax_C.legend(loc="lower right", framealpha=0.9, fontsize=8)

    fig.suptitle(
        r"Closure-defect hierarchy: layer-nested $\mathcal{C}^{(p)}\!=\!\{\Delta_{a}\!>\!a_{p}\}$ "
        r"on the cleaned eleven-regime ladder $N\!\in\![50,512]$",
        fontsize=12, y=0.985,
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
