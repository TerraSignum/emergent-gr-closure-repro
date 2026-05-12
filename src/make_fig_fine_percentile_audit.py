"""Render the bulk-to-matter-core transition figure on the
fine-percentile audit (15 percentile positions across the cleaned
eleven-regime canonical-physics ladder N in [50,512]).

Reads outputs/stage6f_fine_percentile_audit.json and produces a
four-panel figure:
  (A) per-regime values of Delta_rho(N) for all 15 percentiles
      (log-log axes, percentile colour gradient)
  (B) power-law alpha per percentile with 95% bootstrap CI bars
      (colour-coded by classification)
  (C) Symanzik-2 y_inf per percentile with 95% bootstrap CI bars
  (D) classification strip (BULK_BOUND / TRANSITION / MATTER_CORE)

Output: paper/figures/fig_fine_percentile_audit.{pdf,png}
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
IN = REPO / "outputs" / "stage6f_fine_percentile_audit.json"
OUT_PDF = REPO / "paper" / "figures" / "fig_fine_percentile_audit.pdf"
OUT_PNG = REPO / "paper" / "figures" / "fig_fine_percentile_audit.png"

PERC_ORDER = [
    "median", "mean",
    "p90", "p91", "p92", "p93", "p94", "p95",
    "p96", "p97", "p98", "p99",
    "p99_5", "p99_9", "sup",
]

PERC_LABELS = {
    "median": r"$p_{50}$",
    "mean": r"$\overline{\Delta}$",
    "p90": r"$p_{90}$",
    "p91": r"$p_{91}$",
    "p92": r"$p_{92}$",
    "p93": r"$p_{93}$",
    "p94": r"$p_{94}$",
    "p95": r"$p_{95}$",
    "p96": r"$p_{96}$",
    "p97": r"$p_{97}$",
    "p98": r"$p_{98}$",
    "p99": r"$p_{99}$",
    "p99_5": r"$p_{99.5}$",
    "p99_9": r"$p_{99.9}$",
    "sup": r"$\sup$",
}

CLS_COLOUR = {
    "BULK_BOUND": "#2E86AB",
    "TRANSITION": "#F18F01",
    "MATTER_CORE": "#C73E1D",
}


def _percentile_to_xpos(idx, n):
    return idx


def _format_status(pc):
    return (
        "BULK_BOUND" if pc in ("median", "mean", "p90", "p91", "p92",
                                "p93", "p94", "p95", "p96", "p97")
        else ("MATTER_CORE" if pc in ("p99_5", "p99_9", "sup")
              else "TRANSITION")
    )


def main() -> int:
    bundle = json.loads(IN.read_text())
    fits = bundle["fits"]
    regimes = bundle["regimes"]
    cls_map = bundle.get("classification") or {
        pc: _format_status(pc) for pc in PERC_ORDER
    }

    fig = plt.figure(figsize=(12.0, 8.6))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.0, 1.0, 0.18],
        hspace=0.45, wspace=0.28,
        left=0.075, right=0.985, top=0.945, bottom=0.085,
    )
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])
    ax_strip = fig.add_subplot(gs[2, :])

    # ── Panel A: per-regime values (log-log) ─────────────────
    cmap = plt.get_cmap("viridis")
    n_pc = len(PERC_ORDER)
    n_arr = np.array([r["N"] for r in regimes], dtype=float)
    for i, pc in enumerate(PERC_ORDER):
        y = np.array([r["percentiles"][pc] for r in regimes], dtype=float)
        col = cmap(i / (n_pc - 1))
        ax_A.loglog(n_arr, y, marker="o", ms=4.5, lw=1.0,
                     color=col, label=PERC_LABELS[pc])
    ax_A.axhline(0.05, color="gray", ls=":", lw=0.9,
                  alpha=0.7, label="closure threshold $0.05$")
    ax_A.axvline(173.2, color="purple", ls="--", lw=0.9,
                  alpha=0.55, label=r"chirality flip $N_{\mathrm{flip}}\!\approx\!173$")
    ax_A.set_xlabel(r"lattice extension $N$")
    ax_A.set_ylabel(r"$\Delta_{\rho}(N)$")
    ax_A.set_title(r"(A) Per-regime $\Delta_{\rho}(N)$ across the ladder")
    ax_A.grid(True, which="both", alpha=0.25)
    handles, labels = ax_A.get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, labels)
             if l.startswith("$") or "threshold" in l or "flip" in l]
    sel = [(h, l) for h, l in keep
            if l in (
                PERC_LABELS["median"], PERC_LABELS["p90"],
                PERC_LABELS["p95"], PERC_LABELS["p99"],
                PERC_LABELS["p99_5"], PERC_LABELS["sup"],
                "closure threshold $0.05$",
                r"chirality flip $N_{\mathrm{flip}}\!\approx\!173$",
            )]
    ax_A.legend(*zip(*sel), loc="lower left", framealpha=0.9, ncol=2)

    # ── Panel B: alpha vs percentile ─────────────────────────
    xs = np.arange(n_pc)
    a_pt, a_lo, a_hi, a_col = [], [], [], []
    for i, pc in enumerate(PERC_ORDER):
        v = fits[pc]
        a_pt.append(v["alpha"])
        a_lo.append(v["alpha_95_lo"])
        a_hi.append(v["alpha_95_hi"])
        a_col.append(CLS_COLOUR[cls_map.get(pc, _format_status(pc))])
    a_pt = np.array(a_pt)
    a_lo = np.array(a_lo)
    a_hi = np.array(a_hi)
    err_lo = a_pt - a_lo
    err_hi = a_hi - a_pt
    for x, p, eL, eH, c in zip(xs, a_pt, err_lo, err_hi, a_col):
        ax_B.errorbar(x, p, yerr=[[eL], [eH]],
                       fmt="o", color=c, ecolor=c, ms=6, capsize=3,
                       elinewidth=1.4)
    ax_B.axhline(0.0, color="black", lw=0.9, alpha=0.6)
    ax_B.set_xticks(xs)
    ax_B.set_xticklabels([PERC_LABELS[p] for p in PERC_ORDER],
                          rotation=45, ha="right")
    ax_B.set_ylabel(r"power-law $\hat\alpha$ (point + $95\%$ CI)")
    ax_B.set_title(r"(B) Power-law decay exponent $\hat\alpha$")
    ax_B.grid(True, axis="y", alpha=0.25)
    ax_B.set_ylim(-0.2, 2.5)

    # ── Panel C: y_inf vs percentile ─────────────────────────
    y_pt, y_lo, y_hi = [], [], []
    for pc in PERC_ORDER:
        v = fits[pc]
        y_pt.append(v["y_inf"])
        y_lo.append(v["y_inf_95_lo"])
        y_hi.append(v["y_inf_95_hi"])
    y_pt = np.array(y_pt)
    y_lo = np.array(y_lo)
    y_hi = np.array(y_hi)
    e_lo = y_pt - y_lo
    e_hi = y_hi - y_pt
    for x, p, eL, eH, c in zip(xs, y_pt, e_lo, e_hi, a_col):
        ax_C.errorbar(x, p, yerr=[[eL], [eH]],
                       fmt="s", color=c, ecolor=c, ms=6, capsize=3,
                       elinewidth=1.4)
    ax_C.axhline(0.0, color="black", lw=0.9, alpha=0.6)
    ax_C.axhline(0.05, color="gray", ls=":", lw=0.9, alpha=0.7,
                  label=r"closure threshold $0.05$")
    ax_C.set_xticks(xs)
    ax_C.set_xticklabels([PERC_LABELS[p] for p in PERC_ORDER],
                          rotation=45, ha="right")
    ax_C.set_ylabel(r"Symanzik-2 $y_\infty$ (point + $95\%$ CI)")
    ax_C.set_title(r"(C) Continuum asymptote $y_\infty$")
    ax_C.grid(True, axis="y", alpha=0.25)
    ax_C.legend(loc="upper left", framealpha=0.9)

    # ── Panel D: per-regime closure status visualisation ─────
    # Show per-percentile sweep for each regime on a heatmap-like
    # view: x = percentile, y = N, colour = log10 Delta.
    pct_vals = np.array([
        [r["percentiles"][pc] for pc in PERC_ORDER] for r in regimes
    ])
    log_pct = np.log10(np.maximum(pct_vals, 1e-4))
    im = ax_D.imshow(log_pct, aspect="auto", origin="lower",
                       cmap="magma", vmin=-2.0, vmax=0.05)
    ax_D.set_xticks(xs)
    ax_D.set_xticklabels([PERC_LABELS[p] for p in PERC_ORDER],
                          rotation=45, ha="right")
    ax_D.set_yticks(np.arange(len(regimes)))
    ax_D.set_yticklabels([f"{r['regime']} (N={r['N']})" for r in regimes])
    ax_D.set_title(r"(D) $\log_{10}\Delta_{\rho}(N)$ heatmap")
    cbar = fig.colorbar(im, ax=ax_D, fraction=0.04, pad=0.02)
    cbar.set_label(r"$\log_{10}\Delta$")

    # ── Strip: classification ────────────────────────────────
    width = 1.0
    for x, pc in enumerate(PERC_ORDER):
        c = CLS_COLOUR[cls_map.get(pc, _format_status(pc))]
        ax_strip.barh(0, width, left=x - 0.5, height=0.6, color=c,
                       edgecolor="white", linewidth=1.0)
    ax_strip.set_xticks(xs)
    ax_strip.set_xticklabels([PERC_LABELS[p] for p in PERC_ORDER],
                              rotation=45, ha="right")
    ax_strip.set_yticks([])
    ax_strip.set_xlim(-0.5, n_pc - 0.5)
    ax_strip.set_ylim(-0.4, 0.4)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=CLS_COLOUR["BULK_BOUND"]),
        plt.Rectangle((0, 0), 1, 1, color=CLS_COLOUR["TRANSITION"]),
        plt.Rectangle((0, 0), 1, 1, color=CLS_COLOUR["MATTER_CORE"]),
    ]
    legend_labels = [
        r"BULK_BOUND ($\alpha_{95\%\mathrm{lo}}\!>\!0$ "
        r"$\wedge\ y_{\infty,95\%\mathrm{hi}}\!<\!0.05$)",
        r"TRANSITION (one criterion only)",
        r"MATTER_CORE ($y_{\infty,95\%\mathrm{lo}}\!>\!0$, saturated)",
    ]
    ax_strip.legend(legend_handles, legend_labels,
                     loc="upper center", bbox_to_anchor=(0.5, -0.6),
                     ncol=3, frameon=False)
    ax_strip.spines["top"].set_visible(False)
    ax_strip.spines["right"].set_visible(False)
    ax_strip.spines["left"].set_visible(False)
    ax_strip.spines["bottom"].set_visible(False)

    fig.suptitle(
        r"Fine-percentile bulk-to-matter-core transition on the cleaned "
        r"eleven-regime ladder $N\!\in\![50,512]$",
        fontsize=12.5, y=0.985,
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
