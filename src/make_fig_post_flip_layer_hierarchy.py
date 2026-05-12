"""Render the POST-flip 5-diagnostic layer hierarchy figure.

Reads outputs/stage6h_layer_shell_structure_test.json and
produces a 2x2 panel:
  (A) all 5 diagnostics overlaid as f(shell rank n=1..5)
      with PRE and POST sub-curves
  (B) shell-pattern fit: best-model overlay (1/sqrt(n), 1/n,
      1/n^2) for the strongest diagnostic |rho_shift|
  (C) slaving-K ratio per layer for PRE vs POST
  (D) per-regime curves for |rho_shift| (5 PRE + 3 POST regimes)

Output: paper/figures/fig_post_flip_layer_hierarchy.{pdf,png}
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
IN = REPO / "outputs" / "stage6h_layer_shell_structure_test.json"
OUT_PDF = REPO / "paper" / "figures" / "fig_post_flip_layer_hierarchy.pdf"
OUT_PNG = REPO / "paper" / "figures" / "fig_post_flip_layer_hierarchy.png"

LAYER_LABELS = [r"$\mathcal{C}^{(\sup)}$", r"$\mathcal{C}^{(99.9)}$",
                  r"$\mathcal{C}^{(99.5)}$", r"$\mathcal{C}^{(99)}$",
                  r"$\mathcal{C}^{(95)}$"]


def main() -> int:
    bundle = json.loads(IN.read_text())
    pre = bundle["phase_summary"]["PRE"]["sequences"]
    post = bundle["phase_summary"]["POST"]["sequences"]

    fig = plt.figure(figsize=(13.5, 9.0))
    gs = fig.add_gridspec(
        2, 2, hspace=0.42, wspace=0.30,
        left=0.07, right=0.985, top=0.93, bottom=0.07,
    )
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    n_arr = np.arange(1, 6)

    # ── Panel A: 5 diagnostics overlaid (POST) with PRE for comparison ─
    diag_keys = ["|rho_shift|", "R<0_fraction", "psi_sq_fraction",
                  "slaving_K_ratio", "t00_dev_ratio"]
    diag_labels = [r"$|\Delta\rho|$",
                    r"$f(R_{00}\!<\!0)$",
                    r"$|\psi|^{2}/M_{\rm tot}$",
                    r"$\delta_{K,\rm layer}/\delta_{K,\rm lat}$",
                    r"$|T_{00}\!-\!\bar T_{00}|/\rm lat$"]
    colours = ["#C73E1D", "#F18F01", "#3FA7D6", "#7B2CBF", "#2A9D8F"]

    # Normalise each diagnostic to its n=5 (POST) value for overlay
    for dn, lab, col in zip(diag_keys, diag_labels, colours):
        post_v = np.array(post[dn])
        pre_v = np.array(pre[dn])
        # Normalise by max-magnitude in POST for overlay clarity
        norm_post = float(np.max(np.abs(post_v)))
        norm_pre = float(np.max(np.abs(pre_v)))
        if norm_post < 1e-12: norm_post = 1.0
        if norm_pre < 1e-12: norm_pre = 1.0
        ax_A.plot(n_arr, post_v / norm_post, marker="o", lw=1.8,
                   ms=6, color=col, label=lab + "  (POST)")
        ax_A.plot(n_arr, pre_v / norm_pre, marker="s", ls="--",
                   lw=1.0, ms=4, color=col, alpha=0.45,
                   label=lab + "  (PRE)")

    ax_A.set_xticks(n_arr)
    ax_A.set_xticklabels([f"$n={i}$\n{LAYER_LABELS[i-1]}" for i in n_arr],
                          fontsize=8)
    ax_A.set_ylabel(r"normalised by $\max|\cdot|$ in phase")
    ax_A.set_title(r"(A) Five diagnostics across shell rank $n$, "
                    r"PRE (dashed) vs POST (solid)")
    ax_A.grid(True, alpha=0.25)
    ax_A.legend(loc="upper right", framealpha=0.9, fontsize=7, ncol=2)
    ax_A.axvline(1.5, color="gray", ls=":", lw=0.8, alpha=0.5)

    # ── Panel B: |rho_shift| with model overlays (POST) ──
    rho_post = np.array(post["|rho_shift|"])
    ax_B.plot(n_arr, rho_post, marker="o", lw=2.0, ms=7, color="#C73E1D",
               label=r"empirical $|\Delta\rho|$ (POST)")
    # Three candidate scalings, normalised to match the n=5 endpoint
    for model_name, model_fn, c in [
        (r"$1/\sqrt{n}$", lambda x: 1.0 / np.sqrt(x), "#3FA7D6"),
        (r"$1/n$",         lambda x: 1.0 / x,         "#F18F01"),
        (r"$1/n^{2}$",     lambda x: 1.0 / x ** 2,    "#7B2CBF"),
    ]:
        m = model_fn(n_arr)
        # Anchor at n=5 endpoint of the empirical curve
        scale = rho_post[-1] / m[-1]
        ax_B.plot(n_arr, scale * m, marker="x", ls="--", lw=1.3,
                   color=c, label=model_name + " (anchored at $n\!=\!5$)")
    ax_B.set_xticks(n_arr)
    ax_B.set_xticklabels(LAYER_LABELS, fontsize=8)
    ax_B.set_ylabel(r"$|\Delta\rho|$  (POST)")
    ax_B.set_title(r"(B) $|\Delta\rho|$ vs candidate $n$-scalings: "
                    r"$1/\sqrt{n}$ best fit")
    ax_B.grid(True, alpha=0.25)
    ax_B.legend(loc="upper right", framealpha=0.9, fontsize=8)

    # ── Panel C: slaving-K layer/lattice ratio per phase ──
    sla_pre = np.array(pre["slaving_K_ratio"])
    sla_post = np.array(post["slaving_K_ratio"])
    ax_C.plot(n_arr, sla_post, marker="o", lw=2.0, ms=7, color="#C73E1D",
               label="POST (3 regimes)")
    ax_C.plot(n_arr, sla_pre, marker="s", lw=1.0, ms=5, ls="--",
               color="#3FA7D6", label="PRE (5 regimes)")
    ax_C.axhline(1.0, color="black", ls=":", lw=0.8, alpha=0.6,
                  label="lattice average")
    ax_C.set_xticks(n_arr)
    ax_C.set_xticklabels(LAYER_LABELS, fontsize=8)
    ax_C.set_ylabel(r"$\langle|K-K_{\rm slaved}|\rangle_{\rm layer}\,/\,\rm lat$")
    ax_C.set_title("(C) Slaving-$K$ residual ratio per layer "
                    "(unslavedness)")
    ax_C.grid(True, alpha=0.25)
    ax_C.legend(loc="upper right", framealpha=0.9)

    # ── Panel D: per-regime |rho_shift| ──
    regimes = bundle["regimes"]
    pre_regs = [r for r in regimes if r["phase"] == "PRE"]
    post_regs = [r for r in regimes if r["phase"] == "POST"]
    layers_seq = ["Csup", "C99_9", "C99_5", "C99", "C95"]
    for r in pre_regs:
        vals = [abs(r["layers"][L]["rho_shift"]) for L in layers_seq]
        ax_D.plot(n_arr, vals, marker="s", ls="--", lw=0.9,
                   alpha=0.55, color="#3FA7D6")
    for r in post_regs:
        vals = [abs(r["layers"][L]["rho_shift"]) for L in layers_seq]
        ax_D.plot(n_arr, vals, marker="o", lw=1.4, ms=6,
                   color="#C73E1D", label=r["regime"])
    ax_D.set_xticks(n_arr)
    ax_D.set_xticklabels(LAYER_LABELS, fontsize=8)
    ax_D.set_ylabel(r"$|\Delta\rho|$ per regime")
    ax_D.set_title(r"(D) Per-regime curves: PRE (dashed blue) "
                    r"vs POST (solid red)")
    ax_D.grid(True, alpha=0.25)
    ax_D.legend(loc="upper right", framealpha=0.9, fontsize=8)

    fig.suptitle(
        r"POST-flip layer hierarchy on $\mathcal{C}^{(p)}\!=\!\{\Delta_{a}\!>\!a_{p}\}$: "
        r"five diagnostics, slaving-$K$ identification, $\rho$-anti-coherence",
        fontsize=12.5, y=0.97,
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
