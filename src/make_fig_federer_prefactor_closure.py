"""Generate fig:federer_prefactor_closure: 3-panel comprehensive figure.

  Panel A: empirical c, d vs closure values (PRECISE tier)
  Panel B: |d|/c ratio empirical vs (1+gamma) closure
  Panel C: Lambda_t per regime over 12-point ladder vs alpha_xi^2 = 0.81
           (direct (A1) anchor empirical confirmation)
  Panel D: AICc comparison: closure vs free vs Federer-only
           (statistical preference of parameter-free closure)

Output: paper/figures/fig_federer_prefactor_closure.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless CI / no-DISPLAY
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType (vector, arXiv-friendly)
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np


# Empirical M3 fit
C_EMPIRICAL = 0.161304
D_EMPIRICAL = -0.176953
RATIO_EMPIRICAL = abs(D_EMPIRICAL) / C_EMPIRICAL

# Closure prediction
C_PREDICTED = 81.0 / 500.0
D_PREDICTED = -891.0 / 5000.0
RATIO_PREDICTED = 11.0 / 10.0

ALPHA_XI_SQ = 0.81


def main() -> None:
    repo = Path(__file__).resolve().parents[1]

    # Load Lambda_t per regime + AICc data
    pf = json.load(open(repo.parent / "audit"
                          / "federer_physical_factorization_results_2026_05_04.json"))
    el = json.load(open(repo.parent / "audit"
                          / "federer_extended_ladder_results_2026_05_04.json"))

    # ── Build figure (2x2 grid) ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    ax_A, ax_B = axes[0]
    ax_C, ax_D = axes[1]

    # ── Panel A: c, d ──
    labels = ["$c$ (leading)", "$d$ (sub-leading)"]
    emp = [C_EMPIRICAL, D_EMPIRICAL]
    pred = [C_PREDICTED, D_PREDICTED]
    x = np.arange(len(labels))
    w = 0.35
    ax_A.bar(x - w / 2, emp, w, label="empirical (10-pt fit)",
              color="#3c6ea7", edgecolor="black", linewidth=0.6)
    ax_A.bar(x + w / 2, pred, w,
              label=r"closure $2\gamma\Lambda_t$, $-(1+\gamma)c$",
              color="#d97f4a", edgecolor="black", linewidth=0.6)
    for i, (e, p) in enumerate(zip(emp, pred)):
        diff_pct = (e - p) / p * 100
        ax_A.annotate(f"{diff_pct:+.2f}%",
                       xy=(i, max(e, p) + 0.005 if p > 0 else min(e, p) - 0.015),
                       ha="center", fontsize=9, fontstyle="italic", color="#444")
    ax_A.set_xticks(x); ax_A.set_xticklabels(labels)
    ax_A.set_ylabel("prefactor value")
    ax_A.set_title(r"\textbf{(A)} Federer prefactors: empirical vs closure",
                     loc="left")
    ax_A.axhline(0, color="black", linewidth=0.5)
    ax_A.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax_A.grid(True, axis="y", alpha=0.3)

    # ── Panel B: |d|/c ratio ──
    bars = ax_B.bar(["empirical", r"$1+\gamma$"],
                     [RATIO_EMPIRICAL, RATIO_PREDICTED],
                     color=["#3c6ea7", "#d97f4a"],
                     edgecolor="black", linewidth=0.6, width=0.55)
    band_low = RATIO_PREDICTED * 0.99
    band_high = RATIO_PREDICTED * 1.01
    ax_B.axhspan(band_low, band_high, color="#d97f4a", alpha=0.15,
                  label="PRECISE band (1\\%)")
    diff_pct = (RATIO_EMPIRICAL - RATIO_PREDICTED) / RATIO_PREDICTED * 100
    ax_B.annotate(f"{diff_pct:+.2f}%", xy=(0, RATIO_EMPIRICAL + 0.005),
                   ha="center", fontsize=10, fontstyle="italic")
    ax_B.annotate("$=11/10$", xy=(1, RATIO_PREDICTED + 0.005),
                   ha="center", fontsize=10, color="#5a3010")
    ax_B.set_ylabel(r"$|d|/c$")
    ax_B.set_title(r"\textbf{(B)} Sub-leading-to-leading ratio $|d|/c$",
                     loc="left")
    ax_B.set_ylim(1.05, 1.13)
    ax_B.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax_B.grid(True, axis="y", alpha=0.3)

    # ── Panel C: Lambda_t per regime ──
    regimes = list(pf["per_regime"].keys())
    Ns_pf = np.array([pf["per_regime"][r]["N"] for r in regimes])
    L_t = np.array([pf["per_regime"][r]["lambda_t_emp"] for r in regimes])
    order = np.argsort(Ns_pf)
    Ns_pf, L_t = Ns_pf[order], L_t[order]
    regimes_sorted = [regimes[i] for i in order]

    ax_C.scatter(Ns_pf, L_t, s=40, color="#3c6ea7",
                  edgecolor="black", linewidth=0.5,
                  label=r"$\Lambda_t^{\mathrm{emp}}(N)$ on bulk-percentile",
                  zorder=3)
    ax_C.axhline(ALPHA_XI_SQ, color="#d97f4a", linewidth=1.5,
                  linestyle="--",
                  label=r"$\alpha_\xi^2 = 81/100$ (closure target)",
                  zorder=2)
    ax_C.axhspan(ALPHA_XI_SQ * 0.95, ALPHA_XI_SQ * 1.05,
                  color="#d97f4a", alpha=0.15,
                  label=r"$\pm 5\%$ band", zorder=1)
    for n, l, r in zip(Ns_pf, L_t, regimes_sorted):
        ax_C.annotate(r, (n, l), fontsize=7,
                       xytext=(3, 4), textcoords="offset points")
    ax_C.set_xlabel(r"$N$")
    ax_C.set_ylabel(r"$\Lambda_t$")
    ax_C.set_xscale("log")
    ax_C.set_title(r"\textbf{(C)} (A1) anchor: $\Lambda_t^{\mathrm{emp}}\!=\!\alpha_\xi^2$ across $12$ regimes",
                     loc="left")
    ax_C.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax_C.grid(True, alpha=0.3)
    ax_C.set_ylim(0.76, 0.86)

    # ── Panel D: AICc bars ──
    aicc_closure = el["tests"]["T_A_closure"]["aicc"]
    aicc_free = el["tests"]["T_B_free_M3"]["aicc"]
    aicc_only = el["tests"]["T_C_federer_only"]["aicc"]
    rmse_closure = el["tests"]["T_A_closure"]["rmse"]
    rmse_free = el["tests"]["T_B_free_M3"]["rmse"]
    rmse_only = el["tests"]["T_C_federer_only"]["rmse"]

    models = ["closure\n($0$ free)", "free $M3$\n($2$ free)",
              "Federer-only\n($1$ free)"]
    aiccs = [aicc_closure, aicc_free, aicc_only]
    colors = ["#d97f4a", "#3c6ea7", "#7a8ca0"]
    bars_d = ax_D.bar(models, aiccs, color=colors,
                       edgecolor="black", linewidth=0.6, width=0.55)
    for bar, val, rmse in zip(bars_d, aiccs, [rmse_closure, rmse_free, rmse_only]):
        ax_D.annotate(f"AICc $={val:+.2f}$\nRMSE $={rmse*1000:.2f}\\!\\cdot\\!10^{{-3}}$",
                       xy=(bar.get_x() + bar.get_width() / 2,
                           val + (1 if val < -160 else -3)),
                       ha="center", fontsize=8.5)
    ax_D.set_ylabel("AICc")
    ax_D.set_title(r"\textbf{(D)} Statistical preference: closure beats free fit",
                     loc="left")
    ax_D.invert_yaxis()  # lower AICc is better - put it at top
    ax_D.grid(True, axis="y", alpha=0.3)
    delta_TA_TB = aicc_closure - aicc_free
    ax_D.text(0.5, 0.05, f"$\\Delta\\mathrm{{AICc}}$(closure $-$ free) $= {delta_TA_TB:+.2f}$",
                transform=ax_D.transAxes, ha="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbe0",
                          edgecolor="black"))

    plt.tight_layout()
    out_dir = repo / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "fig_federer_prefactor_closure.pdf"
    png = out_dir / "fig_federer_prefactor_closure.png"
    plt.savefig(pdf, dpi=200, bbox_inches="tight")
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf.relative_to(repo)}")
    print(f"Saved: {png.relative_to(repo)}")


if __name__ == "__main__":
    main()
