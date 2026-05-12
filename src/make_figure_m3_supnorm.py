"""Two-panel figure for the M3 sup-norm diagnostics.

Left panel: L^2 vs L^infinity residual on the within-P5 ladder
            N in {50, 64, 72, 84, 100, 128, 200, 300}, comparing
            the bundled stabilisation (V1) with the max-path
            envelope refinement (V2).

Right panel: heavy-tail correlation per N -- fraction of top-10
             worst-case triples that overlap the T_00 top-decile
             of the lattice, with random-expectation reference.

Output: paper/figures/fig_m3_supnorm.pdf
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

REPO = Path(__file__).resolve().parent.parent


def main():
    # Multi-N L^2 + sup data (V1 = current, V2 = max-path post-processing)
    multi_n = json.loads((REPO / "outputs" /
                          "audit_M3_violations_multi_N.json").read_text())
    Ns = np.array([r["N"] for r in multi_n["ladder"]], dtype=float)
    L2_v1 = np.array([r["penalty_residual_L2_mean"]
                       for r in multi_n["ladder"]])
    sup_v1 = np.array([r["penalty_residual_sup_mean"]
                        for r in multi_n["ladder"]])

    # V2 from envelope variants audit (only on subset of regimes)
    env = json.loads((REPO / "outputs" /
                      "audit_M3_envelope_variants.json").read_text())
    env_lookup = {r["N"]: r["variants"] for r in env["rows"]}
    sup_v2 = []
    L2_v2 = []
    for N in Ns:
        v = env_lookup.get(int(N))
        if v and "V2_max_path" in v:
            sup_v2.append(v["V2_max_path"]["sup"])
            L2_v2.append(v["V2_max_path"]["L2"])
        else:
            sup_v2.append(np.nan)
            L2_v2.append(np.nan)
    sup_v2 = np.array(sup_v2)
    L2_v2 = np.array(L2_v2)

    # Heavy-tail correlation per N from sup-norm diagnostics
    diag = json.loads((REPO / "outputs" /
                       "audit_M3_supnorm_diagnostics.json").read_text())
    ht_N = np.array([r["N"] for r in diag["rows"]], dtype=float)
    ht_frac = np.array([r["heavy_tail_correlation"]["fraction"]
                         for r in diag["rows"]])
    random_exp = diag["rows"][0]["heavy_tail_correlation"]["expected_random"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # === LEFT: L^2 vs L^infty trend ===
    axL.plot(Ns, sup_v1, "o-", color="#cc3333", lw=1.6, ms=8,
             label=r"V1 (bundled): $G_{\Delta}^{\sup}$")
    axL.plot(Ns, sup_v2, "s-", color="#cc3333", lw=1.6, ms=8,
             alpha=0.55, fillstyle="none",
             label=r"V2 (max-path post-processing): $G_{\Delta}^{\sup}$")
    axL.plot(Ns, L2_v1, "o--", color="#1f77b4", lw=1.4, ms=6,
             label=r"V1: $G_{\Delta}^{L^{2}}$")
    axL.plot(Ns, L2_v2, "s--", color="#1f77b4", lw=1.4, ms=6,
             alpha=0.55, fillstyle="none",
             label=r"V2: $G_{\Delta}^{L^{2}}$")

    axL.set_xlabel(r"lattice size $N$", fontsize=11)
    axL.set_ylabel(r"M3 violation residual",
                   fontsize=11)
    axL.set_xscale("log")
    axL.set_yscale("log")
    axL.set_ylim(1e-3, 1.0)
    axL.grid(True, which="both", alpha=0.25)
    axL.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    axL.set_title(r"(a) Multi-$N$ M3 violation: $L^{2}$ converges, "
                   r"$L^{\sup}$ stays bounded under V1; V2 envelope refinement"
                   "\n"
                   r"reduces $L^{\sup}$ by $\sim 40\%$ with monotone $N$-decay",
                   fontsize=10.5, pad=8)

    # === RIGHT: heavy-tail correlation per N ===
    width = 0.6
    bars = axR.bar(np.arange(len(ht_N)), ht_frac * 100,
                   width=width, color="#2c7a2c", edgecolor="black", lw=1.0,
                   label="observed: top-10 worst-case triples in $T_{00}$ top-decile")
    axR.axhline(random_exp * 100, color="#cc3333", ls="--", lw=1.4,
                label=fr"random expectation $1-0.9^{{3}}\approx{random_exp*100:.1f}\%$")
    for bar, val in zip(bars, ht_frac):
        axR.text(bar.get_x() + bar.get_width() / 2, val * 100 + 1.5,
                 f"{val*100:.0f}%", ha="center", va="bottom", fontsize=9)
    axR.set_xticks(np.arange(len(ht_N)))
    axR.set_xticklabels([f"$N={int(n)}$" for n in ht_N],
                        rotation=20, ha="right", fontsize=9)
    axR.set_ylabel(r"fraction of top-10 worst-case triples"
                   "\n"
                   r"with index in $T_{00}$ top-decile (\%)",
                   fontsize=10.5)
    axR.set_ylim(0, 100)
    axR.grid(True, axis="y", alpha=0.25)
    axR.legend(loc="upper right", fontsize=9, framealpha=0.95)
    axR.set_title(r"(b) Heavy-tail correlation: M3 sup-norm violations are"
                   "\n"
                   r"matter-cluster-localised at $2$--$3\times$ above random null",
                   fontsize=10.5, pad=8)

    fig.suptitle("M3 sub-multiplicative-triangle violations on the "
                 r"within-$P_{5}$ multi-$N$ ladder",
                 fontsize=12.5, y=1.005)
    plt.tight_layout()

    out_dir = REPO / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "fig_m3_supnorm.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_dir / "fig_m3_supnorm.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
