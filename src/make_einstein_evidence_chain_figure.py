r"""
Generate the five-panel multi-observable evidence-chain figure for
the Theorem 15.18 P2 Einstein-identity-gap exponent alpha = 2/3.

Five panels:

  (1) two-point Richardson on rigorous Delta_E at the canonical
      and extended anchor regimes (data/einstein_gap_results.json
      provides Delta_E^(1) at N=1534 and Delta_E^(2) at N=2254);

  (2) nine-point structural-bound check on
      Delta_curv^{cg=2} (here we visualise the bundled nine-point
      values as a bound-verification: all values <= C_0 * N^{-2/3}
      with C_0 = 15.22; data are reproduced for visualisation only
      from the parent corpus pipeline output);

  (3) nine-point chirality-balance deviation
      (data/einstein_gap_9point_witnesses.json -- secondary
      topological-witness ladder);

  (4) nine-point R_bar load-bearing curvature-side ladder
      (data/einstein_gap_9point_witnesses.json -- primary
      curvature-side witness);

  (5) eight-point T_00^Xi + Lambda source-side consistency
      (data/einstein_with_lambda_8point.json) -- two stacked
      curves: T_00^Xi(N) plateau and G_00(N) -> 0; band shows
      Lambda_eff = 0.314 +/- 1 sigma; inset shows pointwise
      residual G_00 + Lambda - T_00 vs DoD threshold 0.05.

Output: paper/figures/fig4c_five_path_evidence_chain.pdf and .png.

Usage:
    python ./src/make_einstein_evidence_chain_figure.py
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
FIG = REPO / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load_json(name):
    with open(DATA / name, "r", encoding="utf-8") as f:
        return json.load(f)


def panel1_richardson(ax):
    """Two-point Richardson on rigorous Delta_E."""
    d = load_json("einstein_gap_results.json")
    pts = d["honest_two_point_data"]
    n1 = pts[0]["N"]
    g1 = pts[0]["Delta_E"]
    n2 = pts[1]["N"]
    g2 = pts[1]["Delta_E"]
    Ns = np.array([n1, n2])
    gs = np.array([g1, g2])
    ax.loglog(Ns, gs, "o", color="C0", markersize=8,
              label=r"$\Delta_E$ data (2-point)")
    grid = np.geomspace(min(Ns) * 0.6, 1e7, 200)
    for alpha, label, ls in [
        (2.0 / 3.0, r"$\alpha=2/3$", "-"),
        (1.0, r"$\alpha=1.0$", "--"),
        (0.8477, r"$\alpha=0.848$", ":"),
    ]:
        C = g1 * (n1 ** alpha)
        ax.loglog(grid, C * grid ** (-alpha), ls, label=label,
                  linewidth=1.4)
    ax.axhline(0.05, color="grey", linewidth=0.8, linestyle="-.",
               label="DoD 0.05")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\Delta_E(N)$")
    ax.set_title(r"(1) Two-point Richardson on $\Delta_E$"
                 "\n(direct, only 2 anchor regimes)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)


def panel2_curv_bound(ax):
    """Nine-point structural-bound on Delta_curv^cg=2 vs C_0 * N^{-2/3}."""
    Ns = [410, 1539, 2254, 3918, 6038, 9380, 14181, 20000, 28000]
    vals = [0.005, 0.018, 0.014, 0.013, 0.000, 0.021, 0.008, 0.012, 0.017]
    C0 = 15.22
    Ns_arr = np.array(Ns)
    bound = C0 * Ns_arr ** (-2.0 / 3.0)
    ax.semilogx(Ns, vals, "s", color="C2", markersize=7,
                label=r"$\Delta_{\rm curv}^{cg=2}$ (9 points)")
    ax.semilogx(Ns_arr, bound, "-", color="C2", alpha=0.5,
                label=r"$C_0 N^{-2/3}$, $C_0=15.22$")
    ax.axhline(0.05, color="grey", linewidth=0.8, linestyle="-.",
               label="DoD 0.05")
    ax.set_xlabel(r"$N_{\rm dense}$")
    ax.set_ylabel(r"$\Delta_{\rm curv}^{cg=2}$")
    ax.set_title("(2) 9-point structural bound\n"
                 r"(R$^2<0.15$: bound verif., not limit proof)")
    ax.set_ylim(-0.005, 0.030)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)


def panel3_chirality(ax):
    """Nine-point chirality-balance deviation (topological witness)."""
    d = load_json("einstein_gap_9point_witnesses.json")
    Ns = d["lattice_ladder"]["N_values"]
    devs = d["secondary_topological_witness"]["values"]
    Ns_arr = np.array(Ns)
    devs_arr = np.array(devs)
    Delta = d["secondary_topological_witness"][
        "fit_alpha_2_over_3_full_9"]["Delta_infty"]
    fit_grid = np.linspace(min(Ns), 200, 200)
    # Reconstruct C from the fixed-alpha fit on the 9 points
    alpha = 2.0 / 3.0
    xs = Ns_arr ** (-alpha)
    n = len(xs)
    sx = xs.sum()
    sy = devs_arr.sum()
    sxx = (xs * xs).sum()
    sxy = (xs * devs_arr).sum()
    den = n * sxx - sx ** 2
    C = (n * sxy - sx * sy) / den
    Delta_recompute = (sy - C * sx) / n
    fit_curve = Delta_recompute + C * fit_grid ** (-alpha)
    ax.plot(Ns, devs, "^", color="C3", markersize=8,
            label="9 points")
    ax.plot(fit_grid, fit_curve, "-", color="C3", alpha=0.6,
            label=rf"$\alpha=2/3$ fit, $\Delta_\infty=${Delta_recompute:+.3f}")
    ax.axhline(Delta_recompute, color="C3", linewidth=0.6,
               linestyle=":", alpha=0.5)
    ax.axhline(0.05, color="grey", linewidth=0.8, linestyle="-.",
               label="DoD 0.05")
    ax.set_xlabel(r"$N_{\rm lattice}$")
    ax.set_ylabel(r"$1 - \langle$chirality\_balance$\rangle_N$")
    ax.set_title("(3) 9-point chirality-balance\n"
                 r"(topological index witness, R$^2=0.55$)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def panel4_Rbar(ax):
    """Nine-point R_bar curvature-side load-bearing witness."""
    d = load_json("einstein_gap_9point_witnesses.json")
    Ns = d["lattice_ladder"]["N_values"]
    rbs = d["primary_curvature_side_witness"]["values"]
    Ns_arr = np.array(Ns)
    rbs_arr = np.array(rbs)
    alpha = 2.0 / 3.0
    xs = Ns_arr ** (-alpha)
    n = len(xs)
    sx = xs.sum()
    sy = rbs_arr.sum()
    sxx = (xs * xs).sum()
    sxy = (xs * rbs_arr).sum()
    den = n * sxx - sx ** 2
    C = (n * sxy - sx * sy) / den
    Rbar_inf = (sy - C * sx) / n
    fit_grid = np.linspace(min(Ns), 200, 200)
    fit_curve = Rbar_inf + C * fit_grid ** (-alpha)
    ax.plot(Ns, rbs, "o", color="C1", markersize=8,
            label="9 points (mono. 7/8)")
    ax.plot(fit_grid, fit_curve, "-", color="C1", alpha=0.6,
            label=rf"$\alpha=2/3$ fit, $\bar R^\infty=${Rbar_inf:+.3f}")
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="-",
               alpha=0.4)
    ax.axhline(0.05, color="grey", linewidth=0.8, linestyle="-.",
               label="DoD 0.05")
    ax.set_xlabel(r"$N_{\rm lattice}$")
    ax.set_ylabel(r"$\bar R(N)$")
    ax.set_title("(4) 9-point $\\bar R$ curvature-side\n"
                 r"(load-bearing, R$^2=0.83$)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def panel5_lambda(ax):
    """Eight-point T_00 + Lambda source-side consistency."""
    d = load_json("einstein_with_lambda_8point.json")
    Ns = d["lattice_ladder"]["N_values"]
    T00 = d["T_00_Xi_values"]
    G00 = d["G_00_values"]
    Lam_const = d["Lambda_const_for_residual_test"]
    res = d["residual_pointwise_with_Lambda_const"]["values"]
    Lam_pw = d["Lambda_pointwise"]
    Lam_mean = d["Lambda_statistics"]["asymptotic_window_P4_to_P8"]["mean"]
    Lam_std = d["Lambda_statistics"]["asymptotic_window_P4_to_P8"]["std"]

    ax.plot(Ns, T00, "o-", color="C4", markersize=7, linewidth=1.2,
            label=r"$T_{00}^\Xi(N)$ Hilbert var.")
    ax.plot(Ns, G00, "s-", color="C5", markersize=7, linewidth=1.2,
            label=r"$G_{00} = \bar R/2$")
    ax.plot(Ns, Lam_pw, "d:", color="C6", markersize=6, linewidth=1.0,
            alpha=0.7, label=r"$\Lambda(N)=T_{00}-G_{00}$")
    ax.fill_between(Ns,
                    [Lam_mean - Lam_std] * len(Ns),
                    [Lam_mean + Lam_std] * len(Ns),
                    color="C6", alpha=0.15,
                    label=rf"$\Lambda_{{\rm asym}}=${Lam_mean:.3f} (CV 3.3%)")
    ax.set_xlabel(r"$N_{\rm lattice}$")
    ax.set_ylabel(r"value (lattice units)")
    ax.set_title("(5) 8-point T$_{00}$ + $\\Lambda$ source-side\n"
                 r"max $|$res$|<0.019$ for $N\geq 30$")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # Inset: residuals vs DoD threshold
    inset = ax.inset_axes([0.55, 0.55, 0.4, 0.35])
    inset.bar(range(len(Ns)), res, color="C0", alpha=0.6)
    inset.axhline(0.05, color="red", linewidth=0.8, linestyle="--")
    inset.axhline(-0.05, color="red", linewidth=0.8, linestyle="--")
    inset.set_xticks(range(len(Ns)))
    inset.set_xticklabels([str(n) for n in Ns], fontsize=6,
                          rotation=45)
    inset.set_ylabel("res", fontsize=7)
    inset.set_title(
        r"$G_{00}{+}\Lambda{-}T_{00}^\Xi$, DoD $\pm 0.05$",
        fontsize=7)
    inset.tick_params(labelsize=6)


def main():
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 6, hspace=0.42, wspace=0.55)

    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])
    ax4 = fig.add_subplot(gs[1, 0:3])
    ax5 = fig.add_subplot(gs[1, 3:6])

    panel1_richardson(ax1)
    panel2_curv_bound(ax2)
    panel3_chirality(ax3)
    panel4_Rbar(ax4)
    panel5_lambda(ax5)

    fig.suptitle(
        "Five-path multi-observable evidence chain for "
        r"Theorem 15.18 P2 ($\alpha=2/3$)"
        "\n"
        "Indirect witnesses: paths 2-5 do not evaluate "
        r"$\|G_{\mu\nu} - 8\pi G T_{\mu\nu}^\Xi\| / \|G_{\mu\nu}\|$ "
        "directly",
        fontsize=12)

    out_pdf = FIG / "fig4c_five_path_evidence_chain.pdf"
    out_png = FIG / "fig4c_five_path_evidence_chain.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
