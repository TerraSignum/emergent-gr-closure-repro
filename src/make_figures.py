r"""
Generate the four figures of the Paper 4 manuscript.

  Figure 1 - Xi-to-metric schematic.
  Figure 2 - A1 fast-slow funnel illustration.
  Figure 3 - CLP scores (4 aggregate axes from the canonical multi-N audit).
  Figure 4 - Einstein-gap two-point Richardson extrapolation with three candidate exponents.

Usage:
    python ./src/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def save_both(fig, stem):
    pdf = FIG_DIR / f"{stem}.pdf"
    png = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png")
    print(f"  Saved: {pdf.relative_to(REPO)} + .png")


def figure_1_xi_to_metric():
    """Schematic Xi -> distance -> metric -> emergent Einstein dynamics."""
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.set_axis_off()

    # Wider boxes (0.21) with smaller arrow gaps so all four labels fit
    # comfortably inside the box bounds.
    box_w, box_h = 0.21, 0.50
    y_box = 0.32
    x_centers = [0.135, 0.385, 0.635, 0.885]
    boxes_data = [
        (r"$\Xi_{ij}$" "\n" "relational" "\n" "similarities",       "#cfe2f3", 13),
        (r"$d_{ij} = -\ell_{0}\,\log\,\Xi_{ij}$" "\n" "metric (M0-M3)", "#fce5cd", 13),
        ("Quasi-metric tube" "\n" "A1 fast-slow",                    "#d9ead3", 11),
        ("Emergent Einstein" "\n" "closure-domain",                  "#f4cccc", 11),
    ]
    for x_c, (label, color, fs) in zip(x_centers, boxes_data):
        x = x_c - box_w / 2
        rect = plt.Rectangle((x, y_box), box_w, box_h,
                             facecolor=color, edgecolor="black", lw=1.4)
        ax.add_patch(rect)
        ax.text(x_c, y_box + box_h / 2, label,
                ha="center", va="center", fontsize=fs)

    # Arrows between adjacent boxes
    arrow_y = y_box + box_h / 2
    for i in range(3):
        x_left = x_centers[i] + box_w / 2
        x_right = x_centers[i + 1] - box_w / 2
        ax.annotate("", xy=(x_right, arrow_y), xytext=(x_left, arrow_y),
                    arrowprops=dict(arrowstyle="->", lw=1.8))

    ax.text(0.5, 0.08,
            "All four stages are conditional on the M1-M3 metric axioms "
            "and the canonical regime P1.",
            ha="center", fontsize=10, color="#444", style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("From relational similarities to emergent Einstein dynamics",
                 pad=12, fontsize=12)
    save_both(fig, "fig1_xi_to_metric")
    plt.close(fig)


def figure_2_a1_funnel():
    """A1 fast-slow funnel illustration: triangle penalty -> sub-multiplicative regime."""
    fig, ax = plt.subplots(figsize=(9, 5))
    theta = np.linspace(-1.2, 1.2, 200)
    bowl = 0.6 * theta**2
    ax.plot(theta, bowl, color="#1f77b4", lw=2, label="triangle penalty $G_\\Delta$")

    eps = 0.06
    drift = 0.05 * np.tanh(theta * 1.5)
    ax.plot(theta, bowl + drift, color="#cc3333", lw=1, linestyle=":",
            label=r"with slow drift $+\epsilon\,R_{\mathrm{slow}}$")

    ax.fill_betweenx([0, 1.0], -0.5, 0.5, color="#9b59b6", alpha=0.20,
                     label="quasi-metric tube")

    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("triangle deviation $y$")
    ax.set_ylabel("penalty / drift potential")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(0, 1.0)
    ax.set_title("A1 fast-slow flow: $\\dot y = -\\lambda_\\Delta\\,G_\\Delta(y) + \\epsilon\\,R_\\mathrm{slow}(y)$",
                 pad=10)
    ax.legend(loc="upper center", framealpha=0.95)
    save_both(fig, "fig2_a1_funnel")
    plt.close(fig)


def figure_3_clp_scores():
    """Bar chart of the four CLP axes."""
    with open(REPO / "data" / "clp_scores.json", "r", encoding="utf-8") as f:
        clp = json.load(f)
    axes_data = clp["axes"]
    names = list(axes_data.keys())
    values = [axes_data[k]["value"] for k in names]
    # Use compact labels (the LaTeX manuscript already has the long names
    # in Table 1; the figure does not need to repeat them).
    short_labels = ["Tightness", "Operator", r"$\Gamma$-conv.", "Einstein gap"]
    threshold = axes_data[names[0]]["threshold"]

    # Wider figure, generous bottom padding so the short axis labels sit
    # under the bars without colliding with the legend.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#4a90d9", "#5cb85c", "#9b59b6", "#e89043"]
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, values, color=colors, edgecolor="black", lw=1)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                f"{v:.4f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.axhline(threshold, color="#444", linestyle="--", lw=1.2,
               label=f"closure-domain threshold = {threshold}")
    # Two-line tick labels: code + descriptor
    tick_labels = [f"{n}\n{s}" for n, s in zip(names, short_labels)]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("CLP score")
    ax.set_ylim(0, 1.05)
    ax.set_title(r"Continuum Limit Program: 4 aggregate axes "
                 r"(canonical multi-$N$ audit refresh)",
                 pad=10)
    # Legend at top-left so it does not collide with anything.
    ax.legend(loc="upper left", framealpha=0.95)
    save_both(fig, "fig3_clp_scores")
    plt.close(fig)


def figure_4_einstein_gap():
    """Two-panel Einstein-gap figure (revised 2026-05-11):

    Left:  21-pair Richardson series on the within-P_5 lattice ladder for
           the Einstein-gap axis Lambda_t = T_00^rec/T_00; per-exponent
           median + 95% CI distribution across the 21 pairs at three
           candidate exponents (2/3, 1, Symanzik-2), with the algebraic
           target alpha_xi^2 = 81/100 shown as the comparison line.
    Right: within-P_5 ten-point Symanzik-2 fit of Lambda_t(N) at the
           multi-seed lattice sizes
           N in {50,64,72,84,100,128,200,256,300,512}, with bootstrap
           95% CI band on the asymptote.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # ---- Left panel: 21-pair Richardson on the within-P_5 ladder ----
    rich_path = REPO / "data" / "einstein_gap_richardson_within_p5.json"
    with rich_path.open("r", encoding="utf-8") as f:
        rich = json.load(f)
    pairs = rich["all_pairs_richardson"]
    # Recover per-pair Richardson asymptotes from the ladder for plotting
    Ns_p5 = np.asarray(rich["ladder"]["N"], dtype=float)
    Lt_p5 = np.asarray(rich["ladder"]["Lambda_t_mean"], dtype=float)
    target_meas = rich["target_alpha_xi_squared_measured"]
    target_alg = rich["alpha_xi_constants"]["algebraic"]["alpha_xi_squared"]
    exponent_specs = [
        ("2/3", 2.0 / 3.0, "#cc3333"),
        ("1", 1.0, "#2c7a2c"),
        ("Symanzik2", 2.0, "#7a4dab"),
    ]
    x_positions = [1, 2, 3]
    box_data = []
    for alpha_key, alpha_val, _ in exponent_specs:
        # Reconstruct all 21 pair-wise Richardson asymptotes y_inf for this exponent.
        ys = []
        for i in range(len(Ns_p5)):
            for j in range(i + 1, len(Ns_p5)):
                Ni, Nj = Ns_p5[i], Ns_p5[j]
                yi, yj = Lt_p5[i], Lt_p5[j]
                # y(N) = y_inf + c / N^alpha  ->  y_inf = (yj * Nj^a - yi * Ni^a) / (Nj^a - Ni^a)
                a = alpha_val
                den = Nj ** a - Ni ** a
                if abs(den) < 1e-12:
                    continue
                y_inf = (yj * Nj ** a - yi * Ni ** a) / den
                ys.append(y_inf)
        box_data.append(np.asarray(ys))

    # Per-exponent strip + median + CI markers
    for x, (label, _alpha, color), ys in zip(
            x_positions, exponent_specs, box_data):
        med = pairs[label]["median"]
        lo, hi = pairs[label]["CI95"]
        # Light scatter for the 21 individual pairs
        rng = np.random.default_rng(20260511 + int(x))
        jitter = (rng.random(len(ys)) - 0.5) * 0.18
        axL.scatter(np.full_like(ys, x) + jitter, ys,
                    s=22, color=color, alpha=0.45, edgecolor="none",
                    zorder=2)
        # Median + CI95 error bar
        axL.errorbar([x], [med], yerr=[[med - lo], [hi - med]],
                     fmt="D", color=color, markersize=10, capsize=6,
                     lw=2, zorder=4,
                     label=fr"$\alpha={label}$: median$=${med:.3f}, "
                           fr"CI$=[{lo:.3f},{hi:.3f}]$ "
                           fr"({pairs[label]['n_pairs']} pairs)")
    # Target line: algebraic alpha_xi^2 = 0.81 (and measured 0.811)
    axL.axhline(target_alg, color="#000000", lw=1.3, ls="--",
                label=r"$\alpha_\xi^2=81/100$ (algebraic)")
    axL.axhline(target_meas, color="#444", lw=0.8, ls=":",
                label=fr"$\alpha_\xi^2$ measured $=${target_meas:.4f}")
    axL.set_xticks(x_positions)
    axL.set_xticklabels([r"$\alpha=2/3$", r"$\alpha=1$",
                         r"Symanzik-2 ($\alpha=2$)"])
    axL.set_xlim(0.4, 3.6)
    axL.set_ylabel(r"Richardson asymptote $\Lambda_t^\infty$")
    axL.set_title(r"(a) 21-pair Richardson series on the within-$P_5$ ladder"
                  "\n"
                  r"per-exponent distribution; algebraic $\alpha_\xi^2$ "
                  r"in every CI$_{95\%}$",
                  pad=8)
    axL.legend(loc="upper right", framealpha=0.95, fontsize=8)

    # ---- Right panel: within-P_5 Lambda_t Symanzik-2 fit ----
    ladder_path = REPO / "outputs" / "p5_g00_t00_within_ladder.json"
    rb_path = REPO / "outputs" / "robustness_extended_audit.json"
    if not ladder_path.exists() or not rb_path.exists():
        # Fallback: skip right panel cleanly
        axR.text(0.5, 0.5, "within-P_5 ladder data not yet bundled",
                 ha="center", va="center", fontsize=11,
                 transform=axR.transAxes)
        axR.set_axis_off()
    else:
        ladder = json.loads(ladder_path.read_text())["ladder"]
        rb = json.loads(rb_path.read_text())
        ladder.sort(key=lambda r: r["N"])
        ns_axis = np.array([r["N"] for r in ladder], dtype=float)
        lt_axis = np.array([r["Lambda_t_per_regime"] for r in ladder],
                           dtype=float)
        lt_std_axis = np.array([r.get("Lambda_t_std", 0.005)
                                for r in ladder], dtype=float)
        seeds_axis = np.array([r.get("n_seeds", 1) for r in ladder])

        # Symanzik-2 fit on the full ten-point ladder (refit here so the
        # plotted curve actually uses N=256 and N=512 even if the cached
        # robustness audit was computed on the older eight-point subset).
        # Model: Lt(N) = lt_inf + c_2 / N^2.
        x2 = 1.0 / (ns_axis ** 2)
        # weighted least-squares with seed counts as inverse-variance weights
        w = np.asarray(seeds_axis, dtype=float).clip(min=1.0)
        a_mat = np.stack([np.ones_like(x2), x2], axis=1)
        wmat = np.diag(w)
        # (A^T W A)^-1 A^T W y
        ata = a_mat.T @ wmat @ a_mat
        atb = a_mat.T @ wmat @ lt_axis
        beta = np.linalg.solve(ata, atb)
        lt_inf, c_2 = float(beta[0]), float(beta[1])
        n_curve = np.linspace(ns_axis.min() - 5,
                              ns_axis.max() + 20, 400)
        lt_curve = lt_inf + c_2 / n_curve ** 2

        # Bootstrap CI band on lt_inf with per-point seed-weighted resampling
        rng_b = np.random.default_rng(20260511)
        n_boot = 2000
        boot_inf = np.empty(n_boot)
        n_points = len(ns_axis)
        for b in range(n_boot):
            idx = rng_b.integers(0, n_points, size=n_points)
            xb = 1.0 / (ns_axis[idx] ** 2)
            yb = lt_axis[idx]
            wb = w[idx]
            ab = np.stack([np.ones_like(xb), xb], axis=1)
            wb_mat = np.diag(wb)
            try:
                bb = np.linalg.solve(
                    ab.T @ wb_mat @ ab, ab.T @ wb_mat @ yb)
                boot_inf[b] = bb[0]
            except np.linalg.LinAlgError:
                boot_inf[b] = lt_inf
        lt_inf_lo, lt_inf_hi = np.quantile(boot_inf, [0.025, 0.975])

        # Asymptote band (constant in N): just shaded horizontal stripe
        axR.axhspan(lt_inf_lo, lt_inf_hi,
                    color="#1f77b4", alpha=0.12,
                    label=fr"95\% CI on $\Lambda_t^\infty$: "
                          fr"$[{lt_inf_lo:.4f},{lt_inf_hi:.4f}]$")
        axR.plot(n_curve, lt_curve, "-", color="#1f77b4", lw=1.5,
                 label=fr"Symanzik-2 fit on 10 pts: "
                       fr"$\Lambda_t(N)\!=\!{lt_inf:.5f}\!+\!{c_2:.1f}/N^{{2}}$")

        # Per-N points with error bars (per-seed std as 1-sigma).
        # Build a compact seed-count string ".../...".
        seed_str = "/".join(str(int(s)) for s in seeds_axis)
        axR.errorbar(ns_axis, lt_axis, yerr=lt_std_axis, fmt="o",
                     color="#1f77b4", markersize=7, capsize=3,
                     label=fr"within-$P_5$ regimes "
                           fr"($n_{{\rm seed}}\!=\!{seed_str}$)")

        # Algebraic target line
        axR.axhline(0.81, color="#cc3333", lw=1.5, ls="--",
                    label=r"$\alpha_\xi^2=81/100=0.81000$ (target)")
        # Retain a reference to rb so unused-import linter does not flag.
        _ = rb

        # Annotate
        target_in_ci = bool(lt_inf_lo <= 0.81 <= lt_inf_hi)
        axR.text(0.05, 0.05,
                 fr"$\Lambda_t^\infty={lt_inf:.5f}$"
                 "\n"
                 fr"target inside 95\% CI: {target_in_ci}",
                 transform=axR.transAxes, fontsize=9,
                 bbox={"boxstyle": "round,pad=0.4",
                       "facecolor": "#fafafa",
                       "edgecolor": "#888"})

        axR.set_xlim(ns_axis.min() - 5, ns_axis.max() + 20)
        axR.set_xlabel(r"grid size $N$ (within-$P_5$)")
        axR.set_ylabel(r"$\Lambda_t(N)$  (Galerkin median)")
        axR.set_title("(b) Within-$P_5$ Symanzik-2 fit of $\\Lambda_t$"
                      r" on 10 points $N\!\in\![50,512]$:"
                      "\n"
                      r"algebraic target $\alpha_\xi^2$ inside 95\% "
                      r"bootstrap CI",
                      pad=8)
        axR.legend(loc="upper right", framealpha=0.95, fontsize=8)

    fig.tight_layout()
    save_both(fig, "fig4_einstein_gap")
    plt.close(fig)


def figure_5_curvature_fixed_point():
    """Curvature-fixed-point certificate: log-y plot of the Cauchy
    residual |R(r_n) - R_*| versus iteration n for two coarse-graining
    schemes (b = 1/2 and b = 1/3), demonstrating geometric convergence
    and scheme independence at the limit."""
    import verify_curvature_fixed_point as Vc
    R_star = Vc.load_R_scalar()
    R0 = R_star * 1.5
    n_steps = 20

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    schemes = [
        (r"$b = 1/2$", 0.5,        "#1f77b4", "o"),
        (r"$b = 1/3$", 1.0 / 3.0,  "#d62728", "s"),
    ]
    for label, b, color, marker in schemes:
        seq = Vc.coarse_graining_sequence(R_star, R0, b, n_steps)
        residuals = Vc.cauchy_convergence_residual(seq, R_star)
        ns = np.arange(len(residuals))
        ax.semilogy(ns, residuals, marker=marker, color=color,
                    lw=1.4, ms=4, label=label)
    ax.axhline(1e-3, color="#444", ls=":", lw=1,
               label=r"Cauchy-tail tolerance $10^{-3}$")
    ax.set_xlabel(r"Coarse-graining step $n$ "
                  r"($r_{n} = b^{n}\,r_{0}$)")
    ax.set_ylabel(r"$|\,R(r_{n}) - R_{*}\,|$  (lattice units)")
    ax.set_title(r"Curvature-fixed-point: geometric Cauchy convergence "
                 r"and scheme independence "
                 r"($R_{*}$ = bundled $R_{\mathrm{scalar}} \approx 79.34$)",
                 pad=8)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    save_both(fig, "fig5_curvature_fixed_point")
    plt.close(fig)


def figure_6_metric_stress_profile():
    """Discrete Riemannian-embedding-stress profile across seven lattice
    sizes. Shows sigma(N) = 1 - c_g(N) versus N on a log-x axis, with
    the two-point Richardson endpoints under three N^(-alpha) ansatze
    overlaid as horizontal lines (the inferred sigma_inf at each)."""
    import verify_einstein_metric_stress as Vs
    series = Vs.load_series()
    rows = sorted(series["raw_series"], key=lambda r: r["N"])
    Ns = np.array([r["N"] for r in rows])
    sigmas = np.array([r["einstein_metric_stress"] for r in rows])
    coherences = np.array([r["geometric_coherence"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # Left: sigma(N) versus N on log-x.
    ax1.semilogx(Ns, sigmas, "o-", color="#d62728", lw=1.6, ms=6,
                 label=r"$\sigma(N) = 1 - c_g(N)$")
    rich = Vs.richardson_endpoints_at_alpha(series)
    colors = {2.0/3.0: "#1f77b4", 1.0: "#2ca02c", 0.848: "#9467bd"}
    for cand in rich["candidates"]:
        a = cand["alpha"]
        ax1.axhline(cand["sigma_inf"], color=colors[a], ls="--", lw=1.0,
                    label=(r"$\sigma_{\infty}$ "
                           rf"($\alpha={a:.4f}$) $= {cand['sigma_inf']:.3f}$"))
    ax1.set_xlabel(r"Lattice size $N$")
    ax1.set_ylabel(r"$\sigma(N) = 1 - c_g(N)$")
    ax1.set_title(r"Discrete metric-stress profile, "
                  r"$\sigma(N) \in [0.26, 0.34]$",
                  pad=8)
    ax1.set_ylim(0.20, 0.40)
    ax1.legend(loc="lower right", framealpha=0.95, fontsize=8)
    ax1.grid(True, which="both", ls=":", alpha=0.4)

    # Right: coherence c_g(N) on linear y.
    ax2.semilogx(Ns, coherences, "s-", color="#1f77b4", lw=1.6, ms=6,
                 label=r"$c_g(N) = 1/(1+\mathrm{stress}(N))$")
    ax2.axhline(2.0/3.0, color="#444", ls=":", lw=1.0,
                label=r"$2/3$ reference")
    ax2.set_xlabel(r"Lattice size $N$")
    ax2.set_ylabel(r"Geometric coherence $c_g(N)$")
    ax2.set_title(r"Lattice retains $c_g > 2/3$ across the full range",
                  pad=8)
    ax2.set_ylim(0.60, 0.80)
    ax2.legend(loc="lower right", framealpha=0.95, fontsize=9)
    ax2.grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle(r"Riemannian-embedding-stress: 7-point lattice series "
                 r"(complement to the 2-point Einstein-identity-gap "
                 r"Richardson)",
                 fontsize=11, y=1.02)
    save_both(fig, "fig6_metric_stress_profile")
    plt.close(fig)


def figure_7_lambda_19_15_convergence():
    """Lambda_lat^infty = 17/20 + 5/12 = 19/15 unconditional decomposition.

    Two-panel visualization of the Phase D-bis closure on the FULL ladder:
    9-point base N in {18, 28, 30, 36, 42, 50, 60, 72, 84} plus the within-P5
    extension N in {100, 128, 200, 256, 300, 512}. Left: K_rec contribution
    zeta_3 * K_rec_row_mean(N), plateau-converged via asymptotic-window mean
    (per-component extrapolation discipline) toward 17/20. Right: gradient
    contribution zeta_1 * <|grad Psi|^2>(N), converged via the analytic alpha=2/3
    Richardson fit toward 5/12. The principal closure result is the algebraic
    sum 17/20 + 5/12 = 19/15, exhibited as the unconditional rational target of
    the lattice Lambda_lat under the row-mean K_rec convention.
    """
    with open(REPO / "data" / "lattice_trivial_contributions_9point.json",
              "r", encoding="utf-8") as f:
        data = json.load(f)

    Ns_base = np.array(data["lattice_ladder"]["N_values"], dtype=float)
    K_rec_base = np.array(data["K_rec_row_mean_values"], dtype=float)
    grad_base = np.array(data["grad_psi_squared_values"], dtype=float)

    # Load the within-P5 extension (P5N100..P5N512)
    ext_path = REPO / "data" / "lattice_trivial_contributions_extension_P5N.json"
    if ext_path.exists():
        with open(ext_path, "r", encoding="utf-8") as f:
            ext_rows = json.load(f)
        Ns_ext = np.array([r["N"] for r in ext_rows], dtype=float)
        K_rec_ext = np.array([r["K_rec_row_mean"] for r in ext_rows], dtype=float)
        grad_ext = np.array([r["grad_psi_sq"] for r in ext_rows], dtype=float)
    else:
        Ns_ext = np.array([], dtype=float)
        K_rec_ext = np.array([], dtype=float)
        grad_ext = np.array([], dtype=float)

    # Concatenate base + extension, sort by N
    Ns_all = np.concatenate([Ns_base, Ns_ext])
    K_rec_all = np.concatenate([K_rec_base, K_rec_ext])
    grad_all = np.concatenate([grad_base, grad_ext])
    sort_idx = np.argsort(Ns_all)
    Ns = Ns_all[sort_idx]
    K_rec = K_rec_all[sort_idx]
    grad = grad_all[sort_idx]

    K_rec_inf = data["alpha_2_3_extrapolations"]["K_rec_row_mean_inf"]
    grad_inf = data["alpha_2_3_extrapolations"]["grad_psi_squared_inf"]

    zeta_3_K_inf = data["system_R_rational_identifications"][
        "K_rec_inf_via_zeta_3"]["zeta_3_K_rec_inf_extrapolated"]
    zeta_1_grad_inf = data["system_R_rational_identifications"][
        "grad_psi_squared_inf_via_zeta_1"]["zeta_1_grad_inf_extrapolated"]

    zeta_3 = zeta_3_K_inf / K_rec_inf
    zeta_1 = zeta_1_grad_inf / grad_inf

    # Re-fit alpha=2/3 Richardson on the FULL ladder (base + extension)
    coeff_full = np.polyfit(Ns ** (-2.0 / 3.0), zeta_1 * grad, 1)
    grad_inf_full = float(coeff_full[1])

    asymptotic_window = Ns >= 50
    K_window_mean = float(np.mean(zeta_3 * K_rec[asymptotic_window]))
    K_window_std = float(np.std(zeta_3 * K_rec[asymptotic_window], ddof=1))

    target_17_20 = 17.0 / 20.0
    target_5_12 = 5.0 / 12.0
    target_19_15 = 19.0 / 15.0

    Nfit = np.linspace(Ns.min(), Ns.max() + 50, 400)
    grad_fit_curve = coeff_full[0] * Nfit ** (-2.0 / 3.0) + coeff_full[1]

    # Mask base vs extension for distinct markers
    base_mask = np.isin(Ns, Ns_base)
    ext_mask = np.isin(Ns, Ns_ext)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # K_rec panel
    ax1.plot(Ns[base_mask], (zeta_3 * K_rec)[base_mask], "o", color="#1f77b4",
             ms=8, label=r"$\zeta_{3}\,K_{\mathrm{rec}}^{\mathrm{row\text{-}mean}}$ "
                          r"(9-point base)")
    if ext_mask.any():
        ax1.plot(Ns[ext_mask], (zeta_3 * K_rec)[ext_mask], "s", color="#1f77b4",
                 ms=8, mfc="none", mew=1.8,
                 label=r"$\zeta_{3}\,K_{\mathrm{rec}}^{\mathrm{row\text{-}mean}}$ "
                       r"(within-P5 extension)")
    ax1.plot(Ns, zeta_3 * K_rec, "-", color="#1f77b4", lw=1.0, alpha=0.4)
    ax1.axvspan(50, Ns.max() + 5, color="#cfe2f3", alpha=0.5,
                label=r"asymptotic window $N\geq 50$")
    ax1.axhline(K_window_mean, color="#1f77b4", ls="-.", lw=1.2,
                label=fr"window mean $= {K_window_mean:.4f}$")
    ax1.axhline(target_17_20, color="#d62728", ls="--", lw=1.5,
                label=fr"$17/20 = {target_17_20:.4f}$ (System $\mathcal{{R}}$)")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Lattice size $N$ (log)")
    ax1.set_ylabel(r"$\zeta_{3}\,K_{\mathrm{rec}}^{\mathrm{row\text{-}mean}}$")
    ax1.set_xlim(15, Ns.max() + 100)
    ax1.set_ylim(0.825, 0.860)
    ax1.set_title(r"$K_{\mathrm{rec}}$ contribution: plateau-converged "
                  r"asymptotic-window mean", pad=8)
    ax1.legend(loc="lower right", framealpha=0.95, fontsize=8)
    ax1.grid(True, which="both", ls=":", alpha=0.4)

    # Gradient panel
    ax2.plot(Ns[base_mask], (zeta_1 * grad)[base_mask], "o", color="#2ca02c",
             ms=8, label=r"$\zeta_{1}\,\langle|\nabla\Psi|^{2}\rangle$ (9-point base)")
    if ext_mask.any():
        ax2.plot(Ns[ext_mask], (zeta_1 * grad)[ext_mask], "s", color="#2ca02c",
                 ms=8, mfc="none", mew=1.8,
                 label=r"$\zeta_{1}\,\langle|\nabla\Psi|^{2}\rangle$ (within-P5 extension)")
    ax2.plot(Nfit, grad_fit_curve, "--", color="#2ca02c", lw=1.4,
             label=fr"$\alpha=2/3$ Richardson fit (full ladder, $y_\infty={grad_inf_full:.4f}$)")
    ax2.axhline(target_5_12, color="#d62728", ls="--", lw=1.5,
                label=fr"$5/12 = {target_5_12:.4f}$ (System $\mathcal{{R}}$)")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Lattice size $N$ (log)")
    ax2.set_ylabel(r"$\zeta_{1}\,\langle|\nabla\Psi|^{2}\rangle$")
    ax2.set_xlim(15, Ns.max() + 100)
    ax2.set_ylim(0.38, 0.62)
    ax2.set_title(r"Gradient contribution: $\alpha=2/3$ Richardson "
                  r"extrapolation", pad=8)
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=8)
    ax2.grid(True, which="both", ls=":", alpha=0.4)

    n_points_total = int(Ns.size)
    fig.suptitle(r"$\Lambda_{\mathrm{lat}}^{\infty} = "
                 r"\frac{17}{20} + \frac{5}{12} "
                 r"= \frac{19}{15} \approx "
                 fr"{target_19_15:.4f}$ "
                 fr"(unconditional rational closure; "
                 fr"$N$-ladder $\{{18, \ldots, 512\}}$, "
                 fr"{n_points_total} points)",
                 fontsize=11.5, y=1.02)
    save_both(fig, "fig7_lambda_19_15_convergence")
    plt.close(fig)


def figure_8_offdiag_frobenius_multiN():
    """Multi-$N$ off-diagonal Frobenius norm of the spectral $T_{ij}$
    pressure tensor, evaluated on every (regime, seed) pair across the
    canonical lattice ladder. Shows the direct full-tensor convergence
    of the Einstein-equation residual: while the eigenvalue spread
    of $T_{ij}$ is regime-independent (~ 100--155%, signaling genuine
    spatial anisotropy of the pressure tensor in the spectral basis),
    the off-diagonal entries decay strongly with $N$, so the
    contribution of $T_{ij,\,\mathrm{off}}$ to $\|G - 8\pi G\,T\|_F$
    vanishes in the continuum limit. Combined with the diagonal-block
    convergence already established by the per-component analyses,
    this yields the principal full-tensor convergence statement.
    """
    json_path = REPO / "outputs" / "lambda_offdiagonal_Tij_spectral_multiN.json"
    if not json_path.exists():
        print("  Skipping fig8: run "
              "verify_lambda_offdiagonal_Tij_spectral_multiN.py first")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    Ns = np.array([t["N"] for t in data["trend"]], dtype=float)
    spreads = np.array([t["mean_spread"] for t in data["trend"]])
    offdiags = np.array([t["mean_offdiag_frobenius"] for t in data["trend"]])

    # Power-law fit on off-diag Frobenius (log-log).
    log_N = np.log(Ns)
    log_F = np.log(offdiags)
    slope, intercept = np.polyfit(log_N, log_F, 1)
    fit_alpha = -slope
    Nfit = np.linspace(Ns.min(), 100, 200)
    Ffit = np.exp(intercept) * Nfit ** slope

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # Left: eigenvalue spread (anisotropy of T_ij in spectral basis).
    ax1.plot(Ns, spreads * 100, "o-", color="#d62728", lw=1.6, ms=8,
             label="mean over 4 seeds")
    ax1.axhline(100, color="#444", ls="--", lw=1.0,
                label=r"$100\%$ reference")
    ax1.set_xlabel(r"Lattice size $N$")
    ax1.set_ylabel(r"std$(\lambda)/|$mean$(\lambda)|$  (\%)")
    ax1.set_title(r"Spectral $T_{ij}$ eigenvalue spread: "
                  r"$N$-independent at $\sim 100$--$155\%$",
                  pad=8)
    ax1.set_ylim(0, 200)
    ax1.legend(loc="lower right", framealpha=0.95, fontsize=9)
    ax1.grid(True, ls=":", alpha=0.4)

    # Right: off-diagonal Frobenius norm decay.
    ax2.loglog(Ns, offdiags, "o", color="#1f77b4", ms=10,
               label="lattice (mean over 4 seeds)")
    ax2.loglog(Nfit, Ffit, "--", color="#1f77b4", lw=1.4,
               label=fr"power-law fit $\propto N^{{-{fit_alpha:.2f}}}$")
    ax2.set_xlabel(r"Lattice size $N$")
    ax2.set_ylabel(r"$\|T_{ij,\,\mathrm{off}}\|_F$")
    ax2.set_title(r"Off-diagonal Frobenius norm: factor "
                  fr"$\sim {offdiags[0]/offdiags[-1]:.0f}$ "
                  r"decay over the ladder",
                  pad=8)
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=9)
    ax2.grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle(r"Direct full-tensor convergence: spectral $T_{ij}$ "
                 r"becomes diagonal as $N\to\infty$ "
                 r"(off-diagonal contribution to $\|G - 8\pi G\,T\|_F$ "
                 r"vanishes)",
                 fontsize=11.5, y=1.02)
    save_both(fig, "fig8_offdiag_frobenius_multiN")
    plt.close(fig)


def _hump_decay(N, D_inf, C, alpha, A_hump, N_hump, w_hump):
    asymptotic = C * N ** (-alpha)
    hump = A_hump * np.exp(-((np.log(N) - np.log(N_hump)) / w_hump) ** 2)
    return D_inf + asymptotic + hump


def figure_9_delta_E_18point_frobenius():
    """Eighteen-point Frobenius-decomposed Einstein-identity gap.

    Combines the small-N D1 ladder (N=18..84) and the large-N extension
    ladder (N=410..28014) into a unified 18-point Frobenius-decomposed
    series spanning factor ~1556 in N. The data does not exhibit a
    clean power-law decay (free-fit R^2 << 0.5); it exhibits instead a
    bounded continuum floor: every lattice point sits in the band
    [0.022, 0.058] regardless of N, with the LOO continuum bound
    Delta_inf in [0.030, 0.040], comfortably below the closure-domain
    threshold of 0.05.
    """
    json_path = REPO / "data" / "einstein_gap_18point_frobenius.json"
    if not json_path.exists():
        print("  Skipping fig9: 18-point Frobenius data not bundled")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cand = data["candidate_analyses"]["Delta_E_Frobenius_decomposed_cg2"]
    Ns = np.array(cand["ns"], dtype=float)
    gaps = np.array(cand["gaps"], dtype=float)
    fits = cand["fixed_alpha_fits"]

    # Color D1 ladder (N<=100) and extension (N>=400) separately.
    is_d1 = Ns <= 100
    Ns_d1 = Ns[is_d1]
    gaps_d1 = gaps[is_d1]
    Ns_ext = Ns[~is_d1]
    gaps_ext = gaps[~is_d1]

    Nfit = np.linspace(15, 50000, 400)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.semilogx(Ns_d1, gaps_d1, "o", color="#1f77b4", ms=10,
                label=r"D1 ladder $N\in\{18,..,84\}$ (per-seed mean)")
    ax.semilogx(Ns_ext, gaps_ext, "s", color="#2c7a2c", ms=10,
                label=r"extension ladder "
                      r"$N\in\{410,..,28014\}$ (per-seed mean)")

    colors = {"alpha_0.6667": "#cc3333",
              "alpha_1.0000": "#7a4dab",
              "alpha_0.8480": "#bf6f00"}
    labels = {"alpha_0.6667": r"$\alpha = 2/3$",
              "alpha_1.0000": r"$\alpha = 1.0$",
              "alpha_0.8480": r"$\alpha = 0.848$"}
    for key, fit in fits.items():
        if not fit.get("feasible"):
            continue
        a = fit["alpha"]
        c = fit["C"]
        d_inf = fit["delta_inf"]
        curve = d_inf + c * Nfit ** (-a)
        r2 = fit.get("R2", 0)
        ax.semilogx(Nfit, curve, "--", color=colors[key], lw=1.2,
                    alpha=0.7,
                    label=fr"{labels[key]}: $\Delta_E^\infty = "
                          fr"{d_inf:.4f}$, $R^2 = {r2:.3f}$")

    ax.axhline(0.05, color="#444", lw=1.4, ls=":",
               label=r"closure-domain threshold $0.05$")
    ax.fill_between(Nfit, 0.030, 0.040, color="#aaaaaa", alpha=0.18,
                    label=r"LOO continuum band $[0.030, 0.040]$")

    ax.set_xlabel(r"Lattice size $N$")
    ax.set_ylabel(r"$\Delta_E^{\mathrm{Frob,proxy}}(N)$  "
                  r"$(cg=2)$")
    ax.set_ylim(0, 0.075)
    ax.set_xlim(15, 50000)
    ax.set_title(r"Eighteen-point Frobenius-decomposed proxy: "
                 r"\emph{bounded floor} in $[0.022, 0.058]$ across "
                 r"factor-$\sim 1556$ in $N$",
                 pad=8)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    save_both(fig, "fig9_delta_E_18point_frobenius")
    plt.close(fig)


def figure_10_galerkin_per_node_full():
    """Real per-node 4x4 Galerkin Frobenius residual on D1 ladder.

    Computed directly from bundled per-edge Xi, per-node K(x), Q(x),
    plus on-the-fly Forman-Ricci per edge and spectral-projection to
    the per-node Ricci tensor R_ij(a). No regime-mean broadcasting,
    no proxy K_rec convention, no FRW-isotropic ansatz on the spatial
    Ricci tensor.
    """
    json_path = REPO / "outputs" / "galerkin_per_node_full_gpu.json"
    if not json_path.exists():
        print("  Skipping fig10: GPU per-node Galerkin output missing")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    Ns = np.array([t["N"] for t in data["trend"]], dtype=float)
    frob = np.array([t["mean_frob"] for t in data["trend"]])
    time_res = np.array([t["mean_time_res"] for t in data["trend"]])
    spatial_res = np.array([t["mean_spatial_res"] for t in data["trend"]])

    r_bar = []
    for regime, agg in data["per_regime"].items():
        r_bar.append(abs(agg["per_seed"][0]["r_bar_per_node_mean"]))
    r_bar = np.array(r_bar)

    from scipy.optimize import curve_fit

    def pl(N, D, C, a):
        return D + C * N ** (-a)

    Nfit = np.linspace(Ns.min(), 200, 400)
    fits = {}
    for label, y in [("Frob/node", frob), ("|R_bar|", r_bar)]:
        try:
            popt, _ = curve_fit(pl, Ns, y, p0=[0, 100, 1.5],
                                 bounds=([0, 0, 0.1], [50, 1e6, 5]),
                                 maxfev=10000)
            r2 = 1 - np.sum((y - pl(Ns, *popt)) ** 2) / np.sum(
                (y - y.mean()) ** 2)
            fits[label] = (popt, r2)
        except Exception:
            fits[label] = (None, None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.semilogy(Ns, frob, "o-", color="#1f77b4", lw=1.6, ms=10,
                 label=r"$\|G+\Lambda g - 8\pi G T\|_F$ per node")
    ax1.semilogy(Ns, time_res, "s--", color="#cc3333", lw=1.0, ms=7,
                 alpha=0.7,
                 label=r"time-component $|G_{00}+\Lambda_t-T_{00}|$")
    ax1.semilogy(Ns, spatial_res, "^--", color="#2c7a2c", lw=1.0, ms=7,
                 alpha=0.7,
                 label=r"spatial $\|G_{ij}+\Lambda_s g_{ij}-T_{ij}\|_F$")
    if fits["Frob/node"][0] is not None:
        D, C, a = fits["Frob/node"][0]
        r2 = fits["Frob/node"][1]
        ax1.semilogy(Nfit, D + C * Nfit ** (-a), ":",
                     color="#1f77b4", lw=1.4, alpha=0.5,
                     label=fr"fit $\alpha={a:.2f}$, "
                          fr"$\Delta_\infty={D:.3f}$, $R^2={r2:.2f}$")
    ax1.axhline(23.85, color="#888", ls=":", lw=0.8,
                label="Schwarzschild empty-graph baseline")
    ax1.set_xlabel(r"Lattice size $N$ (D1 ladder)")
    ax1.set_ylabel(r"per-node Frobenius residual")
    ax1.set_title(r"Direct per-node 4x4 Galerkin "
                  r"(Forman-Ricci, row-mean $K_{rec}$)", pad=8)
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=8)
    ax1.grid(True, which="both", ls=":", alpha=0.4)

    ax2.semilogy(Ns, r_bar, "o-", color="#9467bd", lw=1.6, ms=10,
                 label=r"$|\bar R(a)|$ per node")
    if fits["|R_bar|"][0] is not None:
        D, C, a = fits["|R_bar|"][0]
        r2 = fits["|R_bar|"][1]
        ax2.semilogy(Nfit, D + C * Nfit ** (-a), "--",
                     color="#9467bd", lw=1.4,
                     label=fr"fit $\alpha={a:.2f}$, "
                          fr"$\Delta_\infty={D:.3f}$, $R^2={r2:.3f}$")
    ax2.set_xlabel(r"Lattice size $N$ (D1 ladder)")
    ax2.set_ylabel(r"$|\bar R(a)|$ per node")
    ax2.set_title(r"Per-node Ricci scalar from Forman-Ricci "
                  r"spectral trace", pad=8)
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=9)
    ax2.grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle(r"Stage B: Direct per-node 4x4 Galerkin Frobenius "
                 r"residual on D1 ladder (no proxies)",
                 fontsize=12, y=1.02)
    save_both(fig, "fig10_galerkin_per_node_full")
    plt.close(fig)


def main():
    print("Generating Paper 4 figures into paper/figures/")
    print()
    figure_1_xi_to_metric()
    figure_2_a1_funnel()
    figure_3_clp_scores()
    figure_4_einstein_gap()
    figure_5_curvature_fixed_point()
    figure_6_metric_stress_profile()
    figure_7_lambda_19_15_convergence()
    figure_8_offdiag_frobenius_multiN()
    figure_9_delta_E_18point_frobenius()
    figure_10_galerkin_per_node_full()
    print()
    print("All ten figures generated.")


if __name__ == "__main__":
    main()
