"""Figures for the triangle phase-class asymmetry findings (P4 Outlook).

Produces four PDF figures in paper/figures/:

  fig_phase_class_asym_N_trend.pdf
    Per-seed asym(PPN-PNN) values vs N=64..300, weighted Symanzik
    a + b/N fit, asymptote band, target -pi/200 line.

  fig_winding_class_split_57.pdf
    Per-regime f_neg and f_pos with 2/7 and 5/7 target lines plus
    cross-regime mean +/- std bands.

  fig_per_bin_asym_drift.pdf
    asym_neg and asym_pos per regime showing the asym_pos N-drift that
    drives the all-asym N-trend; asym_neg ~constant, asym_pos drifts
    monotonically from -0.013 (N=64) to -0.058 (N=300).

  fig_threshold_robustness.pdf
    Asymptote a_inf as a function of c_info threshold factor in
    {1.5x, 2.0x, 2.5x, 3.0x} median-|Delta Xi| with target -pi/200 line.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless CI / no-DISPLAY
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType (vector, arXiv-friendly)
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = REPO / "outputs"


def load_json(name):
    fp = OUT_DIR / name
    return json.loads(fp.read_text())


def fig_phase_class_asym_N_trend():
    """Figure 1: triangle phase-class asym N-trend with Symanzik fit."""
    bundle = load_json("audit_asym_per_seed_n_trend.json")
    rows = bundle["rows"]
    N = np.array([r["N"] for r in rows], dtype=float)
    asym_mean = np.array([r["asym_mean"] for r in rows])
    asym_unc = np.array([r["asym_uncertainty_of_mean"] for r in rows])
    n_seeds = np.array([r["n_seeds"] for r in rows])
    # Symanzik fit
    x = 1.0 / N
    w = 1.0 / np.maximum(asym_unc, 1e-6) ** 2
    A = np.column_stack([np.ones_like(x), x])
    AtWA = A.T @ (w[:, None] * A)
    AtWy = A.T @ (w * asym_mean)
    coef = np.linalg.solve(AtWA, AtWy)
    a_inf, b = coef
    cov = np.linalg.inv(AtWA)
    a_unc = float(np.sqrt(cov[0, 0]))
    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.errorbar(N, asym_mean, yerr=asym_unc, fmt='o', capsize=3,
                color='C0', ecolor='C0', label='Per-regime mean (full seeds)',
                markersize=6, zorder=4)
    # Annotate seeds
    for ni, ai, ns in zip(N, asym_mean, n_seeds):
        ax.annotate(f"{int(ns)}s", (ni, ai), xytext=(5, 5),
                     textcoords='offset points', fontsize=7,
                     color='gray')
    # Symanzik fit curve
    N_fine = np.linspace(50, 350, 200)
    fit_curve = a_inf + b / N_fine
    ax.plot(N_fine, fit_curve, '-', color='C1', lw=2,
            label=fr"Symanzik fit: $a + b/N$, $a_\infty={a_inf:+.4f}$",
            zorder=3)
    # Asymptote band (1 sigma)
    ax.axhspan(a_inf - a_unc, a_inf + a_unc,
                color='C1', alpha=0.15, zorder=2)
    # Target -pi/200
    target = -np.pi / 200
    ax.axhline(target, color='C3', linestyle='--', lw=1.5,
                label=fr"$-\pi\gamma^{{2}}/2 = -\pi/200 = {target:+.5f}$",
                zorder=2)
    # Zero line
    ax.axhline(0, color='gray', lw=0.6, zorder=1)
    ax.set_xlabel('Lattice size $N$', fontsize=11)
    ax.set_ylabel(r'$\mathrm{asym}(\mathrm{PPN}-\mathrm{PNN})$',
                   fontsize=11)
    ax.set_title('Triangle phase-class asymmetry $N$-trend',
                  fontsize=12)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(55, 315)
    fig.tight_layout()
    out = FIG_DIR / "fig_phase_class_asym_N_trend.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}: a_inf = {a_inf:+.5f} +/- {a_unc:.5f}, "
          f"target = {target:+.5f}, diff = {a_inf - target:+.5f}")
    return a_inf, a_unc


def fig_winding_class_split_57():
    """Figure 2: f_neg/f_pos winding-class fractions with 2/7 and 5/7 targets."""
    bundle = load_json("audit_winding_fraction_mechanism.json")
    rows = bundle["rows"]
    regimes = [r["regime"] for r in rows]
    N = np.array([r["N"] for r in rows], dtype=float)
    f_neg = np.array([r["frac_winding_neg_mean"] for r in rows])
    f_neg_unc = np.array([r["frac_winding_neg_unc"] for r in rows])
    f_pos = 1 - f_neg
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x_pos = np.arange(len(regimes))
    width = 0.35
    bars_neg = ax.bar(x_pos - width / 2, f_neg, width,
                        yerr=f_neg_unc, capsize=3,
                        color='C0', alpha=0.8,
                        label=r'$f_{\mathrm{neg}}$ (negative principal-sum)')
    bars_pos = ax.bar(x_pos + width / 2, f_pos, width,
                        yerr=f_neg_unc, capsize=3,
                        color='C2', alpha=0.8,
                        label=r'$f_{\mathrm{pos}}$ (positive principal-sum)')
    ax.axhline(2 / 7, color='C0', linestyle='--', lw=1.8,
                label=r'$2/7=0.2857$')
    ax.axhline(5 / 7, color='C2', linestyle='--', lw=1.8,
                label=r'$5/7=0.7143$')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regimes, fontsize=9, rotation=30)
    ax.set_xlabel('Lattice regime', fontsize=11)
    ax.set_ylabel(r'Winding-class fraction',
                   fontsize=11)
    title = (r'Winding-class split $f_{\mathrm{neg}}{:}f_{\mathrm{pos}}'
              r'\to 2{:}5$ (cross-regime $0.07\%$ match to $2/7,5/7$)')
    ax.set_title(title, fontsize=11)
    ax.legend(loc='center right', fontsize=8.5, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out = FIG_DIR / "fig_winding_class_split_57.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}: f_neg cross-regime mean = "
          f"{f_neg.mean():.4f} (target 2/7 = 0.2857)")


def fig_per_bin_asym_drift():
    """Figure 3: per-bin asym_neg and asym_pos drift with N."""
    bundle = load_json("audit_winding_fraction_mechanism.json")
    rows = bundle["rows"]
    N = np.array([r["N"] for r in rows], dtype=float)
    asym_neg = np.array([r["asym_winding_neg_mean"] for r in rows])
    asym_neg_unc = np.array([r["asym_winding_neg_unc"] for r in rows])
    asym_pos = np.array([r["asym_winding_pos_mean"] for r in rows])
    asym_pos_unc = np.array([r["asym_winding_pos_unc"] for r in rows])
    asym_all = np.array([r["asym_all_measured_mean"] for r in rows])
    asym_all_unc = np.array([r["asym_all_unc"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.errorbar(N, asym_neg, yerr=asym_neg_unc, fmt='o-', capsize=3,
                color='C0', label=r'$\mathrm{asym}_{\mathrm{neg}}$ '
                                    r'(negative principal-sum)',
                markersize=6)
    ax.errorbar(N, asym_pos, yerr=asym_pos_unc, fmt='s-', capsize=3,
                color='C2', label=r'$\mathrm{asym}_{\mathrm{pos}}$ '
                                    r'(positive principal-sum)',
                markersize=6)
    ax.errorbar(N, asym_all, yerr=asym_all_unc, fmt='^-', capsize=3,
                color='C3', label=r'$\mathrm{asym}_{\mathrm{all}}$ (measured)',
                markersize=6)
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_xlabel('Lattice size $N$', fontsize=11)
    ax.set_ylabel(r'$\mathrm{asym}(\mathrm{PPN}-\mathrm{PNN})$ per subset',
                   fontsize=11)
    title = ('Per-bin asym drift with $N$: '
              r'$\mathrm{asym}_{\mathrm{pos}}$ '
              'monotone drift drives all-asym N-trend')
    ax.set_title(title, fontsize=11)
    ax.legend(loc='center left', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig_per_bin_asym_drift.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")


def fig_threshold_robustness():
    """Figure 4: asymptote stability over c_info threshold factors."""
    bundle = load_json("audit_asym_robustness_threshold.json")
    by_factor = bundle["results_by_factor"]
    factors_str = sorted(by_factor.keys(),
                          key=lambda x: float(x))
    factors = [float(f) for f in factors_str]
    asymptotes = [by_factor[f]["asymptote"]
                    if "asymptote" in by_factor[f] else np.nan
                    for f in factors_str]
    asym_unc = [by_factor[f]["asymptote_unc"]
                  if "asymptote_unc" in by_factor[f] else np.nan
                  for f in factors_str]
    chi2 = [by_factor[f]["chi2"]
              if "chi2" in by_factor[f] else np.nan
              for f in factors_str]
    dof = [by_factor[f]["dof"] if "dof" in by_factor[f] else 1
             for f in factors_str]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.5))
    # Left: asymptotes
    ax1.errorbar(factors, asymptotes, yerr=asym_unc, fmt='o', capsize=4,
                  markersize=8, color='C0')
    target = -np.pi / 200
    ax1.axhline(target, color='C3', linestyle='--', lw=1.5,
                  label=fr'$-\pi/200={target:+.5f}$')
    ax1.axhline(-1 / 64, color='C1', linestyle=':', lw=1.5,
                  label=r'$-1/64=-0.01563$')
    ax1.axhline(0, color='gray', lw=0.6)
    ax1.set_xlabel(r'$c_{\mathrm{info}}$ threshold factor',
                    fontsize=11)
    ax1.set_ylabel(r'$a_\infty$ (Symanzik asymptote)', fontsize=11)
    ax1.set_title('Threshold robustness',
                    fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    # Right: chi2/dof per threshold
    chi2_per_dof = [c / max(d, 1) for c, d in zip(chi2, dof)]
    bars = ax2.bar(factors, chi2_per_dof, width=0.3,
                     color=['C3' if c > 5 else 'C0'
                              for c in chi2_per_dof])
    ax2.axhline(1.0, color='gray', linestyle='--', lw=1)
    ax2.set_xlabel(r'$c_{\mathrm{info}}$ threshold factor',
                    fontsize=11)
    ax2.set_ylabel(r'$\chi^{2}/\mathrm{dof}$ of fit', fontsize=11)
    ax2.set_title('Fit quality (red: $> 5$ unreliable)',
                    fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    out = FIG_DIR / "fig_threshold_robustness.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")


def main():
    print("Generating P4 phase-class-asym figures...")
    a_inf, a_unc = fig_phase_class_asym_N_trend()
    fig_winding_class_split_57()
    fig_per_bin_asym_drift()
    fig_threshold_robustness()
    print(f"\nAll 4 figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
