r"""Generate fig17 and fig18: the Q1/Q2 closure-audit figures.

fig17 -- per-summand continuum decomposition of T_00^Xi
  (a) the noisy aggregate T_00 and the dominant K/Q-recoil summand
      S4 vs N, with Symanzik extrapolations, the direct-fit
      bootstrap CI band, and the two candidate landings
      alpha_xi^2 and alpha_xi^2*(1-gamma^2).
  (b) the three sub-leading summands S1, S2, S3 vs N with their
      System-R structural targets (S1 -> alpha_xi^2*gamma^2,
      S2,S3 -> 0).

fig18 -- empirical witnesses of the curvature-dimension lower bound
  (a) signed Hessian-Ricci route: the robust curvature lower
      bounds K_p1, K_p5 and the hyperbolic-node fraction f_neg
      vs N.
  (b) intrinsic Bakry-Emery route: the pointwise Gamma_2 curvature
      percentiles K_inf, K_p5, K_mean vs N.

Both figures read the bundled audit JSONs produced by
verify_t00_summand_decomposition.py,
verify_signed_ricci_lower_bound.py and
verify_carrier_bakry_emery_cd.py; if an output is missing the
corresponding audit is run first.

Usage:
    python ./src/make_fig17_q1q2_audits.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
FIGDIR = REPO / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

ALPHA_XI_SQ = 81.0 / 100.0
GAMMA_SQ = 1.0 / 100.0
ALPHA_XI_SQ_SHIFT = ALPHA_XI_SQ * (1.0 - GAMMA_SQ)   # 8019/10000


def _load(name, module_name):
    path = OUTPUTS / name
    if not path.exists():
        mod = __import__(module_name)
        mod.main()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _symanzik_curve(fit, n_grid):
    """Reconstruct y(N) from a stored Symanzik fit dict (y_inf plus
    the implied per-order coefficients are not stored, so re-fit the
    light curve through y_inf and the endpoint is unnecessary --
    instead just draw a guide through y_inf as N->inf)."""
    return None  # curves drawn from y_inf horizontal guide only


def fig17():
    d = _load("t00_summand_decomposition_audit.json",
              "verify_t00_summand_decomposition")
    regimes = d["per_regime"]
    ns = np.array([r["N"] for r in regimes], dtype=float)
    rm = {k: np.array([r["regime_median"][k] for r in regimes])
          for k in ("S1_half_var_xi", "S2_var_amp", "S3_grad_psi_sq",
                    "S4_kq_recoil", "T00")}
    sym = d["summand_symanzik"]
    ic = d["internal_consistency"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # Panel (a): aggregate T_00 and the dominant summand S4.
    ax1.axhspan(ic["direct_T00_bootstrap_ci95"][0],
                ic["direct_T00_bootstrap_ci95"][1],
                color="0.85", label="direct $T_{00}$ fit 95% CI")
    ax1.axhline(ALPHA_XI_SQ, color="C3", ls="--", lw=1.2,
                label=r"$\alpha_\xi^2=81/100$")
    ax1.axhline(ALPHA_XI_SQ_SHIFT, color="C0", ls=":", lw=1.5,
                label=r"$\alpha_\xi^2(1-\gamma^2)=8019/10000$")
    ax1.plot(ns, rm["T00"], "o-", color="k", ms=4, label=r"$T_{00}$ (sum)")
    ax1.plot(ns, rm["S4_kq_recoil"], "s-", color="C2", ms=4,
             label=r"$S_4$ ($K/Q$ recoil)")
    ax1.scatter([ns.max() * 1.06], [sym["T00"]["y_inf"]], marker=">",
                color="k", zorder=5)
    ax1.scatter([ns.max() * 1.06], [float(ic["structural_sum_value"])],
                marker="*", color="C0", s=90, zorder=5,
                label="structural sum $S_1{+}S_2{+}S_3{+}S_4$")
    ax1.set_xlabel("lattice size $N$")
    ax1.set_ylabel(r"per-regime median")
    ax1.set_title(r"(a) $T_{00}^{\Xi}$ aggregate and dominant summand")
    ax1.legend(fontsize=7, loc="lower right")
    ax1.grid(alpha=0.3)

    # Panel (b): the three sub-leading summands.
    ax2.axhline(ALPHA_XI_SQ * GAMMA_SQ, color="C1", ls="--", lw=1.1,
                label=r"$\alpha_\xi^2\gamma^2=81/10^4$")
    ax2.axhline(0.0, color="0.5", ls=":", lw=1.0)
    ax2.plot(ns, rm["S1_half_var_xi"], "o-", color="C1", ms=4,
             label=r"$S_1=\frac{1}{2}\sigma^2_\Xi$")
    ax2.plot(ns, rm["S2_var_amp"], "^-", color="C4", ms=4,
             label=r"$S_2=\sigma^2_{|\psi|}$")
    ax2.plot(ns, rm["S3_grad_psi_sq"], "v-", color="C5", ms=4,
             label=r"$S_3=|\nabla\psi|^2$")
    for key, col in (("S1_half_var_xi", "C1"), ("S2_var_amp", "C4"),
                     ("S3_grad_psi_sq", "C5")):
        ax2.scatter([ns.max() * 1.06], [sym[key]["y_inf"]], marker=">",
                    color=col, zorder=5)
    ax2.set_xlabel("lattice size $N$")
    ax2.set_ylabel(r"per-regime median")
    ax2.set_title("(b) sub-leading summands $S_1,S_2,S_3$")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(alpha=0.3)

    fig.suptitle("Per-summand continuum decomposition of "
                 r"$T_{00}^{\Xi}$ "
                 f"({d['verdict'].replace('_', ' ').lower()})",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGDIR / "fig17_t00_summand_decomposition.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def fig18():
    dr = _load("signed_ricci_lower_bound_audit.json",
               "verify_signed_ricci_lower_bound")
    db = _load("carrier_bakry_emery_cd_audit.json",
               "verify_carrier_bakry_emery_cd")
    rr = dr["per_regime"]
    rb = db["per_regime"]
    nr = np.array([r["N"] for r in rr], dtype=float)
    nb = np.array([r["N"] for r in rb], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # Panel (a): signed Hessian-Ricci route.
    ax1.axhline(0.0, color="0.5", ls=":", lw=1.0)
    ax1.plot(nr, [r["K_p1"] for r in rr], "o-", color="C0", ms=4,
             label=r"$K_{p1}=\mathrm{p1}\,\lambda_{\min}(R^{\rm Hess})$")
    ax1.plot(nr, [r["K_p5"] for r in rr], "s-", color="C2", ms=4,
             label=r"$K_{p5}=\mathrm{p5}\,\lambda_{\min}(R^{\rm Hess})$")
    ax1.set_xlabel("lattice size $N$")
    ax1.set_ylabel(r"curvature lower bound")
    ax1.set_title("(a) signed Hessian-Ricci route (strengthened A5)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7, loc="lower right")
    axr = ax1.twinx()
    axr.plot(nr, [r["f_neg"] for r in rr], "d--", color="C3", ms=4,
             label=r"$f_{\rm neg}$ (hyperbolic-node fraction)")
    axr.set_ylabel(r"$f_{\rm neg}$", color="C3")
    axr.tick_params(axis="y", labelcolor="C3")
    axr.set_ylim(bottom=-0.002)
    axr.legend(fontsize=7, loc="upper right")

    # Panel (b): intrinsic Bakry-Emery route.
    ax2.axhline(0.0, color="0.5", ls=":", lw=1.0)
    ax2.plot(nb, [r["K_inf"] for r in rb], "o-", color="C0", ms=4,
             label=r"$K_{\inf}=\inf_x K_x$")
    ax2.plot(nb, [r["K_p5"] for r in rb], "s-", color="C2", ms=4,
             label=r"$K_{p5}$")
    ax2.plot(nb, [r["K_mean"] for r in rb], "^-", color="C1", ms=4,
             label=r"$K_{\rm mean}$")
    be = db["cd_curvature"]
    ax2.scatter([nb.max() * 1.06], [be["K_p5_inf"]], marker="*",
                color="C2", s=90, zorder=5)
    ax2.set_xlabel("lattice size $N$")
    ax2.set_ylabel(r"Bakry-Emery curvature $K_x$")
    ax2.set_title(r"(b) intrinsic Bakry-Emery $\Gamma_2$ route")
    ax2.set_ylim(bottom=0.0)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7, loc="lower right")

    fig.suptitle("Empirical witnesses of the curvature-dimension "
                 r"lower bound CD($K_{\rm CD}$,N), $K_{\rm CD}\geq 0$",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGDIR / "fig18_cd_curvature_witnesses.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    fig17()
    fig18()


if __name__ == "__main__":
    main()
