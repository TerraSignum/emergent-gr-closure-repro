"""Make the principal-result figure for the bulk-percentile full-tensor
closure (Stage 6f norm audit).

x-axis: lattice size N
y-axis: per-node relative-Frobenius residual percentile value
6 lines: median, mean, p90, p95, p99, sup
Includes:
  - All regime data points per percentile, regime/seed counts read
    dynamically from outputs/stage6f_full_tensor_norm_audit.json
  - Symanzik-2 fit y(N) = y_inf + b/N for the bulk-percentile lines
  - Threshold marker at 0.05 (median closure) and 0.10 (mean closure)

Output: paper/figures/fig_full_tensor_norm_audit.pdf
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless CI / no-DISPLAY
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType (vector, arXiv-friendly)
matplotlib.rcParams["ps.fonttype"] = 42

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
AUDIT_PATH = REPO / "outputs" / "stage6f_full_tensor_norm_audit.json"
FIG_OUT = REPO / "paper" / "figures" / "fig_full_tensor_norm_audit.pdf"


def main() -> int:
    with open(AUDIT_PATH) as f:
        data = json.load(f)

    rows = data["per_regime"]
    sym = data.get("symanzik_fits", {})

    # Compute regime / seed counts dynamically from the bundled
    # audit JSON so the figure title cannot drift out of sync with
    # the underlying lattice data.
    canonical_rows = [r for r in rows
                      if str(r.get("regime", "")).startswith("P5")]
    alt_rows = [r for r in rows
                if str(r.get("regime", "")).startswith(("P6", "P7", "P8"))]
    n_canonical = len(canonical_rows)
    n_alt = len(alt_rows)
    seeds_canonical = sum(r.get("n_seeds", 0) for r in canonical_rows)
    seeds_alt = sum(r.get("n_seeds", 0) for r in alt_rows)
    n_min = int(min(r["N"] for r in rows))
    n_max = int(max(r["N"] for r in rows))

    # Pull out per-percentile values per regime
    n_arr = np.array([r["N"] for r in rows], dtype=float)
    pct_keys = [
        ("median", "med", "C0", "o", "-"),
        ("mean", "mean", "C1", "s", "-"),
        ("p90", "$p_{90}$", "C2", "^", "-"),
        ("p95", "$p_{95}$", "C3", "D", "-"),
        ("p99", "$p_{99}$", "C4", "v", "--"),
        ("sup", "sup", "C5", "*", ":"),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    # Plot data + Symanzik-2 extrapolation
    n_extrap = np.linspace(40, 1500, 200)
    for stat_key, label, color, marker, ls in pct_keys:
        y = np.array([r["delta_full"][stat_key] for r in rows])
        ax.plot(n_arr, y, marker=marker, linestyle="None",
                color=color, ms=7, label=label)
        if stat_key in sym:
            fit = sym[stat_key]
            y_fit = fit["y_inf"] + fit["b"] / n_extrap
            r2 = fit["r_squared"]
            ax.plot(n_extrap, y_fit, ls=ls, color=color, alpha=0.5,
                    lw=1.0,
                    label=fr"   $\to\,${fit['y_inf']:+.3f} ($R^2={r2:.2f}$)")

    # Threshold markers
    ax.axhline(0.05, color="k", linestyle=":", alpha=0.4, lw=0.8)
    ax.axhline(0.10, color="k", linestyle=":", alpha=0.4, lw=0.8)
    ax.text(35, 0.06, "median closure 0.05", fontsize=8,
             color="k", alpha=0.5)
    ax.text(35, 0.11, "mean closure 0.10", fontsize=8,
             color="k", alpha=0.5)

    ax.axhline(0.0, color="k", linestyle="-", alpha=0.2, lw=0.5)

    ax.set_xscale("log")
    ax.set_xlabel(r"lattice size $N$")
    ax.set_ylabel(r"$\Delta_a = \|R_a\|_F\,/\,\|T_a\|_F$")
    ax.set_title(
        "Bulk-percentile full-tensor closure: per-node "
        "relative-Frobenius residual\n"
        f"canonical $\\mathcal{{P}}_5/\\mathcal{{P}}_5N$ ladder "
        f"$N\\in[{n_min},{n_max}]$ "
        f"({n_canonical} regimes, {seeds_canonical} seeds)"
        + (f" + alt-anchor cross-check "
           f"({n_alt} regimes, {seeds_alt} seeds)"
           if n_alt > 0 else "")
    )
    ax.set_xlim(40, 1500)
    ax.set_ylim(-0.3, 1.1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2,
              framealpha=0.9)

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_OUT, format="pdf", bbox_inches="tight")
    print(f"Saved {FIG_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
