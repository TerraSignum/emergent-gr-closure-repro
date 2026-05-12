"""Generate fig12: Within-regime P5-physics convergence at three
lattice sizes N=50, 64, 100, all sharing the canonical regime
parameter set (lambda_triangle, epsilon, defect_params).

This is the within-regime falsification companion to fig11
(across-regime ladder): if the residual reduction we see across
the canonical ladder were merely a regime-physics coincidence,
it would not appear within a single regime as N increases. The
sequence here shows it does — monotone decrease from 0.109 at
N=50 to 0.036 at N=100, within the same regime."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(
    str(REPO)
)

with open(REPO / "outputs" / "within_p5_runner_A_hessian_ricci.json") as f:
    d = json.load(f)

trend = d["trend"]
by_regime = {t["regime"]: t for t in trend}
within = []
for label in ("P5", "P5N64", "P5N72", "P5N84", "P5N100", "P5N128", "P5N200", "P5N256", "P5N300", "P5N512"):
    if label not in by_regime:
        continue
    t = by_regime[label]
    within.append((t["N"], t["blind_frob_median"], t["blind_frob_mean"]))
within = sorted(within, key=lambda r: r[0])
ns = np.array([r[0] for r in within])
fmed = np.array([r[1] for r in within])
fmean = np.array([r[2] for r in within])

if len(ns) >= 2:
    log_n = np.log(ns)
    log_med = np.log(fmed)
    slope, intercept = np.polyfit(log_n, log_med, 1)
    fit_alpha = -slope
    log_pred = intercept + slope * log_n
    ss_res = float(np.sum((log_med - log_pred) ** 2))
    ss_tot = float(np.sum((log_med - log_med.mean()) ** 2))
    fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
else:
    fit_alpha = None
    fit_r2 = None

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(ns, fmed, "o-", color="#1f77b4", markersize=10,
             label="median (blind $\\Lambda$)")
ax.semilogy(ns, fmean, "s--", color="#ff7f0e", markersize=10,
             label="mean (blind $\\Lambda$)")
ax.axhline(0.05, color="gray", linestyle=":",
            label="closure threshold 0.05")
if fit_alpha is not None:
    ax.text(0.55, 0.85,
             rf"log-log fit: $\alpha={fit_alpha:.2f}$, $R^2={fit_r2:.2f}$"
             f" ({len(ns)} pts)",
             transform=ax.transAxes,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
ax.set_xscale("log")
ax.set_xticks(ns)
ax.set_xticklabels([f"$N={int(n)}$" for n in ns], rotation=45)
ax.set_ylabel("per-node 4x4 Frobenius residual")
# Title built dynamically from the actually-loaded N values so it cannot
# drift from the underlying data file.
n_list_str = ",".join(str(int(n)) for n in ns)
ax.set_title("Within-regime convergence at fixed P5 physics "
             rf"({len(ns)} sizes, $N\in\{{{n_list_str}\}}$)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3, which="both")

# Annotate ratios
for i, n in enumerate(ns):
    ax.annotate(f"{fmed[i]:.3f}",
                 (n, fmed[i]),
                 textcoords="offset points", xytext=(0, 12),
                 ha="center", fontsize=10, color="#1f77b4")

plt.tight_layout()
plt.savefig(REPO / "paper" / "figures" / "fig12_within_regime_p5.pdf",
             bbox_inches="tight")
plt.savefig(REPO / "paper" / "figures" / "fig12_within_regime_p5.png",
             bbox_inches="tight", dpi=150)
print(f"Saved fig12 with {len(within)} within-regime points")
print(f"Sequence: {[(int(n), float(m)) for n,m in zip(ns, fmed)]}")
if fit_alpha is not None:
    print(f"Fitted alpha = {fit_alpha:.3f}")
