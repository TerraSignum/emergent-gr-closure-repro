"""Generate fig11: Runner A Hessian-Ricci Galerkin convergence."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless CI / no-DISPLAY
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType (vector, arXiv-friendly)
matplotlib.rcParams["ps.fonttype"] = 42

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent

with open(
    f"{REPO}/"
    "outputs/galerkin_runner_A_hessian_ricci.json"
) as f:
    d = json.load(f)

trend = d["trend"]
ns = np.array([t["N"] for t in trend])
mean_blind = np.array([t["blind_frob_mean"] for t in trend])
median_blind = np.array([t["blind_frob_median"] for t in trend])
mean_struct = np.array([t["struct_frob_mean"] for t in trend])
median_struct = np.array([t["struct_frob_median"] for t in trend])
t_off = np.array([t["t_off_F_mean"] for t in trend])
lam_t = np.array([t["blind_lam_t"] for t in trend])

# Power-law fits on N >= 42.
mask = ns >= 42

def pl(N, D, C, a):
    return D + C * N ** (-a)

fits_summary = {}
for label, y in [
    ("mean_blind", mean_blind), ("median_blind", median_blind),
    ("mean_struct", mean_struct), ("median_struct", median_struct),
]:
    try:
        popt, _ = curve_fit(
            pl, ns[mask], y[mask], p0=[0.02, 1, 1.5],
            bounds=([0, 0, 0.1], [0.2, 1e6, 5]), maxfev=20000,
        )
        pred = pl(ns[mask], *popt)
        ss_res = np.sum((y[mask] - pred) ** 2)
        ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        fits_summary[label] = (popt, r2)
        print(f"{label:<20}: D_inf={popt[0]:.4f}, "
              f"alpha={popt[2]:.3f}, R^2={r2:.3f}")
    except Exception as e:
        print(f"{label}: fit failed: {e}")
        fits_summary[label] = (None, None)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: Frob residual (mean + median for both Lambda variants)
ax = axes[0]
ax.semilogy(ns, mean_blind, "o-", color="#1f77b4", ms=10, lw=1.6,
            label=r"mean (blind $\Lambda$)")
ax.semilogy(ns, median_blind, "s--", color="#1f77b4", ms=8, lw=1.0,
            alpha=0.7, label=r"median (blind $\Lambda$)")
ax.semilogy(ns, mean_struct, "^-", color="#cc3333", ms=10, lw=1.6,
            label=r"mean (System-R $\Lambda$)")
ax.semilogy(ns, median_struct, "d--", color="#cc3333", ms=8, lw=1.0,
            alpha=0.7, label=r"median (System-R $\Lambda$)")
ax.axhline(0.05, color="#444", ls=":", lw=1.2,
           label="closure threshold 0.05")
if fits_summary["median_struct"][0] is not None:
    popt = fits_summary["median_struct"][0]
    r2 = fits_summary["median_struct"][1]
    nfit = np.linspace(20, 200, 200)
    yfit = pl(nfit, *popt)
    ax.semilogy(nfit, yfit, ":", color="#cc3333", lw=1.4, alpha=0.5,
                label=fr"fit $\alpha={popt[2]:.2f}$, "
                     fr"$\Delta_\infty={popt[0]:.3f}$, $R^2={r2:.2f}$")
ax.set_xlabel("Lattice size N")
ax.set_ylabel(r"$\|G+\Lambda g - 8\pi G T\|_F$ per node")
ax.set_title("Per-node 4x4 Galerkin Frobenius residual\n(Hessian-Ricci)",
             pad=8)
ax.legend(loc="upper right", framealpha=0.95, fontsize=8)
ax.grid(True, which="both", ls=":", alpha=0.4)

# Panel 2: Lambda_t blind-fit convergence to alpha_xi^2
ax = axes[1]
ax.plot(ns, lam_t, "o-", color="#2c7a2c", ms=10, lw=1.6,
        label=r"blind-fit $\Lambda_t$")
ax.axhline(0.81, color="#444", ls="--", lw=1.4,
           label=r"System-R $\alpha_\xi^2 = 81/100$")
ax.set_xlabel("Lattice size N")
ax.set_ylabel(r"$\Lambda_t$")
ax.set_title(r"$\Lambda_t$ asymptotic convergence to $\alpha_\xi^2$",
             pad=8)
ax.set_ylim(0.5, 2.0)
ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
ax.grid(True, ls=":", alpha=0.4)

# Panel 3: T_off Frobenius decay
ax = axes[2]
ax.loglog(ns, t_off, "o-", color="#9467bd", ms=10, lw=1.6,
          label=r"$\|T_{ij,\mathrm{off}}\|_F$")
try:
    nfit = np.linspace(15, 150, 200)
    log_n = np.log(ns)
    log_y = np.log(np.maximum(t_off, 1e-12))
    slope, icpt = np.polyfit(log_n, log_y, 1)
    yfit = np.exp(icpt) * nfit ** slope
    ax.loglog(nfit, yfit, "--", color="#9467bd", lw=1.4,
              label=fr"fit $\propto N^{{{slope:.2f}}}$")
except Exception:
    pass
ax.set_xlabel("Lattice size N")
ax.set_ylabel(r"$\|T_{ij,\mathrm{off}}\|_F$ mean")
ax.set_title("Off-diagonal Frobenius decay", pad=8)
ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
ax.grid(True, which="both", ls=":", alpha=0.4)

fig.suptitle("Direct per-node Galerkin closure with Hessian-Ricci tensor "
             "(Runner A)",
             fontsize=12, y=1.02)
fig.tight_layout()
out = (f"{REPO}/"
       "paper/figures/fig11_runner_A_hessian_ricci.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
fig.savefig(out.replace(".pdf", ".png"), format="png",
            bbox_inches="tight", dpi=150)
print(f"Saved {out}")
