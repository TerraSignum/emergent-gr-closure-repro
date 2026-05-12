"""Regenerate fig11 with TrueType fonts for tectonic compatibility."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Force TrueType (Type 42) fonts so tectonic can parse the PDF.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["pdf.use14corefonts"] = False

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

mask = ns >= 42

def pl(N, D, C, a):
    return D + C * N ** (-a)

fits_summary = {}
for label, y in [
    ("median_struct", median_struct),
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
    except Exception:
        fits_summary[label] = (None, None)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.semilogy(ns, mean_blind, "o-", color="#1f77b4", ms=10, lw=1.6,
            label="mean (blind L)")
ax.semilogy(ns, median_blind, "s--", color="#1f77b4", ms=8, lw=1.0,
            alpha=0.7, label="median (blind L)")
ax.semilogy(ns, mean_struct, "^-", color="#cc3333", ms=10, lw=1.6,
            label="mean (System-R L)")
ax.semilogy(ns, median_struct, "d--", color="#cc3333", ms=8, lw=1.0,
            alpha=0.7, label="median (System-R L)")
ax.axhline(0.05, color="#444", ls=":", lw=1.2,
           label="closure threshold 0.05")
if fits_summary["median_struct"][0] is not None:
    popt = fits_summary["median_struct"][0]
    r2 = fits_summary["median_struct"][1]
    nfit = np.linspace(20, 200, 200)
    yfit = pl(nfit, *popt)
    ax.semilogy(nfit, yfit, ":", color="#cc3333", lw=1.4, alpha=0.5,
                label=f"fit a={popt[2]:.2f}, D_inf={popt[0]:.3f}, R2={r2:.2f}")
ax.set_xlabel("Lattice size N")
ax.set_ylabel("|G + L g - 8 pi G T|_F per node")
ax.set_title("Per-node 4x4 Galerkin Frob (Hessian-Ricci)", pad=8)
ax.legend(loc="upper right", framealpha=0.95, fontsize=8)
ax.grid(True, which="both", ls=":", alpha=0.4)

ax = axes[1]
ax.plot(ns, lam_t, "o-", color="#2c7a2c", ms=10, lw=1.6,
        label="blind-fit L_t")
ax.axhline(0.81, color="#444", ls="--", lw=1.4,
           label="System-R alpha_xi^2 = 81/100")
ax.set_xlabel("Lattice size N")
ax.set_ylabel("L_t")
ax.set_title("L_t asymptotic convergence to alpha_xi^2", pad=8)
ax.set_ylim(0.5, 2.0)
ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
ax.grid(True, ls=":", alpha=0.4)

ax = axes[2]
ax.loglog(ns, t_off, "o-", color="#9467bd", ms=10, lw=1.6,
          label="|T_ij,off|_F")
try:
    nfit = np.linspace(15, 150, 200)
    log_n = np.log(ns)
    log_y = np.log(np.maximum(t_off, 1e-12))
    slope, icpt = np.polyfit(log_n, log_y, 1)
    yfit = np.exp(icpt) * nfit ** slope
    ax.loglog(nfit, yfit, "--", color="#9467bd", lw=1.4,
              label=f"fit prop N^{slope:.2f}")
except Exception:
    pass
ax.set_xlabel("Lattice size N")
ax.set_ylabel("|T_ij,off|_F mean")
ax.set_title("Off-diagonal Frob decay", pad=8)
ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
ax.grid(True, which="both", ls=":", alpha=0.4)

fig.suptitle("Runner A: per-node Galerkin closure (Hessian-Ricci)",
             fontsize=12, y=1.02)
fig.tight_layout()
out = (f"{REPO}/"
       "paper/figures/fig11_runner_A_hessian_ricci.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
fig.savefig(out.replace(".pdf", ".png"), format="png",
            bbox_inches="tight", dpi=150)
print("Saved fig11 with TrueType fonts.")
