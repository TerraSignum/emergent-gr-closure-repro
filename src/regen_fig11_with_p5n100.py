"""Regenerate fig11 (Runner A Hessian-Ricci convergence) including
the P5N100 within-regime point."""

import json

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent

with open(
    f"{REPO}/"
    "outputs/galerkin_runner_A_hessian_ricci.json"
) as f:
    d = json.load(f)

trend = sorted(d["trend"], key=lambda t: t["N"])
ns = np.array([t["N"] for t in trend])
mean_blind = np.array([t["blind_frob_mean"] for t in trend])
median_blind = np.array([t["blind_frob_median"] for t in trend])
mean_struct = np.array([t["struct_frob_mean"] for t in trend])
median_struct = np.array([t["struct_frob_median"] for t in trend])
t_off = np.array([t["t_off_F_mean"] for t in trend])
lam_t = np.array([t["blind_lam_t"] for t in trend])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.semilogy(ns, mean_blind, "o-", label=r"mean (blind $\Lambda$)", color="#1f77b4")
ax.semilogy(ns, median_blind, "s-", label=r"median (blind $\Lambda$)", color="#ff7f0e")
ax.semilogy(ns, mean_struct, "^--", label=r"mean (struct $\Lambda$)", color="#2ca02c")
ax.semilogy(ns, median_struct, "v--", label=r"median (struct $\Lambda$)", color="#d62728")
ax.axhline(0.05, color="gray", linestyle=":", label="closure threshold 0.05")
ax.set_xlabel(r"lattice size $N$")
ax.set_ylabel("per-node 4x4 Frobenius residual")
ax.set_title("Runner A: Hessian-Ricci Galerkin convergence")
ax.legend(fontsize=8, loc="lower left")
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(ns, lam_t, "o-", color="#1f77b4")
ax.axhline(0.81, color="gray", linestyle=":", label=r"structural $\alpha_\xi^2 = 81/100$")
ax.set_xlabel(r"lattice size $N$")
ax.set_ylabel(r"blind-fit $\Lambda_t^*$")
ax.set_title(r"Asymptotic $\Lambda_t \to \alpha_\xi^2$")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.5, 2.0)

ax = axes[2]
ax.semilogy(ns, t_off, "o-", color="#1f77b4")
ax.set_xlabel(r"lattice size $N$")
ax.set_ylabel(r"off-diagonal $\|T_{ij}\|_F$ mean")
ax.set_title("Off-diagonal stress decay")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    f"{REPO}/"
    "paper/figures/fig11_runner_A_hessian_ricci.pdf",
    bbox_inches="tight",
)
plt.savefig(
    f"{REPO}/"
    "paper/figures/fig11_runner_A_hessian_ricci.png",
    bbox_inches="tight",
    dpi=150,
)
print(f"Saved fig11 with {len(trend)} data points (N range: {ns.min()}-{ns.max()})")
