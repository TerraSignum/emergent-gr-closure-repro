"""Figures for the bulk-core signed-balance / NFW-halo signature.

Produces:
  fig_signed_balance_vs_N.pdf
      Per-regime S_core(N) and S_bulk(N) across the cleaned ladder,
      showing the regime-stable bulk-negative amplitude and the
      decreasing matter-core amplitude.

  fig_nfw_halo_profile.pdf
      Mean |tr(R)|_bulk binned vs distance to nearest defect, with
      an NFW reference profile overlaid for a representative
      regime.

  fig_3d_halo_around_defect.pdf
      3D scatter on a representative regime of the per-node bulk
      residual amplitude in spatial Fiedler-frame coordinates,
      colour-coded so the halo extending from defect cores is
      visible.

Reads inputs:
  outputs/stage6f_signed_bulk_core_balance.json  (per-regime sums)
  Recomputes per-node arrays for the radial-profile and 3D scatter
  on a single representative regime (P5N100 by default) using the
  same per_seed_galerkin pipeline as the audit scripts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


class _BlockCupy:
    def find_module(self, name, path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self

    def load_module(self, _name):
        raise ImportError("cupy disabled")


sys.meta_path.insert(0, _BlockCupy())

from stage6f_full_tensor_norm_audit import (  # noqa: E402
    LAMBDA_T, LAMBDA_S, per_node_relative_delta,
    load_canonical, load_snapshots)
from verify_galerkin_runner_A_hessian_ricci import (  # noqa: E402
    XI_THRESH, ELL_0, per_seed_galerkin)
from _d1_npz_discovery import find_d1_npz  # noqa: E402

OUT_DIR = REPO / "paper" / "figures"
SUMMARY = REPO / "outputs" / "stage6f_signed_bulk_core_balance.json"
FIG1 = OUT_DIR / "fig_signed_balance_vs_N.pdf"
FIG2 = OUT_DIR / "fig_nfw_halo_profile.pdf"
FIG3 = OUT_DIR / "fig_3d_halo_around_defect.pdf"

CORE_TAU = 0.05
REPRESENTATIVE_REGIME = "P5N100"
REPRESENTATIVE_N = 100


def _signed_trace(prep, lam_t=LAMBDA_T, lam_s=LAMBDA_S):
    g_00 = prep["g_00_h"]
    g_ij = prep["g_ij_h"]
    t00 = prep["t00"]
    t_ij = prep["t_ij"]
    res_00 = g_00 + lam_t - t00
    eye3 = np.eye(3)
    res_d = (g_ij + lam_s * eye3[None, :, :]) - t_ij
    return res_00 + res_d[:, 0, 0] + res_d[:, 1, 1] + res_d[:, 2, 2]


def _spatial_frame(xi_mat, n_lat):
    np.fill_diagonal(xi_mat, 1.0)
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(float)
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1) + 1e-12
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    l_norm = (np.eye(n_lat)
               - deg_inv_sqrt[:, None] * weight_adj * deg_inv_sqrt[None, :])
    eigvals_l, eigvecs_l = np.linalg.eigh(l_norm)
    return eigvecs_l[:, 1:4], xi_off, adj


def _shortest_dist_to_set(adj, xi_off, target_idx):
    n = adj.shape[0]
    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat = np.where(adj > 0, d_mat, np.inf)
    dist = np.full(n, np.inf)
    dist[target_idx] = 0.0
    visited = np.zeros(n, dtype=bool)
    for _ in range(n):
        u = -1
        best = np.inf
        for i in range(n):
            if not visited[i] and dist[i] < best:
                best = dist[i]
                u = i
        if u < 0 or best == np.inf:
            break
        visited[u] = True
        for v in range(n):
            if not visited[v] and adj[u, v] > 0 and dist[u] + d_mat[u, v] < dist[v]:
                dist[v] = dist[u] + d_mat[u, v]
    return dist


def _gather_representative(regime, n_lat, max_seeds=24):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    seeds = (load_snapshots(p, n_lat)
             if "snapshots" in p.name.lower()
             else load_canonical(p, n_lat))
    out = []
    for s_idx, (xi_mat, psi, k_field, q_field) in enumerate(seeds):
        if s_idx >= max_seeds:
            break
        try:
            prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
            tr = _signed_trace(prep)
            df = per_node_relative_delta(prep, LAMBDA_T, LAMBDA_S)["delta_full"]
            spatial, xi_off, adj = _spatial_frame(np.asarray(xi_mat), n_lat)
            mask_core = df > CORE_TAU
            t00 = prep["t00"]
            target_idx = int(np.argmax(t00))
            dist = _shortest_dist_to_set(adj, xi_off, target_idx)
            out.append({
                "tr": tr, "df": df, "t00": t00, "spatial": spatial,
                "mask_core": mask_core, "dist_to_defect": dist,
                "target_idx": target_idx,
            })
        except Exception as exc:  # noqa: BLE001
            print(f"  seed {s_idx} skipped: {exc}")
            continue
    return out


def fig1_balance_vs_N():
    raw = json.loads(SUMMARY.read_text(encoding="utf-8"))
    # Canonical P5/P5N N-ordered ladder only; alt-anchor regimes
    # (P6/P7/P8) are reported separately as cross-anchor consistency
    # checks and do not belong on this canonical-ladder figure.
    rows = sorted(
        [r for r in raw["per_regime"]
         if str(r.get("regime", "")).startswith("P5")],
        key=lambda r: r["N"])
    n_arr = np.array([r["N"] for r in rows], dtype=float)
    s_core = np.array([r["S_core_per_node"] for r in rows], dtype=float)
    s_bulk = np.array([r["S_bulk_per_node"] for r in rows], dtype=float)
    s_total = s_core + s_bulk

    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=160)
    ax.axhline(0, color="gray", lw=0.8, alpha=0.6)
    ax.plot(n_arr, s_core, "o-", color="C3", lw=1.5, ms=7,
             label=r"$S_{\mathrm{core}}/|V_N|$  (matter cores, $\Delta>\tau$)")
    ax.plot(n_arr, s_bulk, "s-", color="C0", lw=1.5, ms=7,
             label=r"$S_{\mathrm{bulk}}/|V_N|$  (NFW-halo region)")
    ax.plot(n_arr, s_total, "^--", color="black", lw=1.0, ms=5,
             label=r"$\sum$ (Bianchi-consistent $\to 0$)")
    ax.set_xlabel(r"lattice size $N$", fontsize=11)
    ax.set_ylabel(r"per-node signed trace residual",
                   fontsize=11)
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 200, 300, 512])
    ax.set_xticklabels(["50", "100", "200", "300", "512"])
    ax.set_title(r"Signed bulk-core balance on $\mathcal{P}_{5}/\mathcal{P}_{5}N$ ladder",
                  fontsize=11)
    ax.legend(loc="best", fontsize=9, framealpha=0.92)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG1, format="pdf", bbox_inches="tight")
    fig.savefig(FIG1.with_suffix(".png"), format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG1}")


def fig2_radial_profile(seeds):
    if not seeds:
        return
    # Pool over seeds: |tr(R)| in bulk vs distance to nearest defect
    bulk_amp = []
    bulk_dist = []
    for s in seeds:
        m_bulk = ~s["mask_core"]
        finite = np.isfinite(s["dist_to_defect"]) & m_bulk
        bulk_amp.append(np.abs(s["tr"][finite]))
        bulk_dist.append(s["dist_to_defect"][finite])
    amp = np.concatenate(bulk_amp)
    dist = np.concatenate(bulk_dist)
    # Bin
    n_bins = 14
    bin_edges = np.linspace(dist.min(), np.percentile(dist, 95), n_bins + 1)
    bin_mid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_amp = np.zeros(n_bins)
    bin_err = np.zeros(n_bins)
    bin_n = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        msk = (dist >= bin_edges[i]) & (dist < bin_edges[i + 1])
        if msk.sum() > 0:
            bin_amp[i] = amp[msk].mean()
            bin_err[i] = amp[msk].std() / np.sqrt(max(msk.sum(), 1))
            bin_n[i] = msk.sum()

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=160)
    ax.errorbar(bin_mid, bin_amp, yerr=bin_err, fmt="o", color="C0",
                 capsize=3, label=f"binned $\\langle|tr(R)|\\rangle_{{\\rm bulk}}$ on {REPRESENTATIVE_REGIME}")
    # NFW reference: rho(r) ~ 1/(r * (r_s + r)^2)
    valid = bin_n > 5
    if valid.sum() >= 4:
        from scipy.optimize import curve_fit
        def nfw(r, c0, r_s):
            r_safe = np.maximum(r, 1e-9)
            return c0 / (r_safe * (r_s + r_safe) ** 2)
        try:
            popt, _ = curve_fit(nfw, bin_mid[valid], bin_amp[valid],
                                  p0=[bin_amp[valid][0] * bin_mid[valid][0]
                                      * (bin_mid[valid][0] + 1) ** 2,
                                      bin_mid[valid].mean()],
                                  maxfev=4000)
            r_plot = np.linspace(bin_mid[valid].min(),
                                  bin_mid[valid].max(), 80)
            ax.plot(r_plot, nfw(r_plot, *popt), "-", color="C3", lw=1.5,
                     label=f"NFW fit, $r_s={popt[1]:.2f}$")
        except Exception as exc:  # noqa: BLE001
            print(f"  NFW fit failed: {exc}")
    ax.set_xlabel(r"distance to nearest matter-core defect (lattice units)",
                   fontsize=11)
    ax.set_ylabel(r"$\langle|\mathrm{tr}(R)|\rangle_{\mathrm{bulk}}$",
                   fontsize=11)
    ax.set_title(f"Radial profile of the bulk signed-trace amplitude "
                  f"around defects ({REPRESENTATIVE_REGIME})",
                  fontsize=11)
    ax.legend(loc="best", fontsize=9, framealpha=0.92)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG2, format="pdf", bbox_inches="tight")
    fig.savefig(FIG2.with_suffix(".png"), format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG2}")


def fig3_3d_halo(seeds):
    if not seeds:
        return
    s = seeds[0]
    spatial = s["spatial"]
    tr = s["tr"]
    target_idx = s["target_idx"]
    # Use sign and amplitude for colour; mark core nodes with markers
    finite = np.isfinite(tr)
    spatial = spatial[finite]
    tr = tr[finite]
    mask_core = s["mask_core"][finite]

    fig = plt.figure(figsize=(8.0, 6.0), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    bulk = ~mask_core
    sc = ax.scatter(spatial[bulk, 0], spatial[bulk, 1], spatial[bulk, 2],
                     c=tr[bulk], cmap="coolwarm",
                     vmin=-0.10, vmax=+0.10,
                     s=14, alpha=0.55,
                     label="bulk (NFW halo region)")
    ax.scatter(spatial[mask_core, 0], spatial[mask_core, 1],
                spatial[mask_core, 2],
                c="black", marker="x", s=40, alpha=0.85,
                label="matter-core defect nodes ($\\Delta>\\tau$)")
    if target_idx < len(spatial):
        ax.scatter([spatial[target_idx, 0]], [spatial[target_idx, 1]],
                    [spatial[target_idx, 2]],
                    c="red", marker="*", s=240,
                    edgecolors="k", lw=1.0, alpha=1.0,
                    label="reference defect ($\\arg\\max T_{00}$)")
    cb = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.07)
    cb.set_label(r"$\mathrm{tr}(R)$ at node", fontsize=10)
    ax.set_xlabel(r"Fiedler $x$", fontsize=10)
    ax.set_ylabel(r"Fiedler $y$", fontsize=10)
    ax.set_zlabel(r"Fiedler $z$", fontsize=10)
    ax.set_title(f"3D spatial layout of signed trace residual ({REPRESENTATIVE_REGIME})",
                  fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(FIG3, format="pdf", bbox_inches="tight")
    fig.savefig(FIG3.with_suffix(".png"), format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG3}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig1_balance_vs_N()
    seeds = _gather_representative(REPRESENTATIVE_REGIME, REPRESENTATIVE_N)
    fig2_radial_profile(seeds)
    fig3_3d_halo(seeds)


if __name__ == "__main__":
    main()
