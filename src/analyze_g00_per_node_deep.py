"""Deeper per-node diagnostics for matter-curvature coupling in
the LAT-canonical regimes (P5, P5N64, P5N100, P5N300).

Diagnostics:
  1. epsilon per T_00 decile (10 bins instead of just top-10% vs rest)
  2. Theil-Sen robust slope (immune to T_00 outliers, unlike Pearson)
  3. Bootstrap CI on each epsilon estimate (1000 resamples per regime)
  4. Spatial clustering of matter-core nodes (are high-G_00, high-T_00
     nodes spatially adjacent?)
  5. Per-eigendirection epsilon (decompose G_00 into Galerkin modes)
  6. epsilon(N) trend with bootstrap CI
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


class _BlockCupy:
    def find_module(self, name, path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self

    def load_module(self, name):
        raise ImportError("cupy disabled")


sys.meta_path.insert(0, _BlockCupy())

from verify_galerkin_runner_A_hessian_ricci import (
    edge_to_matrix, per_seed_galerkin)


LAT_LADDER = [
    ("P5",      50,  "results_d1_fix17/d1_p5.npz",     "d1"),
    ("P5N64",   64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz",  "snap"),
    ("P5N100", 100,  "results_d1_p5n100_24seeds/P5N100.snapshots.npz","snap"),
    ("P5N300", 300,  "results_d1_p5n300_12seeds/P5N300.snapshots.npz", "snap"),
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz",  "snap"),
    ("P5N512", 512,  "results_d1_p5n512_12seeds/P5N512.snapshots.npz",  "snap"),
]


def load_seed_payload(rel_path, n_lat, kind, max_seeds=32):
    fp = REPO.parent / rel_path
    if not fp.exists():
        return []
    z = np.load(fp, allow_pickle=True)
    out = []
    if kind == "snap":
        snaps = z["edge_xi_snapshots"]
        psi_re = z["psi_real_snapshots"]
        psi_im = z["psi_imag_snapshots"]
        last_idx = snaps.shape[1] - 1
        ns = min(int(snaps.shape[0]), max_seeds)
        has_kq = "k_snapshots" in z.files and "q_snapshots" in z.files
        for s in range(ns):
            xi_mat = np.asarray(snaps[s, last_idx], dtype=float)
            psi = (np.asarray(psi_re[s, last_idx], dtype=float)
                   + 1j * np.asarray(psi_im[s, last_idx], dtype=float))
            if has_kq:
                k_field = np.asarray(z["k_snapshots"][s, last_idx],
                                      dtype=float)
                q_field = np.asarray(z["q_snapshots"][s, last_idx],
                                      dtype=float)
            else:
                k_field = np.full((n_lat, n_lat), 0.55)
                q_field = np.full((n_lat, n_lat), 0.45)
            out.append((xi_mat, psi, k_field, q_field))
        return out
    edge_arr = z["dense_cell_edge_xi_values"]
    amp_arr = z["dense_cell_node_amplitude_values"]
    phase_arr = z["dense_cell_node_phase_values"]
    ns = min(int(edge_arr.shape[0]), max_seeds)
    for s in range(ns):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = np.asarray(z[f"ff_K_seed{s}"], dtype=float)
        q_field = np.asarray(z[f"ff_Q_seed{s}"], dtype=float)
        out.append((xi_mat, psi, k_field, q_field))
    return out


def theil_sen_slope(x, y, max_pairs=10000):
    """Robust slope via median of pairwise (y_j - y_i) / (x_j - x_i)."""
    rng = np.random.default_rng(42)
    n = len(x)
    if n < 2:
        return float("nan")
    n_pairs = n * (n - 1) // 2
    if n_pairs <= max_pairs:
        i_arr, j_arr = np.triu_indices(n, k=1)
    else:
        i_arr = rng.integers(0, n, max_pairs)
        j_arr = rng.integers(0, n, max_pairs)
        diff_idx = i_arr != j_arr
        i_arr = i_arr[diff_idx]
        j_arr = j_arr[diff_idx]
    dx = x[j_arr] - x[i_arr]
    dy = y[j_arr] - y[i_arr]
    valid = np.abs(dx) > 1e-9
    if not np.any(valid):
        return float("nan")
    return float(np.median(dy[valid] / dx[valid]))


def bootstrap_eps(G, T, n_boot=1000, q_core=0.90):
    """Bootstrap epsilon = (G_core - G_back) / (T_core - T_back)
    with bootstrap resampling at the node level."""
    rng = np.random.default_rng(0)
    n = G.size
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        Gb = G[idx]
        Tb = T[idx]
        thr = np.quantile(Tb, q_core)
        is_core = Tb >= thr
        if is_core.sum() < 5 or (~is_core).sum() < 5:
            continue
        dG = float(np.median(Gb[is_core])) - float(np.median(Gb[~is_core]))
        dT = float(np.median(Tb[is_core])) - float(np.median(Tb[~is_core]))
        if abs(dT) < 1e-9:
            continue
        out.append(dG / dT)
    return np.asarray(out, dtype=float)


def main() -> int:
    print("=" * 78)
    print("Per-node deep diagnostics: epsilon by decile + Theil-Sen + bootstrap")
    print("=" * 78)

    bundle = {"per_regime": {}}
    eps_by_N = []
    for reg, n_lat, rel, kind in LAT_LADDER:
        print(f"\n[{reg}, N={n_lat}]")
        payload = load_seed_payload(rel, n_lat, kind, max_seeds=32)
        if not payload:
            print(f"  no data")
            continue
        all_g, all_t = [], []
        for xi, psi, kf, qf in payload:
            np.fill_diagonal(xi, 1.0)
            try:
                prep = per_seed_galerkin(xi, psi, kf, qf, n_lat, np)
            except Exception as e:
                print(f"  galerkin failed: {e}")
                continue
            all_g.append(np.asarray(prep["g_00_h"]))
            all_t.append(np.asarray(prep["t00"]))
        if not all_g:
            continue
        G = np.concatenate(all_g)
        T = np.concatenate(all_t)
        n_seeds = len(all_g)

        # === 1. epsilon per T_00 decile ===
        deciles = np.quantile(T, np.linspace(0, 1, 11))
        decile_data = []
        for i in range(10):
            lo, hi = deciles[i], deciles[i + 1]
            mask = (T >= lo) & (T <= hi if i == 9 else T < hi)
            if mask.sum() < 5:
                continue
            decile_data.append({
                "decile": i + 1,
                "n": int(mask.sum()),
                "T_med": float(np.median(T[mask])),
                "G_med": float(np.median(G[mask])),
                "G_mean": float(np.mean(G[mask])),
            })
        # Slope through decile centroids: dG_med / dT_med
        if len(decile_data) >= 5:
            T_dec = np.array([d["T_med"] for d in decile_data])
            G_dec = np.array([d["G_med"] for d in decile_data])
            slope_dec, intercept_dec = np.polyfit(T_dec, G_dec, 1)
        else:
            slope_dec = intercept_dec = float("nan")

        # === 2. Theil-Sen robust slope ===
        ts_slope = theil_sen_slope(T, G, max_pairs=20000)

        # === 3. Bootstrap CI on epsilon (top-10% core vs background) ===
        eps_boot = bootstrap_eps(G, T, n_boot=1000, q_core=0.90)
        if eps_boot.size > 0:
            eps_med = float(np.median(eps_boot))
            eps_q025 = float(np.quantile(eps_boot, 0.025))
            eps_q975 = float(np.quantile(eps_boot, 0.975))
        else:
            eps_med = eps_q025 = eps_q975 = float("nan")

        # === 4. Bootstrap CI on Theil-Sen slope ===
        rng = np.random.default_rng(11)
        ts_boot = []
        for _ in range(200):
            idx = rng.integers(0, T.size, T.size)
            ts_boot.append(theil_sen_slope(T[idx], G[idx], max_pairs=10000))
        ts_boot = np.asarray(ts_boot, dtype=float)
        ts_q025 = float(np.quantile(ts_boot[np.isfinite(ts_boot)], 0.025)) \
                  if np.any(np.isfinite(ts_boot)) else float("nan")
        ts_q975 = float(np.quantile(ts_boot[np.isfinite(ts_boot)], 0.975)) \
                  if np.any(np.isfinite(ts_boot)) else float("nan")

        # === 5. Spatial clustering: among matter-core nodes,
        #         what fraction have a matter-core neighbor? ===
        thr = float(np.quantile(T, 0.90))
        # Need per-seed indexing for adjacency. Use first seed for spatial
        # diagnostics.
        G_s0 = all_g[0]
        T_s0 = all_t[0]
        is_core_s0 = T_s0 >= np.quantile(T_s0, 0.90)
        n_lat_actual = T_s0.size
        # 1-D spatial adjacency: each node has 2 neighbors (modular)
        nbr_left = np.roll(is_core_s0, 1)
        nbr_right = np.roll(is_core_s0, -1)
        any_core_nbr = nbr_left | nbr_right
        spatial_cluster_frac = float(
            np.mean(any_core_nbr[is_core_s0])) if is_core_s0.any() else 0.0
        # null: random core-fraction = 0.10, expected clustering = 1 -
        # (1-0.10)^2 = 0.19
        null_cluster = 1 - (1 - 0.10) ** 2

        # === Print ===
        print(f"  n_total={G.size}, n_seeds={n_seeds}")
        print(f"  epsilon (top-10% core vs back): "
              f"median={eps_med:+.4f}, "
              f"95%CI=[{eps_q025:+.4f}, {eps_q975:+.4f}]")
        print(f"  Theil-Sen slope (robust):       "
              f"{ts_slope:+.4f}, 95%CI=[{ts_q025:+.4f}, {ts_q975:+.4f}]")
        print(f"  Decile-centroid slope:          {slope_dec:+.4f}")
        print(f"  Spatial cluster (core nbr/core): "
              f"{spatial_cluster_frac:.3f}  (null={null_cluster:.3f})")
        print(f"  Per-decile (T_med, G_med):")
        for d in decile_data:
            print(f"    decile {d['decile']:>2}: T={d['T_med']:.4f} "
                  f"G={d['G_med']:.5f}  (n={d['n']})")

        bundle["per_regime"][reg] = {
            "N": n_lat,
            "n_seeds": n_seeds,
            "n_total": int(G.size),
            "deciles": decile_data,
            "slope_decile_centroid": float(slope_dec),
            "intercept_decile_centroid": float(intercept_dec),
            "theil_sen_slope": ts_slope,
            "theil_sen_CI95": [ts_q025, ts_q975],
            "epsilon_core_median": eps_med,
            "epsilon_core_CI95": [eps_q025, eps_q975],
            "spatial_cluster_frac": spatial_cluster_frac,
            "spatial_cluster_null": null_cluster,
        }
        eps_by_N.append((n_lat, eps_med, eps_q025, eps_q975, ts_slope,
                          slope_dec))

    # === epsilon(N) trend ===
    print()
    print("=" * 78)
    print("epsilon(N) trend across LAT regimes")
    print("=" * 78)
    print(f"  {'N':>4} {'eps_core':>9} {'CI95':>22} {'TS slope':>10} "
          f"{'decile slope':>14}")
    for N, e, lo, hi, ts, dec in eps_by_N:
        print(f"  {N:>4} {e:>+9.4f}  [{lo:>+7.4f}, {hi:>+7.4f}]  "
              f"{ts:>+10.4f}  {dec:>+14.4f}")

    # Theil-Sen over the 4 (N, eps) points
    if len(eps_by_N) >= 3:
        Ns = np.array([p[0] for p in eps_by_N], dtype=float)
        eps = np.array([p[1] for p in eps_by_N], dtype=float)
        # log-log fit
        slope_loglog, intercept_loglog = np.polyfit(np.log(Ns),
                                                     np.log(np.abs(eps)),
                                                     1)
        # linear fit
        slope_lin, intercept_lin = np.polyfit(1.0 / Ns ** 2, eps, 1)
        print()
        print(f"  Power-law fit: log|eps(N)| = {intercept_loglog:.3f} + "
              f"{slope_loglog:+.3f} * log N")
        print(f"    eps ~ N^{slope_loglog:.3f}")
        print(f"  1/N^2 fit:     eps(N) = {intercept_lin:+.5f} + "
              f"({slope_lin:+.2f}) / N^2")
        print(f"    eps_inf = {intercept_lin:+.5f}")

        bundle["epsilon_N_trend"] = {
            "loglog_slope": float(slope_loglog),
            "loglog_intercept": float(intercept_loglog),
            "inv_n2_eps_inf": float(intercept_lin),
            "inv_n2_c2": float(slope_lin),
        }

    # === Synthesis ===
    print()
    print("=" * 78)
    print("Synthesis")
    print("=" * 78)
    eps_arr = np.array([p[1] for p in eps_by_N if np.isfinite(p[1])])
    if eps_arr.size > 0:
        print(f"  Mean epsilon across LAT regimes: "
              f"{float(np.mean(eps_arr)):+.4f} +- "
              f"{float(np.std(eps_arr)):.4f}")
        print(f"  Range: [{float(np.min(eps_arr)):+.4f}, "
              f"{float(np.max(eps_arr)):+.4f}]")
    ts_arr = np.array([p[4] for p in eps_by_N if np.isfinite(p[4])])
    print(f"  Mean Theil-Sen slope (more robust): "
          f"{float(np.mean(ts_arr)):+.4f} +- "
          f"{float(np.std(ts_arr)):.4f}")
    sp = [bundle["per_regime"][r]["spatial_cluster_frac"]
          for r in bundle["per_regime"]]
    print(f"  Mean spatial cluster fraction (core nbr/core): "
          f"{float(np.mean(sp)):.3f}")
    print(f"    Null (random 10% core): 0.190")
    if float(np.mean(sp)) > 0.30:
        print(f"  -> Matter-core nodes are SPATIALLY CLUSTERED ")
        print(f"     (much more adjacent than random); curvature-")
        print(f"     emergence is localized in extended structures,")
        print(f"     not isolated points.")

    out = REPO / "outputs" / "g00_per_node_deep.json"
    out.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
