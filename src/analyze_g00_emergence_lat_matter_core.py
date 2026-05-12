"""Corrected deeper analysis: drop the K/Q-bug DEF regimes
(P5N72/84/128/200 ran with default K=0.55, Q=0.45 -- not a different
physics, just a buggy lattice run halving T_00). Use only the LAT
canonical regimes (P5, P5N64, P5N100, P5N300), and probe the matter-
curvature coupling at the PER-NODE level by stratifying nodes into
matter-core (top-10% T_00) vs background.

If the 'anti-wave from matter-wave' picture is correct:
  Matter-core nodes (high T_00) should show ELEVATED G_00 vs
  background nodes WITHIN THE SAME REGIME (i.e. independent of K/Q
  ambiguity).

This is a real per-node Einstein-coupling test, not a cross-population
artifact.
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


# LAT-canonical (proper K/Q) regimes; updated 2026-05-02 with new
# 12-seed K/Q-fixed P5N128 + 24-seed P5N64/72/84/100 and 8-seed P5N200.
# Discovery layer auto-picks the highest-seed-count source per regime.
LAT_LADDER = [
    ("P5",      50,  "results_d1_fix17/d1_p5.npz",                 "d1"),
    ("P5N64",   64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz",   "snap"),
    ("P5N100", 100,  "results_d1_p5n100_24seeds/P5N100.snapshots.npz", "snap"),
    ("P5N128", 128,  "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz","snap"),
    ("P5N200", 200,  "results_d1_p5n200_8seeds/P5N200.snapshots.npz",  "snap"),
    ("P5N300", 300,  "results_d1_p5n300_12seeds/P5N300.snapshots.npz",         "snap"),
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz",  "snap"),
    ("P5N512", 512,  "results_d1_p5n512_12seeds/P5N512.snapshots.npz",  "snap"),
]


def load_seed_payload(rel_path, n_lat, kind, max_seeds=32):
    fp = REPO.parent / rel_path
    if not fp.exists():
        return []
    z = np.load(fp, allow_pickle=True)
    K_DEF, Q_DEF = 0.55, 0.45
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
                k_field = np.full((n_lat, n_lat), K_DEF)
                q_field = np.full((n_lat, n_lat), Q_DEF)
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


def main() -> int:
    print("=" * 78)
    print("Per-node G_00 vs T_00 stratified by matter-core, LAT only")
    print("=" * 78)

    summary_rows = []
    per_regime_node_data = {}
    for reg, n_lat, rel, kind in LAT_LADDER:
        print(f"\n[{reg}, N={n_lat}]")
        payload = load_seed_payload(rel, n_lat, kind, max_seeds=32)
        if not payload:
            print(f"  no payload at {rel}")
            continue
        all_g, all_t = [], []
        for s, (xi_mat, psi, k_f, q_f) in enumerate(payload):
            np.fill_diagonal(xi_mat, 1.0)
            try:
                prep = per_seed_galerkin(xi_mat, psi, k_f, q_f, n_lat, np)
            except Exception as e:
                print(f"  seed {s} failed: {e}")
                continue
            g00 = np.asarray(prep["g_00_h"])
            t00 = np.asarray(prep["t00"])
            all_g.append(g00)
            all_t.append(t00)
        if not all_g:
            continue
        G = np.concatenate(all_g)
        T = np.concatenate(all_t)
        # Pearson + Spearman
        r_pearson = float(np.corrcoef(G, T)[0, 1])
        # Spearman via rank
        rg = np.argsort(np.argsort(G))
        rt = np.argsort(np.argsort(T))
        r_spearman = float(np.corrcoef(rg, rt)[0, 1])

        # Matter-core: top 10% T_00
        thr = float(np.quantile(T, 0.90))
        is_core = T >= thr
        G_core = G[is_core]
        T_core = T[is_core]
        G_back = G[~is_core]
        T_back = T[~is_core]

        # Per-node linear regression G = a + b * T
        slope, intercept = np.polyfit(T, G, 1)

        per_regime_node_data[reg] = {
            "n_lat": n_lat, "n_nodes": int(G.size),
            "G_med": float(np.median(G)), "T_med": float(np.median(T)),
            "G_mean": float(np.mean(G)), "T_mean": float(np.mean(T)),
            "G_core_med": float(np.median(G_core)),
            "T_core_med": float(np.median(T_core)),
            "G_back_med": float(np.median(G_back)),
            "T_back_med": float(np.median(T_back)),
            "r_pearson": r_pearson,
            "r_spearman": r_spearman,
            "slope_G_vs_T": float(slope),
            "intercept_G_vs_T": float(intercept),
            "G_core_minus_G_back": float(np.median(G_core)
                                          - np.median(G_back)),
            "T_core_minus_T_back": float(np.median(T_core)
                                          - np.median(T_back)),
        }

        print(f"  n_nodes total  = {G.size}")
        print(f"  T_00 median    = {float(np.median(T)):.4f}, "
              f"max = {float(np.max(T)):.4f}")
        print(f"  G_00 median    = {float(np.median(G)):.5f}, "
              f"max = {float(np.max(G)):.5f}")
        print(f"  Pearson r(G,T) = {r_pearson:+.3f}")
        print(f"  Spearman r     = {r_spearman:+.3f}")
        print(f"  Slope G = a + b*T:  b = {slope:+.4f}, a = {intercept:+.5f}")
        print()
        print(f"  Matter-core (top 10%): T_med = {float(np.median(T_core)):.4f}, "
              f"G_med = {float(np.median(G_core)):.5f}")
        print(f"  Background:           T_med = {float(np.median(T_back)):.4f}, "
              f"G_med = {float(np.median(G_back)):.5f}")
        print(f"  G_core - G_back =     {float(np.median(G_core)) - float(np.median(G_back)):+.5f}")
        print(f"  T_core - T_back =     {float(np.median(T_core)) - float(np.median(T_back)):+.5f}")
        if (np.median(T_core) - np.median(T_back)) > 1e-6:
            local_eps = (float(np.median(G_core)) - float(np.median(G_back))) \
                        / (float(np.median(T_core)) - float(np.median(T_back)))
            per_regime_node_data[reg]["local_epsilon_core_vs_back"] = local_eps
            print(f"  Local eps = (dG/dT)|core_vs_back = {local_eps:+.4f}")
        summary_rows.append((reg, n_lat, per_regime_node_data[reg]))

    # === Cross-regime comparison ===
    print()
    print("=" * 78)
    print("Cross-regime trend (LAT only, 4 N-points)")
    print("=" * 78)
    print(f"{'reg':>8} {'N':>4} {'r(G,T)':>8} {'slope':>8} "
          f"{'eps_core':>10} {'G_med':>9} {'T_med':>9}")
    for reg, n_lat, d in summary_rows:
        eps = d.get("local_epsilon_core_vs_back", float("nan"))
        print(f"{reg:>8} {n_lat:>4} {d['r_pearson']:>+8.3f} "
              f"{d['slope_G_vs_T']:>+8.4f} {eps:>+10.4f} "
              f"{d['G_med']:>9.5f} {d['T_med']:>9.4f}")

    # Mean local epsilon across regimes
    eps_arr = [d.get("local_epsilon_core_vs_back")
               for _, _, d in summary_rows
               if "local_epsilon_core_vs_back" in d]
    eps_arr = np.asarray(eps_arr, dtype=float)
    if eps_arr.size > 0:
        print()
        print(f"  Mean local epsilon (core vs background): "
              f"{float(np.mean(eps_arr)):+.4f} +- {float(np.std(eps_arr)):.4f}")
        print(f"  Range: [{float(np.min(eps_arr)):+.4f}, "
              f"{float(np.max(eps_arr)):+.4f}]")
    slope_arr = np.array([d["slope_G_vs_T"] for _, _, d in summary_rows])
    if slope_arr.size > 0:
        print(f"  Per-node regression slope (G = a + b*T): "
              f"mean b = {float(np.mean(slope_arr)):+.4f} +- "
              f"{float(np.std(slope_arr)):.4f}")
    r_arr = np.array([d["r_pearson"] for _, _, d in summary_rows])
    if r_arr.size > 0:
        print(f"  Per-node Pearson r(G,T): "
              f"mean = {float(np.mean(r_arr)):+.3f} +- "
              f"{float(np.std(r_arr)):.3f}")

    print()
    print("=" * 78)
    print("Synthesis")
    print("=" * 78)
    print()
    if abs(float(np.mean(eps_arr))) < 0.001:
        verdict = ("eps ~ 0: matter-core nodes do NOT carry elevated "
                   "G_00 vs background. The per-node Galerkin G_00 is "
                   "K/Q-noise-floor + lattice-cutoff, NOT matter-coupled.")
    elif abs(float(np.mean(eps_arr))) < 0.05:
        verdict = (f"eps = {float(np.mean(eps_arr)):+.4f} (tiny but "
                   "consistent across regimes): there IS a small "
                   "matter-coupled response at the per-node level, "
                   "two orders of magnitude weaker than Einstein 8 pi G "
                   "= 1. Real, but not standard GR-strength.")
    elif abs(float(np.mean(eps_arr))) < 0.5:
        verdict = (f"eps = {float(np.mean(eps_arr)):+.4f}: substantial "
                   "matter-coupled response, intermediate between zero "
                   "and Einstein-strength.")
    else:
        verdict = (f"eps = {float(np.mean(eps_arr)):+.4f}: full "
                   "Einstein-coupling-strength matter response per-node.")
    print(verdict)
    print()
    print("Implication for the 'anti-wave from matter-wave' picture:")
    print("  (a) If r(G,T) > 0.3 within-regime: per-node G_00 IS")
    print("      matter-coupled, anti-wave-from-matter-wave is real.")
    print("  (b) If r(G,T) ~ 0 within-regime: per-node G_00 is NOT")
    print("      matter-coupled, the anti-wave dies as pure lattice")
    print("      cutoff noise, NOT as Einstein response.")

    out = REPO / "outputs" / "g00_emergence_LAT_matter_core.json"
    out.write_text(json.dumps({
        "ladder": [d for _, _, d in summary_rows],
        "mean_local_epsilon": float(np.mean(eps_arr)) if eps_arr.size else None,
        "mean_per_node_slope": float(np.mean(slope_arr)) if slope_arr.size else None,
        "mean_pearson_r": float(np.mean(r_arr)) if r_arr.size else None,
        "verdict": verdict,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
