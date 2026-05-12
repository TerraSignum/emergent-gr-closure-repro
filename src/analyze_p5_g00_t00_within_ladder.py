"""Compute G_00(N) and T_00(N) for the full within-P5 ladder
(N = 50, 64, 72, 84, 100, 128, 200, 300) using the per-node 4x4
Galerkin (Hessian-Ricci) pipeline, then assess:

  1. T_00(N) "matter wave"        - asymptotic plateau
  2. G_00(N) "anti-wave / QFT"    - asymptotic plateau
  3. (G_00 - T_00 + Lambda_t)(N)  - "convergence at zero" residual
  4. Bianchi residual ||div G^00||(N) (already known, included)

Within-regime, single physics, only N varies.
DO NOT modify the manuscript; report only.
"""
from __future__ import annotations
import json
import math
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


P5_LADDER = [
    ("P5",      50,  "results_d1_fix17/d1_p5.npz",                       "d1"),
    ("P5N64",   64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz",     "snap"),
    ("P5N72",   72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz",     "snap"),
    ("P5N84",   84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz",     "snap"),
    ("P5N100", 100,  "results_d1_p5n100_24seeds/P5N100.snapshots.npz",   "snap"),
    ("P5N128", 128,  "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz",  "snap"),
    ("P5N200", 200,  "results_d1_p5n200_8seeds/P5N200.snapshots.npz",    "snap"),
    ("P5N300", 300,  "results_d1_p5n300_12seeds/P5N300.snapshots.npz",           "snap"),
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz",  "snap"),
    ("P5N512", 512,  "results_d1_p5n512_12seeds/P5N512.snapshots.npz",  "snap"),
]


def load_seed_payload(rel_path: str, n_lat: int, kind: str,
                       max_seeds: int = 32):
    """Yield (xi_mat, psi, k_field, q_field) for each seed."""
    fp = REPO.parent / rel_path
    if not fp.exists():
        return []
    z = np.load(fp, allow_pickle=True)
    K_DEF = 0.55
    Q_DEF = 0.45
    out = []
    if kind == "snap":
        if "edge_xi_snapshots" not in z.files:
            return []
        snaps = z["edge_xi_snapshots"]
        psi_re = z["psi_real_snapshots"]
        psi_im = z["psi_imag_snapshots"]
        last_idx = snaps.shape[1] - 1
        ns = min(int(snaps.shape[0]), max_seeds)
        has_kq_snap = "k_snapshots" in z.files and "q_snapshots" in z.files
        for s in range(ns):
            xi_mat = np.asarray(snaps[s, last_idx], dtype=float)
            psi = (np.asarray(psi_re[s, last_idx], dtype=float)
                   + 1j * np.asarray(psi_im[s, last_idx], dtype=float))
            if has_kq_snap:
                k_field = np.asarray(z["k_snapshots"][s, last_idx],
                                      dtype=float)
                q_field = np.asarray(z["q_snapshots"][s, last_idx],
                                      dtype=float)
            else:
                k_field = np.full((n_lat, n_lat), K_DEF)
                q_field = np.full((n_lat, n_lat), Q_DEF)
            out.append((xi_mat, psi, k_field, q_field))
        return out

    # kind == "d1"
    if "dense_cell_edge_xi_values" in z.files:
        edge_arr = z["dense_cell_edge_xi_values"]
        amp_arr = z["dense_cell_node_amplitude_values"]
        phase_arr = z["dense_cell_node_phase_values"]
        ns = min(int(edge_arr.shape[0]), max_seeds)
        for s in range(ns):
            xi_mat = edge_to_matrix(edge_arr[s], n_lat)
            psi = amp_arr[s] * np.exp(1j * phase_arr[s])
            if f"ff_K_seed{s}" in z.files:
                k_field = np.asarray(z[f"ff_K_seed{s}"], dtype=float)
            else:
                k_field = np.full((n_lat, n_lat), K_DEF)
            if f"ff_Q_seed{s}" in z.files:
                q_field = np.asarray(z[f"ff_Q_seed{s}"], dtype=float)
            else:
                q_field = np.full((n_lat, n_lat), Q_DEF)
            out.append((xi_mat, psi, k_field, q_field))
        return out
    return []


def main() -> int:
    print("=" * 78)
    print("Within-P5 Galerkin G_00(N) and T_00(N) ladder")
    print("=" * 78)
    rows = []
    for reg, n_lat, rel, kind in P5_LADDER:
        print(f"\n[{reg}, N={n_lat}, kind={kind}]")
        try:
            payload = load_seed_payload(rel, n_lat, kind, max_seeds=32)
        except Exception as e:
            print(f"  load error: {e}")
            continue
        if not payload:
            print(f"  no payload found at {rel}")
            continue
        # Compute kq_default from the actual snapshot payload presence of
        # ff_K_seed* / ff_Q_seed* arrays — replaces the earlier hardcoded
        # flag that was stale after the K/Q-bug fix on P5N200.
        try:
            _z = np.load(REPO.parent / rel, allow_pickle=True)
            kq_default = not any(k.startswith("ff_K_seed") for k in _z.files)
        except FileNotFoundError:
            kq_default = True
        print(f"  seeds loaded: {len(payload)}, K/Q-default: {kq_default}")
        g00_all, t00_all = [], []
        for s, (xi_mat, psi, k_field, q_field) in enumerate(payload):
            np.fill_diagonal(xi_mat, 1.0)
            try:
                prep = per_seed_galerkin(xi_mat, psi, k_field, q_field,
                                          n_lat, np)
            except Exception as e:
                print(f"  seed {s} galerkin failed: {e}")
                continue
            g00 = np.asarray(prep["g_00_h"])
            t00 = np.asarray(prep["t00"])
            g00_all.append(g00)
            t00_all.append(t00)
            print(f"  seed{s}: G_00 med={float(np.median(g00)):+.4f}, "
                  f"T_00 med={float(np.median(t00)):+.4f}")
        if not g00_all:
            continue
        g00_concat = np.concatenate(g00_all)
        t00_concat = np.concatenate(t00_all)
        # Across-seed medians
        g00_med = float(np.median(g00_concat))
        t00_med = float(np.median(t00_concat))
        # Lambda_t = median(T_00 - G_00) per regime
        lt_per_seed = [float(np.median(t - g))
                       for t, g in zip(t00_all, g00_all)]
        lt_mean = float(np.mean(lt_per_seed))
        lt_std = float(np.std(lt_per_seed))
        # The "anti-wave from matter-wave" prediction: G_00 = T_00 - Lambda_t
        # Test: how well does this hold per-node?
        residual_per_seed = []
        for t, g in zip(t00_all, g00_all):
            lt_pred = float(np.median(t - g))
            r = g - (t - lt_pred)
            residual_per_seed.append(float(np.median(np.abs(r))))
        residual_med = float(np.mean(residual_per_seed))
        # Co-scaling test: G_00 / T_00 ratio
        ratio = g00_med / t00_med if t00_med != 0 else float("nan")
        rows.append({
            "regime": reg, "N": n_lat,
            "kind": kind,
            "kq_default": kq_default,
            "n_seeds": len(g00_all),
            "G_00_med": g00_med,
            "T_00_med": t00_med,
            "Lambda_t_per_regime": lt_mean,
            "Lambda_t_std": lt_std,
            "G_over_T_ratio": ratio,
            "residual_med": residual_med,
        })
        print(f"  REGIME: T_00={t00_med:.4f}, G_00={g00_med:.4f}, "
              f"Lambda_t={lt_mean:.4f}+-{lt_std:.4f}, G/T={ratio:.4f}")

    # Print final ladder table
    print()
    print("=" * 78)
    print("Within-P5 G_00 / T_00 ladder summary")
    print("=" * 78)
    print(f"{'reg':>8} {'N':>4} {'KQ':>4} {'T_00':>8} {'G_00':>8} "
          f"{'Lambda_t':>10} {'G/T':>8} {'|G-(T-Lt)|_med':>16}")
    for r in rows:
        kq = "DEF" if r["kq_default"] else "lat"
        print(f"{r['regime']:>8} {r['N']:>4} {kq:>4} "
              f"{r['T_00_med']:>8.4f} {r['G_00_med']:>8.4f} "
              f"{r['Lambda_t_per_regime']:>10.4f} "
              f"{r['G_over_T_ratio']:>8.4f} "
              f"{r['residual_med']:>16.5f}")

    # Symanzik-2 fit for T_00, G_00, Lambda_t
    print()
    print("=" * 78)
    print("Symanzik-2 fits y(N) = y_inf + c2 / N^2 (within-P5 only)")
    print("=" * 78)
    Ns = np.array([r["N"] for r in rows], dtype=float)

    def fit_s2(y):
        Y = np.asarray(y, dtype=float)
        A = np.column_stack([np.ones_like(Ns), Ns ** -2])
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        pred = A @ coef
        rss = float(np.sum((Y - pred) ** 2))
        tss = float(np.sum((Y - Y.mean()) ** 2))
        return float(coef[0]), float(coef[1]), 1.0 - rss / tss if tss > 0 else 0.0

    fits = {}
    for key, label in [("T_00_med", "T_00"),
                        ("G_00_med", "G_00"),
                        ("Lambda_t_per_regime", "Lambda_t")]:
        ys = [r[key] for r in rows]
        a, c2, r2 = fit_s2(ys)
        fits[label] = {"y_inf": a, "c2": c2, "r2": r2}
        print(f"  {label:>9}: y_inf = {a:+.5f}, c2 = {c2:+.4f}, "
              f"R^2 = {r2:.3f}")

    # The "wave-anti-wave" identity:
    # G_00 + Lambda_t == T_00 ?
    # Asymptote check
    a_T = fits["T_00"]["y_inf"]
    a_G = fits["G_00"]["y_inf"]
    a_L = fits["Lambda_t"]["y_inf"]
    print()
    print("=" * 78)
    print("Wave-anti-wave convergence point")
    print("=" * 78)
    print(f"  T_00 asymptote (matter wave):       {a_T:+.5f}")
    print(f"  G_00 asymptote (anti-wave / QFT):   {a_G:+.5f}")
    print(f"  Lambda_t asymptote:                 {a_L:+.5f}")
    print(f"  (T_00 - G_00) asymptote:            {a_T - a_G:+.5f}")
    print(f"  Identity check (T - G - Lambda_t):  "
          f"{a_T - a_G - a_L:+.6f}")
    print()
    print("  Per-N residual |T - G - Lambda_t| (anti-wave-from-matter):")
    print(f"    {'N':>4} {'T-G':>9} {'Lambda_t':>10} {'T-G-Lt':>10}")
    for r in rows:
        diff = r["T_00_med"] - r["G_00_med"]
        gap = diff - r["Lambda_t_per_regime"]
        print(f"    {r['N']:>4} {diff:>+9.5f} "
              f"{r['Lambda_t_per_regime']:>+10.5f} {gap:>+10.6f}")

    # By construction Lambda_t = median(T-G) per regime, so the
    # per-regime gap is mostly zero. The interesting thing is whether
    # the asymptotes match the framework's claim:
    #   Lambda_t_inf = alpha_xi^2 = 81/100 = 0.81
    #   T_00 plateau at canonical ~0.85
    #   G_00 plateau at ~0.04 (= T_00 - Lambda_t)
    print()
    print("  Framework predictions:")
    print(f"    Lambda_t_inf = alpha_xi^2 = 81/100 = 0.81000")
    print(f"    Measured Lambda_t_inf:              {a_L:.5f}  "
          f"(diff vs 0.81: {a_L - 0.81:+.5f})")
    print(f"    Measured T_00_inf:                  {a_T:.5f}")
    print(f"    Measured G_00_inf:                  {a_G:.5f}")
    print(f"    G_00_inf prediction (T_00 - 0.81):  {a_T - 0.81:.5f}")

    bundle = {
        "ladder": rows,
        "symanzik2_fits": fits,
        "asymptotes": {
            "T_00_inf": a_T,
            "G_00_inf": a_G,
            "Lambda_t_inf": a_L,
            "T_minus_G_inf": a_T - a_G,
        },
        "framework_target": 0.81,
    }
    out = REPO / "outputs" / "p5_g00_t00_within_ladder.json"
    out.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
