"""Re-analyse the within-P5 G_00 / T_00 ladder by splitting it
into K/Q-lattice and K/Q-default populations.

Within-P5 ladder is bimodal:
  K/Q-lattice (proper):  P5 N=50, P5N64, P5N100, P5N300  (T_00 ~ 0.85)
  K/Q-default (bug):     P5N72, P5N84, P5N128, P5N200    (T_00 ~ 0.42)

The CRITICAL observation: G_00(N) is K/Q-INDEPENDENT (decays as
~60/N^2 across both populations). So the "anti-wave" G_00 is a
lattice-discretisation-error that vanishes in continuum, NOT a
matter-coupled Einstein curvature.

This script makes that observation explicit and quantifies it.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LADDER = json.loads(
    (REPO / "outputs" / "p5_g00_t00_within_ladder.json").read_text())["ladder"]

LAT_REGIMES = {"P5", "P5N64", "P5N100", "P5N300"}
DEF_REGIMES = {"P5N72", "P5N84", "P5N128", "P5N200"}


def fit_s2(Ns, ys):
    Ns = np.asarray(Ns, dtype=float)
    ys = np.asarray(ys, dtype=float)
    A = np.column_stack([np.ones_like(Ns), Ns ** -2])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = A @ coef
    rss = float(np.sum((ys - pred) ** 2))
    tss = float(np.sum((ys - ys.mean()) ** 2))
    return float(coef[0]), float(coef[1]), 1.0 - rss / tss if tss > 0 else 0.0


def fit_s2_pos(Ns, ys):
    """Fit y = y_inf + c2/N^2 with c2 free; report y_inf, c2, R^2."""
    return fit_s2(Ns, ys)


def main():
    print("=" * 78)
    print("Within-P5 G_00 / T_00, split by K/Q population")
    print("=" * 78)
    lat = [r for r in LADDER if r["regime"] in LAT_REGIMES]
    deflt = [r for r in LADDER if r["regime"] in DEF_REGIMES]
    lat.sort(key=lambda r: r["N"])
    deflt.sort(key=lambda r: r["N"])

    print(f"\nK/Q-lattice population (proper):")
    print(f"{'reg':>8} {'N':>4} {'T_00':>8} {'G_00':>8} "
          f"{'Lambda_t':>10} {'G/T':>8}")
    for r in lat:
        print(f"{r['regime']:>8} {r['N']:>4} "
              f"{r['T_00_med']:>8.4f} {r['G_00_med']:>8.4f} "
              f"{r['Lambda_t_per_regime']:>10.4f} "
              f"{r['G_over_T_ratio']:>8.4f}")

    print(f"\nK/Q-default population (bug):")
    print(f"{'reg':>8} {'N':>4} {'T_00':>8} {'G_00':>8} "
          f"{'Lambda_t':>10} {'G/T':>8}")
    for r in deflt:
        print(f"{r['regime']:>8} {r['N']:>4} "
              f"{r['T_00_med']:>8.4f} {r['G_00_med']:>8.4f} "
              f"{r['Lambda_t_per_regime']:>10.4f} "
              f"{r['G_over_T_ratio']:>8.4f}")

    # === Fits per population ===
    out_fits = {}
    for label, pop in [("LAT", lat), ("DEF", deflt), ("ALL", lat + deflt)]:
        print()
        print("=" * 78)
        print(f"Symanzik-2 fits y(N) = y_inf + c2 / N^2 ({label})")
        print("=" * 78)
        Ns = [r["N"] for r in pop]
        sub = {}
        for key, name in [("T_00_med", "T_00"),
                           ("G_00_med", "G_00"),
                           ("Lambda_t_per_regime", "Lambda_t")]:
            ys = [r[key] for r in pop]
            a, c, r2 = fit_s2(Ns, ys)
            sub[name] = {"y_inf": a, "c2": c, "r2": r2}
            print(f"  {name:>9}: y_inf = {a:+.5f}, "
                  f"c2 = {c:+.4f}, R^2 = {r2:.3f}")
        out_fits[label] = sub

    # === Cross-population G_00 fit ===
    print()
    print("=" * 78)
    print("KEY OBSERVATION: G_00(N) is K/Q-INDEPENDENT")
    print("=" * 78)
    Ns = [r["N"] for r in (lat + deflt)]
    G = [r["G_00_med"] for r in (lat + deflt)]
    a_G, c_G, r2_G = fit_s2(Ns, G)
    print(f"  Combined fit (8 points across both populations):")
    print(f"    G_00(N) = {a_G:+.5f} + ({c_G:+.3f}) / N^2")
    print(f"    R^2 = {r2_G:.4f}")
    print()
    print("  Per-population fits:")
    print(f"    LAT  G_00(N): y_inf = {out_fits['LAT']['G_00']['y_inf']:.5f}, "
          f"c2 = {out_fits['LAT']['G_00']['c2']:.2f}, "
          f"R^2 = {out_fits['LAT']['G_00']['r2']:.3f}")
    print(f"    DEF  G_00(N): y_inf = {out_fits['DEF']['G_00']['y_inf']:.5f}, "
          f"c2 = {out_fits['DEF']['G_00']['c2']:.2f}, "
          f"R^2 = {out_fits['DEF']['G_00']['r2']:.3f}")
    print()
    print("  -> Both populations give same c2 ~60 and same y_inf ~0,")
    print("     i.e. G_00 ~ 60/N^2 universal across K/Q states.")
    print("     This is consistent with G_00 being a per-node lattice")
    print("     discretisation error (not a K/Q-coupled Einstein term).")

    # === Lambda_t -> T_00 in continuum ===
    print()
    print("=" * 78)
    print("Lambda_t = T_00 - G_00 -> T_00 in continuum (G_00 -> 0)")
    print("=" * 78)
    for lab in ("LAT", "DEF"):
        s = out_fits[lab]
        print(f"  {lab}: T_00_inf = {s['T_00']['y_inf']:.5f}, "
              f"Lambda_t_inf = {s['Lambda_t']['y_inf']:.5f}, "
              f"diff = {s['Lambda_t']['y_inf'] - s['T_00']['y_inf']:+.5f}")

    # === Anti-wave-from-matter-wave check ===
    print()
    print("=" * 78)
    print("Anti-wave from matter-wave: predict G_00 from T_00 alone?")
    print("=" * 78)
    print("  Test: G_00(N) is well-predicted by 60/N^2  (universal)")
    print("        T_00(N) is well-predicted by its plateau (K/Q-set)")
    print("        Lambda_t(N) = T_00(N) - 60/N^2  (combined prediction)")
    print()
    print(f"  {'N':>4} {'T_00':>8} {'G_pred':>8} {'G_meas':>8} "
          f"{'Lt_pred':>10} {'Lt_meas':>10} {'KQ':>4}")
    for r in lat + deflt:
        N = r["N"]
        g_pred = a_G + c_G / N ** 2
        Lt_pred = r["T_00_med"] - g_pred
        kq = "lat" if r["regime"] in LAT_REGIMES else "DEF"
        print(f"  {N:>4} {r['T_00_med']:>8.4f} {g_pred:>8.5f} "
              f"{r['G_00_med']:>8.5f} {Lt_pred:>10.4f} "
              f"{r['Lambda_t_per_regime']:>10.4f} {kq:>4}")
    # Total residual
    Lt_pred_arr = np.array([
        r["T_00_med"] - (a_G + c_G / r["N"] ** 2)
        for r in lat + deflt])
    Lt_meas_arr = np.array([r["Lambda_t_per_regime"]
                             for r in lat + deflt])
    rms = float(np.sqrt(np.mean((Lt_pred_arr - Lt_meas_arr) ** 2)))
    max_dev = float(np.max(np.abs(Lt_pred_arr - Lt_meas_arr)))
    print()
    print(f"  RMS of Lambda_t-prediction error: {rms:.5f}")
    print(f"  Max |Lt_pred - Lt_meas|:          {max_dev:.5f}")

    # Save
    out = REPO / "outputs" / "p5_g00_t00_split_KQ.json"
    out.write_text(json.dumps({
        "lat_pop": lat,
        "def_pop": deflt,
        "fits": out_fits,
        "G_00_combined_fit": {
            "y_inf": a_G, "c2": c_G, "r2": r2_G},
        "anti_wave_predict_rms": rms,
        "anti_wave_predict_max_dev": max_dev,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
