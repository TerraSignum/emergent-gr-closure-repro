"""Symanzik-2 bootstrap CI on the within-P5 7-point Frobenius
asymptote.

Reads per-seed Frobenius median/mean values from
outputs/within_p5_runner_A_hessian_ricci.json (seven canonical
P5-physics sizes N in {50, 64, 72, 84, 100, 200, 300} with four
seeds each) and:

  1. Symanzik-2 fit on the per-N means: y_inf, c2, R^2
  2. Bootstrap (n=5000): resample the four seeds at each N with
     replacement, refit Symanzik-2, collect asymptote
  3. Compare against the 0.05 closure-domain threshold

Writes:
  data/within_p5_frobenius_bootstrap.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent.parent
OUTPUTS = PAPER / "outputs"
DATA = PAPER / "data"


def symanzik2_fit(N, y):
    Ns = np.asarray(N, dtype=float)
    ys = np.asarray(y, dtype=float)
    A = np.column_stack([np.ones_like(Ns), Ns ** -2])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = A @ coef
    ss = float(np.sum((ys - pred) ** 2))
    tss = float(np.sum((ys - ys.mean()) ** 2))
    return float(coef[0]), float(coef[1]), 1.0 - ss / tss if tss > 0 else 0.0


def free_power_law_fit(N, y):
    """Fit y = a + b * N^{-p} via grid scan of p in [0.2, 4]."""
    Ns = np.asarray(N, dtype=float)
    ys = np.asarray(y, dtype=float)
    p_grid = np.linspace(0.20, 4.00, 381)
    best = None
    for p in p_grid:
        A = np.column_stack([np.ones_like(Ns), Ns ** -p])
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        pred = A @ coef
        ss = float(np.sum((ys - pred) ** 2))
        if best is None or ss < best["rss"]:
            tss = float(np.sum((ys - ys.mean()) ** 2))
            best = {"p": float(p), "a": float(coef[0]),
                    "b": float(coef[1]), "rss": ss,
                    "r2": 1.0 - ss / tss if tss > 0 else 0.0}
    return best


def bootstrap(seeds_by_N, n_boot=5000, rng=None):
    if rng is None:
        rng = np.random.default_rng(20260501)
    Ns = np.asarray(sorted(seeds_by_N.keys()), dtype=int)
    asymp_s2, asymp_free, p_free = [], [], []
    for _ in range(n_boot):
        means = []
        for N in Ns:
            arr = np.asarray(seeds_by_N[int(N)], dtype=float)
            idx = rng.integers(0, len(arr), size=len(arr))
            means.append(float(np.mean(arr[idx])))
        y_inf, _, _ = symanzik2_fit(Ns, means)
        if np.isfinite(y_inf):
            asymp_s2.append(y_inf)
        free = free_power_law_fit(Ns, means)
        if np.isfinite(free["a"]):
            asymp_free.append(free["a"])
            p_free.append(free["p"])
    return {
        "symanzik2": {
            "n_valid": len(asymp_s2),
            "median": float(np.median(asymp_s2)),
            "CI95": [float(np.percentile(asymp_s2, 2.5)),
                     float(np.percentile(asymp_s2, 97.5))],
            "CI68": [float(np.percentile(asymp_s2, 16)),
                     float(np.percentile(asymp_s2, 84))],
        },
        "free_power_law": {
            "n_valid": len(asymp_free),
            "asymp_median": float(np.median(asymp_free)),
            "asymp_CI95": [float(np.percentile(asymp_free, 2.5)),
                           float(np.percentile(asymp_free, 97.5))],
            "p_median": float(np.median(p_free)),
            "p_CI95": [float(np.percentile(p_free, 2.5)),
                       float(np.percentile(p_free, 97.5))],
        },
    }


def main() -> int:
    bundle = json.loads(
        (OUTPUTS / "within_p5_runner_A_hessian_ricci.json").read_text())
    seeds_by_N = {}
    for r in bundle["trend"]:
        seeds_by_N[int(r["N"])] = r["per_seed_blind_median"]

    print("=" * 78)
    print("Within-P5 7-point Symanzik-2 bootstrap on Frobenius residual")
    print("=" * 78)
    print(f"{'N':>5} {'n_seeds':>9} {'mean':>10} {'std':>10}")
    Ns_list, means_list = [], []
    for N in sorted(seeds_by_N.keys()):
        arr = np.asarray(seeds_by_N[N], dtype=float)
        Ns_list.append(N)
        means_list.append(float(np.mean(arr)))
        print(f"{N:>5} {len(arr):>9} {means_list[-1]:>10.5f} "
              f"{float(np.std(arr)):>10.5f}")

    y_inf, c2, r2 = symanzik2_fit(Ns_list, means_list)
    print()
    print(f"Symanzik-2 on the 7 means:")
    print(f"  y_inf = {y_inf:.5f}")
    print(f"  c2    = {c2:.3f}")
    print(f"  R^2   = {r2:.4f}")

    free = free_power_law_fit(Ns_list, means_list)
    print()
    print(f"Free-power-law fit a + b*N^(-p):")
    print(f"  a = {free['a']:.5f}, b = {free['b']:.3f}, p = {free['p']:.3f}")
    print(f"  R^2 = {free['r2']:.4f}")

    bs = bootstrap(seeds_by_N, n_boot=5000)
    print()
    print(f"Bootstrap (n=5000):")
    print(f"  Symanzik-2 asymptote median = {bs['symanzik2']['median']:.5f}")
    print(f"    CI95 = [{bs['symanzik2']['CI95'][0]:.5f}, "
          f"{bs['symanzik2']['CI95'][1]:.5f}]")
    print(f"    CI68 = [{bs['symanzik2']['CI68'][0]:.5f}, "
          f"{bs['symanzik2']['CI68'][1]:.5f}]")
    print(f"  Free-power-law asymptote median = "
          f"{bs['free_power_law']['asymp_median']:.5f}")
    print(f"    CI95 = [{bs['free_power_law']['asymp_CI95'][0]:.5f}, "
          f"{bs['free_power_law']['asymp_CI95'][1]:.5f}]")
    print(f"  Free-power-law exponent p = "
          f"{bs['free_power_law']['p_median']:.3f}")
    print(f"    CI95 = [{bs['free_power_law']['p_CI95'][0]:.3f}, "
          f"{bs['free_power_law']['p_CI95'][1]:.3f}]")

    threshold = 0.05
    s2_below = bs["symanzik2"]["CI95"][1] < threshold
    print()
    print(f"Closure threshold {threshold}:")
    print(f"  Symanzik-2 CI95 upper {bs['symanzik2']['CI95'][1]:.5f} "
          f"<= {threshold}: {s2_below}")

    out = {
        "method": "within_p5_frobenius_bootstrap_symanzik2",
        "title": ("Symanzik-2 bootstrap CI on the within-P5 7-point "
                  "Frobenius asymptote at fixed P5 physics."),
        "ladder": {
            "N": Ns_list, "n_seeds_per_N": [len(seeds_by_N[N]) for N in Ns_list],
            "blind_frob_median_per_N": means_list,
        },
        "symanzik2_fit_on_means": {"y_inf": y_inf, "c2": c2, "r2": r2},
        "free_power_law_fit_on_means": free,
        "bootstrap_n5000": bs,
        "closure_threshold": threshold,
        "symanzik2_ci95_upper_below_threshold": bool(s2_below),
    }
    out_path = DATA / "within_p5_frobenius_bootstrap.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
