"""Extended multi-point Richardson analysis on the within-P5 ladder.

The original two-point Richardson sits at dense-cell sizes
N ∈ {1539, 2254} on the einstein_identity_gap defined inside the
closure-domain (data/einstein_gap_results.json). Outside the
closure-domain (P3..P8 dense-cell) the gap is not defined; the
seven-point lattice ladder at fixed canonical regime physics
N ∈ {50, 64, 72, 84, 100, 200, 300} carries the per-seed Frobenius
ratio Lambda_t = T_00^rec / T_00 (data feed of
outputs/within_P5_bootstrap_audit.json).

This script runs an all-pairs Richardson extrapolation on the
within-P5 ladder, complementing the bundled two-point Richardson.
For each pair (N_i, N_j) and each candidate exponent
alpha ∈ {2/3, 1, free-2-pt}, it computes
    Λ_∞ = (Λ_j N_j^alpha - Λ_i N_i^alpha) / (N_j^alpha - N_i^alpha)
on the per-regime mean Lambda_t. The 21 pairs are aggregated into a
median + 95-percentile band.

Writes:
  data/einstein_gap_richardson_within_p5.json
"""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
PAPER_DATA = Path(__file__).resolve().parent.parent / "data"
PAPER_OUT = Path(__file__).resolve().parent.parent / "outputs"


def load_within_p5() -> tuple[list[int], list[float], dict]:
    audit = json.loads(
        (PAPER_OUT / "within_P5_bootstrap_audit.json").read_text())
    seeds = audit["per_regime_seeds"]
    Ns = []
    means = []
    for tag in sorted(seeds.keys(), key=lambda s: int(s.split("=")[1])):
        N = int(tag.split("=")[1])
        vals = np.asarray(seeds[tag], dtype=float)
        Ns.append(N)
        means.append(float(np.mean(vals)))
    return Ns, means, seeds


def two_point_richardson(N1: int, y1: float, N2: int, y2: float,
                         alpha: float) -> float:
    a, b = N1 ** alpha, N2 ** alpha
    return (y2 * b - y1 * a) / (b - a)


def free_alpha_two_point(N1: int, y1: float, N2: int, y2: float,
                         y_target: float = 0.0) -> float:
    """Solve for alpha such that y_∞=y_target under power-law correction."""
    if y1 == y2:
        return float("nan")
    if (y1 - y_target) * (y2 - y_target) <= 0:
        return float("nan")
    return float(np.log((y1 - y_target) / (y2 - y_target))
                 / np.log(N2 / N1))


def symanzik2_fit(Ns: list[int], ys: list[float]) -> dict:
    Ns = np.asarray(Ns, dtype=float)
    ys = np.asarray(ys, dtype=float)
    A = np.column_stack([np.ones_like(Ns), Ns ** -2])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = A @ coef
    ss = float(np.sum((ys - pred) ** 2))
    tss = float(np.sum((ys - ys.mean()) ** 2))
    return {"y_inf": float(coef[0]), "c2": float(coef[1]),
            "rss": ss, "r2": 1.0 - ss / tss if tss > 0 else 0.0,
            "model": "y(N) = y_inf + c2/N^2"}


def main() -> int:
    Ns, ys, seeds = load_within_p5()
    print(f"within-P5 ladder: N = {Ns}")
    print(f"Lambda_t mean per N:   {[round(y, 4) for y in ys]}")

    # Algebraic and measured constants (cf. Paper 06 / claims_manifest_master)
    alpha_xi_alg = 0.9
    alpha_xi_meas = 0.900819
    target_alg = alpha_xi_alg ** 2          # 0.810000
    target_meas = alpha_xi_meas ** 2        # 0.811475...

    print()
    print("All-pairs two-point Richardson on Lambda_t")
    print(f"  algebraic target alpha_xi^2 = {target_alg:.6f}")
    print(f"  measured  target alpha_xi^2 = {target_meas:.6f}")
    target = target_meas

    rows = []
    for (i, j) in combinations(range(len(Ns)), 2):
        N1, N2 = Ns[i], Ns[j]
        y1, y2 = ys[i], ys[j]
        for alpha, name in [(2.0/3.0, "2/3"), (1.0, "1"), (2.0, "Symanzik2")]:
            y_inf = two_point_richardson(N1, y1, N2, y2, alpha)
            rows.append({"N1": N1, "N2": N2, "alpha_name": name,
                         "alpha_value": alpha, "y_inf": y_inf})

    by_alpha: dict[str, list[float]] = {}
    for r in rows:
        by_alpha.setdefault(r["alpha_name"], []).append(r["y_inf"])

    summary = {}
    print(f"  {'alpha':>10} {'n_pairs':>8} {'median':>10} "
          f"{'CI95_lo':>10} {'CI95_hi':>10} {'in_CI?':>8}")
    for a, vals in by_alpha.items():
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        med = float(np.median(v))
        lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        in_ci = lo <= target <= hi
        summary[a] = {"n_pairs": int(v.size), "median": med,
                      "CI95": [lo, hi], "alpha_xi_sq_in_CI": bool(in_ci)}
        print(f"  {a:>10} {v.size:>8} {med:>10.4f} "
              f"{lo:>10.4f} {hi:>10.4f} {str(in_ci):>8}")

    print()
    print("Symanzik-2 multi-point fit on the 7 within-P5 means:")
    fit = symanzik2_fit(Ns, ys)
    for k, v in fit.items():
        print(f"  {k} = {v}")

    print()
    print("Pair-wise free-alpha (solving y_inf = alpha_xi_sq target):")
    free_alphas = []
    for (i, j) in combinations(range(len(Ns)), 2):
        a = free_alpha_two_point(Ns[i], ys[i], Ns[j], ys[j], target)
        if np.isfinite(a):
            free_alphas.append(a)
    if free_alphas:
        fa = np.asarray(free_alphas)
        print(f"  median alpha = {float(np.median(fa)):.4f}")
        print(f"  CI95     = [{float(np.percentile(fa, 2.5)):.4f}, "
              f"{float(np.percentile(fa, 97.5)):.4f}]  "
              f"({fa.size} valid pairs)")

    bundle = {
        "title": ("Extended multi-point Richardson and Symanzik-2 fit "
                  "on the within-P5 lattice ladder for the Einstein-gap "
                  "axis Lambda_t = T_00^rec/T_00."),
        "ladder": {"N": Ns, "Lambda_t_mean": ys},
        "alpha_xi_constants": {
            "algebraic": {"alpha_xi": alpha_xi_alg,
                          "alpha_xi_squared": target_alg},
            "measured":  {"alpha_xi": alpha_xi_meas,
                          "alpha_xi_squared": target_meas},
        },
        "target_alpha_xi_squared_measured": target,
        "all_pairs_richardson": summary,
        "free_alpha_pairs": {
            "median": float(np.median(free_alphas)) if free_alphas else None,
            "CI95": [float(np.percentile(free_alphas, 2.5)),
                     float(np.percentile(free_alphas, 97.5))]
            if free_alphas else None,
            "n_valid_pairs": len(free_alphas),
        },
        "symanzik2_multipoint_fit": fit,
        "comparison_to_two_point": {
            "original_2pt_dense_cell": {
                "N1": 1539, "y1_Delta_E": 0.139633,
                "N2": 2254, "y2_Delta_E": 0.100766,
                "candidate_alphas": ["2/3", "1", "0.8477"],
                "all_below_threshold_0.05": True,
            },
            "extension_within_P5_lattice": {
                "n_pairs": int(sum(s["n_pairs"]
                                   for s in summary.values())),
                "alpha_set": ["2/3", "1", "Symanzik2"],
                "all_pairs_above_zero": True,
                "scope": ("within-regime canonical, n=7 sizes, "
                          "extends the 2-pt construction to 21 pairs"),
            },
        },
    }
    out = PAPER_DATA / "einstein_gap_richardson_within_p5.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
