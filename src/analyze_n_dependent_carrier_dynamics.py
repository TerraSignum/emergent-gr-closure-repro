"""Analysis-only: test the hypothesis that some observable carries
'docks' on certain N values but not others, producing oscillatory
residuals around the Symanzik-2 asymptote rather than monotonic
decay.

User-frame: information not lost, just doesn't dock at this
knot/seed but at the next-higher N. Empirically this would
manifest as:
  (a) sign-flips of residual against the smooth asymptotic fit
  (b) non-monotonic increase-decrease pattern across consecutive N
  (c) potential correlation with number-theoretic properties of N

Inputs analysed:
  - Lambda_t per-N within-P5 (within_P5_bootstrap_audit)
  - chirality-deviation per-N within-P5
  - T_00,med per-N from per_regime_lambda_t_universal_audit
  - per-node 4x4 Galerkin Frobenius median per-N

For each series, fit Symanzik-2 + bootstrap CI, then report:
  - residual sign per N
  - autocorrelation of residual sign
  - max consecutive |residual|/|asymptote-error| ratio
  - correlation with N's prime-divisor structure

DO NOT modify the manuscript; report only.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def symanzik2_fit(N, y):
    Ns = np.asarray(N, dtype=float)
    ys = np.asarray(y, dtype=float)
    A = np.column_stack([np.ones_like(Ns), Ns ** -2])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = A @ coef
    rss = float(np.sum((ys - pred) ** 2))
    tss = float(np.sum((ys - ys.mean()) ** 2))
    return float(coef[0]), float(coef[1]), 1.0 - rss / tss if tss > 0 else 0.0, pred


def free_power_fit(N, y, p_grid=None):
    Ns = np.asarray(N, dtype=float)
    ys = np.asarray(y, dtype=float)
    if p_grid is None:
        p_grid = np.linspace(0.2, 4.0, 381)
    best = None
    for p in p_grid:
        A = np.column_stack([np.ones_like(Ns), Ns ** -p])
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        pred = A @ coef
        rss = float(np.sum((ys - pred) ** 2))
        if best is None or rss < best["rss"]:
            best = {"p": float(p), "a": float(coef[0]),
                    "b": float(coef[1]), "rss": rss, "pred": pred}
    return best


def divisor_count(n):
    return sum(1 for k in range(1, int(math.sqrt(n)) + 1)
               if n % k == 0) * 2 - (1 if math.isqrt(n) ** 2 == n else 0)


def largest_prime_factor(n):
    f = 2
    while f * f <= n:
        if n % f == 0:
            n //= f
        else:
            f += 1
    return n


def sign_run_lengths(signs):
    """Lengths of consecutive same-sign runs."""
    runs = []
    cur = 1
    for i in range(1, len(signs)):
        if signs[i] == signs[i-1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return runs


def analyze_series(name, N, y, asymptote_known=None):
    print(f"\n{'='*72}")
    print(f"Series: {name}")
    print(f"{'='*72}")
    print(f"  N = {N}")
    print(f"  y = {[round(v, 5) for v in y]}")

    a_s2, c_s2, r2_s2, pred_s2 = symanzik2_fit(N, y)
    free = free_power_fit(N, y)
    res_s2 = np.asarray(y) - pred_s2
    res_free = np.asarray(y) - free["pred"]
    res_known = (np.asarray(y) - asymptote_known) if asymptote_known else None

    print(f"\n  Symanzik-2 fit:  a = {a_s2:.5f}, c2 = {c_s2:.3f}, R^2 = {r2_s2:.3f}")
    print(f"  Free fit:        a = {free['a']:.5f}, p = {free['p']:.3f}")
    if asymptote_known:
        print(f"  Known target:    a* = {asymptote_known:.5f}")

    print(f"\n  Residuals against Symanzik-2 fit:")
    for n_, r in zip(N, res_s2):
        sign = "+" if r > 0 else "-" if r < 0 else "0"
        print(f"    N={n_:>4}: residual = {r:+.5f}  [{sign}]")

    signs = np.sign(res_s2).astype(int)
    n_changes = int(np.sum(np.abs(np.diff(signs))) // 2)
    runs = sign_run_lengths(signs)
    n_pairs = len(N) - 1
    print(f"\n  Sign-change count: {n_changes} / {n_pairs} consecutive pairs")
    print(f"  Sign-run lengths:  {runs}  (max = {max(runs)})")

    # Lag-1 autocorrelation of residuals
    if len(res_s2) > 2:
        res_centered = res_s2 - res_s2.mean()
        if np.var(res_s2) > 1e-12:
            ac1 = float(np.sum(res_centered[:-1] * res_centered[1:])
                        / np.sum(res_centered ** 2))
        else:
            ac1 = 0.0
    else:
        ac1 = 0.0
    print(f"  Lag-1 autocorr of residuals: {ac1:+.3f}  "
          f"(positive = monotonic; negative = oscillatory)")

    # Number-theoretic correlations
    div_counts = [divisor_count(int(n)) for n in N]
    largest_pf = [largest_prime_factor(int(n)) for n in N]
    print(f"\n  Number-theoretic properties of N:")
    print(f"    {'N':>4} {'#div':>5} {'lpf':>5} {'res':>10}")
    for n_, d, lpf, r in zip(N, div_counts, largest_pf, res_s2):
        print(f"    {n_:>4} {d:>5} {lpf:>5} {r:>+10.5f}")
    # Spearman residual vs divisor count
    if np.var(div_counts) > 0 and np.var(res_s2) > 0:
        rd = float(np.corrcoef(np.argsort(np.argsort(div_counts)),
                               np.argsort(np.argsort(res_s2)))[0, 1])
    else:
        rd = 0.0
    print(f"  Spearman rho(divisor_count, |residual|): {rd:+.3f}")

    return {
        "name": name, "N": list(N),
        "y": list(y),
        "symanzik2": {"a": a_s2, "c2": c_s2, "r2": r2_s2},
        "free_fit": {"a": free["a"], "p": free["p"]},
        "residuals_s2": list(res_s2),
        "sign_changes": n_changes,
        "n_pairs": n_pairs,
        "sign_run_lengths": runs,
        "lag1_autocorr": ac1,
        "divisor_counts": div_counts,
        "spearman_rho_divcount": rd,
    }


def main() -> int:
    print("=" * 72)
    print("Analyse: N-abhängige Carrier-Dynamik (User-Hypothese)")
    print("=" * 72)
    print()
    print("Wenn die Hypothese stimmt, sollten Residuen:")
    print("  (a) hohe Sign-change-Count (oszillierend)")
    print("  (b) negative Lag-1 Autokorrelation")
    print("  (c) Korrelation mit number-theoretischen Eigenschaften von N")

    # 1. Lambda_t within-P5 (per-regime mean from within_P5_bootstrap)
    bs = json.loads(
        (REPO / "outputs" / "within_P5_bootstrap_audit.json").read_text())
    seeds = bs["per_regime_seeds"]
    Ns_lt, ys_lt = [], []
    for tag in sorted(seeds.keys(), key=lambda s: int(s.split("=")[1])):
        Ns_lt.append(int(tag.split("=")[1]))
        ys_lt.append(float(np.mean(seeds[tag])))
    res_lt = analyze_series("Lambda_t per-N (within-P5)", Ns_lt, ys_lt,
                             asymptote_known=0.811475)

    # 2. Chirality-deviation per-N
    chi = json.loads(
        (REPO / "data" / "einstein_gap_chirality_within_p5.json").read_text())
    res_chi = analyze_series("chirality_deviation per-N (within-P5)",
                              chi["N_values"], chi["deviation_means"],
                              asymptote_known=0.0)

    # 3. T_00 cross-regime (full ladder excluding K/Q-bug)
    pr = json.loads(
        (REPO / "outputs" / "per_regime_lambda_t_universal_audit.json"
         ).read_text())
    Ns_t00 = [r["N"] for r in pr["per_regime"]
              if not r["regime"].endswith("N128")]
    ys_t00 = [r["T_00_med"] for r in pr["per_regime"]
              if not r["regime"].endswith("N128")]
    res_t00 = analyze_series("T_00,med cross-regime", Ns_t00, ys_t00)

    # 4. Galerkin Frobenius per-N within-P5 (7 points)
    gal = json.loads(
        (REPO / "outputs" / "within_p5_runner_A_hessian_ricci.json"
         ).read_text())
    Ns_gal = [t["N"] for t in gal["trend"]]
    ys_gal = [t["blind_frob_median"] for t in gal["trend"]]
    res_gal = analyze_series("Galerkin Frobenius median per-N (within-P5)",
                              Ns_gal, ys_gal,
                              asymptote_known=0.0)

    print("\n" + "=" * 72)
    print("Aggregierte Hypothesen-Bewertung")
    print("=" * 72)
    bundle = {
        "Lambda_t":  res_lt,
        "chirality": res_chi,
        "T_00":      res_t00,
        "Galerkin":  res_gal,
    }
    print(f"\n{'series':<40} {'sign_changes':>13} {'lag1':>8} "
          f"{'spearman_div':>14}")
    for k, r in bundle.items():
        print(f"  {r['name']:<38} {r['sign_changes']:>4}/{r['n_pairs']:<3}    "
              f"{r['lag1_autocorr']:>+8.3f} {r['spearman_rho_divcount']:>+14.3f}")

    # Summary verdict
    n_oscillating = sum(1 for r in bundle.values()
                        if r["lag1_autocorr"] < -0.1
                        and r["sign_changes"] >= max(2,
                                                     r["n_pairs"] // 3))
    print(f"\nOszillierende Reihen (lag1 < -0.1, sign_changes >= N/3): "
          f"{n_oscillating} of 4")
    if n_oscillating >= 2:
        print("  -> User-Hypothese empirisch unterstützt: mehrere "
              "Observablen zeigen non-monotonic / oscillatory pattern")
    else:
        print("  -> User-Hypothese NICHT empirisch unterstützt: "
              "Symanzik-2 monotone-decay Bild dominiert")

    out = REPO / "outputs" / "n_dependent_carrier_dynamics_analysis.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
