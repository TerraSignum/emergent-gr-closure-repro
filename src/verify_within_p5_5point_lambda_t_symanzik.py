"""Within-P5 5-point Symanzik fit on Lambda_t = T_00^rec / T_00.

The within-P5 ladder restricted to the canonical-configuration
range N <= 100 carries five points
{P5 N=50, P5N64, P5N72, P5N84, P5N100}, each with four lattice
seeds. The Lambda_t per-seed values are bundled in
outputs/within_P5_bootstrap_audit.json. This script does:

  1. Symanzik-2 fit on the per-seed pooled values
  2. Bootstrap CI by resampling seeds (4 per regime, 5 regimes)
  3. Comparison of the asymptote against the algebraic and
     measured alpha_xi^2 targets
  4. Comparison against the bundled 7-point fit which extends to
     N in {200, 300} via snapshot-source data

Writes:
  data/lambda_t_within_p5_5point.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent.parent
OUTPUTS = PAPER / "outputs"
DATA = PAPER / "data"


def symanzik2_fit(N: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    A = np.column_stack([np.ones_like(N, dtype=float), N.astype(float) ** -2])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss = float(np.sum((y - pred) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    return float(coef[0]), float(coef[1]), 1.0 - ss / tss if tss > 0 else 0.0


def bootstrap(seeds_by_N: dict[int, list[float]], n_boot: int = 5000,
              rng: np.random.Generator | None = None) -> dict:
    if rng is None:
        rng = np.random.default_rng(20260501)
    Ns = np.array(sorted(seeds_by_N.keys()), dtype=int)
    asymptotes, c2s = [], []
    for _ in range(n_boot):
        means = []
        for N in Ns:
            arr = np.asarray(seeds_by_N[int(N)], dtype=float)
            idx = rng.integers(0, len(arr), size=len(arr))
            means.append(float(np.mean(arr[idx])))
        y_inf, c2, _ = symanzik2_fit(Ns, np.asarray(means))
        if np.isfinite(y_inf) and -2 < y_inf < 2:
            asymptotes.append(y_inf)
            c2s.append(c2)
    asym = np.asarray(asymptotes)
    return {
        "n_boot_valid": int(asym.size),
        "median": float(np.median(asym)),
        "CI95": [float(np.percentile(asym, 2.5)),
                 float(np.percentile(asym, 97.5))],
        "CI68": [float(np.percentile(asym, 16)),
                 float(np.percentile(asym, 84))],
        "c2_median": float(np.median(c2s)),
    }


def main() -> int:
    bundle = json.loads(
        (OUTPUTS / "within_P5_bootstrap_audit.json").read_text())
    seeds_full = bundle["per_regime_seeds"]
    seeds_5 = {int(k.split("=")[1]): v for k, v in seeds_full.items()
               if int(k.split("=")[1]) <= 100}

    print("=" * 80)
    print("Within-P5 5-point Symanzik-2 fit on Lambda_t (canonical N<=100)")
    print("=" * 80)
    print(f"{'N':>5} {'n_seeds':>9} {'mean':>10} {'std':>10}")
    Ns, means = [], []
    for N in sorted(seeds_5.keys()):
        arr = np.asarray(seeds_5[N], dtype=float)
        Ns.append(N)
        means.append(float(np.mean(arr)))
        print(f"{N:>5} {len(arr):>9} {means[-1]:>10.4f} "
              f"{float(np.std(arr)):>10.4f}")

    Ns_arr = np.asarray(Ns, dtype=int)
    means_arr = np.asarray(means, dtype=float)
    y_inf, c2, r2 = symanzik2_fit(Ns_arr, means_arr)
    print()
    print(f"Symanzik-2 on the 5 means:")
    print(f"  y_inf = {y_inf:.6f}")
    print(f"  c2    = {c2:.4f}")
    print(f"  R^2   = {r2:.4f}")

    bs = bootstrap(seeds_5)
    print()
    print(f"Bootstrap (n={bs['n_boot_valid']}):")
    print(f"  median  = {bs['median']:.6f}")
    print(f"  CI95    = [{bs['CI95'][0]:.6f}, {bs['CI95'][1]:.6f}]")
    print(f"  CI68    = [{bs['CI68'][0]:.6f}, {bs['CI68'][1]:.6f}]")

    targets = {
        "alpha_xi_sq_algebraic_0.81": 0.81,
        "alpha_xi_sq_measured_0.811475": 0.811475,
        "alpha_xi_sq_minus_half_gamma_sq_0.805": 0.805,
        "alpha_xi_sq_plus_gamma_sq_0.82": 0.82,
    }
    in_ci_95 = {n: bool(bs["CI95"][0] <= v <= bs["CI95"][1])
                for n, v in targets.items()}
    in_ci_68 = {n: bool(bs["CI68"][0] <= v <= bs["CI68"][1])
                for n, v in targets.items()}

    print()
    print("Targets in CI:")
    for name, val in targets.items():
        cf95 = "in CI95" if in_ci_95[name] else "out CI95"
        cf68 = "in CI68" if in_ci_68[name] else "out CI68"
        print(f"  {name:<45} = {val:.6f}   {cf95}, {cf68}")

    out = {
        "method": "within_P5_5point_symanzik2_lambda_t",
        "ladder": {"N": Ns, "n_seeds_per_N": [len(seeds_5[N]) for N in Ns],
                   "mean_per_N": means},
        "symanzik2_fit_on_means": {"y_inf": y_inf, "c2": c2, "r2": r2},
        "bootstrap_5000": bs,
        "targets_in_CI95": in_ci_95,
        "targets_in_CI68": in_ci_68,
        "comparison_to_7point_fit": {
            "asymptote_median_7pt": 0.811742741607193,
            "CI95_7pt": [0.801092781932126, 0.8226373790033402],
            "ladder_7pt_N": [50, 64, 72, 84, 100, 200, 300],
        },
        "purpose": ("Within-regime canonical-configuration ladder "
                    "(N<=100) Symanzik fit on Lambda_t. The 5-point fit "
                    "is orthogonal to the 7-point fit by restricting to "
                    "uniformly-configured runs and excluding the "
                    "snapshot-source large-N tail."),
    }
    out_path = DATA / "lambda_t_within_p5_5point.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
