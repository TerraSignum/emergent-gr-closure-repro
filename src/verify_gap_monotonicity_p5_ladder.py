"""Gap-monotonicity audit on the canonical $\\mathcal P_5/\\mathcal P_5 N$
physics ladder.

Companion to `verify_cone_tightness_p5_ladder.py'. The
weighted-gap monotonicity identity reads

    G(L) := 1 - mean_k s_k(L),    G'(L) = -mean_k r_k

so $G'(L) < 0$ iff $\\bar r := \\mathrm{mean}_k r_k > 0$.

This is the lattice-scaling claim that the uniformly weighted
closure gap is strictly decreasing in the lattice scale
$L = N^{1/3}$. We test it on the same five-percentile score
vector and the same ten-regime physics ladder as the
cone-tightness audit.

Output: outputs/verify_gap_monotonicity_p5_ladder.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "outputs" / "stage6f_fine_percentile_audit.json"
OUT = REPO / "outputs" / "verify_gap_monotonicity_p5_ladder.json"

PCT_KEYS = ["median", "mean", "p95", "p99", "sup"]
P5_REGIMES = {"P5", "P5N64", "P5N72", "P5N84", "P5N100",
              "P5N128", "P5N200", "P5N256", "P5N300", "P5N512"}


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    p5_regimes = [r for r in d["regimes"]
                  if r["regime"] in P5_REGIMES and "percentiles" in r]
    p5_regimes.sort(key=lambda r: r["N"])

    n_values = np.array([float(r["N"]) for r in p5_regimes])
    l_values = n_values ** (1.0 / 3.0)

    # Score matrix s_k(N) = 1 - delta_{rho_k}(N)
    score_mat = np.zeros((len(p5_regimes), len(PCT_KEYS)))
    for i, r in enumerate(p5_regimes):
        for k_idx, k_name in enumerate(PCT_KEYS):
            v = r["percentiles"].get(k_name)
            score_mat[i, k_idx] = 1.0 - float(v) if v is not None else np.nan

    # Per-score linear fit: s_k(L) = a_k + b_k L, rate r_k = b_k
    rates = np.zeros(score_mat.shape[1])
    for k in range(score_mat.shape[1]):
        b, _ = np.polyfit(l_values, score_mat[:, k], 1)
        rates[k] = b

    r_bar = float(np.mean(rates))
    g_prime = -r_bar  # G'(L) = -r_bar

    # Gap G(L_i) at each regime
    gap_per_regime = [
        {"regime": p5_regimes[i]["regime"],
         "N": int(n_values[i]),
         "L": float(l_values[i]),
         "G": float(1.0 - np.mean(score_mat[i, :]))}
        for i in range(len(p5_regimes))
    ]

    # Numerical monotone check on G(L_i)
    g_vals = np.array([row["G"] for row in gap_per_regime])
    g_diffs = np.diff(g_vals)  # G(L_{i+1}) - G(L_i), should be < 0
    n_steps_decreasing = int(np.sum(g_diffs < 0))
    n_steps_total = len(g_diffs)

    # Bootstrap CI95 on r_bar
    rng = np.random.default_rng(42)
    r_bar_boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(p5_regimes), size=len(p5_regimes))
        if len(set(idx)) < 3:
            continue
        s_boot = score_mat[idx, :]
        l_boot = l_values[idx]
        rates_b = np.array([np.polyfit(l_boot, s_boot[:, j], 1)[0]
                            for j in range(s_boot.shape[1])])
        r_bar_boot.append(float(np.mean(rates_b)))
    r_bar_boot = np.array(r_bar_boot)
    r_bar_ci95 = [float(np.percentile(r_bar_boot, 2.5)),
                  float(np.percentile(r_bar_boot, 97.5))]

    # Per-score rate breakdown
    per_score = [{"score_key": PCT_KEYS[k],
                  "rate": float(rates[k]),
                  "rate_positive": bool(rates[k] > 0)}
                 for k in range(len(PCT_KEYS))]
    n_positive_score_rates = int(np.sum(rates > 0))

    hypothesis_cert = (
        r_bar > 0
        and r_bar_ci95[0] > 0
        and n_positive_score_rates == len(PCT_KEYS)
    )

    out = {
        "method": "Gap-monotonicity audit on the canonical "
                  "$\\mathcal P_5/\\mathcal P_5 N$ physics ladder; "
                  "scores s_k = 1 - Delta_{rho_k}(N), rates r_k = "
                  "ds_k/dL where L = N^{1/3}.",
        "n_regimes": len(p5_regimes),
        "score_keys": PCT_KEYS,
        "r_bar": r_bar,
        "r_bar_bootstrap_CI95": r_bar_ci95,
        "g_prime": g_prime,
        "g_prime_bootstrap_CI95": [-r_bar_ci95[1], -r_bar_ci95[0]],
        "per_score": per_score,
        "n_positive_score_rates": n_positive_score_rates,
        "gap_per_regime": gap_per_regime,
        "g_monotone_decreasing_steps": n_steps_decreasing,
        "g_total_steps": n_steps_total,
        "hypothesis_r_bar_gt_zero_certified": hypothesis_cert,
        "verdict": ("GAP_MONOTONE_DECREASING_CERTIFIED" if hypothesis_cert
                    else "GAP_MONOTONE_PARTIAL"),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{len(p5_regimes)} P5/P5N regimes loaded")
    print(f"  r_bar = {r_bar:+.4e}, CI95 = "
          f"[{r_bar_ci95[0]:+.4e}, {r_bar_ci95[1]:+.4e}]")
    print(f"  G'(L) = {g_prime:+.4e}")
    print(f"  per-score rate > 0: {n_positive_score_rates}/{len(PCT_KEYS)}")
    print(f"  G(L) monotone-decreasing steps: "
          f"{n_steps_decreasing}/{n_steps_total}")
    print(f"  hypothesis r_bar > 0 certified: {hypothesis_cert}")
    print(f"  verdict: {out['verdict']}")
    print(f"\nSaved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
