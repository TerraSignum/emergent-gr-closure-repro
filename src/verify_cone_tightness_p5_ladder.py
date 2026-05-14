"""Cone-tightness audit on the canonical $\\mathcal P_5/\\mathcal P_5 N$
physics ladder.

Companion to the model-point-ladder audit
`verify_cone_tightness_extended_ladder.py' (which reports a
negative result on the cross-model-point sequence). Here we
evaluate the score-rate covariance on the canonical-physics
closure-domain ladder
$N\\in\\{50, 64, 72, 84, 100, 128, 200, 256, 300, 512\\}$
using as `scores' the per-regime closure quality of the
five-percentile cut $\\rho\\in\\{$median, mean, p95, p99, sup$\\}$:

    s_k(N)  :=  1 - Delta_{rho_k}(N)

so that s_k -> 1 corresponds to perfect closure at percentile
rho_k. The lattice scale is L = N^{1/3}; the rate vector is
the linear-fit slope r_k = ds_k / dL across the ladder; the
cone-tightness hypothesis is Cov(s, r) < 0.

Output: outputs/verify_cone_tightness_p5_ladder.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "outputs" / "stage6f_fine_percentile_audit.json"
OUT = REPO / "outputs" / "verify_cone_tightness_p5_ladder.json"

PCT_KEYS = ["median", "mean", "p95", "p99", "sup"]
P5_REGIMES = {"P5", "P5N64", "P5N72", "P5N84", "P5N100",
              "P5N128", "P5N200", "P5N256", "P5N300", "P5N512"}


def cov(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def cv(x):
    x = np.asarray(x, dtype=float)
    return float(np.std(x) / np.mean(x)) if np.mean(x) > 1e-15 else float("nan")


def build_score_matrix(p5_regimes):
    """Score matrix S[i, k] = 1 - delta at percentile k for regime i."""
    n_values = np.array([float(r["N"]) for r in p5_regimes])
    l_values = n_values ** (1.0 / 3.0)
    score_mat = np.zeros((len(p5_regimes), len(PCT_KEYS)))
    for i, r in enumerate(p5_regimes):
        for k_idx, k_name in enumerate(PCT_KEYS):
            v = r["percentiles"].get(k_name)
            score_mat[i, k_idx] = 1.0 - float(v) if v is not None else np.nan
    return score_mat, n_values, l_values


def fit_rates(score_mat, l_values):
    """Per-score linear fit s_k(L) = a_k + b_k L; rate r_k = b_k."""
    rates = np.zeros(score_mat.shape[1])
    for k in range(score_mat.shape[1]):
        b, _ = np.polyfit(l_values, score_mat[:, k], 1)
        rates[k] = b
    return rates


def compute_per_regime(p5_regimes, score_mat, n_values, l_values, rates):
    """Per-regime cov(s, r) and dCV/dL with full-ladder rates."""
    r_bar_full = float(np.mean(rates))
    per_regime = []
    for i in range(len(p5_regimes)):
        s_vec = score_mat[i, :]
        cov_i = cov(s_vec, rates)
        mu_i = float(np.mean(s_vec))
        sig_i = float(np.std(s_vec))
        cv_i = sig_i / mu_i if mu_i > 1e-15 else float("nan")
        sig_prime = cov_i / sig_i if sig_i > 1e-15 else 0.0
        dcv_dl = ((sig_prime * mu_i - sig_i * r_bar_full) / (mu_i ** 2)
                  if mu_i > 1e-15 else float("nan"))
        per_regime.append({
            "regime": p5_regimes[i]["regime"],
            "N": int(n_values[i]),
            "L": float(l_values[i]),
            "cv": cv_i,
            "cov_sr": cov_i,
            "mu": mu_i,
            "sigma": sig_i,
            "dcv_dl": dcv_dl,
        })
    return per_regime


def compute_pairwise(p5_regimes, score_mat, n_values, l_values):
    """Adjacent-pair cov with local rates."""
    pairwise = []
    for i in range(len(p5_regimes) - 1):
        s_i = score_mat[i, :]
        s_j = score_mat[i + 1, :]
        dl = l_values[i + 1] - l_values[i]
        if abs(dl) < 1e-12:
            continue
        local_rates = (s_j - s_i) / dl
        cov_local = cov(s_i, local_rates)
        pairwise.append({
            "regime_lo": p5_regimes[i]["regime"],
            "regime_hi": p5_regimes[i + 1]["regime"],
            "N_lo": int(n_values[i]),
            "N_hi": int(n_values[i + 1]),
            "dL": float(dl),
            "cov_sr_local": cov_local,
            "sign_negative": bool(cov_local < 0),
        })
    return pairwise


def bootstrap_cov_ci(score_mat, l_values, n_boot=2000, seed=42):
    """Bootstrap CI95 on the global Cov(s, r)."""
    n = score_mat.shape[0]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(set(idx)) < 3:
            continue
        s_boot = score_mat[idx, :]
        l_boot = l_values[idx]
        rates_b = np.array([np.polyfit(l_boot, s_boot[:, j], 1)[0]
                            for j in range(s_boot.shape[1])])
        s_means_b = s_boot.mean(axis=0)
        boot.append(cov(s_means_b, rates_b))
    boot = np.array(boot)
    return [float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5))]


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    p5_regimes = [r for r in d["regimes"]
                  if r["regime"] in P5_REGIMES and "percentiles" in r]
    p5_regimes.sort(key=lambda r: r["N"])

    score_mat, n_values, l_values = build_score_matrix(p5_regimes)
    rates = fit_rates(score_mat, l_values)

    s_means = score_mat.mean(axis=0)
    cov_global = cov(s_means, rates)
    r_bar_global = float(np.mean(rates))

    per_regime = compute_per_regime(
        p5_regimes, score_mat, n_values, l_values, rates)
    pairwise = compute_pairwise(p5_regimes, score_mat, n_values, l_values)
    cov_ci95 = bootstrap_cov_ci(score_mat, l_values)

    n_pair_negative = sum(1 for p in pairwise if p["sign_negative"])
    n_pair_total = len(pairwise)
    n_regime_dcv_negative = sum(1 for r in per_regime
                                 if r["dcv_dl"] is not None
                                 and r["dcv_dl"] < 0)

    # Robust certification: global Cov(s, r) < 0 with bootstrap-CI95
    # strictly below zero, plus dCV/dL < 0 on every regime in the
    # ladder. The adjacent-pair-local Cov sign-check is reported
    # but not part of the certification criterion (local
    # fluctuations are expected; the lattice-scaling claim is
    # about the global / per-regime average rate, not the
    # adjacent-pair local rate).
    hypothesis_cert = (
        cov_global < 0
        and cov_ci95[1] < 0
        and n_regime_dcv_negative == len(p5_regimes)
    )

    out = {
        "method": "Cone-tightness audit on the canonical "
                  "$\\mathcal P_5/\\mathcal P_5 N$ physics ladder; "
                  "scores s_k = 1 - Delta_{rho_k}(N) for "
                  "rho_k in {median, mean, p95, p99, sup}.",
        "ladder": [{"regime": r["regime"], "N": int(n_values[i]),
                    "L": float(l_values[i])} for i, r in enumerate(p5_regimes)],
        "n_regimes": len(p5_regimes),
        "score_keys": PCT_KEYS,
        "n_score_keys": len(PCT_KEYS),
        "global_cov_sr": cov_global,
        "global_cov_sr_bootstrap_CI95": cov_ci95,
        "global_r_bar": r_bar_global,
        "per_regime": per_regime,
        "pairwise_adjacent": pairwise,
        "n_adjacent_pairs": n_pair_total,
        "n_adjacent_pairs_with_neg_cov": n_pair_negative,
        "n_regimes_with_neg_dcv_dl": n_regime_dcv_negative,
        "hypothesis_cov_sr_lt_zero_certified": hypothesis_cert,
        "verdict": ("CONE_TIGHTENING_CERTIFIED" if hypothesis_cert
                    else "CONE_TIGHTENING_PARTIAL"),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{len(p5_regimes)} P5/P5N regimes loaded")
    print(f"  global Cov(s, r) = {cov_global:+.4e}, "
          f"CI95 = [{cov_ci95[0]:+.4e}, {cov_ci95[1]:+.4e}]")
    print(f"  adjacent pairs with negative cov: "
          f"{n_pair_negative}/{n_pair_total}")
    print(f"  per-regime dCV/dL negative: "
          f"{n_regime_dcv_negative}/{len(p5_regimes)}")
    print(f"  hypothesis Cov(s,r) < 0 certified: {hypothesis_cert}")
    print(f"  verdict: {out['verdict']}")
    print(f"\nSaved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
