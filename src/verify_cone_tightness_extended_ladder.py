"""Extended cone-tightness audit on the canonical closure-domain
ladder.

The cone-anticorrelation tightening identity reads

    Cov(s, r) < 0    ⟹    d/dL CV(L) < 0

where L = N^{1/3} is the effective lattice scale, {s_k} is the
projector-score vector at a given regime, and {r_k} is the
linear rate of score change vs L. The identity itself is
algebraic (Cauchy--Schwarz on covariance + standard CV
calculus); the *hypothesis* Cov(s, r) < 0 is empirical and
requires numerical verification on the ladder of interest.

The bundled internal proof (`proof_h203_cone_tightness` in the
framework code) verifies the hypothesis on a 2-regime pair
{P1, P2'}. This script extends the verification to the canonical
9-payload model-point ladder {P0, ..., P8} via the same six
closure-score keys used by the Dal Maso Γ-convergence audit
(`verify_clp_c_gamma_convergence_detailed.py`), and reports the
global linear-fit rate vector r_k = ds_k/dL and the
Cov(s, r) sign across (i) every adjacent regime pair on the
ladder and (ii) the global fit. Hypothesis Cov(s, r) < 0 is
considered empirically certified if every adjacent pair
satisfies it AND the global Cov(s, r) is negative with
bootstrap-CI95 strictly below zero.

Output: outputs/verify_cone_tightness_extended_ladder.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
D1_DIRS = [
    REPO / "results_d1_fix17",
    REPO / "results_d1_fix16" / "p6",
    REPO / "results_d1_fix16" / "p7",
    REPO / "results_d1_fix16" / "p8",
]
OUT = REPO / "emergent-gr-closure-repro" / "outputs" / \
      "verify_cone_tightness_extended_ladder.json"

SCORE_KEYS = [
    "d1_fixpoint_proximity_score",
    "d1_fixpoint_transport_score",
    "d1_gamma_full_macroclass_joint_closure_score",
    "d1_gamma_full_macroclass_joint_nonuniform_closure_score",
    "d1_gamma_ir_variational_closure_score",
    "d1_gamma_ir_residual_locality_score",
]


def _sf(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def load_payloads():
    payloads = []
    for d in D1_DIRS:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("d1_p*.json")):
            if f.name.endswith(".metadata.json") or "report" in f.name:
                continue
            payloads.append(json.loads(f.read_text(encoding="utf-8")))
    seen = {}
    for p in payloads:
        n = p.get("dense_cell_node_count")
        if n is None:
            continue
        seen[int(round(float(n)))] = p
    return [seen[k] for k in sorted(seen.keys())]


def cov(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def cv(x):
    x = np.asarray(x, dtype=float)
    return float(np.std(x) / np.mean(x)) if np.mean(x) > 1e-15 else float("nan")


def main():
    payloads = load_payloads()
    print(f"{len(payloads)} payloads loaded")
    Ns = np.array([float(p["dense_cell_node_count"]) for p in payloads])
    Ls = Ns ** (1.0 / 3.0)

    # Build the (regime x score) matrix
    S = np.zeros((len(payloads), len(SCORE_KEYS)))
    for i, p in enumerate(payloads):
        for j, k in enumerate(SCORE_KEYS):
            v = _sf(p.get(k))
            S[i, j] = v if v is not None else np.nan

    # Drop scores with any NaN (must be defined on every regime).
    good_cols = ~np.any(np.isnan(S), axis=0)
    used_keys = [SCORE_KEYS[j] for j, ok in enumerate(good_cols) if ok]
    S = S[:, good_cols]
    print(f"  usable score keys: {len(used_keys)} / {len(SCORE_KEYS)}")

    # Per-score linear fit s_k(L) = a_k + b_k L -> rate r_k = b_k
    rates = np.zeros(S.shape[1])
    for j in range(S.shape[1]):
        b, a = np.polyfit(Ls, S[:, j], 1)
        rates[j] = b

    # Global cov(s, r) using s = mean score per key, r = b_k
    s_means = S.mean(axis=0)
    cov_global = cov(s_means, rates)
    r_bar_global = float(np.mean(rates))

    # Per-regime cov(s, r) using s = scores at this regime, r = b_k
    per_regime_cov = []
    per_regime_dcv = []
    for i in range(len(payloads)):
        s_vec = S[i, :]
        cov_i = cov(s_vec, rates)
        cv_i = cv(s_vec)
        mu_i = float(np.mean(s_vec))
        sig_i = float(np.std(s_vec))
        r_bar_i = float(np.mean(rates))
        sig_prime = cov_i / sig_i if sig_i > 1e-15 else 0.0
        dcv_dl = ((sig_prime * mu_i - sig_i * r_bar_i) / (mu_i ** 2)
                  if mu_i > 1e-15 else float("nan"))
        per_regime_cov.append({
            "N": int(Ns[i]),
            "L": float(Ls[i]),
            "cv": cv_i,
            "cov_sr": cov_i,
            "mu": mu_i,
            "sigma": sig_i,
            "r_bar": r_bar_i,
            "dcv_dl": dcv_dl,
        })

    # Adjacent-pair cov(s, r) using local rates
    pairwise = []
    for i in range(len(payloads) - 1):
        s_i = S[i, :]
        s_j = S[i + 1, :]
        dl = Ls[i + 1] - Ls[i]
        if abs(dl) < 1e-12:
            continue
        local_rates = (s_j - s_i) / dl
        cov_local = cov(s_i, local_rates)
        pairwise.append({
            "N_lo": int(Ns[i]),
            "N_hi": int(Ns[i + 1]),
            "dL": float(dl),
            "cov_sr_local": cov_local,
            "sign_negative": bool(cov_local < 0),
        })

    # Bootstrap CI95 on global cov(s, r)
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(payloads), size=len(payloads))
        if len(set(idx)) < 3:
            continue
        Sb = S[idx, :]
        Lb = Ls[idx]
        rates_b = np.array([np.polyfit(Lb, Sb[:, j], 1)[0]
                            for j in range(Sb.shape[1])])
        s_means_b = Sb.mean(axis=0)
        boot.append(cov(s_means_b, rates_b))
    boot = np.array(boot)
    cov_ci95 = [float(np.percentile(boot, 2.5)),
                float(np.percentile(boot, 97.5))]

    n_pair_negative = sum(1 for p in pairwise if p["sign_negative"])
    n_pair_total = len(pairwise)
    n_regime_dcv_negative = sum(1 for r in per_regime_cov
                                 if r["dcv_dl"] is not None
                                 and r["dcv_dl"] < 0)

    hypothesis_cert = (
        cov_global < 0
        and cov_ci95[1] < 0
        and n_pair_negative == n_pair_total
    )

    out = {
        "method": "Extended cone-tightness audit on the canonical "
                  "closure-domain ladder; verifies the empirical "
                  "hypothesis Cov(s, r) < 0 underlying the analytic "
                  "cone-anticorrelation tightening identity.",
        "ladder": [{"N": int(Ns[i]), "L": float(Ls[i])} for i in range(len(payloads))],
        "n_regimes": len(payloads),
        "score_keys_used": used_keys,
        "n_score_keys": len(used_keys),
        "global_cov_sr": cov_global,
        "global_cov_sr_bootstrap_CI95": cov_ci95,
        "global_r_bar": r_bar_global,
        "per_regime": per_regime_cov,
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
    print(f"  global Cov(s, r) = {cov_global:+.4e}, "
          f"CI95 = [{cov_ci95[0]:+.4e}, {cov_ci95[1]:+.4e}]")
    print(f"  adjacent pairs with negative cov: "
          f"{n_pair_negative}/{n_pair_total}")
    print(f"  per-regime dCV/dL negative: "
          f"{n_regime_dcv_negative}/{len(payloads)}")
    print(f"  hypothesis Cov(s,r) < 0 certified: {hypothesis_cert}")
    print(f"  verdict: {out['verdict']}")
    print(f"\nSaved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
