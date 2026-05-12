"""Closure-derivation H-H: structural identification q = 1/alpha_xi^2 = 100/81
on the canonical P5/P5N ladder, supersedes the earlier 11/9 reading.

H-A v3 family-separated fit revealed that the canonical 9-regime
P5/P5N N-ordered ladder gives q_emp = 1.2359; the closest
System-R rational is

    q  =  1 / alpha_xi^2  =  100 / 81  =  1.23457

with rel-err 0.109%, **10x tighter** than 11/9 = 1.222 (rel-err
1.108%).

Structural reading: the convergence exponent of
delta(N) = 1 - Lambda_t/T_00 is the **inverse of the asymptote**:
since Lambda_t = alpha_xi^2 = 81/100, the rate of approach
goes as q = 1/Lambda_t = 100/81. This is a self-consistent
relation: the back-channel projection sets both where the
ratio Lambda_t/T_00 lands AND how fast it gets there.

The earlier H-F multiple-form derivation (1+2gamma/alpha_xi,
(alpha_xi+2gamma)/alpha_xi, polynomials-in-d at d=4, ...) all
evaluated to 11/9 and were structurally suggestive, but on
canonical-only data they are NOT the best rational.

Writes peer_reviews/HH_q_inverse_alpha_xi_squared.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "emergent-gr-closure-repro" / "outputs" / "per_regime_lambda_t_universal_audit.json"
OUT = REPO / "data" / "closure_derivations" / "HH_q_inverse_alpha_xi_squared.json"

GAMMA = 1.0 / 10.0
ALPHA_XI = 9.0 / 10.0


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    canonical = {"P5", "P5N64", "P5N72", "P5N84", "P5N100",
                 "P5N128", "P5N200", "P5N256", "P5N300", "P5N512"}
    rows = [(r["regime"], r["N"], r["Lambda_t_over_T_00_ratio"])
            for r in d["per_regime"] if r["regime"] in canonical]
    rows.sort(key=lambda x: x[1])
    ns = np.array([r[1] for r in rows], dtype=float)
    rs = np.array([r[2] for r in rows], dtype=float)
    deltas = 1.0 - rs

    q_pred = 1.0 / ALPHA_XI**2
    print(f"=== Hypothesis: q = 1/alpha_xi^2 = {q_pred:.5f} = 100/81 ===")
    print()

    # 0-parameter check: delta(N) = c / N^q with q fixed at 1/alpha_xi^2
    # and find c = best fit
    log_invn = np.log(1.0 / ns)
    log_delta = np.log(deltas)
    # Holding q fixed, fit c only:
    # log(delta) = log(c) - q log(N) = log(c) + q log(1/N)
    # => log(c) = mean(log(delta) - q log(1/N))
    log_c = float(np.mean(log_delta - q_pred * log_invn))
    c = math.exp(log_c)
    pred = c / ns ** q_pred
    rms_rel = float(np.sqrt(np.mean(((pred - deltas)/deltas)**2)))
    print(f"=== Fit at fixed q = 1/alpha_xi^2 ===")
    print(f"  c (best fit) = {c:.5f}")
    print(f"  RMS-rel = {100*rms_rel:.2f}%")
    print()

    print(f"{'regime':<10s} {'N':>5s} {'delta_emp':>10s} {'pred':>10s} {'rel_err':>10s}")
    rows_out = []
    for (reg, N, _r), de, dp in zip(rows, deltas, pred):
        rr = 100 * (dp - de) / de
        rows_out.append({"regime": reg, "N": int(N),
                         "delta_emp": float(de), "delta_pred": float(dp),
                         "rel_err_pct": float(rr)})
        print(f"  {reg:<8s} {int(N):>5d} {de:>10.5f} {dp:>10.5f} {rr:>+10.2f}%")
    print()

    # Test rational candidates for c at fixed q = 1/alpha_xi^2
    candidates_c = [
        ("3 = N_gen", 3.0),
        ("alpha_xi^(-3) = 1000/729", 1.0 / ALPHA_XI**3),
        ("4 = d", 4.0),
        ("(d-1)^q = 3^(100/81)", 3.0 ** q_pred),
        ("100/27", 100.0 / 27.0),
        ("d - 1 + alpha_xi = 39/10", 3 + ALPHA_XI),
        ("4 alpha_xi = 18/5", 4 * ALPHA_XI),
        ("alpha_xi^2 * 4 = 81/25", ALPHA_XI**2 * 4),
        ("(d-1)*alpha_xi^(-2) = 300/81", 3 / ALPHA_XI**2),
    ]
    print(f"=== c candidates at fixed q = 1/alpha_xi^2 ===")
    candidates_c.sort(key=lambda x: abs(x[1] - c))
    for label, val in candidates_c[:6]:
        rel = abs(val - c) / c
        print(f"  {label:<32s} = {val:.5f}  rel-err = {100*rel:.2f}%")
    best_c = candidates_c[0]
    print()

    # Compare to 11/9 reading
    q_alt = 11.0/9.0
    log_c_alt = float(np.mean(log_delta - q_alt * log_invn))
    c_alt = math.exp(log_c_alt)
    pred_alt = c_alt / ns ** q_alt
    rms_alt = float(np.sqrt(np.mean(((pred_alt - deltas)/deltas)**2)))
    print(f"=== Comparison: 11/9 reading on canonical ===")
    print(f"  q = 11/9 = {q_alt:.5f}")
    print(f"  c (best fit at q=11/9) = {c_alt:.5f}")
    print(f"  RMS-rel = {100*rms_alt:.2f}%")
    print()
    print(f"=== Verdict ===")
    if rms_rel < rms_alt * 0.95:
        verdict = (f"Q_INVERSE_ALPHA_XI_SQUARED_PREFERRED: "
                   f"q = 1/alpha_xi^2 = 100/81 (RMS {100*rms_rel:.2f}%) "
                   f"fits canonical ladder better than 11/9 "
                   f"(RMS {100*rms_alt:.2f}%). The convergence exponent "
                   f"equals the inverse of the asymptote Lambda_t = alpha_xi^2.")
    elif rms_rel * 1.05 < rms_alt:
        verdict = (f"Q_INVERSE_ALPHA_XI_SQUARED_PREFERRED")
    else:
        verdict = (f"BOTH_COMPARABLE: q=1/alpha_xi^2 (RMS {100*rms_rel:.2f}%) "
                   f"and q=11/9 (RMS {100*rms_alt:.2f}%) fit comparably "
                   f"on canonical ladder; structural ambiguity remains.")
    print(f"  {verdict}")

    bundle = {
        "method": "HH_q_inverse_alpha_xi_squared",
        "framework_constants": {"gamma": GAMMA, "alpha_xi": ALPHA_XI},
        "canonical_ladder_n": len(rows),
        "fit_at_q_inverse_alpha_xi_squared": {
            "q_predicted": q_pred,
            "q_form": "1/alpha_xi^2 = 100/81",
            "c_best_fit": c,
            "rms_rel_pct": 100*rms_rel,
        },
        "fit_at_q_11_9_for_comparison": {
            "q": q_alt,
            "c_best_fit": c_alt,
            "rms_rel_pct": 100*rms_alt,
        },
        "best_c_match": {"label": best_c[0], "value": float(best_c[1]),
                          "c_fit": c,
                          "rel_err_pct": float(100*abs(best_c[1]-c)/c)},
        "per_regime": rows_out,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
