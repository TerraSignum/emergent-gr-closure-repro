r"""Theory-experiment hybrid prior over candidate rationals (Tier 4).

Combines:
  1. Occam-style structural simplicity: 1 / (q * log(q+1))
  2. Theoretical-plausibility score from analytic bounds:
       Cheeger lower bound, Kesten-McKay upper-edge, cavity prediction
  3. Algebraic-form bonus for closures expressible as
     simple-formulae in the (d, N_gen) anchor.

Output: Prior(p/q) for any rational p/q with q <= q_max.
"""
from __future__ import annotations

import math
from fractions import Fraction


# Analytic anchors (theory-side).
# Each is a (rational, score, justification) triple.
_THEORY_ANCHORS = [
    # lambda_inf^vac on the (4, 3) anchor: (d-1)/(2d) = 3/8.
    # Weight 3.0 in log-prior gives 20x bonus -- moderate, defensible:
    # reflects Cheeger upper-bound chain + KM tree-like baseline + cavity
    # consistency, but does not predetermine the answer.
    (Fraction(3, 8), 3.0, "(d-1)/(2d) at d=4; Cheeger + KM + cavity chain"),
    # lambda_inf^skel: (d+N_gen)/(2 d N_gen) = 7/24
    (Fraction(7, 24), 2.5, "(d+N_gen)/(2 d N_gen) skeleton harmonic-mean"),
    # lambda_inf^mat closure: 79/200 = 3/8 + d gamma^2 / 2
    (Fraction(79, 200), 2.5, "3/8 + d*gamma^2/2 matter-branch lift"),
    # Vacuum-near alternatives that are structurally less plausible
    # but algebraically simple:
    (Fraction(2, 5), 1.0, "2/5 = matter-branch coincidence; weak theory support"),
    (Fraction(1, 3), 1.0, "1/3 = 1/N_gen; trivial generation-mode"),
    (Fraction(5, 12), 0.8, "5/12 = bundle-fraction (d, N_gen)-specific"),
    (Fraction(11, 28), 0.3, "weighted-gap alternative, no closed-form"),
    (Fraction(7, 19), 0.3, "near 3/8 numerical coincidence"),
]

# Universal-leakage shift: lambda_2^KM(d=12) - 7/100 ~ 0.377
# This is the rigorous KM baseline tied to the carrier (d_eff = 12).
_KM_SHIFTED_VALUE = 0.37723  # 1 - 2*sqrt(11)/12 - 7/100
_KM_WIDTH = 0.012             # plausibility window around the KM anchor

# Cavity-method window (Pham-Peron-Metz numerics; empirical observation)
_CAVITY_LO, _CAVITY_HI = 0.37, 0.39


def occam_score(p_q: Fraction) -> float:
    """Log-prior weight from Kolmogorov-style simplicity."""
    q = p_q.denominator
    return -math.log(q * math.log(q + 1))


def theory_score(p_q: Fraction) -> float:
    """Sum of analytical-plausibility contributions for candidate p/q."""
    score = 0.0
    val = float(p_q)
    # Match against named anchors
    for anchor, weight, _just in _THEORY_ANCHORS:
        if p_q == anchor:
            score += weight
    # KM-shifted upper-edge baseline (continuous proximity)
    delta_km = abs(val - _KM_SHIFTED_VALUE)
    if delta_km < _KM_WIDTH:
        score += 2.0 * (1.0 - delta_km / _KM_WIDTH)
    # Cavity-window membership
    if _CAVITY_LO <= val <= _CAVITY_HI:
        score += 1.0
    return score


def log_prior(p_q: Fraction) -> float:
    """Combined Occam + theory log-prior (unnormalised)."""
    return occam_score(p_q) + theory_score(p_q)


def prior_table(rationals: list[Fraction]) -> dict[Fraction, dict[str, float]]:
    """Return per-rational decomposition of the log-prior."""
    rows = {}
    for r in rationals:
        oc = occam_score(r)
        th = theory_score(r)
        rows[r] = {
            "occam": oc,
            "theory": th,
            "log_prior_raw": oc + th,
        }
    # Normalise prior to sum to 1
    log_raw = [v["log_prior_raw"] for v in rows.values()]
    max_log = max(log_raw)
    exp_norm = sum(math.exp(lr - max_log) for lr in log_raw)
    log_norm = max_log + math.log(exp_norm)
    for r in rows:
        rows[r]["log_prior_normed"] = rows[r]["log_prior_raw"] - log_norm
        rows[r]["prior"] = math.exp(rows[r]["log_prior_normed"])
    return rows


if __name__ == "__main__":
    from .. import _d1_ladder_discovery  # noqa: F401
    import sys
    qmax = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    seen = set()
    rats = []
    for q in range(2, qmax + 1):
        for p in range(1, q):
            f = Fraction(p, q)
            if f.denominator > qmax:
                continue
            if f not in seen:
                seen.add(f)
                rats.append(f)
    table = prior_table(rats)
    rows = sorted(table.items(), key=lambda kv: kv[1]["prior"], reverse=True)
    print("Top-10 theory-hybrid prior weights:")
    print(f"  {'p/q':>10} {'value':>9} {'occam':>+7} {'theory':>+7} {'prior':>10}")
    for r, info in rows[:10]:
        print(f"  {f'{r.numerator}/{r.denominator}':>10} "
              f"{float(r):>9.5f} {info['occam']:+7.3f} {info['theory']:+7.3f} "
              f"{info['prior']:>10.5f}")
