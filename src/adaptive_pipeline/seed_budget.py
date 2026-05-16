r"""Adaptive seed budget (Tier 1).

Given target CI95 half-width and observed per-seed sigma, return the
minimum seed count needed.
"""
from __future__ import annotations

import math


def required_seeds(per_seed_std: float, target_ci95_half: float,
                    min_seeds: int = 1, max_seeds: int = 24) -> int:
    """Minimum seeds for target CI95 from observed per-seed std.

    CI95_half = 1.96 * sigma_mean = 1.96 * per_seed_std / sqrt(n_seeds).
    Solve for n_seeds: n = (1.96 * per_seed_std / target_ci95_half)^2.
    """
    if target_ci95_half <= 0:
        return max_seeds
    n = (1.96 * per_seed_std / target_ci95_half) ** 2
    return max(min_seeds, min(max_seeds, int(math.ceil(n))))


def bracket_check_seeds(prior_bracket_half: float, per_seed_std: float,
                          confidence: float = 0.99) -> int:
    """Seeds needed to verify next-N landing in prior bracket with given confidence.

    If true value is at edge of prior bracket, we want the sample mean's
    CI to overlap the bracket. Sample mean has sigma = sigma_seed/sqrt(n).
    Need sigma_mean <= prior_bracket_half / z_alpha.
    """
    from scipy.stats import norm
    z = float(norm.ppf((1 + confidence) / 2))
    return required_seeds(per_seed_std, prior_bracket_half / z, min_seeds=2)


def plan_next_run(current_asymptote: float, current_ci95_half: float,
                   target_n: int, per_seed_std: float = 0.017) -> dict:
    """Plan the next run at target_n given prior knowledge."""
    # Bracket-check seeds: verify result lands in [prior - 2*ci95, prior + 2*ci95]
    bracket_half = 2.0 * current_ci95_half
    seeds_check = bracket_check_seeds(bracket_half, per_seed_std)
    # Full-precision seeds: target CI halved
    target_ci_half = current_ci95_half / 2
    seeds_full = required_seeds(per_seed_std, target_ci_half)
    return {
        "target_n": target_n,
        "prior_mean": current_asymptote,
        "prior_bracket": [current_asymptote - bracket_half,
                          current_asymptote + bracket_half],
        "seeds_bracket_check": seeds_check,
        "seeds_full_precision": seeds_full,
        "recommendation": ("bracket-check first, then expand to full-precision "
                           "if needed"),
    }


if __name__ == "__main__":
    # Demo: at N=1024 we have asymptote 0.398 ± 0.005
    plan = plan_next_run(current_asymptote=0.398, current_ci95_half=0.005,
                         target_n=2048)
    print(f"Plan for N=2048:")
    for k, v in plan.items():
        print(f"  {k}: {v}")
