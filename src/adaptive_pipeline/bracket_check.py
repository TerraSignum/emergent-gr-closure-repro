r"""Bracket-check runner (Tier 1).

Run a small number of seeds at the new N, verify the result lands in the
predicted bracket [prior_mean - k*sigma, prior_mean + k*sigma].

If inside: declare bracket-confirmed, no need for full 12-seed run.
If outside: alarm + full re-run with diagnostics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BracketCheckResult:
    new_n: int
    n_seeds_run: int
    observed_mean: float
    observed_std: float
    prior_mean: float
    prior_bracket: tuple[float, float]
    z_score: float            # (observed - prior) / sigma_combined
    inside_bracket: bool
    verdict: str              # "CONFIRMED", "OUTSIDE_ALARM", "BORDERLINE"


def check_bracket(observed_lambdas: list[float], prior_mean: float,
                    prior_ci95_half: float, sigma_alarm_z: float = 3.0) -> BracketCheckResult:
    """Verify if observed mean lands inside prior bracket."""
    n_seeds = len(observed_lambdas)
    if n_seeds == 0:
        raise ValueError("at least 1 seed required")
    observed_mean = sum(observed_lambdas) / n_seeds
    if n_seeds == 1:
        observed_std = 0.0
        sigma_mean = 0.02  # heuristic single-seed sigma
    else:
        var = sum((x - observed_mean) ** 2 for x in observed_lambdas) / (n_seeds - 1)
        observed_std = math.sqrt(var)
        sigma_mean = observed_std / math.sqrt(n_seeds)

    bracket = (prior_mean - 2 * prior_ci95_half,
                prior_mean + 2 * prior_ci95_half)
    # Combined sigma: prior CI + sample mean sigma
    sigma_combined = math.sqrt((prior_ci95_half / 1.96) ** 2 + sigma_mean ** 2)
    z = (observed_mean - prior_mean) / max(sigma_combined, 1e-9)

    inside = bracket[0] <= observed_mean <= bracket[1]
    if abs(z) > sigma_alarm_z:
        verdict = "OUTSIDE_ALARM"
    elif inside:
        verdict = "CONFIRMED"
    else:
        verdict = "BORDERLINE"

    return BracketCheckResult(
        new_n=0,
        n_seeds_run=n_seeds,
        observed_mean=observed_mean,
        observed_std=observed_std,
        prior_mean=prior_mean,
        prior_bracket=bracket,
        z_score=z,
        inside_bracket=inside,
        verdict=verdict,
    )


if __name__ == "__main__":
    # Demo
    obs = [0.397, 0.399, 0.401]
    res = check_bracket(obs, prior_mean=0.398, prior_ci95_half=0.005)
    print(f"Observed: {res.observed_mean:.4f} +/- {res.observed_std:.4f}")
    print(f"Prior bracket: {res.prior_bracket}")
    print(f"z-score: {res.z_score:.2f}, verdict: {res.verdict}")
