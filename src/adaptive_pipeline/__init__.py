"""Adaptive carrier-pipeline modules for Tier 1-4 sequential refinement.

Tier 1 (immediate):
    seed_budget       — adaptive seed-count from prior CI target
    bracket_check     — verify next-N landing in predicted bracket
    bayesian_stopping — entropy-based stop criterion

Tier 2 (medium):
    cross_n_coupling  — multi-grid initialisation for higher-N
    information_gain  — info-gain estimators per candidate action
    mim               — marginal-info maximisation for next-action choice

Tier 3 (advanced):
    bandit            — Thompson-sampling for strategy selection
    universal_bracket — analytic bound combination
    doubt_score       — weakest-link variance attribution

Tier 4 (deep frameworks):
    theory_prior      — Cheeger + KM + cavity structured prior
    hierarchical_bayes — joint posterior over (lambda, b, c, sigma)
    gittins           — info/cost optimal next-action
    multi_closure     — joint posterior over System-R primitives
    surrogate         — mean-field + generative surrogate models
"""
from __future__ import annotations

__all__ = [
    "seed_budget",
    "bracket_check",
    "bayesian_stopping",
    "theory_prior",
    "hierarchical_bayes",
    "gittins",
    "multi_closure",
]
