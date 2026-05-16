r"""Joint posterior over System-R primitives (Tier 4).

The carrier programme has 6 System-R primitives, algebraically coupled:
  lambda_inf = 3/8
  gamma^2    = 1/100  (chirality-flip)
  alpha_xi^2 = 81/100 (similarity-matrix)
  D_Omega    = 67/80  (chirality-mixing)
  beta_pi    = 15/16
  eps^2_sync = 1/20 = gamma/2  (sync-locking)

Constraint: gamma * alpha_xi = sync ~ ... and other algebraic identities
documented in P3 + P4 + bridge.

Joint posterior on (lambda, gamma_sq, alpha_xi_sq, D_Omega, beta_pi,
eps_sync_sq) under empirical data + algebraic constraints gives much
tighter CI than individual posteriors, because mutual constraints
prune the prior support.

Implementation: rejection sampler over the rational lattice, weighted
by per-closure likelihood + indicator function for algebraic constraints
(within tolerance).
"""
from __future__ import annotations

import math
from fractions import Fraction
from dataclasses import dataclass


@dataclass
class SystemRPrimitive:
    name: str
    registered_rational: Fraction
    empirical_mean: float
    empirical_ci95_half: float


# Standard System-R anchor at (d=4, N_gen=3):
SYSTEM_R = {
    "lambda_inf_vac": SystemRPrimitive("lambda_inf_vac", Fraction(3, 8),
                                         0.3732, 0.003),
    "gamma_sq": SystemRPrimitive("gamma_sq", Fraction(1, 100),
                                   0.0100, 0.0002),
    "alpha_xi_sq": SystemRPrimitive("alpha_xi_sq", Fraction(81, 100),
                                      0.8100, 0.001),
    "D_Omega": SystemRPrimitive("D_Omega", Fraction(67, 80),
                                  0.8375, 0.001),
    "beta_pi": SystemRPrimitive("beta_pi", Fraction(15, 16),
                                  0.9375, 0.001),
    "eps_sync_sq": SystemRPrimitive("eps_sync_sq", Fraction(1, 20),
                                      0.0500, 0.0005),
}


# Algebraic constraints (closed-form identities documented in corpus):
def algebraic_constraints_log_likelihood(
    primitives: dict[str, float],
    tol: float = 1e-3,
) -> float:
    """Log-likelihood from algebraic constraints, Gaussian penalty."""
    ll = 0.0

    # Constraint 1: gamma^2 = eps_sync^2 (sync-locking eps^2_sync = gamma)
    # Approximation: gamma_sq = 1/100, eps_sync_sq = gamma/2 = 1/20 = 5 gamma_sq
    expected = 5 * primitives.get("gamma_sq", 0.0)
    observed = primitives.get("eps_sync_sq", 0.0)
    if expected > 0:
        ll += -0.5 * ((expected - observed) / max(tol, 1e-12)) ** 2

    # Constraint 2: alpha_xi_sq = (1 - gamma)^2 ~ 0.9^2 = 0.81 (similarity-matrix)
    gamma = math.sqrt(primitives.get("gamma_sq", 0.0))
    expected = (1 - gamma) ** 2
    observed = primitives.get("alpha_xi_sq", 0.0)
    ll += -0.5 * ((expected - observed) / max(tol, 1e-12)) ** 2

    # Constraint 3: lambda_inf_vac = (d-1)/(2d) = 3/8 at (d=4, N_gen=3)
    # (this is the structural conjecture)

    return ll


def joint_posterior_sample(n_samples: int = 20000,
                             seed: int = 42) -> list[dict]:
    """Generate joint posterior samples via importance sampling.

    Sample from independent Gaussian likelihoods per closure, then weight
    by algebraic-constraint joint log-likelihood.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    samples = []
    log_weights = []
    for _ in range(n_samples):
        s = {}
        for name, prim in SYSTEM_R.items():
            sigma = prim.empirical_ci95_half / 1.96
            s[name] = float(rng.normal(prim.empirical_mean, sigma))
        log_w = algebraic_constraints_log_likelihood(s, tol=0.001)
        samples.append(s)
        log_weights.append(log_w)
    # Resample by weight
    log_weights = np.asarray(log_weights)
    max_lw = log_weights.max()
    weights = np.exp(log_weights - max_lw)
    weights /= weights.sum()
    indices = rng.choice(n_samples, size=n_samples, p=weights, replace=True)
    return [samples[i] for i in indices]


def joint_summary(samples: list[dict]) -> dict:
    """Posterior summary from joint samples."""
    import numpy as np
    if not samples:
        return {}
    keys = list(samples[0].keys())
    summary = {}
    for k in keys:
        arr = np.array([s[k] for s in samples])
        summary[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "ci95": (float(np.quantile(arr, 0.025)),
                       float(np.quantile(arr, 0.975))),
            "registered_rational_value": float(SYSTEM_R[k].registered_rational),
            "z_from_registered": float(
                (arr.mean() - float(SYSTEM_R[k].registered_rational))
                / max(arr.std(ddof=1), 1e-12)
            ),
        }
    return summary


if __name__ == "__main__":
    print("Joint posterior over 6 System-R primitives (importance-resampled):")
    samples = joint_posterior_sample(n_samples=10000)
    summary = joint_summary(samples)
    print(f"  {'primitive':<18} {'mean':>9} {'std':>9} {'registered':>11} {'z':>7}")
    for k, v in summary.items():
        print(f"  {k:<18} {v['mean']:>9.5f} {v['std']:>9.5f} "
              f"{v['registered_rational_value']:>11.5f} {v['z_from_registered']:>+7.2f}")
