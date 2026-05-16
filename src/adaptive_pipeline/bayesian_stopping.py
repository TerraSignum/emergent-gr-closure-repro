r"""Bayesian stopping criterion (Tier 1).

Stop when posterior of top rational > target_posterior, or
when posterior entropy < target_entropy_bits.
"""
from __future__ import annotations

import math


def posterior_entropy_bits(posterior: dict) -> float:
    """Shannon entropy of a discrete posterior, in bits."""
    h = 0.0
    for p in posterior.values():
        if p > 0:
            h -= p * math.log2(p)
    return h


def should_stop(posterior: dict, target_posterior: float = 0.95,
                 target_entropy_bits: float = 0.3) -> tuple[bool, str]:
    """Decide whether to stop the adaptive pipeline."""
    if not posterior:
        return False, "no posterior"
    top = max(posterior.values())
    h = posterior_entropy_bits(posterior)
    if top >= target_posterior:
        return True, f"top posterior {top:.4f} >= {target_posterior}"
    if h <= target_entropy_bits:
        return True, f"entropy {h:.4f} bits <= {target_entropy_bits}"
    return False, f"continue (top={top:.4f}, H={h:.4f} bits)"


if __name__ == "__main__":
    posterior = {"3/8": 0.71, "11/29": 0.09, "7/19": 0.04, "other": 0.16}
    stop, reason = should_stop(posterior)
    print(f"Stop: {stop}, reason: {reason}")
    print(f"Entropy: {posterior_entropy_bits(posterior):.3f} bits")
