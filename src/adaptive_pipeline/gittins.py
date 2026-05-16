r"""Information-theoretic optimal next-action (Tier 4 Gittins index).

For each candidate action (run at N=X with k seeds, parameter-variation,
tau-sweep, etc.), compute:
  Expected Information Gain (EIG) / Compute Cost (hours)

The optimal next action maximises this ratio (Gittins-index optimality
for the multi-armed-bandit setting where each arm gives stochastic info).

EIG estimation: predict reduction in posterior entropy under a
hypothetical-data model. We use a Gaussian approximation:
  EIG ~ 0.5 * log(1 + sigma_prior^2 / sigma_post_predicted^2)
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Action:
    name: str
    cost_hours: float
    predicted_sigma_post: float    # predicted posterior std after running
    description: str = ""


@dataclass
class GittinsResult:
    action: Action
    eig_bits: float
    gittins_index: float           # eig_bits / cost_hours
    rank: int


def expected_info_gain_gaussian(sigma_prior: float, sigma_post: float) -> float:
    """EIG (in nats) for Gaussian posterior shrinkage from sigma_prior to sigma_post."""
    if sigma_post <= 0 or sigma_prior <= 0:
        return 0.0
    if sigma_post >= sigma_prior:
        return 0.0
    return 0.5 * math.log(sigma_prior ** 2 / sigma_post ** 2)


def predict_sigma_post(action: Action, sigma_prior: float,
                        per_seed_std: float = 0.017) -> float:
    """Predict posterior sigma after action.

    For an action that takes k seeds at higher N:
        sigma_post ~ per_seed_std / sqrt(k) combined with sigma_prior
        via standard Bayesian update (precision sum).
    """
    return action.predicted_sigma_post


def rank_actions(actions: list[Action], sigma_prior: float,
                  per_seed_std: float = 0.017) -> list[GittinsResult]:
    """Rank candidate actions by Gittins index."""
    results = []
    for a in actions:
        sigma_post = predict_sigma_post(a, sigma_prior, per_seed_std)
        eig_nats = expected_info_gain_gaussian(sigma_prior, sigma_post)
        eig_bits = eig_nats / math.log(2)
        gi = eig_bits / max(a.cost_hours, 1e-6)
        results.append(GittinsResult(action=a, eig_bits=eig_bits,
                                       gittins_index=gi, rank=0))
    results.sort(key=lambda r: r.gittins_index, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1
    return results


def standard_action_menu(current_sigma: float, per_seed_std: float = 0.017) -> list[Action]:
    """Generate a standard candidate-action menu for the carrier program."""
    import math as _m
    actions = []

    # Higher-N actions at current parameter point (d=4, N_gen=3, tau=0.10)
    for n_lat, base_cost in [(2048, 0.5), (4096, 4.0), (8192, 32.0)]:
        for k in (1, 2, 3, 12):
            # sigma_post: combine prior precision and new-data precision
            sigma_data = per_seed_std / _m.sqrt(k)
            sigma_post = 1.0 / _m.sqrt(1.0 / current_sigma**2 + 1.0 / sigma_data**2)
            # Cost scales linearly in k for moderate k
            cost = base_cost * k / 12 if k > 1 else base_cost / 12
            actions.append(Action(
                name=f"N={n_lat}, {k} seeds",
                cost_hours=cost,
                predicted_sigma_post=sigma_post,
                description=f"Lever-arm extension at d=4, N_gen=3",
            ))

    # Parameter-variation actions (test d-only hypothesis)
    for d_test in (3, 5, 6):
        for k in (3, 12):
            # New d-point: new posterior dimension. EIG is harder to estimate;
            # use heuristic that d-variation halves d-only hypothesis uncertainty.
            sigma_post = current_sigma * 0.6  # heuristic: large info from d-variation
            cost = 0.05 * k / 3  # cheap at N=512 with full GPU stack
            actions.append(Action(
                name=f"(d={d_test}, N_gen=3) test, {k} seeds at N=512",
                cost_hours=cost,
                predicted_sigma_post=sigma_post,
                description="d-only hypothesis test",
            ))

    # Tau-sweep
    actions.append(Action(
        name="tau-sweep 6 points at N=512, 3 seeds each",
        cost_hours=0.15,
        predicted_sigma_post=current_sigma * 0.8,
        description="Threshold-robustness of 7/24 closure",
    ))

    return actions


if __name__ == "__main__":
    sigma_prior = 0.005  # current N=1024 CI95_half ~ 0.005
    actions = standard_action_menu(sigma_prior)
    ranked = rank_actions(actions, sigma_prior)
    print(f"Current posterior sigma = {sigma_prior:.4f}")
    print(f"Top-10 Gittins-optimal actions:")
    print(f"  {'rank':>4} {'action':<45} {'cost(h)':>9} {'eig(bits)':>10} {'gittins':>9}")
    for r in ranked[:10]:
        print(f"  {r.rank:>4} {r.action.name:<45} "
              f"{r.action.cost_hours:>9.3f} {r.eig_bits:>10.3f} {r.gittins_index:>9.2f}")
