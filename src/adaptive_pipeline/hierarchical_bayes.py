r"""Hierarchical Bayesian Symanzik fit (Tier 4).

Standard practice (currently): per-N seed-mean lambda_2(N), then
ordinary least-squares fit lambda(N) = a + b/N.

Tier-4 upgrade: joint posterior over (lambda_inf, b, c, sigma_seed)
where each per-seed observation contributes likelihood. Captures:
  * within-N seed variance (sigma_seed) as a parameter, not assumed
  * Symanzik-2 (c/N^2) as separate parameter, decoupled from b
  * full posterior covariance for the Bayesian rational-ID

Model:
    lambda_2(N, seed) ~ Normal(a + b/N + c/N^2, sigma_seed)
    a, b, c          ~ Normal(prior_mean, prior_std)
    sigma_seed       ~ HalfNormal(0.05)

Inference: MCMC via emcee or analytic Laplace approx around MAP.
This module implements the Laplace approximation (cheap, gives Gaussian
posterior approximation, sufficient for Symanzik-fit purposes).

For full MCMC: install emcee, swap _laplace_posterior for _mcmc_posterior.
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class HierarchicalSymanzikFit:
    """Posterior summary of a hierarchical Symanzik fit."""
    a_mean: float           # posterior mean of asymptote
    a_std: float            # posterior std of asymptote
    b_mean: float
    b_std: float
    c_mean: float
    c_std: float
    sigma_seed_mean: float
    log_evidence: float     # marginal log-likelihood for model comparison
    n_data: int             # total per-seed observations
    n_regimes: int          # distinct N values

    def asymptote_ci95(self) -> tuple[float, float]:
        return (self.a_mean - 1.96 * self.a_std,
                self.a_mean + 1.96 * self.a_std)

    def predict(self, N: float) -> tuple[float, float]:
        """Predictive mean + std at given N."""
        mean = self.a_mean + self.b_mean / N + self.c_mean / (N * N)
        # Propagated variance (approximate, ignoring covariance for simplicity)
        var = (self.a_std ** 2
               + (self.b_std / N) ** 2
               + (self.c_std / (N * N)) ** 2
               + self.sigma_seed_mean ** 2)
        return mean, math.sqrt(var)


def _build_design_matrix(N_per_obs: np.ndarray, model_order: int = 2) -> np.ndarray:
    """Design matrix for Symanzik-{model_order} fit. Each row = [1, 1/N, 1/N^2, ...]."""
    cols = [np.ones_like(N_per_obs, dtype=np.float64)]
    inv_N = 1.0 / N_per_obs
    cols.append(inv_N)
    if model_order >= 2:
        cols.append(inv_N ** 2)
    if model_order >= 3:
        cols.append(inv_N ** 3)
    return np.column_stack(cols)


def laplace_fit(
    N_per_obs: np.ndarray,
    lambda_obs: np.ndarray,
    prior_a: float = 0.40,
    prior_a_std: float = 0.05,
    prior_b: float = 0.0,
    prior_b_std: float = 5.0,
    prior_c: float = 0.0,
    prior_c_std: float = 50.0,
    model_order: int = 2,
) -> HierarchicalSymanzikFit:
    """Compute Laplace approximation to the Bayesian Symanzik fit posterior.

    Inputs:
      N_per_obs : array of length n_obs, the N value for each observation
      lambda_obs: array of length n_obs, the observed lambda_2 per (N, seed)
      prior_*   : Normal-prior hyperparameters for the regression coefficients
      model_order: 1 (a + b/N), 2 (a + b/N + c/N^2), or 3 (cubic)

    Output:
      HierarchicalSymanzikFit with posterior means + stds, predictive eval.
    """
    n_obs = len(lambda_obs)
    if n_obs < model_order + 1:
        raise ValueError(f"Need at least {model_order+1} observations for "
                          f"Symanzik-{model_order} fit, have {n_obs}")

    X = _build_design_matrix(N_per_obs, model_order=model_order)
    y = np.asarray(lambda_obs, dtype=np.float64)

    # MAP estimate (closed-form Gaussian-prior MAP = ridge regression).
    # Pad prior arrays to match design-matrix columns for model_order=3 case.
    prior_means_full = [prior_a, prior_b, prior_c, 0.0]  # d-coeff defaults to 0
    prior_stds_full = [prior_a_std, prior_b_std, prior_c_std, 500.0]
    prior_means = np.array(prior_means_full[:X.shape[1]])
    prior_stds = np.array(prior_stds_full[:X.shape[1]])

    # Estimate sigma_seed from residuals iteratively (one EM step).
    # First pass: assume sigma_seed = 0.02.
    sigma_seed = 0.02
    for _ in range(3):
        Lambda = np.diag(1.0 / prior_stds ** 2)  # prior precision
        XtX = X.T @ X / sigma_seed ** 2
        Xty = X.T @ y / sigma_seed ** 2 + Lambda @ prior_means
        post_cov = np.linalg.inv(XtX + Lambda)
        post_mean = post_cov @ Xty
        residuals = y - X @ post_mean
        sigma_seed = max(np.std(residuals, ddof=1), 0.001)

    # Final posterior
    Lambda = np.diag(1.0 / prior_stds ** 2)
    XtX = X.T @ X / sigma_seed ** 2
    Xty = X.T @ y / sigma_seed ** 2 + Lambda @ prior_means
    post_cov = np.linalg.inv(XtX + Lambda)
    post_mean = post_cov @ Xty

    # Marginal log-likelihood (Laplace approx)
    # log_evidence = -0.5 * (n_obs * log(2 pi sigma^2)
    #               + sum residuals^2 / sigma^2
    #               + log|prior_cov^-1 + XtX|
    #               - log|prior_cov^-1|
    #               + (post - prior)^T prior_cov^-1 (post - prior))
    residuals = y - X @ post_mean
    sse = float((residuals ** 2).sum())
    log_evidence = (
        -0.5 * n_obs * math.log(2 * math.pi * sigma_seed ** 2)
        - 0.5 * sse / sigma_seed ** 2
        - 0.5 * float(np.log(np.linalg.det(XtX + Lambda)))
        + 0.5 * float(np.log(np.linalg.det(Lambda)))
        - 0.5 * float((post_mean - prior_means) @ Lambda @ (post_mean - prior_means))
    )

    post_std = np.sqrt(np.diag(post_cov))

    out = HierarchicalSymanzikFit(
        a_mean=float(post_mean[0]),
        a_std=float(post_std[0]),
        b_mean=float(post_mean[1]),
        b_std=float(post_std[1]),
        c_mean=float(post_mean[2]) if X.shape[1] > 2 else 0.0,
        c_std=float(post_std[2]) if X.shape[1] > 2 else 0.0,
        sigma_seed_mean=sigma_seed,
        log_evidence=log_evidence,
        n_data=n_obs,
        n_regimes=len(set(N_per_obs.tolist())),
    )
    return out


def model_comparison(
    N_per_obs: np.ndarray,
    lambda_obs: np.ndarray,
    **prior_kwargs,
) -> dict:
    """Compare Symanzik-1 vs Symanzik-2 vs Symanzik-3 via log-evidence."""
    results = {}
    for k in (1, 2, 3):
        if len(lambda_obs) >= k + 1:
            fit = laplace_fit(N_per_obs, lambda_obs, model_order=k, **prior_kwargs)
            results[f"symanzik_{k}"] = {
                "log_evidence": fit.log_evidence,
                "asymptote_mean": fit.a_mean,
                "asymptote_std": fit.a_std,
                "asymptote_ci95": fit.asymptote_ci95(),
                "sigma_seed": fit.sigma_seed_mean,
            }
    # Compute Bayes factors
    if "symanzik_1" in results and "symanzik_2" in results:
        results["bayes_factor_2_vs_1"] = math.exp(
            results["symanzik_2"]["log_evidence"] - results["symanzik_1"]["log_evidence"]
        )
    if "symanzik_2" in results and "symanzik_3" in results:
        results["bayes_factor_3_vs_2"] = math.exp(
            results["symanzik_3"]["log_evidence"] - results["symanzik_2"]["log_evidence"]
        )
    # Select best model by log-evidence
    best = max((k for k in results if k.startswith("symanzik_")),
               key=lambda k: results[k]["log_evidence"])
    results["selected"] = best
    return results


if __name__ == "__main__":
    # Demo: fit synthetic data
    rng = np.random.default_rng(42)
    true_a = 0.375
    true_b = 4.0
    true_c = 0.0
    N_vals = np.array([100, 128, 200, 256, 300, 512, 1024])
    seeds_per_N = 8
    n_obs = []
    lam_obs = []
    for N in N_vals:
        for _ in range(seeds_per_N):
            n_obs.append(N)
            lam_obs.append(true_a + true_b / N + true_c / N**2 + rng.normal(0, 0.015))
    n_obs = np.array(n_obs)
    lam_obs = np.array(lam_obs)
    print("Demo data: true a=0.375, b=4.0, c=0.0, sigma=0.015")
    print(f"  {len(n_obs)} obs across {len(N_vals)} regimes")
    print()
    cmp = model_comparison(n_obs, lam_obs)
    for k in ("symanzik_1", "symanzik_2", "symanzik_3"):
        if k in cmp:
            v = cmp[k]
            print(f"  {k}: a = {v['asymptote_mean']:.5f} +/- {v['asymptote_std']:.5f}, "
                  f"log_evidence = {v['log_evidence']:+.2f}")
    print(f"  Bayes factor 2v1 = {cmp.get('bayes_factor_2_vs_1', float('nan')):.3f}")
    print(f"  Bayes factor 3v2 = {cmp.get('bayes_factor_3_vs_2', float('nan')):.3f}")
    print(f"  Selected: {cmp['selected']}")
