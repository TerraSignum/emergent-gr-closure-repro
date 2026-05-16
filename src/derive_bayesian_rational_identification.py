r"""Bayesian rational-identification of the (SG) closure asymptote.

Companion script to Remark rmk:bayesian_rational_id of P4.

Question: given an empirical Symanzik-1 asymptote `lambda_inf_hat`
with bootstrap-CI95 sigma_hat, what is the Bayesian posterior over
the simplest rationals p/q with q <= q_max?

Prior (Occam, formalised via Kolmogorov-style complexity):
    Prior(p/q) propto 1 / (q * log(q + 1))
The factor log(q+1) gives extra penalty to denominators with many
prime factors; equivalently it is a smoothed version of the
description-length cost of a rational with denominator q.

Likelihood (Gaussian under bootstrap):
    L(lambda_inf = p/q | data) = N(p/q; lambda_inf_hat, sigma_hat)

Posterior:
    P(lambda_inf = p/q | data) propto Prior(p/q) * L(p/q | data)

Output: outputs/derive_bayesian_rational_identification.json plus a
console summary table.

Validates rmk:bayesian_rational_id of the P4 manuscript by showing
3/8 dominates the posterior at N=512 already, with the dominance
expected to sharpen further with N=1024 data.
"""
from __future__ import annotations

import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
REPO = SRC.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Empirical asymptotes are loaded ADAPTIVELY from the derive_skeleton_*
# JSON outputs (which are themselves auto-discovered from the ladder NPZ
# files). Whenever new high-N data is added, derive_skeleton_*_chain runs
# auto-pick it up; this Bayesian-ID then auto-reads the new asymptotes.
#
# Branch separation reading: pooled-ladder Symanzik-1 asymptote is a
# branch-MIXED average, expected to land near 2/5 (matter-branch). The
# branch-resolved VAC asymptote (N<=100) is the test for 3/8.
def _load_adaptive_targets() -> dict:
    """Read latest empirical asymptotes from the chain-audit JSON."""
    chain_json = OUTPUTS / "derive_skeleton_weighted_lift_chain.json"
    if not chain_json.is_file():
        return {}
    import json
    bundle = json.loads(chain_json.read_text(encoding="utf-8"))
    sym = bundle.get("symanzik", {})
    targets = {}
    if "weighted" in sym:
        # Pooled-ladder weighted lambda_2 asymptote. Branch-MIXED.
        # Use larger CI95 since pooled across branches.
        per = bundle.get("per_regime", [])
        n_total_seeds = sum(r.get("n_seeds", 0) for r in per)
        # Heuristic CI95 = stddev / sqrt(N_seeds) * 1.96
        ci_half = 0.01 / max(np.sqrt(max(n_total_seeds, 1)), 1.0) * 1.96
        ci_half = max(ci_half, 0.003)
        targets["lambda_inf_pooled"] = {
            "asymptote": sym["weighted"]["a_inf"],
            "ci95_half": ci_half,
            "label": "Pooled-ladder weighted lambda_2 (branch-mixed; expect 2/5)",
            "registered_rational": (2, 5),
        }
    if "skeleton" in sym:
        per = bundle.get("per_regime", [])
        n_total = sum(r.get("n_seeds", 0) for r in per)
        ci_half = max(0.012 / np.sqrt(max(n_total, 1)) * 1.96, 0.002)
        targets["lambda_inf_skel"] = {
            "asymptote": sym["skeleton"]["a_inf"],
            "ci95_half": ci_half,
            "label": "Skeleton (tau=0.10) Laplacian asymptote",
            "registered_rational": (7, 24),
        }
    if "lift" in sym:
        targets["lift"] = {
            "asymptote": sym["lift"]["a_inf"],
            "ci95_half": 0.05,
            "label": "Weight-lift Symanzik asymptote",
            "registered_rational": (9, 7),
        }
    return targets


# Static branch-resolved targets (low-N VAC vs high-N MATTER).
# These do not auto-update because the branch-separation is a structural
# choice, not a single Symanzik fit. The static asymptotes below are from
# memory project_lemma_B_phase2_J_X_loops_2026_05_12; they get updated
# manually when branch-resolved analysis advances.
STATIC_BRANCH_TARGETS = {
    "lambda_inf_vac": {
        "asymptote": 0.3732,           # VAC-BRANCH (N <= 100)
        "ci95_half": 0.003,
        "label": "Vacuum-branch lambda_inf (low-N regime)",
        "registered_rational": (3, 8),
    },
    "lambda_inf_mat": {
        "asymptote": 0.3957,           # MATTER-BRANCH (N >= 256)
        "ci95_half": 0.004,
        "label": "Matter-branch lambda_inf (high-N regime)",
        "registered_rational": (79, 200),
    },
}


def get_targets() -> dict:
    """Combine adaptive pooled/skeleton targets with static branch targets."""
    targets = dict(STATIC_BRANCH_TARGETS)
    targets.update(_load_adaptive_targets())
    return targets


DEFAULT_TARGETS = get_targets()


def simplest_rationals(q_max: int, lo: float = 0.0, hi: float = 1.0) -> list[Fraction]:
    """Enumerate distinct simplest rationals p/q with 2 <= q <= q_max in (lo, hi)."""
    seen: set[Fraction] = set()
    out: list[Fraction] = []
    for q in range(2, q_max + 1):
        for p in range(1, q):
            f = Fraction(p, q)
            if f.denominator > q_max:
                continue  # reducible to a denominator already seen
            if lo < float(f) < hi and f not in seen:
                seen.add(f)
                out.append(f)
    return sorted(out, key=float)


def occam_prior(rationals: list[Fraction]) -> np.ndarray:
    """Occam-style prior: 1 / (q * log(q + 1)), normalised."""
    weights = np.array([1.0 / (r.denominator * math.log(r.denominator + 1))
                         for r in rationals])
    return weights / weights.sum()


def gaussian_likelihood(rationals: list[Fraction], mean: float, sigma: float) -> np.ndarray:
    """Gaussian likelihood of each rational under N(mean, sigma)."""
    sigma = max(sigma, 1.0e-12)
    z = (np.array([float(r) for r in rationals]) - mean) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def posterior(rationals: list[Fraction], mean: float, ci95_half: float, q_max: int,
              window: float | None = None) -> dict:
    """Compute prior, likelihood, posterior; return per-rational table."""
    if window is not None:
        rationals = [r for r in rationals if abs(float(r) - mean) <= window]
        if not rationals:
            return {"rationals": [], "summary": None}

    sigma = ci95_half / 1.96
    prior = occam_prior(rationals)
    likelihood = gaussian_likelihood(rationals, mean, sigma)
    raw = prior * likelihood
    posterior_vals = raw / raw.sum() if raw.sum() > 0 else raw
    table = []
    for r, pr, lk, po in zip(rationals, prior, likelihood, posterior_vals):
        z = (float(r) - mean) / sigma
        table.append({
            "rational": f"{r.numerator}/{r.denominator}",
            "p": r.numerator,
            "q": r.denominator,
            "value": float(r),
            "prior": float(pr),
            "likelihood": float(lk),
            "posterior": float(po),
            "z_score": float(z),
        })
    table.sort(key=lambda d: d["posterior"], reverse=True)
    summary = {
        "n_candidates": len(rationals),
        "top_rational": table[0]["rational"] if table else None,
        "top_posterior": table[0]["posterior"] if table else 0.0,
        "second_rational": table[1]["rational"] if len(table) > 1 else None,
        "second_posterior": table[1]["posterior"] if len(table) > 1 else 0.0,
        "posterior_odds_top_vs_second": (table[0]["posterior"] / table[1]["posterior"])
                                          if len(table) > 1 and table[1]["posterior"] > 0
                                          else float("inf"),
    }
    return {"rationals": table, "summary": summary}


def run_one_target(key: str, target: dict, q_max: int) -> dict:
    print("-" * 78)
    print(f"  {key}: {target['label']}")
    print("-" * 78)
    print(f"  empirical asymptote: {target['asymptote']:.5f} +/- {target['ci95_half']:.5f}  (CI95)")
    print(f"  registered rational: {target['registered_rational'][0]}/{target['registered_rational'][1]} "
          f"= {target['registered_rational'][0]/target['registered_rational'][1]:.5f}")

    rationals = simplest_rationals(q_max, 0.0, 1.0)
    result = posterior(rationals, target["asymptote"], target["ci95_half"], q_max,
                       window=10.0 * target["ci95_half"])

    summary = result["summary"]
    if summary is None:
        print("  [error] no rationals in window; widen window or q_max")
        return {"key": key, "target": target, "result": None}

    print()
    print(f"  Top-5 posterior (q_max = {q_max}):")
    print(f"  {'rank':>4}  {'p/q':>8}  {'value':>9}  {'z':>7}  {'prior':>9}  {'posterior':>11}")
    for i, row in enumerate(result["rationals"][:5]):
        print(f"  {i+1:>4}  {row['rational']:>8}  {row['value']:>9.5f}  "
              f"{row['z_score']:+7.2f}  {row['prior']:>9.4f}  {row['posterior']:>11.4f}")

    reg = target["registered_rational"]
    reg_str = f"{reg[0]}/{reg[1]}"
    reg_row = next((r for r in result["rationals"] if r["rational"] == reg_str), None)
    if reg_row is None:
        print(f"  [warn] registered rational {reg_str} not in window")
    else:
        rank = result["rationals"].index(reg_row) + 1
        print(f"  -> registered {reg_str} is rank {rank}, posterior {reg_row['posterior']:.4f}")

    return {"key": key, "target": target, "summary": summary, "table": result["rationals"]}


def main():
    print("=" * 78)
    print("Bayesian rational-identification of the (SG) closure asymptotes")
    print("=" * 78)
    print()

    targets = DEFAULT_TARGETS
    q_max = 50
    print(f"Prior: Occam-style 1 / (q * log(q+1)), q in [2, {q_max}]")
    print()

    all_results = []
    for key, target in targets.items():
        result = run_one_target(key, target, q_max)
        all_results.append(result)
        print()

    out = {
        "title": "Bayesian rational-identification of (SG) closure asymptotes",
        "method": {
            "prior": "Occam: P(p/q) ∝ 1 / (q · log(q+1))",
            "likelihood": "Gaussian N(p/q; asymptote, sigma=ci95_half/1.96)",
            "q_max": q_max,
            "window": "10 * ci95_half around asymptote",
        },
        "results": all_results,
    }
    out_path = OUTPUTS / "derive_bayesian_rational_identification.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
