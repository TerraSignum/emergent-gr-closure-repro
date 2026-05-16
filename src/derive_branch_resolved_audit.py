r"""Branch-resolved Symanzik audit (Phase 2 follow-up).

The corpus has documented (memory project_lemma_B_phase2_J_X_loops) that
the (SG)-axiom value lambda_inf splits into two branches:

  VAC-BRANCH (low-N, pre-flip):   lambda_inf -> 3/8     (target 0.3750)
  MATTER-BRANCH (high-N, post-flip): lambda_inf -> 79/200 (target 0.3950)

The crossover happens at N* ~ 110-120. Pooling the whole ladder MIXES
the branches and gives an asymptote near (3/8 + 79/200)/2 ~ 0.385, often
landing near 2/5 = 0.400 due to the matter-branch dominating at higher N.

This script:
  1. Auto-discovers the ladder
  2. Separates into VAC (N <= 100) and MATTER (N >= 256) branches
  3. Fits Symanzik-1 + Symanzik-2 + Symanzik-3 to each branch separately
  4. Reports branch-resolved asymptotes + CI95
  5. Theory-hybrid Bayesian posterior per branch

Output: outputs/derive_branch_resolved_audit.json + console.
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
GENERATED = REPO / "paper" / "generated"
GENERATED.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC))
from _d1_ladder_discovery import discover_d1_ladder  # noqa: E402
from adaptive_pipeline import hierarchical_bayes, theory_prior  # noqa: E402

# Branch separation thresholds (from corpus memory):
VAC_BRANCH_N_MAX = 100      # N <= 100 -> vacuum branch
MATTER_BRANCH_N_MIN = 256   # N >= 256 -> matter branch
# Intermediate 128 <= N <= 200: crossover region, excluded from branch fits.


def extract_per_seed_lambdas(ladder, n_filter, max_seeds: int = 12):
    """Extract per-seed weighted-Laplacian lambda_2 from ladder regimes."""
    all_n = []
    all_lam = []
    sources = []
    for regime, n_lat, npz_path in ladder:
        if not n_filter(n_lat):
            continue
        d = np.load(npz_path, allow_pickle=True)
        if "edge_xi_snapshots" not in d.files:
            continue
        xi_arr = d["edge_xi_snapshots"][:, -1]
        n_seeds = min(max_seeds, xi_arr.shape[0])
        sources.append((regime, n_lat, n_seeds))
        for s in range(n_seeds):
            xi = xi_arr[s].astype(np.float64)
            np.fill_diagonal(xi, 0.0)
            deg = np.maximum(xi.sum(axis=1), 1e-12)
            d_inv = 1.0 / np.sqrt(deg)
            laplacian = np.eye(xi.shape[0]) - (d_inv[:, None] * xi * d_inv[None, :])
            laplacian = 0.5 * (laplacian + laplacian.T)
            try:
                eigs = np.linalg.eigvalsh(laplacian)
                all_n.append(n_lat)
                all_lam.append(float(eigs[1]))
            except np.linalg.LinAlgError:
                continue
    return np.asarray(all_n), np.asarray(all_lam), sources


def fit_and_report(branch_name, n_arr, lam_arr, target_rational):
    """Hierarchical Bayes + Symanzik-{1,2} fit for one branch."""
    if len(lam_arr) < 4:
        return {"branch": branch_name, "error": f"insufficient data ({len(lam_arr)} obs)"}
    cmp = hierarchical_bayes.model_comparison(n_arr, lam_arr)
    best_key = cmp["selected"]
    best = cmp[best_key]
    mu = best["asymptote_mean"]
    sigma = best["asymptote_std"]

    target = float(Fraction(*target_rational))
    z_target = (mu - target) / max(sigma, 1e-12)
    rel_err = (mu - target) / target * 100

    print("-" * 78)
    print(f"  {branch_name}")
    print("-" * 78)
    print(f"  {len(lam_arr)} obs across N in [{n_arr.min()}, {n_arr.max()}]")
    for k in ("symanzik_1", "symanzik_2", "symanzik_3"):
        if k in cmp:
            v = cmp[k]
            print(f"  {k}: a = {v['asymptote_mean']:.5f} +/- {v['asymptote_std']:.5f}  "
                  f"log_ev = {v['log_evidence']:+.2f}")
    print(f"  Selected: {best_key}, target {target_rational[0]}/{target_rational[1]}"
          f" = {target:.5f}, residual {rel_err:+.3f}%, z = {z_target:+.2f}")
    # Bayes factor 2v1
    bf_2v1 = cmp.get("bayes_factor_2_vs_1", float('nan'))
    print(f"  Bayes factor 2v1 = {bf_2v1:.3f}")
    return {
        "branch": branch_name,
        "n_obs": len(lam_arr),
        "n_range": [int(n_arr.min()), int(n_arr.max())],
        "selected_model": best_key,
        "asymptote_mean": mu,
        "asymptote_std": sigma,
        "asymptote_ci95": [mu - 1.96 * sigma, mu + 1.96 * sigma],
        "target_rational": f"{target_rational[0]}/{target_rational[1]}",
        "target_value": target,
        "relative_error_pct": rel_err,
        "z_score_vs_target": z_target,
        "bayes_factor_2v1": bf_2v1,
    }


def bayesian_id_branch(branch_name, asymptote_mean, asymptote_std,
                         registered_rational, q_max: int = 50):
    """Theory-hybrid posterior over rationals for one branch."""
    print("-" * 78)
    print(f"  Bayesian Rational-ID for {branch_name}")
    print("-" * 78)
    print(f"  Asymptote {asymptote_mean:.5f} +/- {asymptote_std:.5f}")
    # Enumerate rationals, restrict to window
    seen = set()
    rationals = []
    for q in range(2, q_max + 1):
        for p in range(1, q):
            f = Fraction(p, q)
            if f.denominator > q_max:
                continue
            if f not in seen:
                seen.add(f)
                rationals.append(f)
    window = 10 * asymptote_std
    rationals = [r for r in rationals if abs(float(r) - asymptote_mean) <= window]
    if not rationals:
        return {"branch": branch_name, "error": "no rationals in window"}
    prior_table = theory_prior.prior_table(rationals)
    posterior = {}
    for r in rationals:
        prior = prior_table[r]["prior"]
        z = (float(r) - asymptote_mean) / max(asymptote_std, 1e-12)
        likelihood = math.exp(-0.5 * z * z) / (asymptote_std * math.sqrt(2 * math.pi))
        posterior[r] = prior * likelihood
    total = sum(posterior.values()) or 1.0
    posterior = {r: p / total for r, p in posterior.items()}
    rows = sorted(posterior.items(), key=lambda kv: kv[1], reverse=True)
    reg = registered_rational
    reg_frac = Fraction(*reg)
    reg_in = next((i for i, (r, _) in enumerate(rows) if r == reg_frac), -1)
    print(f"  Top-5 posterior:")
    for r, p in rows[:5]:
        z = (float(r) - asymptote_mean) / asymptote_std
        marker = "<-- registered" if r == reg_frac else ""
        print(f"    {r.numerator}/{r.denominator}  value={float(r):.5f}  "
              f"z={z:+.2f}  posterior={p:.5f}  {marker}")
    print(f"  Registered {reg[0]}/{reg[1]}: posterior {posterior.get(reg_frac, 0.0):.5f}"
          f" (rank {reg_in + 1 if reg_in >= 0 else 'N/A'})")
    return {
        "branch": branch_name,
        "top_rational": f"{rows[0][0].numerator}/{rows[0][0].denominator}",
        "top_posterior": rows[0][1],
        "registered": f"{reg[0]}/{reg[1]}",
        "registered_posterior": posterior.get(reg_frac, 0.0),
        "registered_rank": reg_in + 1 if reg_in >= 0 else -1,
        "top_5": [{"rational": f"{r.numerator}/{r.denominator}",
                     "value": float(r), "posterior": p,
                     "z": (float(r) - asymptote_mean) / asymptote_std}
                    for r, p in rows[:5]],
    }


def main():
    print("=" * 78)
    print("Branch-resolved Symanzik audit (VAC vs MATTER)")
    print("=" * 78)
    ladder = discover_d1_ladder(REPO)

    # VAC branch (N <= 100)
    n_vac, lam_vac, src_vac = extract_per_seed_lambdas(
        ladder, lambda n: n <= VAC_BRANCH_N_MAX
    )
    print(f"\n  VAC branch sources: {src_vac}")
    vac_fit = fit_and_report("VAC-branch (N <= 100)", n_vac, lam_vac, (3, 8))
    if "asymptote_mean" in vac_fit:
        vac_bayes = bayesian_id_branch(
            "VAC-branch", vac_fit["asymptote_mean"], vac_fit["asymptote_std"],
            (3, 8)
        )
    else:
        vac_bayes = {}

    # MATTER branch (N >= 256)
    n_mat, lam_mat, src_mat = extract_per_seed_lambdas(
        ladder, lambda n: n >= MATTER_BRANCH_N_MIN
    )
    print(f"\n  MATTER branch sources: {src_mat}")
    mat_fit = fit_and_report("MATTER-branch (N >= 256)", n_mat, lam_mat, (79, 200))
    if "asymptote_mean" in mat_fit:
        mat_bayes = bayesian_id_branch(
            "MATTER-branch", mat_fit["asymptote_mean"], mat_fit["asymptote_std"],
            (79, 200)
        )
    else:
        mat_bayes = {}

    report = {
        "title": "Branch-resolved Symanzik audit + Bayesian Rational-ID",
        "vac_branch_fit": vac_fit,
        "vac_branch_bayes": vac_bayes,
        "matter_branch_fit": mat_fit,
        "matter_branch_bayes": mat_bayes,
        "vac_branch_sources": src_vac,
        "matter_branch_sources": src_mat,
    }
    out_path = OUTPUTS / "derive_branch_resolved_audit.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Emit tex macros for manuscript
    macros = GENERATED / "branch_resolved_macros.tex"
    if "asymptote_mean" in vac_fit and "asymptote_mean" in mat_fit:
        lines = [
            "% AUTO-GENERATED by src/derive_branch_resolved_audit.py",
            f"\\newcommand{{\\vacBranchAsymptote}}{{{vac_fit['asymptote_mean']:.5f}}}",
            f"\\newcommand{{\\vacBranchStd}}{{{vac_fit['asymptote_std']:.5f}}}",
            f"\\newcommand{{\\vacBranchRelErr}}{{{vac_fit['relative_error_pct']:+.3f}\\%}}",
            f"\\newcommand{{\\vacBranchPosterior}}{{{vac_bayes.get('registered_posterior', 0.0):.3f}}}",
            f"\\newcommand{{\\matterBranchAsymptote}}{{{mat_fit['asymptote_mean']:.5f}}}",
            f"\\newcommand{{\\matterBranchStd}}{{{mat_fit['asymptote_std']:.5f}}}",
            f"\\newcommand{{\\matterBranchRelErr}}{{{mat_fit['relative_error_pct']:+.3f}\\%}}",
            f"\\newcommand{{\\matterBranchPosterior}}"
            f"{{{mat_bayes.get('registered_posterior', 0.0):.3f}}}",
        ]
        macros.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nSaved: {macros}")

    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
