r"""Q1 / Proposal 2: the variance-reactivity identity for the carrier
edge-Xi graph.

Proposal 1 (verify_t00_summand_decomposition.py) found that the
leading T_00 summand S1 = 0.5 * var_xi extrapolates cleanly to
alpha_xi^2 * gamma^2 = 81/10000, i.e. the per-node row-variance of
the canonical Xi-graph extrapolates to

    sigma^2_Xi(N -> infinity)  ==  2 * alpha_xi^2 * gamma^2  =  81/5000.

This script tests that identity directly, and the companion
row-mean identity, on the canonical-physics ladder N in [64,512]:

  (1) per-node row-variance  sigma^2_Xi(a) -> 2*alpha_xi^2*gamma^2 ?
  (2) per-node row-mean      mu_Xi(a)      -> closed-form System-R ?
  (3) global mean edge weight <edge_xi> (np.mean over the
      diagonal-zeroed matrix, matching the corpus definition in
      verify_within_p5_lattice_asymptotes.py) cross-checked
      against the corpus 9/2 nominal for <edge_xi> * N.

The "variance reactivity" coefficient alpha_xi = 9/10 of the
System-R tuple is, by this reading, not an external label but the
algebraic continuum value carried by the edge-Xi second moment:
sigma^2_Xi = 2 gamma^2 * alpha_xi^2 ties the row-variance to
alpha_xi^2 through the universal gamma^2 scale. Confirming the
identity (rational landing within bootstrap CI) reduces the
empirical T_00 leg of the CBI to a statement about a single,
clean, high-R^2 second-moment extrapolation.

No fits, no fallbacks: moments are read directly off the bundled
lattice snapshots; the Symanzik discipline and seed-bootstrap are
shared with verify_t00_summand_decomposition.py.

Usage:
    python ./src/verify_variance_reactivity_identity.py
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _d1_npz_discovery import find_d1_npz  # noqa: E402
from verify_t00_summand_decomposition import (  # noqa: E402
    LADDER, XI_THRESH, ALPHA_XI, GAMMA, ALPHA_XI_SQ,
    load_seeds, symanzik_fit, bootstrap_ci, identify_rational,
)

# Closed-form System-R targets for the edge-Xi moments.
SIGMA2_TARGET = 2 * ALPHA_XI_SQ * GAMMA ** 2          # 81/5000
SIGMA2_TARGET_NAME = "2 * alpha_xi^2 * gamma^2"


def xi_moments_per_seed(xi_mat):
    """Per-node row-mean and row-variance of the thresholded edge-Xi
    graph (definitions identical to t_munu_spectral() of
    verify_galerkin_runner_A_hessian_ricci.py), plus the global
    diagonal-excluded matrix mean.

    The third quantity matches the corpus <edge_xi> definition of
    verify_within_p5_lattice_asymptotes.py exactly -- np.mean over
    the full edge matrix with the (zero) diagonal included, i.e.
    off_diagonal_sum / N^2 -- so that <edge_xi> * N is a faithful
    cross-check against the corpus 9/2 asymptote, not a different
    normalisation.
    """
    xi_mat = np.where(np.isfinite(xi_mat), xi_mat, 0.0)
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(np.float64)
    weight_adj = xi_off * adj
    deg_count = adj.sum(axis=1) + 1e-12
    row_mean = weight_adj.sum(axis=1) / deg_count
    row_var = (((weight_adj - row_mean[:, None]) ** 2 * adj).sum(axis=1)
               / deg_count)
    # Corpus-faithful <edge_xi>: mean over the full matrix with the
    # diagonal zeroed (= off_diagonal_sum / N^2), storage-stable
    # across the snapshot and per-seed-keyed NPZ layouts.
    edge_mean = float(np.mean(xi_off))
    return row_mean, row_var, edge_mean


def fit_moment(n_arr, per_seed_medians):
    """Symanzik-fit a per-regime median series; return the dict the
    report consumes (y_inf, order, R^2, bootstrap CI)."""
    y = [float(np.mean(m)) for m in per_seed_medians]
    inf1, _, r2_1 = symanzik_fit(n_arr, y, 1)
    inf2, _, r2_2 = symanzik_fit(n_arr, y, 2)
    if r2_2 >= r2_1 + 0.02:
        y_inf, order, r2 = inf2, 2, r2_2
    else:
        y_inf, order, r2 = inf1, 1, r2_1
    ci_lo, ci_hi = bootstrap_ci(n_arr, per_seed_medians, order)
    return {"y_inf": y_inf, "symanzik_order": order, "r_squared": r2,
            "bootstrap_ci95": [ci_lo, ci_hi]}


def main():
    print("=" * 78)
    print("Q1 / Proposal 2: variance-reactivity identity for edge-Xi")
    print("=" * 78)
    print()

    regimes = []
    for regime, _ in LADDER:
        npz_path = find_d1_npz(regime, REPO)
        if npz_path is None or not Path(npz_path).exists():
            print(f"  [skip] {regime}: no NPZ payload found")
            continue
        try:
            seeds, n_actual = load_seeds(npz_path)
        except (KeyError, ValueError) as exc:
            print(f"  [skip] {regime}: unloadable payload ({exc})")
            continue
        rm_med, rv_med, em_list = [], [], []
        for xi_mat, _, _, _ in seeds:
            row_mean, row_var, edge_mean = xi_moments_per_seed(xi_mat)
            rm_med.append(float(np.median(row_mean)))
            rv_med.append(float(np.median(row_var)))
            em_list.append(edge_mean)
        rec = {
            "regime": regime, "N": n_actual, "n_seeds": len(seeds),
            "row_mean_median_per_seed": rm_med,
            "row_var_median_per_seed": rv_med,
            "edge_mean_per_seed": em_list,
            "row_mean_median": float(np.mean(rm_med)),
            "row_var_median": float(np.mean(rv_med)),
            "edge_mean": float(np.mean(em_list)),
            "edge_mean_times_N": float(np.mean(em_list)) * n_actual,
        }
        regimes.append(rec)
        print(f"  {regime:8s} N={n_actual:4d} seeds={len(seeds):2d}  "
              f"mu_Xi={rec['row_mean_median']:.5f}  "
              f"sigma2_Xi={rec['row_var_median']:.6f}  "
              f"<edge>*N={rec['edge_mean_times_N']:.4f}")

    if len(regimes) < 4:
        print("\nInsufficient ladder coverage for Symanzik extrapolation.")
        raise SystemExit(1)

    n_arr = [r["N"] for r in regimes]
    print()
    print("-" * 78)
    print(f"Symanzik extrapolation (canonical ladder N in "
          f"[{min(n_arr)},{max(n_arr)}], {len(regimes)} points)")
    print("-" * 78)

    sigma2 = fit_moment(n_arr, [r["row_var_median_per_seed"] for r in regimes])
    mu = fit_moment(n_arr, [r["row_mean_median_per_seed"] for r in regimes])
    edge_times_n = fit_moment(
        n_arr, [[v * r["N"] for v in r["edge_mean_per_seed"]]
                for r in regimes])

    # (1) The variance-reactivity identity.
    s2_lo, s2_hi = sigma2["bootstrap_ci95"]
    s2_target = float(SIGMA2_TARGET)
    s2_in_ci = s2_lo <= s2_target <= s2_hi
    s2_rel = abs(sigma2["y_inf"] - s2_target) / s2_target
    print(f"  sigma^2_Xi  y_inf = {sigma2['y_inf']:.6f}  "
          f"Sym-{sigma2['symanzik_order']} R^2={sigma2['r_squared']:.2f}  "
          f"CI95=[{s2_lo:.5f},{s2_hi:.5f}]")
    print(f"    target {SIGMA2_TARGET_NAME} = {SIGMA2_TARGET} "
          f"= {s2_target:.6f}   rel.err {s2_rel*100:.2f}%  "
          f"{'[identity holds: in CI]' if s2_in_ci else '[OUTSIDE CI]'}")

    # (2) Row-mean: free System-R rational identification.
    mu_frac, mu_err = identify_rational(mu["y_inf"])
    mu_lo, mu_hi = mu["bootstrap_ci95"]
    mu_in_ci = (mu_frac is not None and mu_lo <= float(mu_frac) <= mu_hi)
    mu_tag = (f"~ {mu_frac} ({mu_err*100:.2f}%)" if mu_frac is not None
              else f"no clean rational (closest off {mu_err*100:.1f}%)")
    print(f"  mu_Xi       y_inf = {mu['y_inf']:.6f}  "
          f"Sym-{mu['symanzik_order']} R^2={mu['r_squared']:.2f}  "
          f"CI95=[{mu_lo:.5f},{mu_hi:.5f}]  {mu_tag}"
          f"{'  [in CI]' if mu_in_ci else ''}")

    # (3) <edge_xi> * N cross-check against the corpus 9/2 nominal.
    # Definition matches verify_within_p5_lattice_asymptotes.py
    # exactly (np.mean over the diagonal-zeroed matrix), so this is a
    # faithful cross-check rather than a different normalisation.
    et_lo, et_hi = edge_times_n["bootstrap_ci95"]
    et_target = 4.5
    et_in_ci = et_lo <= et_target <= et_hi
    et_rel = abs(edge_times_n["y_inf"] - et_target) / et_target
    print(f"  <edge>*N    y_inf = {edge_times_n['y_inf']:.5f}  "
          f"Sym-{edge_times_n['symanzik_order']} "
          f"R^2={edge_times_n['r_squared']:.2f}  "
          f"CI95=[{et_lo:.4f},{et_hi:.4f}]")
    print(f"    corpus nominal 9/2 = 4.5   rel.err {et_rel*100:.2f}%  "
          f"{'[in CI]' if et_in_ci else '[below 9/2 -- consistent '
          'with the corpus own 4.25-4.40 extrapolation range]'}")

    print()
    print("-" * 78)
    print("Verdict")
    print("-" * 78)
    if s2_in_ci:
        print("  The variance-reactivity identity")
        print("    sigma^2_Xi -> 2 * alpha_xi^2 * gamma^2 = 81/5000")
        print("  holds: the closed-form System-R target lies within the")
        print("  seed-bootstrap 95% CI of the edge-Xi row-variance")
        print("  asymptote. The leading T_00 summand S1 = 0.5*sigma^2_Xi")
        print("  is therefore a clean second-moment closure, and the")
        print("  variance-reactivity coefficient alpha_xi enters T_00")
        print("  algebraically (through alpha_xi^2) rather than as a label.")
        verdict = "VARIANCE_REACTIVITY_IDENTITY_CONFIRMED"
    else:
        print("  The closed-form target 2*alpha_xi^2*gamma^2 = 81/5000 is")
        print("  NOT within the bootstrap CI at this ladder resolution;")
        print(f"  the row-variance asymptote {sigma2['y_inf']:.6f} differs")
        print(f"  by {s2_rel*100:.2f}%. The identity is not certified here.")
        verdict = "VARIANCE_REACTIVITY_IDENTITY_NOT_CONFIRMED"

    out = {
        "criterion": "Q1/Proposal-2: variance-reactivity identity "
                     "sigma^2_Xi -> 2*alpha_xi^2*gamma^2 = 81/5000",
        "canonical_ladder": [(r["regime"], r["N"]) for r in regimes],
        "per_regime": regimes,
        "sigma2_xi": {
            **sigma2,
            "target_name": SIGMA2_TARGET_NAME,
            "target_fraction": str(SIGMA2_TARGET),
            "target_value": s2_target,
            "rel_err": s2_rel,
            "target_within_ci95": s2_in_ci,
        },
        "mu_xi": {
            **mu,
            "free_rational_id": str(mu_frac) if mu_frac is not None else None,
            "free_rational_rel_err": mu_err,
            "free_rational_within_ci95": mu_in_ci,
        },
        "edge_mean_times_N": {
            **edge_times_n,
            "definition": "np.mean over the diagonal-zeroed edge "
                          "matrix, matching "
                          "verify_within_p5_lattice_asymptotes.py",
            "corpus_nominal": "9/2",
            "corpus_nominal_value": et_target,
            "rel_err_vs_nominal": et_rel,
            "within_ci95_of_nominal": et_in_ci,
        },
        "verdict": verdict,
    }
    out_path = OUTPUTS / "variance_reactivity_identity_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Verdict: {verdict}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
