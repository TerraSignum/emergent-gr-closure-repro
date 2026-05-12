"""Lemma B Phase-2 Step 3a: edge-weight distribution + threshold-
dependent effective-degree audit.

Step 2 finding: mean weighted degree mu_deg = 4.24 (constant in N)
with CV = 0.168. Phase-1 finding: lambda_2(L_N) -> 0.3789. The
Alon-Boppana bound for regular graphs with d_eff = 4.24 predicts
lambda_2 <= 1 - 2*sqrt(d_eff - 1)/d_eff = 0.151 for the normalised
Laplacian, but the empirical value 0.379 is more than twice that.
The resolution is one of:

  (i)  the graph is irregular and Alon-Boppana does not apply
       directly (Kahale's irregular-expander bound is needed),
  (ii) the unweighted "effective degree" (count of non-trivial
       neighbours, threshold-dependent) is different from the
       weighted mean.

This audit computes:

  (E1) Histogram of off-diagonal Xi values per snapshot, summarised
       by quartile + decile statistics across the ladder.

  (E2) Threshold-dependent effective degree:
         d_eff(tau, N) = mean_i |{j : Xi(i,j) > tau}|
       for a sweep tau in {0.01, 0.05, 0.1, 0.2, 0.5}.

  (E3) "Significant-neighbour" count: number of edges per vertex
       carrying >= 10% of the row's total weight.

  (E4) Comparison between unweighted-d_eff (at multiple thresholds)
       and weighted mu_deg = 4.24. If they agree at some natural
       threshold, that threshold defines the "structural neighbour"
       relation; if they disagree, the carrier is a true weighted
       graph with no natural unweighted skeleton.

Output: outputs/verify_lemma_B_edge_weight_structure.json
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REPO_ROOT = REPO.parent
OUT = REPO / "outputs" / "verify_lemma_B_edge_weight_structure.json"

LADDER = [
    ("P5",     50,  "results_d1_fix17/d1_p5.npz",            "xi_seedK"),
    ("P5N64",  64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz", "edge_xi_snapshots"),
    ("P5N72",  72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz", "edge_xi_snapshots"),
    ("P5N84",  84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz", "edge_xi_snapshots"),
    ("P5N100", 100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz", "edge_xi_snapshots"),
    ("P5N128", 128, "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz", "edge_xi_snapshots"),
    ("P5N200", 200, "results_d1_p5n200_8seeds/P5N200.snapshots.npz", "edge_xi_snapshots"),
    ("P5N256", 256, "results_d1_p5n256_12seeds/P5N256.snapshots.npz", "edge_xi_snapshots"),
    ("P5N300", 300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz", "edge_xi_snapshots"),
    ("P5N512", 512, "results_d1_p5n512_12seeds/P5N512.snapshots.npz", "edge_xi_snapshots"),
]

THRESHOLDS = [0.01, 0.05, 0.1, 0.2, 0.5]


def load_all_xi(npz_path: Path, hint: str) -> list[np.ndarray]:
    if not npz_path.exists():
        return []
    z = np.load(npz_path, allow_pickle=True)
    matrices: list[np.ndarray] = []
    if hint == "edge_xi_snapshots" and "edge_xi_snapshots" in z.files:
        snaps = np.asarray(z["edge_xi_snapshots"])
        last = snaps.shape[1] - 1
        for s in range(snaps.shape[0]):
            xi = np.asarray(snaps[s, last], dtype=float).copy()
            np.fill_diagonal(xi, 1.0)
            matrices.append(xi)
        return matrices
    if hint == "xi_seedK":
        n_seeds = sum(1 for k in z.files if k.startswith("xi_seed"))
        for s in range(n_seeds):
            key = f"xi_seed{s}"
            if key not in z.files:
                continue
            xi = np.asarray(z[key], dtype=float).copy()
            np.fill_diagonal(xi, 1.0)
            matrices.append(xi)
        return matrices
    return matrices


def per_snapshot_diagnostics(xi: np.ndarray) -> dict[str, Any]:
    """Edge-weight distribution + threshold-dependent effective
    degree on a single Xi_N snapshot."""
    n = xi.shape[0]
    # Off-diagonal entries
    iu = np.triu_indices(n, k=1)
    edges = xi[iu]
    edges = edges[np.isfinite(edges)]
    if edges.size == 0:
        return {}
    # Distribution summary
    q = np.quantile(edges, [0.05, 0.25, 0.5, 0.75, 0.95])
    dist = {
        "min": float(edges.min()),
        "p5": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
        "max": float(edges.max()),
        "mean": float(edges.mean()),
        "std": float(edges.std(ddof=1)) if edges.size > 1 else 0.0,
    }
    # Threshold-dependent effective degree (count, mean across i)
    deg_at_thresh: dict[str, float] = {}
    w = xi - np.eye(n)
    w = np.maximum(w, 0.0)
    for tau in THRESHOLDS:
        # row-wise count of entries > tau
        count_per_row = (w > tau).sum(axis=1).astype(float)
        deg_at_thresh[f"d_eff_tau_{tau}"] = float(count_per_row.mean())

    # Significant-neighbour count: edges carrying >= 10% of row total
    rowsum = w.sum(axis=1, keepdims=True)
    rowsum_safe = np.where(rowsum > 0, rowsum, 1.0)
    sig_count = ((w / rowsum_safe) >= 0.10).sum(axis=1).astype(float)
    deg_at_thresh["d_significant_10pct"] = float(sig_count.mean())

    return {"edge_dist": dist, "d_eff_thresholds": deg_at_thresh}


def audit_regime(regime: str, n_lat: int, rel: str, hint: str
                 ) -> dict[str, Any]:
    npz = REPO_ROOT / rel
    xis = load_all_xi(npz, hint)
    if not xis:
        return {"regime": regime, "N": n_lat,
                "n_seeds_loaded": 0,
                "status": "SNAPSHOT_NOT_AVAILABLE"}
    diags = [per_snapshot_diagnostics(xi) for xi in xis]
    diags = [d for d in diags if d]
    if not diags:
        return {"regime": regime, "N": n_lat,
                "n_seeds_loaded": len(xis),
                "status": "ALL_SEEDS_DEGENERATE"}

    out = {"regime": regime, "N": n_lat,
           "n_seeds_loaded": len(xis),
           "n_seeds_valid": len(diags),
           "status": "OK"}
    # Average over seeds
    edge_keys = list(diags[0]["edge_dist"].keys())
    out["edge_dist"] = {
        k: float(np.mean([d["edge_dist"][k] for d in diags]))
        for k in edge_keys
    }
    out["d_eff_thresholds"] = {
        k: float(np.mean([d["d_eff_thresholds"][k] for d in diags]))
        for k in diags[0]["d_eff_thresholds"]
    }
    return out


def fit_d_eff_scaling(per_regime: list[dict[str, Any]], tau_key: str
                      ) -> dict[str, Any]:
    """Fit d_eff at fixed threshold across N. Want: is d_eff(tau, N)
    bounded as N -> inf?"""
    valid = [(r["N"], r["d_eff_thresholds"][tau_key])
             for r in per_regime if r["status"] == "OK"
             and tau_key in r["d_eff_thresholds"]]
    if len(valid) < 3:
        return {"verdict": "INSUFFICIENT_DATA", "n_points": len(valid)}
    n_arr = np.array([v[0] for v in valid], dtype=float)
    y = np.array([v[1] for v in valid], dtype=float)
    # Const vs Symanzik-1
    c_const = float(y.mean())
    sse_c = float(((y - c_const) ** 2).sum())
    A = np.column_stack([np.ones_like(n_arr), 1.0 / n_arr])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    c_inf, a_sym = float(sol[0]), float(sol[1])
    sse_s = float(((y - (c_inf + a_sym / n_arr)) ** 2).sum())
    # Linear in N
    A = np.column_stack([np.ones_like(n_arr), n_arr])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    c_lin, b_lin = float(sol[0]), float(sol[1])
    sse_l = float(((y - (c_lin + b_lin * n_arr)) ** 2).sum())
    n_pts = len(valid)
    # AICc selection
    def aicc(sse, k):
        return n_pts * np.log(sse / n_pts + 1e-30) + 2 * k + 2 * k * (k + 1) / max(n_pts - k - 1, 1)
    aiccs = {"const": aicc(sse_c, 1),
             "symanzik_1": aicc(sse_s, 2),
             "linear": aicc(sse_l, 2)}
    best = min(aiccs, key=lambda k: aiccs[k])
    return {
        "n_points": n_pts,
        "AICc": aiccs,
        "preferred": best,
        "params": ({"c": c_const} if best == "const"
                   else {"c_inf": c_inf, "a": a_sym} if best == "symanzik_1"
                   else {"c_lin": c_lin, "b": b_lin}),
        "y_values": list(y),
        "N_values": list(n_arr),
    }


def alon_boppana_bound_normalised(d_eff: float) -> float:
    """Alon-Boppana upper bound for the second-smallest eigenvalue
    of the normalised Laplacian on a d-regular graph:
        lambda_2(L_norm) <= 1 - 2*sqrt(d-1)/d  + o(1)
    Returns the leading-order RHS. Only meaningful for d > 1.
    """
    if d_eff <= 1:
        return 0.0
    return float(1.0 - 2.0 * np.sqrt(d_eff - 1.0) / d_eff)


def main():
    per_regime = [audit_regime(reg, n_lat, rel, hint)
                  for reg, n_lat, rel, hint in LADDER]

    # N-scaling fits of d_eff at each threshold
    threshold_fits = {}
    for tau in THRESHOLDS:
        threshold_fits[f"d_eff_tau_{tau}"] = fit_d_eff_scaling(
            per_regime, f"d_eff_tau_{tau}")
    threshold_fits["d_significant_10pct"] = fit_d_eff_scaling(
        per_regime, "d_significant_10pct")

    # Alon-Boppana applicability: at which (if any) threshold tau does
    # d_eff(tau, N) approach a finite constant d_eff^inf such that
    # alon_boppana(d_eff^inf) >= 0.3789 (the observed lambda_inf)?
    lambda_inf_obs = 0.3789  # from Phase 1, prop:sg_empirical
    ab_candidates = []
    for tau in THRESHOLDS:
        fit = threshold_fits[f"d_eff_tau_{tau}"]
        if fit.get("preferred") in ("const", "symanzik_1"):
            params = fit["params"]
            d_inf = params.get("c") or params.get("c_inf")
            if d_inf is not None and d_inf > 1:
                ab = alon_boppana_bound_normalised(d_inf)
                ab_candidates.append({
                    "tau": tau,
                    "d_eff_inf": d_inf,
                    "alon_boppana_normalised": ab,
                    "AB_consistent_with_lambda_obs":
                        bool(ab >= lambda_inf_obs),
                })

    out = {
        "headline": ("Lemma B Phase-2 Step 3a: edge-weight "
                     "distribution + threshold-dependent effective "
                     "degree. Tests whether an unweighted skeleton "
                     "of Xi_N at some natural threshold tau gives "
                     "a d_eff(tau) consistent with the observed "
                     "lambda_inf = 0.3789 via Alon-Boppana."),
        "method": (
            "Per snapshot: 5-95 quantile distribution of off-diag "
            "Xi; d_eff(tau) = mean row-count above tau for tau in "
            "{0.01, 0.05, 0.1, 0.2, 0.5}; d_significant_10pct = "
            "count of edges with >= 10% of row total. Per regime: "
            "cross-seed mean. N-scaling fit (const, Symanzik-1, "
            "linear) with AICc selection. Alon-Boppana check: "
            "lambda_AB(d_eff^inf) = 1 - 2*sqrt(d-1)/d compared to "
            "observed lambda_inf = 0.3789."),
        "per_regime": per_regime,
        "threshold_fits": threshold_fits,
        "alon_boppana_check": {
            "lambda_inf_observed": lambda_inf_obs,
            "candidates": ab_candidates,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_summary(per_regime, threshold_fits, ab_candidates,
                  lambda_inf_obs)
    return 0


def print_summary(per_regime, fits, ab_cands, lambda_inf_obs):
    print("=" * 78)
    print("Lemma B Phase-2 Step 3a: Edge-Weight Distribution + d_eff(tau)")
    print("=" * 78)
    print(f"{'Regime':<8} {'N':>4} {'min':>8} {'p5':>8} {'p50':>8} "
          f"{'p95':>8} {'max':>8} {'mean':>8}")
    print("-" * 78)
    for r in per_regime:
        if r["status"] != "OK":
            print(f"{r['regime']:<8} {r['N']:>4}  {r['status']}")
            continue
        d = r["edge_dist"]
        print(f"{r['regime']:<8} {r['N']:>4} "
              f"{d['min']:>8.4f} {d['p5']:>8.4f} {d['p50']:>8.4f} "
              f"{d['p95']:>8.4f} {d['max']:>8.4f} {d['mean']:>8.4f}")
    print()
    print(f"{'Regime':<8} {'N':>4} "
          f"{'tau=.01':>10} {'tau=.05':>10} {'tau=.1':>10} "
          f"{'tau=.2':>10} {'tau=.5':>10} {'sig10%':>10}")
    print("-" * 78)
    for r in per_regime:
        if r["status"] != "OK":
            continue
        d = r["d_eff_thresholds"]
        print(f"{r['regime']:<8} {r['N']:>4} "
              f"{d['d_eff_tau_0.01']:>10.3f} "
              f"{d['d_eff_tau_0.05']:>10.3f} "
              f"{d['d_eff_tau_0.1']:>10.3f} "
              f"{d['d_eff_tau_0.2']:>10.3f} "
              f"{d['d_eff_tau_0.5']:>10.3f} "
              f"{d['d_significant_10pct']:>10.3f}")
    print()
    print("d_eff(tau, N) scaling fits:")
    for k, fit in fits.items():
        if fit.get("verdict") == "INSUFFICIENT_DATA":
            continue
        best = fit["preferred"]
        params = fit["params"]
        param_str = ", ".join(f"{n}={v:.3f}" for n, v in params.items())
        print(f"  {k:<28s} preferred={best:<12s}  {param_str}")
    print()
    print(f"Alon-Boppana check (observed lambda_inf = {lambda_inf_obs}):")
    for c in ab_cands:
        verdict = "CONSISTENT" if c["AB_consistent_with_lambda_obs"] else "RULED OUT"
        print(f"  tau={c['tau']:<6}  d_eff^inf = {c['d_eff_inf']:>8.3f}  "
              f"lambda_AB = {c['alon_boppana_normalised']:>8.4f}   {verdict}")


if __name__ == "__main__":
    raise SystemExit(main())
