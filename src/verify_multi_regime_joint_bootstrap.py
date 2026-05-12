"""(Opt-8) Multi-regime joint bootstrap combining two independent
N-axes:
  - Dense-cell ladder N ∈ {410, 1539, 2254, ..., 28014} (P0-P8, 9 pts)
  - Lattice-N ladder  N ∈ {28, 36, 42, 50, 64, 60, 72, 84, 100, 72, 84, 200, 128, 128} (P1-P5N200, 14 pts)

These two N-axes are independent (dense_cell sub-lattice vs regime
lattice). A simultaneous bootstrap on BOTH axes for each metric
would test universality of the asymptotic behavior across different
discretization scales.

For metrics that exist on BOTH axes:
  - Lambda_t = mean(T_00 - G_00) on the regime ladder (from existing audit)
  - absorption / spectral on the dense-cell ladder

We can't directly combine these (different metrics), but we can
test joint consistency:
  - Independent Symanzik-2 fits on both axes
  - Bootstrap CI on each
  - Test if the Lambda_t asymptote on regime axis is consistent
    with the operator-convergence asymptotes on dense-cell axis

Output: outputs/multi_regime_joint_bootstrap_audit.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent

D1_DIRS_DENSE = [
    REPO / "results_d1_fix17",
    REPO / "results_d1_fix16" / "p6",
    REPO / "results_d1_fix16" / "p7",
    REPO / "results_d1_fix16" / "p8",
]


def symanzik_2_fit(n_arr, y_arr):
    n_arr = np.asarray(n_arr, float); y_arr = np.asarray(y_arr, float)
    if len(n_arr) < 2: return None
    A = np.column_stack([np.ones_like(n_arr), n_arr**-2])
    coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
    pred = A @ coef
    rss = float(np.sum((y_arr - pred)**2))
    return {"gap_inf": float(coef[0]), "c_2": float(coef[1]),
            "rss": rss, "n_points": int(len(n_arr))}


def bootstrap_gap_inf(n_arr, y_arr, n_boot=2000, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    gaps = []
    n = len(n_arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = symanzik_2_fit(n_arr[idx], y_arr[idx])
        if r is not None and np.isfinite(r["gap_inf"]):
            gaps.append(r["gap_inf"])
    gaps = np.array([g for g in gaps if -2 < g < 2])
    if len(gaps) < 50: return None
    return {"median": float(np.median(gaps)),
            "CI95": [float(np.percentile(gaps, 2.5)),
                     float(np.percentile(gaps, 97.5))],
            "n_resamples": int(len(gaps))}


def load_dense_cell_payloads():
    payloads = []
    for d in D1_DIRS_DENSE:
        if not d.is_dir(): continue
        for f in sorted(d.glob("d1_p*.json")):
            if f.name.endswith(".metadata.json") or "report" in f.name: continue
            with open(f) as fh:
                d_obj = json.load(fh)
            n = d_obj.get("dense_cell_node_count")
            if n is None: continue
            payloads.append({"N": float(n),
                             "absorption": d_obj.get("d1_gamma_ir_residual_absorption_closure_score"),
                             "locality": d_obj.get("d1_gamma_ir_residual_locality_score"),
                             "density": d_obj.get("d1_gamma_ir_residual_density_score"),
                             "spectral": d_obj.get("d1_gamma_full_macroclass_joint_closure_score"),
                             "variational": d_obj.get("d1_gamma_ir_variational_closure_score")})
    seen = {}
    for p in payloads:
        key = int(round(p["N"]))
        if key not in seen: seen[key] = p
    return sorted(seen.values(), key=lambda x: x["N"])


def load_regime_lambda_t():
    """Return [(N, Lambda_t_per_regime)] from existing within_P5_bootstrap_audit.json + per_channel."""
    out = []
    # Within-P5 sequence
    p1 = REPO / "emergent-gr-closure-repro" / "outputs" / "within_P5_bootstrap_audit.json"
    if p1.exists():
        with open(p1) as f:
            d = json.load(f)
        for n_str, seeds in d.get("per_regime_seeds", {}).items():
            try:
                n_val = int(n_str.split("=")[1])
                if seeds:
                    out.append((float(n_val), float(np.mean(seeds))))
            except (ValueError, IndexError):
                continue
    # Per-channel from skeptical audit
    p2 = REPO / "emergent-gr-closure-repro" / "outputs" / "skeptical_audit_pythagorean.json"
    if p2.exists():
        with open(p2) as f:
            d = json.load(f)
        for r in d.get("per_regime", []):
            out.append((float(r["N"]), float(r["lambda_t_star"])))
    # Dedupe by N
    seen = {}
    for n, v in out:
        key = int(round(n))
        if key not in seen:
            seen[key] = v
    return sorted([(float(k), v) for k, v in seen.items()])


def main() -> int:
    print("="*100)
    print("(Opt-8) Multi-regime joint bootstrap: dense-cell + lattice-N axes")
    print("="*100)

    # ─── Dense-cell axis (P0-P8) ─────────────────────────
    dc = load_dense_cell_payloads()
    print(f"\nDense-cell ladder ({len(dc)} points):")
    for p in dc: print(f"  N={int(p['N']):>6}: absorption={p['absorption']:.4f}, spectral={p['spectral']:.4f}")

    n_dc = np.array([p["N"] for p in dc])
    metrics_dc = ["absorption", "locality", "density", "spectral", "variational"]
    print(f"\nDense-cell Symanzik-2 + bootstrap (n_boot=2000):")
    print(f"  {'metric':<14} {'fit gap':>9} {'bootstrap median':>17} {'95% CI':>22}")
    print(f"  " + "-"*72)
    dc_results = {}
    for m in metrics_dc:
        vals = np.array([p[m] for p in dc], float)
        fit = symanzik_2_fit(n_dc, vals)
        bs = bootstrap_gap_inf(n_dc, vals)
        dc_results[m] = {"fit": fit, "bootstrap": bs}
        print(f"  {m:<14} {fit['gap_inf']:>9.4f} {bs['median']:>17.4f} "
              f"[{bs['CI95'][0]:>+7.4f}, {bs['CI95'][1]:>+7.4f}]")

    # ─── Lattice-N axis (P-regime ladder) ─────────────────
    rg = load_regime_lambda_t()
    print(f"\nLattice-N ladder ({len(rg)} points):")
    for n, v in rg: print(f"  N={int(n):>4}: Lambda_t={v:.4f}")

    n_rg = np.array([n for n, _ in rg])
    Lt_rg = np.array([v for _, v in rg])
    fit_lt = symanzik_2_fit(n_rg, Lt_rg)
    bs_lt = bootstrap_gap_inf(n_rg, Lt_rg, n_boot=2000)
    print(f"\nLattice-N Symanzik-2 fit on Lambda_t:")
    print(f"  fit gap_inf = {fit_lt['gap_inf']:.4f}")
    print(f"  bootstrap median = {bs_lt['median']:.4f}, 95% CI [{bs_lt['CI95'][0]:+.4f}, {bs_lt['CI95'][1]:+.4f}]")
    print(f"  α_ξ² = 0.810 in CI? {bs_lt['CI95'][0] <= 0.810 <= bs_lt['CI95'][1]}")

    # ─── Cross-axis consistency ─────────────────────────
    print(f"\n{'='*100}")
    print(f"Cross-axis consistency check")
    print(f"{'='*100}")
    print(f"\nLambda_t (lattice axis) bootstrap CI: [{bs_lt['CI95'][0]:.4f}, {bs_lt['CI95'][1]:.4f}]")
    print(f"Spectral (dense-cell axis) bootstrap CI: [{dc_results['spectral']['bootstrap']['CI95'][0]:.4f}, {dc_results['spectral']['bootstrap']['CI95'][1]:.4f}]")
    print(f"  → these are DIFFERENT physical quantities; check is for asymptote stability")
    print()
    print(f"  Lattice-N median Λ_t = {bs_lt['median']:.4f}, range = {bs_lt['CI95'][1] - bs_lt['CI95'][0]:.4f}")
    print(f"  Dense-cell median spectral = {dc_results['spectral']['bootstrap']['median']:.4f}, range = {dc_results['spectral']['bootstrap']['CI95'][1] - dc_results['spectral']['bootstrap']['CI95'][0]:.4f}")

    out = {
        "method": "multi_regime_joint_bootstrap",
        "dense_cell_axis": {"n_points": len(dc), "metrics": dc_results},
        "lattice_n_axis": {"n_points": len(rg), "lambda_t_fit": fit_lt, "lambda_t_bootstrap": bs_lt},
        "cross_axis_check": {
            "lambda_t_consistent_with_alpha_xi_sq": bool(bs_lt['CI95'][0] <= 0.810 <= bs_lt['CI95'][1]),
            "all_dc_above_threshold_count": sum(1 for v in dc_results.values() if v['bootstrap']['CI95'][0] >= 0.4),
            "n_dc_metrics": len(dc_results),
        },
    }
    out_path = REPO / "emergent-gr-closure-repro" / "outputs" / "multi_regime_joint_bootstrap_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
