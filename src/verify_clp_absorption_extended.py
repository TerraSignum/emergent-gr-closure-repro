"""(Solution-1') Extended CLP-B B4 absorption fit with bootstrap CI.

The current CLP audit uses 5 dense_cell N points; we have 6 in
d1_lattice_payload. Re-fit on full 6 + bootstrap CI on asymptote.

Power-law form: absorption(N) = gap_inf + prefactor * N^(-alpha)

Output: outputs/clp_absorption_extended_audit.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent  # /Emergence
PARENT = REPO

D1_DIRS = [
    PARENT / "d1_lattice_payload",
    PARENT / "d1_lattice_payload" / "p6",
    PARENT / "d1_lattice_payload" / "p7",
    PARENT / "d1_lattice_payload" / "p8",
]


def load_payloads():
    payloads = []
    for d1_dir in D1_DIRS:
        if not d1_dir.is_dir(): continue
        files = sorted([f for f in d1_dir.glob("d1_p*.json")
                       if not f.name.endswith(".metadata.json")
                       and "report" not in f.name])
        for f in files:
            with open(f) as fh:
                d = json.load(fh)
            n = d.get("dense_cell_node_count")
            ab = d.get("d1_gamma_ir_residual_absorption_closure_score")
            if n is not None and ab is not None:
                payloads.append({"file": f.name, "N": float(n), "abs": float(ab),
                                 "dir": d1_dir.name})
    # dedupe by N (keep first found)
    seen = {}
    for p in payloads:
        key = int(round(p["N"]))
        if key not in seen:
            seen[key] = p
    return sorted(seen.values(), key=lambda x: x["N"])


def power_law_fit(N_arr, y_arr):
    """Fit y(N) = gap_inf + prefactor * N^(-alpha). Returns dict."""
    # Try a range of alpha values, pick best fit
    from scipy.optimize import minimize_scalar
    N_arr = np.asarray(N_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)
    def residual(alpha):
        x = N_arr ** (-alpha)
        # Linear fit y = gap + prefactor * x
        try:
            A = np.column_stack([np.ones_like(x), x])
            coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
            pred = A @ coef
            return np.sum((y_arr - pred) ** 2)
        except Exception:
            return 1e20
    try:
        res = minimize_scalar(residual, bounds=(0.1, 5.0), method="bounded")
        alpha = res.x
        x = N_arr ** (-alpha)
        A = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
        gap_inf = float(coef[0]); prefactor = float(coef[1])
        pred = A @ coef
        residual_val = float(np.sum((y_arr - pred) ** 2))
    except Exception:
        return None
    return {"alpha": float(alpha), "prefactor": prefactor, "gap_inf": gap_inf,
            "residual": residual_val, "n_points": int(len(N_arr))}


def main() -> int:
    print("="*100)
    print("(Solution-1') Extended CLP-B B4 absorption fit + bootstrap CI")
    print("="*100)

    payloads = load_payloads()
    print(f"\nFound {len(payloads)} d1_p* payloads:")
    for p in payloads:
        print(f"  {p['file']}: N={p['N']:>8.0f}, absorption={p['abs']:.4f}")

    if len(payloads) < 4:
        print("Need ≥4 payloads"); return 1

    N_arr = np.array([p["N"] for p in payloads])
    y_arr = np.array([p["abs"] for p in payloads])

    # Subset fit (audit-style excluding P0)
    fit_sub = power_law_fit(N_arr[1:], y_arr[1:])
    print(f"\n{len(N_arr)-1}-point fit (excl P0): alpha={fit_sub['alpha']:.3f}, gap_inf={fit_sub['gap_inf']:.4f}, residual={fit_sub['residual']:.5f}")

    # Full N-point fit (all loaded points)
    fit_full = power_law_fit(N_arr, y_arr)
    print(f"{len(N_arr)}-point full fit: alpha={fit_full['alpha']:.3f}, gap_inf={fit_full['gap_inf']:.4f}, residual={fit_full['residual']:.5f}")

    # Bootstrap on full N points
    print(f"\nBootstrap CI on {len(N_arr)}-point fit (n_boot=2000):")
    rng = np.random.default_rng(42)
    n_boot = 2000
    asymptotes, alphas = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(payloads), size=len(payloads))
        try:
            r = power_law_fit(N_arr[idx], y_arr[idx])
            if r is None: continue
            asymptotes.append(r["gap_inf"])
            alphas.append(r["alpha"])
        except Exception:
            continue
    asymptotes = np.array([a for a in asymptotes if np.isfinite(a) and -5 < a < 5])
    alphas = np.array([a for a in alphas if np.isfinite(a) and 0 < a < 10])

    if len(asymptotes) > 100:
        med = float(np.median(asymptotes))
        lo, hi = np.percentile(asymptotes, [2.5, 97.5])
        print(f"  Asymptote median: {med:.4f}")
        print(f"  Asymptote 95% CI: [{lo:.4f}, {hi:.4f}]")
        in_05 = lo <= 0.5 <= hi
        print(f"  0.5 threshold IN CI? {in_05}")
        med_alpha = float(np.median(alphas))
        lo_a, hi_a = np.percentile(alphas, [2.5, 97.5])
        print(f"  Alpha median: {med_alpha:.3f}")
        print(f"  Alpha 95% CI: [{lo_a:.3f}, {hi_a:.3f}]")
    else:
        print(f"  Insufficient bootstrap successful fits: {len(asymptotes)}")
        med, lo, hi, in_05 = None, None, None, None

    print()
    print(f"=== Verdict ===")
    if in_05:
        verdict = "ABSORPTION_THRESHOLD_REACHED_IN_CI"
    elif med is not None and med > 0.5:
        verdict = "ABSORPTION_MEDIAN_ABOVE_THRESHOLD"
    elif med is not None:
        verdict = f"ABSORPTION_BELOW_THRESHOLD (median {med:.4f}, CI excludes 0.5)"
    else:
        verdict = "INSUFFICIENT"
    print(f"  {verdict}")

    out = {
        "method": "clp_absorption_extended",
        "payloads": payloads,
        "fit_subset_excl_P0": fit_sub,
        "fit_full": fit_full,
        "n_points_used": int(len(N_arr)),
        "bootstrap": {
            "n_boot": int(n_boot),
            "asymptote_median": med,
            "asymptote_CI95": [float(lo), float(hi)] if lo is not None else None,
            "0p5_in_CI": bool(in_05) if in_05 is not None else None,
            "alpha_median": float(np.median(alphas)) if len(alphas) else None,
            "alpha_CI95": [float(np.percentile(alphas, 2.5)), float(np.percentile(alphas, 97.5))] if len(alphas) else None,
        },
        "verdict": verdict,
    }
    out_path = REPO / "emergent-gr-closure-repro" / "outputs" / "clp_absorption_extended_audit.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
