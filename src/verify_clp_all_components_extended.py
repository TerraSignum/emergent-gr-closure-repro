"""(Solution-1''  ) Extended CLP-B B4 ALL sub-components fit + bootstrap.

Same 9-point dense-cell ladder, but for all four sub-components:
  - spectral
  - locality
  - density
  - absorption

This gives a complete CLP-B B4 picture, not just the absorption
bottleneck.

Output: outputs/clp_all_components_extended_audit.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parent.parent.parent

D1_DIRS = [
    REPO / "results_d1_fix17",
    REPO / "results_d1_fix16" / "p6",
    REPO / "results_d1_fix16" / "p7",
    REPO / "results_d1_fix16" / "p8",
]

METRICS = {
    "absorption": "d1_gamma_ir_residual_absorption_closure_score",
    "locality":   "d1_gamma_ir_residual_locality_score",
    "density":    "d1_gamma_ir_residual_density_score",
    # NOTE: "spectral" in CLP-B B4 is derived from B2 spectral-gap stability,
    # which itself is power_law_fit on joint_closure_score (continuum_limit_proof.py:411-432, 543).
    # Loading the underlying source key directly here:
    "spectral":   "d1_gamma_full_macroclass_joint_closure_score",
    "variational":"d1_gamma_ir_variational_closure_score",
}


def load_payloads():
    payloads = []
    for d1_dir in D1_DIRS:
        if not d1_dir.is_dir(): continue
        for f in sorted(d1_dir.glob("d1_p*.json")):
            if f.name.endswith(".metadata.json") or "report" in f.name: continue
            with open(f) as fh:
                d = json.load(fh)
            n = d.get("dense_cell_node_count")
            if n is None: continue
            entry = {"file": f.name, "N": float(n), "dir": d1_dir.name}
            for tag, key in METRICS.items():
                v = d.get(key)
                if isinstance(v, (int, float)):
                    entry[tag] = float(v)
            payloads.append(entry)
    seen = {}
    for p in payloads:
        key = int(round(p["N"]))
        if key not in seen:
            seen[key] = p
    return sorted(seen.values(), key=lambda x: x["N"])


def power_law_fit(n_arr, y_arr):
    n_arr = np.asarray(n_arr, float); y_arr = np.asarray(y_arr, float)
    def residual(alpha):
        x = n_arr ** (-alpha)
        try:
            A = np.column_stack([np.ones_like(x), x])
            coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
            pred = A @ coef
            return float(np.sum((y_arr - pred)**2))
        except Exception:
            return 1e20
    try:
        res = minimize_scalar(residual, bounds=(0.1, 5.0), method="bounded")
        alpha = res.x
        x = n_arr ** (-alpha)
        A = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
        return {"alpha": float(alpha), "prefactor": float(coef[1]),
                "gap_inf": float(coef[0]), "n_points": int(len(n_arr))}
    except Exception:
        return None


def bootstrap_asymptote(n_arr, y_arr, n_boot=2000, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    asymptotes, alphas = [], []
    n = len(n_arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = power_law_fit(n_arr[idx], y_arr[idx])
        if r is None: continue
        asymptotes.append(r["gap_inf"])
        alphas.append(r["alpha"])
    asymptotes = np.array([a for a in asymptotes if np.isfinite(a) and -5 < a < 5])
    alphas = np.array([a for a in alphas if np.isfinite(a)])
    if len(asymptotes) < 50:
        return None
    return {
        "n_resamples": int(len(asymptotes)),
        "asymptote_median": float(np.median(asymptotes)),
        "asymptote_CI95": [float(np.percentile(asymptotes, 2.5)),
                           float(np.percentile(asymptotes, 97.5))],
        "alpha_median": float(np.median(alphas)),
        "alpha_CI95": [float(np.percentile(alphas, 2.5)),
                       float(np.percentile(alphas, 97.5))],
    }


def main() -> int:
    print("="*100)
    print("(Solution-1'') CLP-B B4 ALL sub-components extended bootstrap")
    print("="*100)
    payloads = load_payloads()
    print(f"\nFound {len(payloads)} dense-cell payloads:\n")
    print(f"{'file':<22} {'N':>8} | " + "  ".join(f"{m:>10}" for m in METRICS))
    print("-"*100)
    for p in payloads:
        line = f"{p['file']:<22} {p['N']:>8.0f} |"
        for m in METRICS:
            v = p.get(m, None)
            line += f"  {f'{v:.4f}' if v is not None else '—':>10}"
        print(line)

    if len(payloads) < 4: return 1

    n_arr = np.array([p["N"] for p in payloads])
    out = {"method": "clp_all_components_extended", "n_points": int(len(payloads)),
           "by_metric": {}}

    print()
    print(f"{'metric':<12} | {'fit_alpha':>10} {'fit_gap':>9} | {'bs_median':>10} {'bs_CI95':>22} | thr=0.5 in CI?")
    print("-"*100)
    for m in METRICS:
        y = [p.get(m) for p in payloads]
        if any(v is None for v in y): print(f"{m:<12} | NaN (missing)"); continue
        y_arr = np.array(y, float)
        fit = power_law_fit(n_arr, y_arr)
        if fit is None: continue
        bs = bootstrap_asymptote(n_arr, y_arr)
        if bs is None: continue
        in_ci = bs["asymptote_CI95"][0] <= 0.5 <= bs["asymptote_CI95"][1]
        print(f"{m:<12} | {fit['alpha']:>10.3f} {fit['gap_inf']:>9.4f} | "
              f"{bs['asymptote_median']:>10.4f} [{bs['asymptote_CI95'][0]:>+7.4f}, {bs['asymptote_CI95'][1]:>+7.4f}] | "
              f"{'YES' if in_ci else 'NO':>5}")
        out["by_metric"][m] = {"fit": fit, "bootstrap": bs, "threshold_0p5_in_CI": bool(in_ci)}

    print()
    out_path = REPO / "emergent-gr-closure-repro" / "outputs" / "clp_all_components_extended_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
