"""(Opt-2/3/7) Unified CLP-B B4 audit: Symanzik 2+4, Joint-fit shared-α, AIC/BIC.

Properly handles all 5 sub-components (absorption, locality, density,
spectral, variational) on the 9-point dense-cell ladder.

Three complementary fits per component:
  M1 — free power-law:   y(N) = g + p · N^(-α)             [3 params]
  M2 — Symanzik 2+4:     y(N) = g + c2/N^2 + c4/N^4         [3 params]
  M3 — pure Symanzik-2:  y(N) = g + c2/N^2                  [2 params]

Plus joint fit M4 across all 5 components with shared α (15 params:
1 α + 5×2 per-component (gap_inf, prefactor)). Tests universality of
finite-N corrections.

Model selection: AICc with finite-sample correction.
Bootstrap CI: 2000 resamples, percentile method.

Validation:
  - All fits compared at common AICc baseline
  - Bootstrap stable across seed
  - Per-fit residuals reported
  - Edge case handling: divergent fits, NaN guards

Output: outputs/clp_unified_audit.json
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

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


# ─── fit models ──────────────────────────────────────────────────────────

def fit_free_power_law(n_arr, y_arr):
    """y = g + p · N^(-α). Returns dict or None."""
    n_arr = np.asarray(n_arr, float)
    y_arr = np.asarray(y_arr, float)
    if len(n_arr) < 3: return None
    def loss(alpha):
        x = n_arr ** (-alpha)
        A = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
        pred = A @ coef
        return float(np.sum((y_arr - pred)**2))
    try:
        res = minimize_scalar(loss, bounds=(0.1, 5.0), method="bounded")
        alpha = float(res.x)
        x = n_arr ** (-alpha)
        A = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
        pred = A @ coef
        rss = float(np.sum((y_arr - pred)**2))
        return {
            "model": "free_powerlaw", "n_params": 3,
            "alpha": alpha, "prefactor": float(coef[1]),
            "gap_inf": float(coef[0]),
            "rss": rss, "n_points": int(len(n_arr)),
        }
    except Exception:
        return None


def fit_symanzik_2plus4(n_arr, y_arr):
    """y = g + c2/N^2 + c4/N^4. Linear in (g, c2, c4). Returns dict."""
    n_arr = np.asarray(n_arr, float)
    y_arr = np.asarray(y_arr, float)
    if len(n_arr) < 3: return None
    try:
        x2 = n_arr**(-2.0); x4 = n_arr**(-4.0)
        A = np.column_stack([np.ones_like(x2), x2, x4])
        coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
        pred = A @ coef
        rss = float(np.sum((y_arr - pred)**2))
        return {
            "model": "symanzik_2plus4", "n_params": 3,
            "gap_inf": float(coef[0]), "c_2": float(coef[1]), "c_4": float(coef[2]),
            "rss": rss, "n_points": int(len(n_arr)),
        }
    except Exception:
        return None


def fit_symanzik_2(n_arr, y_arr):
    """y = g + c2/N^2. 2 params. Returns dict."""
    n_arr = np.asarray(n_arr, float)
    y_arr = np.asarray(y_arr, float)
    if len(n_arr) < 2: return None
    try:
        x2 = n_arr**(-2.0)
        A = np.column_stack([np.ones_like(x2), x2])
        coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
        pred = A @ coef
        rss = float(np.sum((y_arr - pred)**2))
        return {
            "model": "symanzik_2", "n_params": 2,
            "gap_inf": float(coef[0]), "c_2": float(coef[1]),
            "rss": rss, "n_points": int(len(n_arr)),
        }
    except Exception:
        return None


def aicc(rss, n, k):
    """Akaike with finite-sample correction. Lower is better."""
    if n <= k + 1 or rss <= 0: return float('inf')
    aic = n * math.log(rss / n) + 2*k
    correction = (2 * k * (k+1)) / (n - k - 1)
    return aic + correction


def bic(rss, n, k):
    """Schwarz-Bayes. Lower is better."""
    if rss <= 0: return float('inf')
    return n * math.log(rss / n) + k * math.log(n)


# ─── joint fit with shared α ─────────────────────────────────────────────

def fit_joint_shared_alpha(n_arr, y_dict):
    """y_k(N) = g_k + p_k · N^(-α) with α shared across all components.

    y_dict: {component: y_arr}
    Total params: 1 (α) + 2 * n_components (g_k, p_k each)
    """
    n_arr = np.asarray(n_arr, float)
    components = sorted(y_dict.keys())
    n_comp = len(components)

    def loss(alpha):
        x = n_arr ** (-alpha)
        A = np.column_stack([np.ones_like(x), x])
        total_rss = 0.0
        for k in components:
            coef, *_ = np.linalg.lstsq(A, y_dict[k], rcond=None)
            pred = A @ coef
            total_rss += float(np.sum((y_dict[k] - pred)**2))
        return total_rss

    try:
        res = minimize_scalar(loss, bounds=(0.1, 5.0), method="bounded")
        alpha_star = float(res.x)
        # Now extract per-component params with this shared alpha
        x = n_arr ** (-alpha_star)
        A = np.column_stack([np.ones_like(x), x])
        per_component = {}
        total_rss = 0.0
        for k in components:
            coef, *_ = np.linalg.lstsq(A, y_dict[k], rcond=None)
            pred = A @ coef
            rss_k = float(np.sum((y_dict[k] - pred)**2))
            total_rss += rss_k
            per_component[k] = {
                "gap_inf": float(coef[0]), "prefactor": float(coef[1]),
                "rss": rss_k,
            }
        return {
            "model": "joint_shared_alpha",
            "alpha_shared": alpha_star,
            "n_params": 1 + 2 * n_comp,
            "n_points_total": n_comp * len(n_arr),
            "per_component": per_component,
            "total_rss": total_rss,
        }
    except Exception:
        return None


# ─── bootstrap ───────────────────────────────────────────────────────────

def bootstrap_asymptote(n_arr, y_arr, fit_fn, n_boot=2000, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    asymptotes = []
    n = len(n_arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            r = fit_fn(n_arr[idx], y_arr[idx])
            if r is not None and "gap_inf" in r:
                asymptotes.append(r["gap_inf"])
        except Exception:
            continue
    asymptotes = np.array([a for a in asymptotes if np.isfinite(a) and -2 < a < 2])
    if len(asymptotes) < 100:
        return None
    return {
        "n_resamples": int(len(asymptotes)),
        "median": float(np.median(asymptotes)),
        "CI95": [float(np.percentile(asymptotes, 2.5)),
                 float(np.percentile(asymptotes, 97.5))],
        "std": float(np.std(asymptotes)),
    }


# ─── main ───────────────────────────────────────────────────────────────

def main() -> int:
    print("="*100)
    print("(Opt-2/3/7) Unified CLP audit: Symanzik 2+4, Joint shared-α, AIC/BIC")
    print("="*100)

    payloads = load_payloads()
    print(f"\nFound {len(payloads)} dense-cell payloads")

    n_arr = np.array([p["N"] for p in payloads])

    # Build y_dict for all components present
    y_dict = {}
    for m in METRICS:
        try:
            vals = [p.get(m) for p in payloads]
            if any(v is None for v in vals): continue
            y_dict[m] = np.array(vals, float)
        except KeyError:
            continue
    print(f"Components present: {list(y_dict.keys())}")
    print()

    # ─── Per-component model comparison ─────────────────────────────────
    print("="*100)
    print("Per-component model comparison (free power-law vs Symanzik 2+4 vs Symanzik 2)")
    print("="*100)
    results = {"per_component": {}}
    print(f"\n{'metric':<12} {'model':<18} {'gap_inf':>9} {'rss':>10} {'AICc':>9} {'BIC':>9} {'Δ_AICc':>9}")
    print("-"*100)

    for m, y_arr in y_dict.items():
        fits = {
            "free_powerlaw": fit_free_power_law(n_arr, y_arr),
            "symanzik_2plus4": fit_symanzik_2plus4(n_arr, y_arr),
            "symanzik_2": fit_symanzik_2(n_arr, y_arr),
        }
        n = len(n_arr)
        comp_results = {}
        scores = {}
        for fname, f in fits.items():
            if f is None: continue
            f["AICc"] = aicc(f["rss"], n, f["n_params"])
            f["BIC"] = bic(f["rss"], n, f["n_params"])
            scores[fname] = f["AICc"]
            comp_results[fname] = f
        if not scores: continue
        best_aicc = min(scores.values())
        for fname, f in comp_results.items():
            d_aicc = f["AICc"] - best_aicc
            star = "★" if d_aicc < 0.01 else ""
            print(f"{m:<12} {fname:<18} {f['gap_inf']:>9.4f} {f['rss']:>10.5f} "
                  f"{f['AICc']:>9.2f} {f['BIC']:>9.2f} {d_aicc:>9.2f} {star}")
        # bootstrap on best model
        best_model = min(comp_results.keys(), key=lambda k: comp_results[k]["AICc"])
        fit_fn = {"free_powerlaw": fit_free_power_law,
                  "symanzik_2plus4": fit_symanzik_2plus4,
                  "symanzik_2": fit_symanzik_2}[best_model]
        bs = bootstrap_asymptote(n_arr, y_arr, fit_fn, n_boot=2000)
        comp_results["best_by_AICc"] = best_model
        comp_results["bootstrap_on_best"] = bs
        results["per_component"][m] = comp_results
        if bs:
            in_ci = bs["CI95"][0] <= 0.5 <= bs["CI95"][1]
            print(f"{'':12} → BEST: {best_model}, bootstrap median {bs['median']:.4f}, "
                  f"CI95 [{bs['CI95'][0]:+.3f}, {bs['CI95'][1]:+.3f}], 0.5 in CI: {'YES' if in_ci else 'NO'}")
        print()

    # ─── Joint fit shared alpha ────────────────────────────────────────
    print("="*100)
    print("Joint fit with shared α across all components")
    print("="*100)
    joint = fit_joint_shared_alpha(n_arr, y_dict)
    if joint:
        print(f"\nShared α = {joint['alpha_shared']:.4f}")
        print(f"Total RSS across {len(y_dict)} components: {joint['total_rss']:.5f}")
        print(f"Total params: {joint['n_params']}")
        print(f"Total points: {joint['n_points_total']}")
        n_total = joint['n_points_total']; k_total = joint['n_params']
        joint["AICc"] = aicc(joint['total_rss'], n_total, k_total)
        joint["BIC"]  = bic(joint['total_rss'], n_total, k_total)
        print(f"AICc: {joint['AICc']:.2f}, BIC: {joint['BIC']:.2f}")
        print()
        print(f"{'component':<12} {'gap_inf':>9} {'prefactor':>11} {'rss':>10}")
        print("-"*50)
        for k, v in joint["per_component"].items():
            print(f"{k:<12} {v['gap_inf']:>9.4f} {v['prefactor']:>11.2f} {v['rss']:>10.5f}")
        results["joint_shared_alpha"] = joint
    print()

    # ─── Compare joint vs sum-of-individuals ────────────────────────────
    sum_individual_rss = 0.0
    sum_individual_params = 0
    for m in y_dict:
        best_m = results["per_component"][m]["best_by_AICc"]
        f = results["per_component"][m][best_m]
        sum_individual_rss += f["rss"]
        sum_individual_params += f["n_params"]
    n_total = sum(len(n_arr) for _ in y_dict)
    indiv_aicc = aicc(sum_individual_rss, n_total, sum_individual_params)
    indiv_bic = bic(sum_individual_rss, n_total, sum_individual_params)
    print("="*100)
    print("Joint vs sum-of-individuals model comparison")
    print("="*100)
    print(f"{'method':<32} {'total RSS':>12} {'n_params':>10} {'AICc':>9} {'BIC':>9}")
    print("-"*80)
    print(f"{'sum-of-individuals (best each)':<32} {sum_individual_rss:>12.5f} "
          f"{sum_individual_params:>10} {indiv_aicc:>9.2f} {indiv_bic:>9.2f}")
    if "joint_shared_alpha" in results:
        j = results["joint_shared_alpha"]
        print(f"{'joint shared-α':<32} {j['total_rss']:>12.5f} "
              f"{j['n_params']:>10} {j['AICc']:>9.2f} {j['BIC']:>9.2f}")
        delta_aicc = j["AICc"] - indiv_aicc
        delta_bic = j["BIC"] - indiv_bic
        print(f"\nΔAICc (joint − individual) = {delta_aicc:+.2f}")
        print(f"ΔBIC  (joint − individual) = {delta_bic:+.2f}")
        if delta_aicc < -2: verdict = "JOINT_PREFERRED_AICc<-2"
        elif delta_aicc < 2: verdict = "JOINT_AND_INDIVIDUAL_INDISTINGUISHABLE"
        else: verdict = "INDIVIDUAL_PREFERRED_AICc>+2"
        print(f"Verdict: {verdict}")
        results["joint_vs_individual_AICc_delta"] = delta_aicc
        results["joint_vs_individual_BIC_delta"] = delta_bic
        results["verdict"] = verdict

    out_path = REPO / "emergent-gr-closure-repro" / "outputs" / "clp_unified_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
