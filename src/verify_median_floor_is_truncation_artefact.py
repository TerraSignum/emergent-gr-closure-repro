"""verify_median_floor_is_truncation_artefact.py

Bombenfeste Verifikation: Der "Symanzik-2 y_inf = 0.019" Bulk-Median-Floor
ist KEIN struktureller asymptotischer Floor, sondern ein
Trunkierungs-Artefakt der Symanzik-2-Basis {1, 1/N, 1/N^2}, die das
strukturelle 1/N^(1/d_spatial) = 1/N^(1/3) Federer-Skalengesetz nicht
darstellen kann.

Drei Strenge-Stufen:

(1) Direkter Daten-Test: Power-law α auf der 10-Regime-Leiter
    konsistent mit 1/3 (Federer geometric measure theory bound).

(2) Symanzik-Basis-Argument analytisch: ein N^(-1/3)-Signal kann durch
    {1, 1/N, 1/N^2} nicht ohne konstanten Bias dargestellt werden.

(3) Forward-Test: synthetic data y_synth(N) = c * N^(-1/3) (FORCED zu 0
    asymptotisch); Symanzik-2 auf y_synth gibt apparent y_inf > 0
    auf der gleichen N-Leiter wie die Lattice-Daten. Falls
    apparent_y_inf_synth ≈ 0.019, ist die Empirie konsistent mit reiner
    1/N^(1/3)-Form (kein echter Floor).

(4) Confidence-Intervall-Verifikation: 1/3 liegt im Power-law-α CI.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
BUNDLE_PATH = REPO / "outputs" / "stage6f_bulk_percentile_upper_bounds.json"
PERCENTILE = "median"

# ---------- Stage (1): Bundle-Daten direkt nachrechnen ----------
def load_bundle():
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(BUNDLE_PATH)
    return json.load(open(BUNDLE_PATH, "r", encoding="utf-8"))

def load_per_n_data():
    """Lade per-N median values from stage6f_regular_core_decomposition."""
    p = REPO / "outputs" / "stage6f_regular_core_decomposition.json"
    d = json.load(open(p, "r", encoding="utf-8"))
    Ns, ys = [], []
    for entry in d["per_regime"]:
        n = entry.get("N")
        rsm = entry.get("regular_set_median", {})
        v = rsm.get("0.05") if isinstance(rsm, dict) else None
        if n is not None and v is not None:
            Ns.append(int(n)); ys.append(float(v))
    return np.array(Ns, dtype=float), np.array(ys)

def fit_symanzik2(N, y):
    """y = a + b/N + c/N^2, return (a, b, c, rmse, aic)."""
    A = np.column_stack([np.ones_like(N), 1.0/N, 1.0/N**2])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    rmse = float(np.sqrt(np.mean(res**2)))
    aic = len(N) * np.log(max(rmse**2, 1e-30)) + 2*3
    return coef[0], coef[1], coef[2], rmse, aic

def fit_powerlaw(N, y):
    """y = c * N^(-alpha) via log-linear fit, return (c, alpha, rmse, aic)."""
    log_N = np.log(N); log_y = np.log(np.maximum(y, 1e-15))
    slope, intercept = np.polyfit(log_N, log_y, 1)
    alpha = -slope; c = np.exp(intercept)
    yp = c * N**(-alpha)
    rmse = float(np.sqrt(np.mean((y - yp)**2)))
    aic = len(N) * np.log(max(rmse**2, 1e-30)) + 2*2
    return c, alpha, rmse, aic

def fit_federer(N, y):
    """y = c * N^(-1/3) FORCED, single-parameter fit, return (c, rmse, aic)."""
    basis = N**(-1/3)
    c = float(np.sum(y * basis) / np.sum(basis**2))  # least-squares scalar
    yp = c * basis
    rmse = float(np.sqrt(np.mean((y - yp)**2)))
    aic = len(N) * np.log(max(rmse**2, 1e-30)) + 2*1
    return c, rmse, aic

# ---------- Stage (3): Forward-Test mit synthetic data ----------
def forward_test(N_grid, c_true=1.0, n_seeds=1000, noise_rel=0.05):
    """Generate y_synth(N) = c_true * N^(-1/3), add seed-noise, fit Symanzik-2,
    return distribution of apparent y_inf values.

    If echte Skalierung 1/N^(1/3) ist und Symanzik-2 systematic produces
    apparent y_inf > 0 weil Basis 1/N^(1/3) nicht enthält.
    """
    rng = np.random.default_rng(seed=42)
    apparent_y_inf = []
    apparent_a = []
    for s in range(n_seeds):
        y_clean = c_true * N_grid**(-1/3)
        noise = rng.normal(0, noise_rel * y_clean.mean(), size=len(N_grid))
        y_noisy = y_clean + noise
        a, b, c, rmse, _ = fit_symanzik2(N_grid, y_noisy)
        apparent_y_inf.append(a)
    apparent_y_inf = np.array(apparent_y_inf)
    return apparent_y_inf

def main():
    print("=" * 78)
    print("Bulk-Median Floor: Symanzik-2 Trunkierungs-Artefakt-Test")
    print("=" * 78)

    # Stage (1): Daten laden + Symanzik-2 + Power-law direkt fitten
    Ns, ys = load_per_n_data()
    print(f"\n[Stage 1] Direct fit on {len(Ns)} ladder points (P5-canonical, regular_set_median[0.05]):")
    print(f"  Ns = {[int(n) for n in Ns]}")

    a_S, b_S, c_S, rmse_S, aic_S = fit_symanzik2(Ns, ys)
    c_P, alpha_P, rmse_P, aic_P = fit_powerlaw(Ns, ys)
    c_F, rmse_F, aic_F = fit_federer(Ns, ys)

    print(f"\n  Symanzik-2 (3 params): a={a_S:.6f}, b={b_S:.4f}, c={c_S:.4f}")
    print(f"     rmse={rmse_S:.3e}, AIC={aic_S:.2f}")
    print(f"  Power-law  (2 params): c={c_P:.4f}, alpha={alpha_P:.4f}")
    print(f"     rmse={rmse_P:.3e}, AIC={aic_P:.2f}")
    print(f"  Federer    (1 param ): c={c_F:.4f}, alpha=1/3 FORCED")
    print(f"     rmse={rmse_F:.3e}, AIC={aic_F:.2f}")

    aic_min = min(aic_S, aic_P, aic_F)
    print(f"\n  DeltaAIC vs best: Symanzik-2={aic_S-aic_min:.2f}, "
          f"Power-law={aic_P-aic_min:.2f}, Federer(1/3)={aic_F-aic_min:.2f}")

    # Stage (2): Bundle-Bootstrap-CI anschauen
    bundle = load_bundle()
    bp_med = bundle["per_percentile"][PERCENTILE]
    print(f"\n[Stage 2] Bundle bootstrap CI ({bundle['n_bootstrap']} resamples):")
    print(f"  Symanzik-2 y_inf central = {bp_med['symanzik2_unconstrained_y_inf']:.6f}, "
          f"95% CI = {bp_med['symanzik2_y_inf_95CI']}")
    print(f"  Power-law alpha central = {bp_med['powerlaw_alpha_central']:.6f}, "
          f"95% CI = {bp_med['powerlaw_alpha_95CI']}")
    one_third = 1/3
    a_lo, a_hi = bp_med['powerlaw_alpha_95CI']
    contains = a_lo <= one_third <= a_hi
    print(f"  1/3 = {one_third:.6f} contained in alpha 95% CI = {contains}")

    # Stage (3): Forward-Test - synthetic 1/N^(1/3) data, Symanzik-2 fit
    print("\n[Stage 3] Forward-Test: synthetic y_synth(N) = c * N^(-1/3) (true asymptote = 0).")
    print("  Fit Symanzik-2 on synthetic data, measure apparent_y_inf distribution.")
    # Fit c_true to match observed lattice mean
    c_true = c_F  # from Federer fit
    apparent = forward_test(Ns, c_true=c_true, n_seeds=2000, noise_rel=0.05)
    print(f"  c_true = {c_true:.4f} (matched to observed lattice via Federer fit)")
    print(f"  Symanzik-2 apparent_y_inf distribution over 2000 seeds:")
    print(f"     mean = {np.mean(apparent):.6f}")
    print(f"     median = {np.median(apparent):.6f}")
    print(f"     std = {np.std(apparent):.6f}")
    print(f"     95% CI = [{np.percentile(apparent, 2.5):.6f}, {np.percentile(apparent, 97.5):.6f}]")
    obs = bp_med['symanzik2_unconstrained_y_inf']
    z_score = abs(obs - np.mean(apparent)) / np.std(apparent)
    print(f"\n  Observed Symanzik-2 y_inf = {obs:.6f}")
    print(f"  Z-score vs synthetic-Symanzik-2 distribution: {z_score:.2f}")
    print(f"  -> if |Z| < 2: empirical y_inf is statistically indistinguishable from")
    print(f"     'pure 1/N^(1/3) signal interpreted by Symanzik-2 truncation'")

    # Stage (4): Symanzik-Basis-Argument analytisch
    print("\n[Stage 4] Analytical: Symanzik-2 basis cannot represent N^(-1/3).")
    # Project N^(-1/3) onto {1, 1/N, 1/N^2} basis on the actual N-grid
    target = Ns**(-1/3)
    A = np.column_stack([np.ones_like(Ns), 1.0/Ns, 1.0/Ns**2])
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    proj = A @ coef
    residual = target - proj
    print(f"  L2-projection of N^(-1/3) onto Symanzik-2 basis on actual N-grid:")
    print(f"    a coefficient (= apparent_floor) = {coef[0]:.6f}")
    print(f"    b coefficient = {coef[1]:.4f}")
    print(f"    c coefficient = {coef[2]:.4f}")
    print(f"    residual rms = {np.sqrt(np.mean(residual**2)):.3e}")
    print(f"  -> the projection has a non-zero a-coefficient (= apparent floor)")
    print(f"     of {coef[0]:.6f}, *purely from the basis truncation*, even though")
    print(f"     the true asymptote is 0.")

    # Save
    out = {
        "audit": "median-floor-truncation-artefact",
        "stand": "2026-05-04",
        "ladder": [int(n) for n in Ns],
        "median_values": [float(y) for y in ys],
        "fits": {
            "symanzik2": {"a_inf": a_S, "b": b_S, "c": c_S, "rmse": rmse_S, "aic": aic_S},
            "powerlaw":  {"c": c_P, "alpha": alpha_P, "rmse": rmse_P, "aic": aic_P},
            "federer_one_third": {"c": c_F, "alpha": 1/3, "rmse": rmse_F, "aic": aic_F},
        },
        "bundle_bootstrap": {
            "powerlaw_alpha_central": bp_med['powerlaw_alpha_central'],
            "powerlaw_alpha_95CI": bp_med['powerlaw_alpha_95CI'],
            "one_third_in_CI": bool(contains),
            "symanzik2_y_inf_central": bp_med['symanzik2_unconstrained_y_inf'],
            "symanzik2_y_inf_95CI": bp_med['symanzik2_y_inf_95CI'],
        },
        "forward_test": {
            "synthetic_form": "y_synth(N) = c_true * N^(-1/3), true asymptote = 0",
            "n_seeds": 2000,
            "c_true": c_true,
            "symanzik2_apparent_y_inf_mean": float(np.mean(apparent)),
            "symanzik2_apparent_y_inf_median": float(np.median(apparent)),
            "symanzik2_apparent_y_inf_95CI": [float(np.percentile(apparent, 2.5)),
                                              float(np.percentile(apparent, 97.5))],
            "z_score_obs_vs_synthetic": float(z_score),
        },
        "analytical_basis_projection": {
            "target": "N^(-1/3) on actual N-grid",
            "symanzik2_basis": "{1, 1/N, 1/N^2}",
            "projected_a_coefficient": float(coef[0]),
            "projected_b_coefficient": float(coef[1]),
            "projected_c_coefficient": float(coef[2]),
            "interpretation":
                "Even with true asymptote zero, projection onto {1,1/N,1/N^2} "
                "absorbs N^(-1/3) into a non-zero a-coefficient. This is the "
                "Symanzik-2 truncation artefact.",
        },
    }
    out_path = REPO / "data" / "median_floor_truncation_artefact.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
