"""5-point P5-physics within-regime Symanzik fit + kappa_t algebraic
exploration.

P5-physics sequence: P5 (N=50), P5N64, P5N72, P5N84, P5N100 — all
share identical lambda_triangle, epsilon, defect_params, alpha_scale
settings.

If d_inf < 0.05 (95% bootstrap CI upper) for all 4 components on
this within-regime sequence, the closure is regime-pure (no
cross-regime confound).

Algebraic exploration: try common combinations of System-R
constants (alpha_xi=9/10, gamma=1/10, beta_pi=15/16, D=67/80,
eps_sync2=1/20) against the empirical kappa_t = 0.987 from the
backreaction theorem.

Output: outputs/p5_only_symanzik_audit.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


class _BlockCupy:
    def find_module(self, name, path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self

    def load_module(self, name):
        raise ImportError("cupy disabled")

sys.meta_path.insert(0, _BlockCupy())

from verify_per_eigendirection_residual import gather_regime


P5_SEQUENCE = [
    ("P5", 50), ("P5N64", 64), ("P5N72", 72),
    ("P5N84", 84), ("P5N100", 100),
]
KAPPA_T_OBSERVED = 0.987


def fit_symanzik_2_4(N, y, constrain=True):
    if len(N) < 4: return None
    X = np.column_stack([np.ones_like(N), N ** -2.0, N ** -4.0])
    c, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
    if constrain and c[0] < 0:
        X2 = X[:, 1:]
        c12, *_ = np.linalg.lstsq(X2, y, rcond=1e-10)
        c = np.array([0.0, c12[0], c12[1]])
    pred = X @ c
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"d_inf": float(c[0]), "c_2": float(c[1]), "c_4": float(c[2]),
            "R_squared": r2}


def bootstrap_d_inf(N, y, n_boot=1000):
    rng = np.random.default_rng(seed=42)
    ds = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(N), size=len(N))
        if len(np.unique(N[idx])) < 4: continue
        f = fit_symanzik_2_4(N[idx], y[idx], constrain=True)
        if f: ds.append(f["d_inf"])
    if not ds:
        return None
    ds = np.array(ds)
    return {"median": float(np.median(ds)),
            "ci_2_5": float(np.percentile(ds, 2.5)),
            "ci_97_5": float(np.percentile(ds, 97.5))}


def kappa_algebraic_candidates():
    """Try candidate algebraic forms for kappa_t in System-R constants."""
    alpha_xi = 9.0 / 10.0
    gamma = 1.0 / 10.0
    beta_pi = 15.0 / 16.0
    D = 67.0 / 80.0
    eps_sync2 = 1.0 / 20.0
    pi = np.pi

    cands = [
        ("1 - gamma^2",                   1 - gamma**2),
        ("1 - gamma^2 - gamma^3",          1 - gamma**2 - gamma**3),
        ("1 - gamma^2 * (1 + gamma)",      1 - gamma**2 * (1 + gamma)),
        ("alpha_xi^gamma",                 alpha_xi ** gamma),
        ("alpha_xi^2 + (1-alpha_xi^2)*alpha_xi", alpha_xi**2 + (1-alpha_xi**2)*alpha_xi),
        ("1 - gamma^2 * pi/eps_sync2 / 50", 1 - gamma**2 * pi / eps_sync2 / 50),
        ("alpha_xi + gamma*(1-gamma)",     alpha_xi + gamma*(1-gamma)),
        ("alpha_xi + gamma - 2*gamma^2",   alpha_xi + gamma - 2*gamma**2),
        ("1 - gamma^2 - eps_sync2*gamma",  1 - gamma**2 - eps_sync2*gamma),
        ("1 - gamma * eps_sync2 * pi/4",   1 - gamma * eps_sync2 * pi/4),
        ("(alpha_xi + 1)/2 + gamma*(1-gamma)/2",  (alpha_xi + 1)/2 + gamma*(1-gamma)/2),
        ("alpha_xi * (1 + gamma) - 0.005", alpha_xi * (1 + gamma) - 0.005),
        ("D + (1-D) * (1+gamma) * 0.5",    D + (1-D) * (1+gamma) * 0.5),
        ("beta_pi + (1-beta_pi)*alpha_xi", beta_pi + (1-beta_pi)*alpha_xi),
        ("beta_pi + gamma*(1-beta_pi)*4",   beta_pi + gamma*(1-beta_pi)*4),
        ("(1+alpha_xi)/2 + gamma^2*5",     (1+alpha_xi)/2 + gamma**2*5),
        ("1 - gamma^2 * 4/pi",             1 - gamma**2 * 4/pi),
        ("1 - gamma^3 * pi",               1 - gamma**3 * pi),
    ]
    return [(label, val, abs(val - KAPPA_T_OBSERVED) / KAPPA_T_OBSERVED * 100)
            for label, val in cands]


def main() -> int:
    print("=" * 100)
    print("(1) 5-point P5-physics within-regime Symanzik fit")
    print("=" * 100)
    rows = []
    for reg, n_lat in P5_SEQUENCE:
        try:
            r = gather_regime(reg, n_lat)
        except Exception as e:
            print(f"  {reg}: ERROR: {e}")
            continue
        if r is None:
            print(f"  {reg}: not found")
            continue
        rows.append(r)
    print(f"  Loaded {len(rows)} P5-physics points")

    if len(rows) < 4:
        print(f"  Insufficient points; abort.")
        return 1

    N = np.array([r["N"] for r in rows], dtype=float)
    print()
    print(f"{'reg':<10} {'N':>3} | "
          f"{'R_time_med':>11} {'R_trace_med':>12} {'R_TF_med':>10} {'R_off_med':>10}")
    print("-" * 70)
    for r in rows:
        print(f"{r['regime']:<10} {r['N']:>3} | "
              f"{r['R_time_median_abs']:>11.5f} "
              f"{r['R_trace_median_abs']:>12.5f} "
              f"{r['R_TF_norm_median_abs']:>10.5f} "
              f"{r['R_off_median_abs']:>10.5f}")

    print()
    print("Symanzik 2+4 (constrained-positive) on 5-point P5-only sequence:")
    print(f"{'comp':<22} {'d_inf':>10} {'c_2':>10} {'R^2':>7} | "
          f"{'CI 95% [low, high]':>26}")
    print("-" * 90)
    fits = {}
    for comp in ["R_time_median_abs", "R_trace_median_abs",
                 "R_TF_norm_median_abs", "R_off_median_abs"]:
        y = np.array([r[comp] for r in rows])
        f = fit_symanzik_2_4(N, y, constrain=True)
        b = bootstrap_d_inf(N, y, n_boot=1000)
        fits[comp] = {"fit": f, "bootstrap": b}
        print(f"{comp:<22} {f['d_inf']:>10.5f} {f['c_2']:>+10.2f} "
              f"{f['R_squared']:>7.3f} | "
              f"[{b['ci_2_5']:>+8.5f}, {b['ci_97_5']:>+8.5f}]"
              if b else "  (no bootstrap)")

    n_pass = sum(1 for c, f in fits.items()
                 if f["bootstrap"] and f["bootstrap"]["ci_97_5"] <= 0.05)
    print(f"\n  Components passing 95% CI Upper <= 0.05: {n_pass}/4")

    print()
    print("=" * 100)
    print(f"(2) kappa_t algebraic exploration (target: kappa_t = {KAPPA_T_OBSERVED})")
    print("=" * 100)
    print()
    print(f"{'expression':<50} {'value':>10} {'rel error':>10}")
    print("-" * 80)
    cands = sorted(kappa_algebraic_candidates(), key=lambda x: x[2])
    for label, val, err in cands[:12]:
        print(f"{label:<50} {val:>10.5f} {err:>9.3f}%")
    closest = cands[0]
    print()
    print(f"Closest match: {closest[0]} = {closest[1]:.5f} ({closest[2]:.3f}% off)")

    if closest[2] < 1.0:
        verdict = f"KAPPA_T_RATIONAL_FORM_CANDIDATE_{closest[0].replace(' ', '_')}"
    elif closest[2] < 3.0:
        verdict = "KAPPA_T_RATIONAL_FORM_APPROXIMATE"
    else:
        verdict = "KAPPA_T_NO_CLEAN_RATIONAL_FORM"
    print(f"VERDICT: {verdict}")

    out_path = REPO / "outputs" / "p5_only_symanzik_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "p5_only_5point_symanzik_plus_kappa_algebraic",
            "schema_version": "1.0.0",
            "p5_within_regime_fits": fits,
            "n_components_pass_95ci_upper_below_0p05": int(n_pass),
            "kappa_t_observed": KAPPA_T_OBSERVED,
            "kappa_algebraic_candidates": [
                {"label": label, "value": val, "rel_error_percent": err}
                for label, val, err in cands
            ],
            "closest_match": {"label": closest[0], "value": closest[1],
                              "rel_error_percent": closest[2]},
            "verdict_kappa": verdict,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
