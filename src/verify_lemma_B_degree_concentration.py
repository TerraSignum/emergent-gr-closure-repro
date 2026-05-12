"""Lemma B Phase-2 Step 2: empirical test of the degree-
concentration hypothesis.

Step 1 falsified the radial low-rank hypothesis (see
`verify_lemma_B_radial_hypothesis.py`,
`outputs/verify_lemma_B_radial_hypothesis.json`). The uniform
Laplacian spectral gap of Proposition `prop:sg_empirical`
(lambda_inf ~ 0.3789) therefore must arise from a structural
property other than low-dimensional embedding of Xi_N. The
revised primary candidate (notes/lemma_B_proof_strategy.md
Route 1+, "quantitative Cheeger via degree concentration")
predicts the gap is controlled by degree concentration:

  Degree-concentration hypothesis:
    Var(deg) / Mean(deg)^2 -> 0 as N -> infinity

where deg(i) = sum_{j != i} Xi_N(i, j) is the unnormalised
degree of vertex i in the Xi-weighted graph.

If degree-concentration holds, the random-regular-graph-style
Cheeger argument (Diaconis-Stroock canonical paths) gives a
uniform spectral gap controlled by the mean degree and the
degree-fluctuation amplitude, not by xi_min alone — analytically
tractable, with explicit closed form.

Two complementary concentration metrics are reported:

  (D1) Relative coefficient of variation:
       CV_deg(N) = std(deg) / mean(deg)
       (dimensionless; should -> 0 if concentration holds)

  (D2) Normalised second moment:
       NSM_deg(N) = Var(deg) / Mean(deg)^2 = CV_deg^2
       (the quantity Cheeger arguments control directly)

Both are cross-seed averaged per regime, then N-scaling-fitted
across the ladder with AICc model selection over
{const, linear, log, sqrt, power_law}. The hypothesis is
supported iff the preferred model has a positive asymptote
N -> infinity strictly smaller than the smallest regime value
(i.e., the metric is decreasing, not asymptotically constant).

Output: outputs/verify_lemma_B_degree_concentration.json
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REPO_ROOT = REPO.parent
OUT = REPO / "outputs" / "verify_lemma_B_degree_concentration.json"

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
    ("P5N600", 600, "results_d1_p5n600_12seeds/P5N600.snapshots.npz",   "edge_xi_snapshots"),
    ("P5N700", 700, "results_d1_p5n700_12seeds/P5N700.snapshots.npz",   "edge_xi_snapshots"),
    ("P5N800", 800, "results_d1_p5n800_12seeds/P5N800.snapshots.npz",   "edge_xi_snapshots"),
]

BOOT_RESAMPLES = 1000
RNG_SEED = 42


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


def degree_stats(xi: np.ndarray) -> dict[str, float]:
    """Per-snapshot degree statistics.

    deg(i) = sum_{j != i} max(Xi_N(i,j), 0)  (clamp neg → 0)
    """
    w = xi - np.eye(xi.shape[0])
    w = np.maximum(w, 0.0)
    deg = w.sum(axis=1)  # shape (N,)
    mean = float(deg.mean())
    var = float(deg.var(ddof=1)) if deg.size > 1 else 0.0
    std = float(deg.std(ddof=1)) if deg.size > 1 else 0.0
    if mean <= 0:
        return {
            "mean_deg": mean, "std_deg": std, "var_deg": var,
            "CV_deg": float("nan"),
            "NSM_deg": float("nan"),
            "min_deg": float(deg.min()), "max_deg": float(deg.max()),
        }
    return {
        "mean_deg": mean,
        "std_deg": std,
        "var_deg": var,
        "CV_deg": std / mean,
        "NSM_deg": var / (mean ** 2),
        "min_deg": float(deg.min()),
        "max_deg": float(deg.max()),
    }


def bootstrap_ci(values: list[float], n_boot: int,
                 rng: np.random.Generator) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def _aicc(sse: float, n: int, k: int) -> float:
    return (n * np.log(sse / n + 1e-30) + 2 * k
            + 2 * k * (k + 1) / max(n - k - 1, 1))


def fit_concentration_scaling(per_regime: list[dict[str, Any]],
                              diag_key: str) -> dict[str, Any]:
    """AICc fit over {const, linear, power_law, Symanzik-1}.

    For a concentration metric we expect either:
      - const: metric is N-independent (no concentration)
      - power_law (with positive exponent on 1/N): metric -> 0
      - Symanzik-1: metric = c_inf + a/N (asymptotic constant)

    The preferred model + sign of the asymptote determines whether
    degree concentration holds.
    """
    valid = [(r["N"], r[f"{diag_key}_mean"]) for r in per_regime
             if r.get(f"{diag_key}_mean") is not None
             and np.isfinite(r[f"{diag_key}_mean"])]
    if len(valid) < 3:
        return {"verdict": "INSUFFICIENT_DATA", "n_points": len(valid)}
    n_arr = np.array([v[0] for v in valid], dtype=float)
    y = np.array([v[1] for v in valid], dtype=float)
    n_pts = len(valid)
    models: dict[str, dict[str, Any]] = {}

    # const: y = c
    c_const = float(y.mean())
    sse = float(((y - c_const) ** 2).sum())
    models["const"] = {"params": {"c": c_const}, "SSE": sse,
                       "AICc": _aicc(sse, n_pts, 1), "k": 1}

    # power_law: y = c * N^(-alpha)  (log-log lstsq)
    A = np.column_stack([np.ones_like(n_arr), -np.log(n_arr)])
    sol, *_ = np.linalg.lstsq(A, np.log(y + 1e-30), rcond=None)
    c_pow, alpha_pow = float(np.exp(sol[0])), float(sol[1])
    y_pred = c_pow * n_arr ** (-alpha_pow)
    sse = float(((y - y_pred) ** 2).sum())
    models["power_law"] = {"params": {"c": c_pow, "alpha": alpha_pow},
                           "SSE": sse, "AICc": _aicc(sse, n_pts, 2), "k": 2}

    # Symanzik-1: y = c_inf + a/N
    A = np.column_stack([np.ones_like(n_arr), 1.0 / n_arr])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    c_inf, a_sym = float(sol[0]), float(sol[1])
    y_pred = c_inf + a_sym / n_arr
    sse = float(((y - y_pred) ** 2).sum())
    models["symanzik_1"] = {"params": {"c_inf": c_inf, "a": a_sym},
                            "SSE": sse, "AICc": _aicc(sse, n_pts, 2),
                            "k": 2}

    best = min(models, key=lambda k: models[k]["AICc"])
    deltas = {k: models[k]["AICc"] - models[best]["AICc"] for k in models}

    # Determine asymptote
    if best == "const":
        asym = models[best]["params"]["c"]
        concentration_holds = False  # not approaching zero
    elif best == "power_law":
        # power-law -> 0 iff alpha > 0
        alpha = models[best]["params"]["alpha"]
        asym = 0.0 if alpha > 0 else models[best]["params"]["c"]
        concentration_holds = bool(alpha > 0)
    elif best == "symanzik_1":
        asym = models[best]["params"]["c_inf"]
        # Symanzik-1 -> c_inf; concentration iff c_inf <= 0 or
        # very small relative to data
        concentration_holds = (asym < y.min() * 0.5)
    else:
        asym, concentration_holds = float("nan"), False

    return {
        "n_points": n_pts,
        "models": models,
        "delta_AICc": deltas,
        "preferred_model": best,
        "asymptote": asym,
        "concentration_holds": bool(concentration_holds),
    }


def audit_regime(regime: str, n_lat: int, rel: str, hint: str,
                 rng: np.random.Generator) -> dict[str, Any]:
    npz = REPO_ROOT / rel
    xis = load_all_xi(npz, hint)
    if not xis:
        return {"regime": regime, "N": n_lat,
                "n_seeds_loaded": 0,
                "status": "SNAPSHOT_NOT_AVAILABLE"}
    diags = [degree_stats(xi) for xi in xis]
    diags = [d for d in diags if np.isfinite(d.get("CV_deg", float("nan")))]
    if not diags:
        return {"regime": regime, "N": n_lat,
                "n_seeds_loaded": len(xis),
                "status": "ALL_SEEDS_DEGENERATE"}
    out = {"regime": regime, "N": n_lat,
           "n_seeds_loaded": len(xis),
           "n_seeds_valid": len(diags),
           "status": "OK"}
    for key in ("mean_deg", "std_deg", "CV_deg", "NSM_deg",
                "min_deg", "max_deg"):
        vals = [d[key] for d in diags]
        out[f"{key}_per_seed"] = vals
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci = bootstrap_ci(vals, BOOT_RESAMPLES, rng)
        out[f"{key}_ci95"] = [ci[0], ci[1]]
    return out


def main():
    rng = np.random.default_rng(RNG_SEED)
    per_regime = [audit_regime(reg, n_lat, rel, hint, rng)
                  for reg, n_lat, rel, hint in LADDER]
    fits = {
        "CV_deg": fit_concentration_scaling(per_regime, "CV_deg"),
        "NSM_deg": fit_concentration_scaling(per_regime, "NSM_deg"),
        "mean_deg": fit_concentration_scaling(per_regime, "mean_deg"),
    }

    # Verdict: degree concentration supported iff CV_deg or NSM_deg
    # asymptote is approaching zero (positive alpha or small c_inf).
    nsm = fits["NSM_deg"]
    cv = fits["CV_deg"]
    concentration_supported = (
        nsm.get("concentration_holds", False)
        or cv.get("concentration_holds", False)
    )

    out = {
        "headline": ("Lemma B Phase-2 Step 2: degree-concentration "
                     "audit on the 10-regime canonical-physics "
                     "ladder. Cross-seed mean + CI95 of deg(i), "
                     "CV_deg, NSM_deg = Var(deg)/Mean(deg)^2 per "
                     "regime, with AICc-ranked N-scaling fit."),
        "method": (
            "deg(i) = sum_{j != i} max(Xi_N(i,j), 0) for each "
            "vertex i. Per-snapshot: mean, std, CV = std/mean, "
            "NSM = var/mean^2. Per regime: cross-seed mean + "
            "bootstrap-95% CI. N-scaling: AICc over {const, "
            "power_law, Symanzik-1}."),
        "boot_resamples": BOOT_RESAMPLES,
        "rng_seed": RNG_SEED,
        "per_regime": per_regime,
        "scaling_fits": fits,
        "degree_concentration_supported": bool(concentration_supported),
        "verdict": (
            "DEGREE_CONCENTRATION_SUPPORTED — Route 1+ "
            "(quantitative Cheeger via degree concentration) "
            "remains viable for Phase 2"
            if concentration_supported
            else "DEGREE_CONCENTRATION_NOT_SUPPORTED — Route 1+ "
                 "also fails; analytical route requires further "
                 "revision"
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_summary(per_regime, fits, out["verdict"])
    return 0


def print_summary(per_regime, fits, verdict):
    print("=" * 78)
    print("Lemma B Phase-2 Step 2: Degree-Concentration empirical test")
    print("=" * 78)
    print(f"{'Regime':<8} {'N':>4} {'seeds':>6} "
          f"{'mean deg':>10} {'CV_deg':>10} {'NSM_deg':>14}")
    print("-" * 78)
    for r in per_regime:
        if r["status"] != "OK":
            print(f"{r['regime']:<8} {r['N']:>4}  {r['status']}")
            continue
        print(f"{r['regime']:<8} {r['N']:>4} "
              f"{r['n_seeds_valid']:>6} "
              f"{r['mean_deg_mean']:>10.4f} "
              f"{r['CV_deg_mean']:>10.5f} "
              f"{r['NSM_deg_mean']:>14.6e}")
    print()
    print("N-scaling fits (per diagnostic):")
    for diag_name, fit in fits.items():
        if fit.get("verdict") == "INSUFFICIENT_DATA":
            print(f"  {diag_name}: INSUFFICIENT_DATA")
            continue
        best = fit["preferred_model"]
        params = ", ".join(f"{pn}={pv:.4e}"
                           for pn, pv in fit["models"][best]["params"].items())
        holds = "CONCENTRATES" if fit["concentration_holds"] else "STATIC"
        print(f"  {diag_name:<10s} preferred = {best:<10s}  {params:<45s}  "
              f"-> {holds}, asymptote = {fit['asymptote']:.4e}")
    print()
    print(f"Verdict: {verdict}")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
