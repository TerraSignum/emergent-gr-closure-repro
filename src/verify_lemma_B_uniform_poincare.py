"""Lemma B (uniform Poincare / spectral-gap) empirical certification.

This script is Phase 1 of the program to upgrade the conditional
master closure theorem (P6) into an unconditional continuum theorem
by deriving admissibility conditions A1+A8 from a uniform spectral
gap on the Xi-weighted graph Laplacian.

The corpus already contains:
  - verify_admissibility_counterexample_and_spectral_gap.py:
    constructive counterexample (constant-Xi C_N(alpha) violates A8)
    + first-seed lambda_2 audit on a 4-regime sub-ladder.

This script extends that work with:
  - cross-seed certification (all 24 / 12 / 28 seeds per regime,
    not just first); reports mean lambda_2 + bootstrap 95% CI;
  - 10-regime ladder N in {50, 64, 72, 84, 100, 128, 200, 256, 300, 512};
  - explicit Poincare-constant C_P = 1/lambda_2;
  - N-scaling fit: const vs N^(-alpha) AICc selection;
  - per-seed degenerate-Laplacian filter (drops regimes where W has
    a zero row).

Statement audited (target Lemma B):

  Let {(V_N, Xi_N, mu_N)}_{N>=N_0} be an admissible relational-
  carrier sequence (M0-M3, Xi >= xi_min > 0). Then there exists a
  uniform spectral-gap constant lambda_* > 0 such that

    lambda_2(L_N) >= lambda_*    for all N >= N_0,

  where L_N = I - D_N^{-1/2} W_N D_N^{-1/2} is the symmetric
  normalised Xi-weighted graph Laplacian, with W_N = Xi_N - I
  and D_N = diag(W_N @ 1). Equivalently the Poincare constant
  C_P = 1/lambda_* < infty.

Output:
  outputs/verify_lemma_B_uniform_poincare.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REPO_ROOT = REPO.parent
OUT = REPO / "outputs" / "verify_lemma_B_uniform_poincare.json"

# Canonical-physics ladder. Each row is
# (regime label, N, snapshot_npz, snapshot_loader_hint).
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
    # Matter-branch lever-arm extension (Phase-3, FB-w4 test).
    # Auto-included if the corresponding npz exists.
    ("P5N600", 600, "results_d1_p5n600_12seeds/P5N600.snapshots.npz", "edge_xi_snapshots"),
    ("P5N700", 700, "results_d1_p5n700_12seeds/P5N700.snapshots.npz", "edge_xi_snapshots"),
    ("P5N800", 800, "results_d1_p5n800_12seeds/P5N800.snapshots.npz", "edge_xi_snapshots"),
]

BOOT_RESAMPLES = 1000
RNG_SEED = 42


# ---------------------------------------------------------------
# Snapshot loaders
# ---------------------------------------------------------------

def load_all_xi(npz_path: Path, hint: str) -> list[np.ndarray]:
    """Return a list of Xi matrices (one per seed; or the last
    timestep of each seed, when timesteps are tracked).

    Diagonal is set to 1.0 in all cases.
    """
    if not npz_path.exists():
        return []
    z = np.load(npz_path, allow_pickle=True)

    matrices: list[np.ndarray] = []

    if hint == "edge_xi_snapshots" and "edge_xi_snapshots" in z.files:
        # shape (n_seeds, n_timesteps, n, n) — take last timestep per seed.
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


# ---------------------------------------------------------------
# Spectral-gap of Xi-weighted normalised Laplacian
# ---------------------------------------------------------------

def lambda2_normalised_laplacian(xi: np.ndarray) -> float | None:
    """Smallest non-zero eigenvalue of
       L = I - D^{-1/2} W D^{-1/2},  W = Xi - I,  D = diag(W @ 1).
    Spectrum in [0, 2]; returns None if degenerate (some row sums
    to zero). Symmetrised numerically before eigvalsh.
    """
    n = xi.shape[0]
    w = xi - np.eye(n)
    w = np.maximum(w, 0.0)
    deg = w.sum(axis=1)
    if np.any(deg <= 1e-12):
        return None
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    norm_w = w * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    laplacian = np.eye(n) - norm_w
    laplacian = 0.5 * (laplacian + laplacian.T)
    eigs = np.linalg.eigvalsh(laplacian)
    return float(eigs[1])


def bootstrap_ci(values: list[float], n_boot: int, rng: np.random.Generator
                 ) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


# ---------------------------------------------------------------
# N-scaling fit: lambda_2(N) = c (const) vs c * N^(-alpha)
# ---------------------------------------------------------------

def _aicc(sse: float, n: int, k: int) -> float:
    return (n * np.log(sse / n + 1e-30) + 2 * k
            + 2 * k * (k + 1) / max(n - k - 1, 1))


def fit_scaling(per_regime: list[dict[str, Any]]) -> dict[str, Any]:
    """AICc comparison across five competing N-scaling models for
    lambda_2(N). The Symanzik-1 model lambda = lambda_inf + a/N is
    the natural finite-size correction for a quantity with a
    positive continuum limit; we report all five.
    """
    valid = [(r["N"], r["lambda_2_mean"]) for r in per_regime
             if r["lambda_2_mean"] is not None]
    if len(valid) < 3:
        return {"verdict": "INSUFFICIENT_DATA", "n_points": len(valid)}

    n_arr = np.array([v[0] for v in valid], dtype=float)
    y = np.array([v[1] for v in valid], dtype=float)
    n_pts = len(valid)

    models: dict[str, dict[str, Any]] = {}

    # 1) const: y = c
    c_const = float(y.mean())
    sse = float(((y - c_const) ** 2).sum())
    models["const"] = {"params": {"c": c_const}, "SSE": sse,
                       "AICc": _aicc(sse, n_pts, 1), "k": 1}

    # 2) power_law: y = c * N^(-alpha)  (log-log linear fit)
    A = np.column_stack([np.ones_like(n_arr), -np.log(n_arr)])
    sol, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    c_pow, alpha_pow = float(np.exp(sol[0])), float(sol[1])
    y_pred = c_pow * n_arr ** (-alpha_pow)
    sse = float(((y - y_pred) ** 2).sum())
    models["power_law"] = {"params": {"c": c_pow, "alpha": alpha_pow},
                           "SSE": sse, "AICc": _aicc(sse, n_pts, 2), "k": 2}

    # 3) Symanzik-1: y = lambda_inf + a / N
    A = np.column_stack([np.ones_like(n_arr), 1.0 / n_arr])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    lam_inf1, a1 = float(sol[0]), float(sol[1])
    y_pred = lam_inf1 + a1 / n_arr
    sse = float(((y - y_pred) ** 2).sum())
    models["symanzik_1"] = {"params": {"lambda_inf": lam_inf1, "a": a1},
                            "SSE": sse, "AICc": _aicc(sse, n_pts, 2), "k": 2}

    # 4) Symanzik-2: y = lambda_inf + a / N + b / N^2
    A = np.column_stack([np.ones_like(n_arr), 1.0 / n_arr, 1.0 / n_arr ** 2])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    lam_inf2, a2, b2 = float(sol[0]), float(sol[1]), float(sol[2])
    y_pred = lam_inf2 + a2 / n_arr + b2 / n_arr ** 2
    sse = float(((y - y_pred) ** 2).sum())
    models["symanzik_2"] = {"params": {"lambda_inf": lam_inf2, "a": a2, "b": b2},
                            "SSE": sse, "AICc": _aicc(sse, n_pts, 3), "k": 3}

    # 5) Symanzik-1/2: y = lambda_inf + a / sqrt(N)
    A = np.column_stack([np.ones_like(n_arr), 1.0 / np.sqrt(n_arr)])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    lam_inf3, a3 = float(sol[0]), float(sol[1])
    y_pred = lam_inf3 + a3 / np.sqrt(n_arr)
    sse = float(((y - y_pred) ** 2).sum())
    models["symanzik_half"] = {"params": {"lambda_inf": lam_inf3, "a": a3},
                               "SSE": sse, "AICc": _aicc(sse, n_pts, 2), "k": 2}

    best = min(models, key=lambda k: models[k]["AICc"])
    best_aicc = models[best]["AICc"]
    deltas = {k: models[k]["AICc"] - best_aicc for k in models}

    asymptote = None
    if best in ("const",):
        asymptote = models[best]["params"]["c"]
    elif best.startswith("symanzik"):
        asymptote = models[best]["params"]["lambda_inf"]
    elif best == "power_law":
        # power law has no finite asymptote (alpha > 0) or constant (alpha == 0)
        asymptote = 0.0 if models[best]["params"]["alpha"] > 1e-3 else \
                    models[best]["params"]["c"]

    asymptote_pos = (asymptote is not None and asymptote > 1e-3)

    return {
        "n_points": n_pts,
        "models": models,
        "delta_AICc": deltas,
        "preferred_model": best,
        "asymptote_lambda_inf": asymptote,
        "asymptote_strictly_positive": bool(asymptote_pos),
        "verdict": (
            f"PREFERRED_{best.upper()}; "
            f"continuum asymptote lambda_inf = {asymptote:.4f}"
            if asymptote is not None and asymptote_pos
            else f"PREFERRED_{best.upper()}; ASYMPTOTE_NOT_STRICTLY_POSITIVE"
        ),
    }


# ---------------------------------------------------------------
# Per-regime audit + helpers
# ---------------------------------------------------------------

def _empty_record(regime: str, n_lat: int, n_seeds: int, status: str
                  ) -> dict[str, Any]:
    return {
        "regime": regime, "N": n_lat,
        "n_seeds_loaded": n_seeds,
        "lambda_2_per_seed": None if n_seeds == 0 else [],
        "lambda_2_mean": None,
        "lambda_2_min": None,
        "lambda_2_ci95": None,
        "status": status,
    }


def _record_from_lambdas(regime: str, n_lat: int, n_seeds: int,
                         lambdas: list[float],
                         rng: np.random.Generator) -> dict[str, Any]:
    mean = float(np.mean(lambdas))
    lam_min = float(np.min(lambdas))
    lam_max = float(np.max(lambdas))
    lam_std = float(np.std(lambdas, ddof=1)) if len(lambdas) > 1 else 0.0
    ci = bootstrap_ci(lambdas, BOOT_RESAMPLES, rng)
    return {
        "regime": regime, "N": n_lat,
        "n_seeds_loaded": n_seeds,
        "n_seeds_valid": len(lambdas),
        "lambda_2_per_seed": lambdas,
        "lambda_2_mean": mean,
        "lambda_2_std": lam_std,
        "lambda_2_min": lam_min,
        "lambda_2_max": lam_max,
        "lambda_2_ci95": [ci[0], ci[1]],
        "poincare_constant_max": 1.0 / lam_min,
        "poincare_constant_mean": 1.0 / mean,
        "status": "OK",
    }


def audit_regime(regime: str, n_lat: int, rel: str, hint: str,
                 rng: np.random.Generator) -> dict[str, Any]:
    """Audit a single ladder regime: load all seeds, compute lambda_2,
    return a per-regime record."""
    npz = REPO_ROOT / rel
    xis = load_all_xi(npz, hint)
    if not xis:
        return _empty_record(regime, n_lat, 0, "SNAPSHOT_NOT_AVAILABLE")
    lambdas = [lam for xi in xis
               if (lam := lambda2_normalised_laplacian(xi)) is not None]
    if not lambdas:
        return _empty_record(regime, n_lat, len(xis), "ALL_SEEDS_DEGENERATE")
    return _record_from_lambdas(regime, n_lat, len(xis), lambdas, rng)


def compute_uniform_lower_bound(per_regime: list[dict[str, Any]]
                                ) -> dict[str, Any]:
    """Cross-regime uniform lambda_*: minimum lambda_2 observed, the
    regime where it occurred, and the corresponding C_P upper bound."""
    valid = [r for r in per_regime if r["lambda_2_min"] is not None]
    if not valid:
        return {
            "lambda_star_min_across_ladder": None,
            "lambda_star_worst_regime": None,
            "poincare_constant_max": None,
        }
    lambda_star = float(min(r["lambda_2_min"] for r in valid))
    worst = min(valid, key=lambda r: r["lambda_2_mean"])
    return {
        "lambda_star_min_across_ladder": lambda_star,
        "lambda_star_worst_regime": worst["regime"],
        "poincare_constant_max": 1.0 / lambda_star,
    }


def build_verdict(uniform: dict[str, Any], scaling: dict[str, Any]) -> str:
    """Translate (uniform_lower_bound, scaling fit) into a string verdict."""
    lam_star = uniform.get("lambda_star_min_across_ladder")
    if lam_star is None:
        return "INSUFFICIENT_DATA"
    asym = scaling.get("asymptote_lambda_inf")
    if asym is None or asym <= 1e-3:
        return "ASYMPTOTE_NOT_STRICTLY_POSITIVE"
    return (f"UNIFORM_SPECTRAL_GAP_CERTIFIED "
            f"(lambda_inf = {asym:.4f}, lambda_* = {lam_star:.4f}, "
            f"preferred = {scaling.get('preferred_model')})")


# ---------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------

def main():
    rng = np.random.default_rng(RNG_SEED)
    per_regime = [audit_regime(reg, n_lat, rel, hint, rng)
                  for reg, n_lat, rel, hint in LADDER]
    uniform = compute_uniform_lower_bound(per_regime)
    scaling = fit_scaling(per_regime)
    verdict = build_verdict(uniform, scaling)

    out = {
        "headline": ("Lemma B (uniform spectral gap) empirical "
                     "certification: cross-seed mean and bootstrap CI "
                     "of lambda_2(L_N) on the 10-regime canonical-physics "
                     "ladder N in {50,64,72,84,100,128,200,256,300,512}."),
        "method": (
            "L_N = I - D_N^{-1/2} W_N D_N^{-1/2} with W_N = Xi_N - I "
            "(off-diagonal Xi) and D_N = diag(W_N @ 1). lambda_2 = "
            "smallest non-zero eigenvalue. Per-regime mean, std, min, "
            "max, and bootstrap-95% CI across all available seeds "
            "(snapshot last-timestep per seed)."),
        "boot_resamples": BOOT_RESAMPLES,
        "rng_seed": RNG_SEED,
        "per_regime": per_regime,
        "uniform_lower_bound": uniform,
        "N_scaling_fit": scaling,
        "verdict": verdict,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print_summary(per_regime, uniform, scaling, out["verdict"])
    return 0


def print_summary(per_regime: list[dict[str, Any]],
                  uniform: dict[str, Any],
                  scaling: dict[str, Any],
                  verdict: str) -> None:
    print("=" * 78)
    print("Lemma B (uniform Poincare / spectral-gap) empirical certification")
    print("=" * 78)
    print(f"{'Regime':<8} {'N':>4} {'seeds':>6} "
          f"{'mean lambda_2':>14} {'min':>8} {'max':>8} "
          f"{'CI95':>22} {'C_P max':>10}")
    print("-" * 78)
    for r in per_regime:
        if r["lambda_2_mean"] is None:
            print(f"{r['regime']:<8} {r['N']:>4} {'-':>6}  {r['status']}")
            continue
        ci = r["lambda_2_ci95"]
        print(f"{r['regime']:<8} {r['N']:>4} "
              f"{r.get('n_seeds_valid', 0):>6} "
              f"{r['lambda_2_mean']:>14.6f} "
              f"{r['lambda_2_min']:>8.4f} {r['lambda_2_max']:>8.4f} "
              f"[{ci[0]:.4f},{ci[1]:.4f}] "
              f"{r['poincare_constant_max']:>10.3f}")
    print()
    lam_star = uniform["lambda_star_min_across_ladder"]
    if lam_star is not None:
        print(f"Uniform lower bound lambda_* = {lam_star:.6f}  "
              f"(worst regime: {uniform['lambda_star_worst_regime']})")
        print(f"Poincare constant C_P = 1/lambda_* <= "
              f"{uniform['poincare_constant_max']:.4f}")
    print()
    print("N-scaling fit (5 models, AICc-ranked):")
    for k, m in sorted(scaling["models"].items(),
                       key=lambda kv: kv[1]["AICc"]):
        params = ", ".join(f"{pn}={pv:.4f}" for pn, pv in m["params"].items())
        delta = scaling["delta_AICc"][k]
        marker = "*" if k == scaling["preferred_model"] else " "
        print(f"  {marker} {k:<14s} {params:<48s} AICc={m['AICc']:7.2f}  "
              f"DeltaAICc=+{delta:5.2f}")
    asym = scaling.get("asymptote_lambda_inf")
    if asym is not None:
        print(f"\n  continuum asymptote lambda_inf = {asym:.4f}, "
              f"C_P^inf = {1.0/asym:.4f}")
    print()
    print(f"Verdict: {verdict}")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
