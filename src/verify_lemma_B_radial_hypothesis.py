"""Lemma B Phase-2 Step 1: empirical test of the radial hypothesis.

Complementary to (NOT redundant with) the earlier
`verify_xi_gram_spectral_gap_scaling.py`, which tests the
Wigner-Dyson RMT classification of Xi_N's spectrum (gap-ratio
statistic, Poisson/GOE/GUE) on the P0..P8 cross-regime ladder.
This script asks a different question on the P5/P5N
canonical-physics ladder: whether the singular spectrum of
Xi_N concentrates on a bounded number of dominant components
(low-effective-rank radial structure), which is the structural
property required by the spectral-synthesis route for Lemma B.

The proof-strategy survey (notes/lemma_B_proof_strategy.md) identified
Route 3 (radial spectral synthesis via low-effective-dimension)
as the recommended analytical route for Lemma B. The earlier
"radial" formulation Xi_N(i,j) = f(d_N(i,j)) is tautological
because d_N(i,j) := -log Xi_N(i,j) by definition. The
non-tautological radial-hypothesis is:

  Radial Hypothesis (effective-dimension version):
    The carrier-action UV-closure produces Xi_N matrices whose
    singular-value spectrum concentrates on a number k of
    dominant components, where k = k(d, N_gen, xi_min) is
    bounded uniformly in N.

If the hypothesis holds:
  - Vertices admit an embedding in R^k via the dominant left/right
    singular vectors.
  - The Markov chain P_N = D^{-1} W with W = Xi - I is then a
    "radial" operator on this finite-dimensional embedding.
  - The Laplacian spectrum L_N admits a closed-form expansion in
    terms of the embedding geometry, and the asymptote
    lambda_inf becomes algebraically computable from the System-R
    rationals.

The hypothesis is operationalised through three effective-rank
diagnostics, computed per snapshot and reported with cross-seed
mean + bootstrap-95% CI per regime:

  (R1) r_95(Xi_N): smallest k such that sum_{i<=k} sigma_i^2 /
       sum_i sigma_i^2 >= 0.95.

  (R2) r_99(Xi_N): same at 99% energy threshold.

  (R3) Participation ratio:
       PR = (sum sigma_i^2)^2 / sum sigma_i^4
       (= effective number of components, dimensionless).

For radial hypothesis to hold uniformly, all three should
saturate (be bounded above) as N grows. We fit each effective-
rank diagnostic to four candidate scaling forms and AICc-rank:

  - const  : r(N) = r_inf
  - linear : r(N) = r_inf + b * N
  - log    : r(N) = r_inf + b * log(N)
  - sqrt   : r(N) = r_inf + b * sqrt(N)

If the const model wins, the radial hypothesis is empirically
supported and Route 3 is unlocked. If a non-constant model wins,
the effective rank diverges and Route 3 needs revision.

Output: outputs/verify_lemma_B_radial_hypothesis.json
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REPO_ROOT = REPO.parent
OUT = REPO / "outputs" / "verify_lemma_B_radial_hypothesis.json"

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

BOOT_RESAMPLES = 1000
RNG_SEED = 42


# ---------------------------------------------------------------
# Snapshot loaders (shared with verify_lemma_B_uniform_poincare)
# ---------------------------------------------------------------

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


# ---------------------------------------------------------------
# Effective-rank diagnostics
# ---------------------------------------------------------------

def effective_ranks(xi: np.ndarray) -> dict[str, float]:
    """Compute r_95, r_99, and participation-ratio (PR) effective
    ranks from the SVD spectrum of xi.

    All three quantify how concentrated the singular-value spectrum
    is on a low-dimensional subspace.
    """
    # SVD spectrum (singular values, descending)
    try:
        sv = np.linalg.svd(xi, compute_uv=False)
    except np.linalg.LinAlgError:
        return {"r_95": float("nan"), "r_99": float("nan"),
                "PR": float("nan")}
    energy = sv ** 2
    total = float(energy.sum())
    if total <= 0:
        return {"r_95": float("nan"), "r_99": float("nan"),
                "PR": float("nan")}
    cum = np.cumsum(energy) / total
    r_95 = int(np.searchsorted(cum, 0.95) + 1)
    r_99 = int(np.searchsorted(cum, 0.99) + 1)
    # Participation ratio: dimensionless effective rank
    pr = float(total ** 2 / (energy ** 2).sum())
    # Top singular-value share (sigma_1^2 / total)
    top_share = float(energy[0] / total)
    return {
        "r_95": float(r_95),
        "r_99": float(r_99),
        "PR": pr,
        "top_singular_share": top_share,
        "sigma_top5": [float(s) for s in sv[:5]],
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


# ---------------------------------------------------------------
# N-scaling fit (4 models, AICc-ranked)
# ---------------------------------------------------------------

def _aicc(sse: float, n: int, k: int) -> float:
    return (n * np.log(sse / n + 1e-30) + 2 * k
            + 2 * k * (k + 1) / max(n - k - 1, 1))


def fit_rank_scaling(per_regime: list[dict[str, Any]],
                     diag_key: str) -> dict[str, Any]:
    """AICc model selection on the cross-seed mean of `diag_key`
    across regimes. Returns coefficients of four candidate models
    and the preferred model.
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

    # const: y = r_inf
    r_const = float(y.mean())
    sse = float(((y - r_const) ** 2).sum())
    models["const"] = {"params": {"r_inf": r_const}, "SSE": sse,
                       "AICc": _aicc(sse, n_pts, 1), "k": 1}

    # linear: y = r_inf + b * N
    A = np.column_stack([np.ones_like(n_arr), n_arr])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    r_lin, b_lin = float(sol[0]), float(sol[1])
    y_pred = r_lin + b_lin * n_arr
    sse = float(((y - y_pred) ** 2).sum())
    models["linear"] = {"params": {"r_inf": r_lin, "b": b_lin},
                        "SSE": sse, "AICc": _aicc(sse, n_pts, 2),
                        "k": 2}

    # log: y = r_inf + b * log(N)
    A = np.column_stack([np.ones_like(n_arr), np.log(n_arr)])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    r_log, b_log = float(sol[0]), float(sol[1])
    y_pred = r_log + b_log * np.log(n_arr)
    sse = float(((y - y_pred) ** 2).sum())
    models["log"] = {"params": {"r_inf": r_log, "b": b_log},
                     "SSE": sse, "AICc": _aicc(sse, n_pts, 2),
                     "k": 2}

    # sqrt: y = r_inf + b * sqrt(N)
    A = np.column_stack([np.ones_like(n_arr), np.sqrt(n_arr)])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    r_sqrt, b_sqrt = float(sol[0]), float(sol[1])
    y_pred = r_sqrt + b_sqrt * np.sqrt(n_arr)
    sse = float(((y - y_pred) ** 2).sum())
    models["sqrt"] = {"params": {"r_inf": r_sqrt, "b": b_sqrt},
                      "SSE": sse, "AICc": _aicc(sse, n_pts, 2),
                      "k": 2}

    best = min(models, key=lambda k: models[k]["AICc"])
    best_aicc = models[best]["AICc"]
    deltas = {k: models[k]["AICc"] - best_aicc for k in models}
    bounded = (best == "const") or (
        best in ("log", "sqrt", "linear")
        and models[best]["params"]["b"] < 0.01 * y.mean() / n_arr.max())
    return {
        "n_points": n_pts,
        "models": models,
        "delta_AICc": deltas,
        "preferred_model": best,
        "effective_rank_bounded": bool(bounded),
    }


# ---------------------------------------------------------------
# Per-regime audit
# ---------------------------------------------------------------

def audit_regime(regime: str, n_lat: int, rel: str, hint: str,
                 rng: np.random.Generator) -> dict[str, Any]:
    npz = REPO_ROOT / rel
    xis = load_all_xi(npz, hint)
    if not xis:
        return {
            "regime": regime, "N": n_lat,
            "n_seeds_loaded": 0,
            "status": "SNAPSHOT_NOT_AVAILABLE",
        }
    diags = [effective_ranks(xi) for xi in xis]
    diags = [d for d in diags if np.isfinite(d.get("r_95", float("nan")))]
    if not diags:
        return {
            "regime": regime, "N": n_lat,
            "n_seeds_loaded": len(xis),
            "status": "ALL_SEEDS_DEGENERATE",
        }
    out = {
        "regime": regime, "N": n_lat,
        "n_seeds_loaded": len(xis),
        "n_seeds_valid": len(diags),
        "status": "OK",
    }
    for key in ("r_95", "r_99", "PR", "top_singular_share"):
        vals = [d[key] for d in diags]
        out[f"{key}_per_seed"] = vals
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out[f"{key}_min"] = float(np.min(vals))
        out[f"{key}_max"] = float(np.max(vals))
        ci = bootstrap_ci(vals, BOOT_RESAMPLES, rng)
        out[f"{key}_ci95"] = [ci[0], ci[1]]
    return out


def main():
    rng = np.random.default_rng(RNG_SEED)
    per_regime = [audit_regime(reg, n_lat, rel, hint, rng)
                  for reg, n_lat, rel, hint in LADDER]

    fits = {
        "r_95": fit_rank_scaling(per_regime, "r_95"),
        "r_99": fit_rank_scaling(per_regime, "r_99"),
        "PR":   fit_rank_scaling(per_regime, "PR"),
        "top_singular_share": fit_rank_scaling(per_regime,
                                               "top_singular_share"),
    }

    # Overall verdict: radial hypothesis supported iff at least
    # r_95 and PR are bounded (top_singular_share and r_99 are
    # secondary diagnostics).
    radial_supported = (
        fits["r_95"].get("effective_rank_bounded", False)
        and fits["PR"].get("effective_rank_bounded", False)
    )

    out = {
        "headline": ("Lemma B Phase-2 Step 1: empirical test of the "
                     "radial hypothesis via effective-rank scaling "
                     "of Xi_N on the 10-regime canonical-physics "
                     "ladder N in {50,64,72,84,100,128,200,256,300,512}."),
        "method": (
            "For each snapshot Xi_N (last timestep per seed, all "
            "available seeds), compute the SVD spectrum and three "
            "effective-rank diagnostics: r_95 (95%-energy cutoff), "
            "r_99 (99%-energy cutoff), and participation ratio PR "
            "= (sum sigma^2)^2 / sum sigma^4. Per regime: cross-seed "
            "mean + bootstrap-95% CI. AICc model selection on the "
            "N-scaling of each diagnostic across the four candidate "
            "forms {const, linear, log, sqrt} identifies whether "
            "effective rank is bounded uniformly in N."),
        "boot_resamples": BOOT_RESAMPLES,
        "rng_seed": RNG_SEED,
        "per_regime": per_regime,
        "scaling_fits": fits,
        "radial_hypothesis_supported": bool(radial_supported),
        "verdict": (
            "RADIAL_HYPOTHESIS_SUPPORTED — Route 3 (Hardy / radial "
            "spectral synthesis) unlocked for Phase 2"
            if radial_supported
            else "RADIAL_HYPOTHESIS_NOT_SUPPORTED — Route 3 needs "
                 "revision; effective rank diverges with N"
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_summary(per_regime, fits, out["verdict"])
    return 0


def print_summary(per_regime: list[dict[str, Any]],
                  fits: dict[str, dict[str, Any]],
                  verdict: str) -> None:
    print("=" * 78)
    print("Lemma B Phase-2 Step 1: Radial Hypothesis empirical test")
    print("=" * 78)
    print(f"{'Regime':<8} {'N':>4} {'seeds':>6} "
          f"{'r_95':>8} {'r_99':>8} {'PR':>10} {'sigma_top_share':>16}")
    print("-" * 78)
    for r in per_regime:
        if r["status"] != "OK":
            print(f"{r['regime']:<8} {r['N']:>4}  {r['status']}")
            continue
        print(f"{r['regime']:<8} {r['N']:>4} "
              f"{r['n_seeds_valid']:>6} "
              f"{r['r_95_mean']:>8.2f} "
              f"{r['r_99_mean']:>8.2f} "
              f"{r['PR_mean']:>10.3f} "
              f"{r['top_singular_share_mean']:>16.4f}")
    print()
    print("N-scaling fits (per diagnostic):")
    for diag_name, fit in fits.items():
        if fit.get("verdict") == "INSUFFICIENT_DATA":
            print(f"  {diag_name}: INSUFFICIENT_DATA")
            continue
        best = fit["preferred_model"]
        params = ", ".join(f"{pn}={pv:.3f}"
                           for pn, pv in fit["models"][best]["params"].items())
        bounded = "BOUNDED" if fit.get("effective_rank_bounded") else "DIVERGES"
        print(f"  {diag_name:<22s} preferred = {best:<6s}  {params:<35s}  "
              f"-> {bounded}")
    print()
    print(f"Verdict: {verdict}")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
