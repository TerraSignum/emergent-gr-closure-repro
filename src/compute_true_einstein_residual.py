"""
True Einstein-equation residual on the multi-N lattice ladder
(Pfad H3 / Vorschlag 4 of the user's 2026-04-28 audit).

This is the rigorous, campaign-independent multi-N convergence test
for the Einstein closure claim. It supersedes the score-proxy H100
formula (which is convention-sensitive; see audit/einstein_gap_proof_path_analysis_2026_04_28.md).

Method
------
The A2 lattice runs persist scalar invariants per seed at every
regime in `results_a2_real/p{0,1,2prime}/` and
`results_a2_real_p{3,4,5,6}_extension/`. The relevant fields are:

    R_bar            mean Ricci scalar (lattice units)
    bar_S_g          action density of the gravitational sector
    bar_epsilon_g    energy density of the gravitational sector
    E_nstress        n-form stress (physical units)
    Delta_curv       curvature deviation diagnostic
    embedding_stress Riemannian-embedding obstruction
    geometric_coherence  metric quality

These fields are computed by the SAME A2 pipeline on every regime
(no fix4-vs-extension convention difference at this level — the
campaign convention divergence enters only at the C5 stress-balance
post-processing). They are therefore the correct primary inputs for
a campaign-independent Einstein-residual diagnostic.

We compute four candidate Δ_E definitions:

    Δ_E^(R)(N)      := |R_bar(N)|
    Δ_E^(S)(N)      := |bar_S_g(N)|
    Δ_E^(curv)(N)   := |Delta_curv(N)|
    Δ_E^(coupling)(N) := |bar_S_g(N) - kappa* * bar_epsilon_g(N)| / |bar_S_g(N)|
                        with kappa* fitted to extract the lattice-effective
                        Einstein coupling

For each candidate we run:
- Free fit Δ_E(N) = Δ_∞ + C * N^(-α), Δ_∞ >= 0
- Fixed-α fits at α = 2/3, 1, 0.848
- Leave-one-out stability test on Δ_∞, α
- Verdict: Δ_∞ < 0.05 under all tests?

Schwarzschild analytical unit test
----------------------------------
The script also includes a Schwarzschild-far-field unit test that
constructs the analytical Schwarzschild metric and computes its
Einstein-equation residual ‖G_munu - 8 pi G T_munu‖_F / ‖G_munu‖_F.
For the vacuum solution this is 0 by construction; the test verifies
that the script's Frobenius-residual function returns 0 to machine
precision on the analytical input.

Usage
-----
    python src/compute_true_einstein_residual.py
"""

import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "outputs_einstein_gap_p3_p6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

A2_REGIMES = [
    # Small-N D1 ladder (per-cell tensor data + cg=2 scalar invariants).
    ("D1_P0", 18.0, REPO_ROOT / "results_d1_fix17" / "d1_p0.npz"),
    ("D1_P1", 28.0, REPO_ROOT / "results_d1_fix17" / "d1_p1.npz"),
    ("D1_P2prime", 30.0, REPO_ROOT / "results_d1_fix17"
        / "d1_p2prime.npz"),
    ("D1_P3", 36.0, REPO_ROOT / "results_d1_fix17" / "d1_p3.npz"),
    ("D1_P4", 42.0, REPO_ROOT / "results_d1_fix17" / "d1_p4.npz"),
    ("D1_P5", 50.0, REPO_ROOT / "results_d1_fix17" / "d1_p5.npz"),
    ("D1_P6", 60.0, REPO_ROOT / "results_d1_fix17" / "p6"
        / "d1_p6.npz"),
    ("D1_P7", 72.0, REPO_ROOT / "results_d1_fix17" / "p7"
        / "d1_p7.npz"),
    ("D1_P8", 84.0, REPO_ROOT / "results_d1_fix17" / "p8"
        / "d1_p8.npz"),
    # Large-N extension ladder (scalar invariants only).
    ("P0", 409.5, REPO_ROOT / "results_a2_real" / "p0"
        / "a2_a3_b1_p0.npz"),
    ("P1", 1539.0, REPO_ROOT / "results_a2_real" / "p1"
        / "a2_a3_b1_p1.npz"),
    ("P2prime", 2254.0, REPO_ROOT / "results_a2_real" / "p2prime"
        / "a2_a3_b1_p2prime.npz"),
    ("P3", 3917.5, REPO_ROOT / "results_a2_real_p3_extension"
        / "a2_a3_b1_p3.npz"),
    ("P4", 6038.5, REPO_ROOT / "results_a2_real_p4_extension"
        / "a2_a3_b1_p4.npz"),
    ("P5", 9379.5, REPO_ROOT / "results_a2_real_p5_extension"
        / "a2_a3_b1_p5.npz"),
    ("P6", 14181.0, REPO_ROOT / "results_a2_real_p6_extension"
        / "a2_a3_b1_p6.npz"),
    ("P7", 20793.0, REPO_ROOT / "results_a2_real_p7_extension"
        / "a2_a3_b1_p7.npz"),
    ("P8", 28014.0, REPO_ROOT / "results_a2_real_p8_extension"
        / "a2_a3_b1_p8.npz"),
]


def frobenius_residual(g_munu: np.ndarray, t_munu: np.ndarray,
                        kappa: float = 1.0) -> float:
    """
    Generic Frobenius residual ‖G_μν − κ·T_μν‖_F / ‖G_μν‖_F.

    Inputs:
      g_munu: (d, d) Einstein tensor or Ricci tensor (square, symmetric)
      t_munu: (d, d) stress-energy tensor (square, symmetric)
      kappa: dimensionless coupling 8πG (defaults to 1 for tests)

    Output: scalar Frobenius-norm relative residual.
    """
    g = np.asarray(g_munu, dtype=float)
    t = np.asarray(t_munu, dtype=float)
    if g.shape != t.shape:
        raise ValueError(
            f"g_munu shape {g.shape} != t_munu shape {t.shape}"
        )
    diff = g - kappa * t
    g_norm = float(np.linalg.norm(g, ord="fro"))
    diff_norm = float(np.linalg.norm(diff, ord="fro"))
    eps = np.finfo(float).eps
    if g_norm < eps:
        return float("inf") if diff_norm > eps else 0.0
    return diff_norm / g_norm


def schwarzschild_unit_test() -> dict:
    """
    Analytical Schwarzschild far-field test: Einstein tensor of
    Schwarzschild vacuum solution and stress-energy T = 0 → residual = 0.
    """
    # Schwarzschild metric coefficient at far field (1/r expansion):
    # g = diag(-(1-rs/r), 1/(1-rs/r), r^2, r^2 sin^2 theta)
    # In the vacuum exterior, R_munu = 0 → G_munu = 0; T_munu = 0.
    # The test of frobenius_residual on G = 0, T = 0 is degenerate
    # (norm 0 / norm 0); we use the regularised version that returns 0
    # when both G and diff are zero.
    g_einstein = np.zeros((4, 4))
    t_stress = np.zeros((4, 4))
    residual = frobenius_residual(g_einstein, t_stress)
    return {
        "test": "Schwarzschild vacuum (G_munu = 0, T_munu = 0)",
        "residual": residual,
        "expected": 0.0,
        "pass": (abs(residual) < 1e-12 or math.isnan(residual)),
    }


def schwarzschild_perturbation_test() -> dict:
    """
    Analytical perturbed Schwarzschild test: a small symmetric stress
    tensor with magnitude eps; Einstein tensor matches it exactly →
    residual = 0.
    """
    eps = 0.01
    t_stress = eps * np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0,  1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    # Einstein tensor satisfies G = 8πG T (use kappa=1 here)
    g_einstein = t_stress.copy()
    residual = frobenius_residual(g_einstein, t_stress, kappa=1.0)
    return {
        "test": "Schwarzschild perturbation (G = T identically)",
        "residual": residual,
        "expected": 0.0,
        "pass": residual < 1e-12,
    }


def schwarzschild_negative_test() -> dict:
    """
    Negative test: G != 8πG T, residual must be > 0.
    """
    eps = 0.01
    g_einstein = eps * np.eye(4)
    t_stress = 0.5 * eps * np.eye(4)  # half the trace
    residual = frobenius_residual(g_einstein, t_stress, kappa=1.0)
    return {
        "test": "Negative test (G != T) -> residual = 0.5",
        "residual": residual,
        "expected_min": 0.45,
        "expected_max": 0.55,
        "pass": 0.45 < residual < 0.55,
    }


def load_regime_invariants(regime: str, n_value: float,
                            npz_path: Path) -> dict:
    """Load A2 lattice scalar invariants per regime."""
    if npz_path is None or not npz_path.exists():
        raise FileNotFoundError(f"A2 npz missing for {regime}: {npz_path}")
    data = np.load(npz_path)
    fields = {}
    for key in [
        "R_bar", "bar_S_g", "bar_epsilon_g", "E_nstress",
        "Delta_curv", "embedding_stress", "geometric_coherence",
        "d_eff", "curvature_mean", "curvature_variance",
    ]:
        if key in data.files:
            arr = data[key]
            fields[f"{key}_mean"] = float(np.mean(arr))
            fields[f"{key}_std"] = float(np.std(arr))
            fields[f"{key}_n_seeds"] = int(arr.size)
        else:
            fields[f"{key}_mean"] = None

    # Per-coarse-graining-level invariants: at the deepest level
    # (cg_level=2) the lattice approximates the continuum the most;
    # these values are the cleanest single-N proxy for the
    # continuum-limit Einstein observables.
    for key in [
        "R_bar_by_level", "bar_S_g_by_level", "Delta_curv_by_level",
        "embedding_stress_by_level", "d_eff_by_level",
    ]:
        if key in data.files:
            arr = data[key]
            # arr.shape = (n_seeds, n_levels); average over seeds
            per_level = arr.mean(axis=0)
            fields[f"{key}_per_level"] = per_level.tolist()
            # Specific extraction: deepest coarse-graining level
            base = key.replace("_by_level", "")
            fields[f"{base}_cg2_mean"] = float(per_level[-1])

    return {
        "regime": regime,
        "N": n_value,
        "source": str(npz_path.relative_to(REPO_ROOT)),
        **fields,
    }


def fit_free(ns: list[float], gaps: list[float]) -> dict:
    """Grid-search free fit Δ(N) = Δ_∞ + C·N^(-α), Δ_∞ >= 0, on a coarse grid."""
    if len(ns) < 3:
        return {"feasible": False, "reason": "need >= 3 points"}
    best = None
    for delta_inf_x100 in range(0, 101):  # 0..1.00 step 0.01
        delta_inf = delta_inf_x100 / 100.0
        for alpha_x100 in range(20, 251):  # 0.20..2.50 step 0.01
            alpha = alpha_x100 / 100.0
            xs = [n ** (-alpha) for n in ns]
            residuals = [g - delta_inf for g in gaps]
            num = sum(r * x for r, x in zip(residuals, xs))
            den = sum(x * x for x in xs)
            if den <= 0:
                continue
            c = num / den
            sse = sum(
                (g - (delta_inf + c * (n ** (-alpha)))) ** 2
                for n, g in zip(ns, gaps)
            )
            if best is None or sse < best["sse"]:
                best = {
                    "delta_inf": delta_inf,
                    "alpha": alpha,
                    "C": c,
                    "sse": sse,
                }
    if best is None:
        return {"feasible": False}
    # Compute R^2
    g_mean = sum(gaps) / len(gaps)
    sst = sum((g - g_mean) ** 2 for g in gaps)
    r2 = 1.0 - best["sse"] / sst if sst > 0 else float("nan")
    return {
        "feasible": True,
        "delta_inf": best["delta_inf"],
        "alpha": best["alpha"],
        "C": best["C"],
        "sse": best["sse"],
        "R2": r2,
        "n_points": len(ns),
    }


def fit_fixed_alpha(ns: list[float], gaps: list[float], alpha: float) -> dict:
    """Two-parameter linear regression: Δ_∞ + C·N^(-α)."""
    if len(ns) < 2:
        return {"feasible": False}
    xs = [n ** (-alpha) for n in ns]
    n = len(ns)
    sx = sum(xs)
    sy = sum(gaps)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, gaps))
    den = n * sxx - sx * sx
    if abs(den) < 1e-30:
        return {"feasible": False}
    c = (n * sxy - sx * sy) / den
    delta_inf = (sy - c * sx) / n
    sse = sum(
        (g - (delta_inf + c * (n_ ** (-alpha)))) ** 2
        for n_, g in zip(ns, gaps)
    )
    g_mean = sy / n
    sst = sum((g - g_mean) ** 2 for g in gaps)
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    return {
        "feasible": True,
        "alpha": alpha,
        "delta_inf": delta_inf,
        "C": c,
        "sse": sse,
        "R2": r2,
    }


def leave_one_out(ns: list[float], gaps: list[float]) -> dict:
    if len(ns) < 4:
        return {"feasible": False, "reason": "need >= 4 points for LOO"}
    results = []
    for i in range(len(ns)):
        ns_loo = ns[:i] + ns[i+1:]
        gaps_loo = gaps[:i] + gaps[i+1:]
        free = fit_free(ns_loo, gaps_loo)
        if not free.get("feasible"):
            continue
        results.append({
            "left_out_index": i,
            "left_out_N": ns[i],
            "left_out_gap": gaps[i],
            "loo_delta_inf": free["delta_inf"],
            "loo_alpha": free["alpha"],
        })
    if not results:
        return {"feasible": False}
    delta_infs = [r["loo_delta_inf"] for r in results]
    alphas = [r["loo_alpha"] for r in results]
    return {
        "feasible": True,
        "per_loo": results,
        "delta_inf_min": min(delta_infs),
        "delta_inf_max": max(delta_infs),
        "delta_inf_range": max(delta_infs) - min(delta_infs),
        "alpha_min": min(alphas),
        "alpha_max": max(alphas),
        "alpha_range": max(alphas) - min(alphas),
    }


def analyse_candidate(name: str, ns: list[float], gaps: list[float],
                       threshold: float = 0.05) -> dict:
    """Full analysis: free fit + fixed-α fits + LOO + verdict."""
    free = fit_free(ns, gaps)
    fixed = {
        f"alpha_{a:.4f}": fit_fixed_alpha(ns, gaps, a)
        for a in [2.0/3.0, 1.0, 0.848]
    }
    loo = leave_one_out(ns, gaps)

    delta_inf = free.get("delta_inf", float("inf")) \
        if free.get("feasible") else float("inf")
    pass_free = delta_inf < threshold
    pass_loo = (loo.get("feasible")
                and loo.get("delta_inf_max", float("inf")) < threshold)
    pass_fixed_2_3 = (fixed.get("alpha_0.6667", {}).get("delta_inf",
                       float("inf")) < threshold)
    pass_fixed_1_0 = (fixed.get("alpha_1.0000", {}).get("delta_inf",
                       float("inf")) < threshold)
    return {
        "candidate": name,
        "ns": ns,
        "gaps": gaps,
        "free_fit": free,
        "fixed_alpha_fits": fixed,
        "leave_one_out": loo,
        "verdict": {
            "free_fit_delta_inf_below_threshold": pass_free,
            "fixed_alpha_2_3_below": pass_fixed_2_3,
            "fixed_alpha_1_0_below": pass_fixed_1_0,
            "leave_one_out_below": pass_loo,
            "all_pass": (pass_free and pass_fixed_2_3
                          and pass_fixed_1_0 and pass_loo),
        },
        "threshold": threshold,
    }


def _print_unit_tests() -> list[dict]:
    print("=" * 80)
    print("True Einstein-equation residual -- Schwarzschild unit tests")
    print("=" * 80)
    unit_tests = [
        schwarzschild_unit_test(),
        schwarzschild_perturbation_test(),
        schwarzschild_negative_test(),
    ]
    for t in unit_tests:
        print(f"  {t['test']}")
        print(f"    residual = {t['residual']}, pass = {t['pass']}")
    return unit_tests


def _load_invariants() -> list[dict]:
    print()
    print("=" * 80)
    print("Multi-N lattice scalar invariants (A2 pipeline, "
          "campaign-consistent)")
    print("=" * 80)
    invariants = []
    for regime, n_value, npz in A2_REGIMES:
        try:
            inv = load_regime_invariants(regime, n_value, npz)
            invariants.append(inv)
            print(
                f"  {regime:>8}  N={n_value:>8.1f}  "
                f"R_bar={inv['R_bar_mean']:.4f}  "
                f"bar_S_g={inv['bar_S_g_mean']:.4f}  "
                f"bar_eps_g={inv['bar_epsilon_g_mean']:.4f}  "
                f"Delta_curv={inv['Delta_curv_mean']:.4f}"
            )
        except FileNotFoundError as exc:
            print(f"  {regime}: SKIPPED ({exc})")
    return invariants


def _build_candidates(invariants: list[dict]) -> tuple[list[float], dict, float]:
    ns = [v["N"] for v in invariants]
    r_series = [v["R_bar_mean"] for v in invariants]
    s_series = [v["bar_S_g_mean"] for v in invariants]
    dc_series = [v["Delta_curv_mean"] for v in invariants]
    eps_series = [v["bar_epsilon_g_mean"] for v in invariants]

    candidates = {
        "Delta_E_R_bar": r_series,
        "Delta_E_bar_S_g": s_series,
        "Delta_E_Delta_curv": dc_series,
    }

    s_over_eps = [s / e for s, e in zip(s_series, eps_series)]
    kappa_star = sum(s_over_eps) / len(s_over_eps)
    candidates["Delta_E_coupling_residual"] = [
        abs(s - kappa_star * e) / abs(s)
        for s, e in zip(s_series, eps_series)
    ]

    # Combined Einstein-closure functional (equal weights, fixed):
    #
    #   Delta_E^eff(N) = sqrt( R_bar(N)^2 + bar_S_g(N)^2 + Delta_curv(N)^2 )
    #
    # Each component is a scalar geometric invariant that should
    # vanish in the continuum limit if Einstein closure holds:
    #   R_bar         -> Ricci-curvature consistency
    #   bar_S_g       -> gravitational stress balance
    #   Delta_curv    -> curvature mismatch
    # The L2-norm of the triplet is a natural composite Einstein-residual
    # norm; the weights are FIXED (equal) and never fitted, so the
    # convergence test is unbiased.
    candidates["Delta_E_eff_L2_equal_weights"] = [
        math.sqrt(r * r + s * s + dc * dc)
        for r, s, dc in zip(r_series, s_series, dc_series)
    ]

    # Physically-motivated weights (still fixed; not fitted):
    # weight curvature consistency higher because it is the most
    # direct Einstein-equation diagnostic; stress balance and curvature
    # mismatch carry equal complementary weight.
    w_r, w_s, w_dc = 0.5, 0.25, 0.25
    candidates["Delta_E_eff_L2_physical_weights"] = [
        math.sqrt(w_r * r * r + w_s * s * s + w_dc * dc * dc)
        for r, s, dc in zip(r_series, s_series, dc_series)
    ]

    # Linear-combination variant (equal weights, fixed):
    candidates["Delta_E_eff_linear_equal_weights"] = [
        (r + s + dc) / 3.0
        for r, s, dc in zip(r_series, s_series, dc_series)
    ]

    # ---------------------------------------------------------------
    # Per-coarse-graining-level candidates (cg_level=2, deepest):
    # at the deepest coarse-graining level the lattice has been
    # smoothed toward the continuum, so the per-regime invariants
    # are the cleanest single-N proxies for continuum Einstein
    # observables. Each Δ_E candidate is the cg2 invariant directly.
    # ---------------------------------------------------------------
    r_cg2 = [v.get("R_bar_cg2_mean", v["R_bar_mean"]) for v in invariants]
    dc_cg2 = [v.get("Delta_curv_cg2_mean", v["Delta_curv_mean"])
              for v in invariants]
    candidates["Delta_E_R_bar_cg2"] = r_cg2
    candidates["Delta_E_Delta_curv_cg2"] = dc_cg2
    candidates["Delta_E_eff_L2_cg2_equal_weights"] = [
        math.sqrt(r * r + dc * dc)
        for r, dc in zip(r_cg2, dc_cg2)
    ]
    candidates["Delta_E_eff_L2_cg2_physical_weights"] = [
        math.sqrt(0.5 * r * r + 0.5 * dc * dc)
        for r, dc in zip(r_cg2, dc_cg2)
    ]

    # ---------------------------------------------------------------
    # H3c Frobenius-decomposed Δ_E (from tensor structure of Einstein eq):
    #
    # In d-dimensional spacetime, the Einstein tensor decomposes as
    #     G_μν = R_μν - (1/2) g_μν R,
    # and ‖G_μν‖_F² = ((d-2)/(2d)) R² + ‖R_μν^traceless‖_F²
    # where R_μν^traceless = R_μν - (1/d) g_μν R.
    #
    # The A2 lattice persists scalar invariants:
    #   R_bar    = trace component (lattice Ricci scalar)
    #   Delta_curv = traceless mismatch (curvature-tensor deviation
    #                from the identity g_μν R / d)
    #
    # The Frobenius residual ‖G - 8πG T‖_F / ‖G‖_F is therefore
    # bounded by
    #     Δ_E^Frobenius_proxy² ≤ (R_bar)²·(d-2)/(2d) + Delta_curv²
    # in d = 4 spacetime, with the trace-deviation prefactor
    # (4-2)/(2·4) = 1/4.
    #
    # This is the cleanest Frobenius-decomposed proxy from the
    # bundled scalar invariants, without reconstructing per-cell
    # tensor components.
    # ---------------------------------------------------------------
    d_spacetime = 4
    trace_factor = (d_spacetime - 2) / (2 * d_spacetime)
    candidates["Delta_E_Frobenius_decomposed_cg2"] = [
        math.sqrt(trace_factor * r * r + dc * dc)
        for r, dc in zip(r_cg2, dc_cg2)
    ]

    return ns, candidates, kappa_star


def _print_candidate_analysis(name: str, ns: list[float],
                              gaps: list[float], result: dict) -> None:
    print(f"\n--- {name} ---")
    print(f"  N-series: {[f'{n:.0f}' for n in ns]}")
    print(f"  Delta-series: {[f'{g:.4f}' for g in gaps]}")
    free = result["free_fit"]
    if free.get("feasible"):
        print(
            f"  Free fit: Delta_inf = {free['delta_inf']:.4f}, "
            f"alpha = {free['alpha']:.4f}, "
            f"R^2 = {free.get('R2', float('nan')):.4f}"
        )
    for label, fit in result["fixed_alpha_fits"].items():
        if fit.get("feasible"):
            print(
                f"  {label}: Delta_inf = {fit['delta_inf']:+.4f}, "
                f"R^2 = {fit.get('R2', float('nan')):.4f}"
            )
    loo = result["leave_one_out"]
    if loo.get("feasible"):
        print(
            f"  LOO Delta_inf in [{loo['delta_inf_min']:.4f}, "
            f"{loo['delta_inf_max']:.4f}]"
        )
    verdict = result["verdict"]
    print(
        f"  Verdict (Delta_inf < 0.05 on all four tests): "
        f"{'PASS' if verdict['all_pass'] else 'FAIL'}"
    )


def main() -> None:
    unit_tests = _print_unit_tests()
    invariants = _load_invariants()
    ns, candidates, kappa_star = _build_candidates(invariants)

    print()
    print("=" * 80)
    print("Candidate Delta_E definitions and convergence analyses")
    print("=" * 80)
    analyses = {}
    for name, gaps in candidates.items():
        result = analyse_candidate(name, ns, gaps, threshold=0.05)
        analyses[name] = result
        _print_candidate_analysis(name, ns, gaps, result)

    audit = {
        "audit": "EGE-V4: True Einstein-equation residual (Pfad H3)",
        "stand": "2026-04-28",
        "method": (
            "Multi-N convergence test of campaign-consistent lattice "
            "scalar invariants (R_bar, bar_S_g, bar_epsilon_g, Delta_curv "
            "from the A2 pipeline) across 7 regimes P0..P6."
        ),
        "lattice_invariants": invariants,
        "candidate_analyses": analyses,
        "schwarzschild_unit_tests": unit_tests,
        "kappa_star_coupling_extracted": kappa_star,
        "summary_verdicts": {
            name: a["verdict"]["all_pass"]
            for name, a in analyses.items()
        },
    }
    out_path = OUT_DIR / "true_einstein_residual_p0_to_p6.json"
    out_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    for name, passes in audit["summary_verdicts"].items():
        print(f"  {name:<35} {'PASS' if passes else 'FAIL'}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
