"""Stage 6h: Massart-style ratio test for the layer-rho hierarchy
against shell-rank scaling models 1/n and 1/n^2.

Tests whether the per-layer rho-shift sequence on the within-P5
ladder is consistent with a fixed shell-rank scaling

    |Delta rho|^layer = c * f(n),   n in {1,...,5}

for the candidate models
    f1: f(n) = 1/n
    f2: f(n) = 1/n^2
    fhalf: f(n) = 1/sqrt(n)   (reference, already known best fit)

with shell rank
    n=1 Csup, n=2 C99.9, n=3 C99.5, n=4 C99, n=5 C95

For each model we report

  ratio_n^layer = |Delta rho|^layer / f(n)
  cross-layer mean and CV of these ratios per regime
  CV<0.20 => model holds across layers

and per-regime log-log fit
  |shift| ~ n^(-alpha)
slope alpha: pure 1/n => alpha=1, pure 1/n^2 => alpha=2,
1/sqrt(n) => alpha=0.5.

Output: outputs/stage6h_shell_rank_scaling_test.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


class _BlockCupy:
    def find_spec(self, name, path=None, target=None):
        if name == "cupy" or name.startswith("cupy."):
            raise ImportError("cupy disabled")
        return None

    def load_module(self, name):
        raise ImportError("cupy disabled")


sys.meta_path.insert(0, _BlockCupy())

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from _d1_npz_discovery import find_d1_npz  # noqa: E402
from stage6f_full_tensor_norm_audit import (  # noqa: E402
    LAMBDA_T, LAMBDA_S, load_canonical, load_snapshots,
    per_node_relative_delta,
)
from verify_galerkin_runner_A_hessian_ricci import per_seed_galerkin  # noqa: E402

N_STAR = 50
N_GEN = 3
D = 4
N_FLIP = N_STAR * math.sqrt(D * N_GEN)

LADDER = [
    ("P5",     50),
    ("P5N64",  64),
    ("P5N72",  72),
    ("P5N84",  84),
    ("P5N100", 100),
    ("P5N200", 200),
    ("P5N300", 300),
    ("P5N512", 512),
    ("P5N256", 256),
]

LAYER_DEFS = [
    ("Csup",  "max", 1),
    ("C99_9", 99.9,  2),
    ("C99_5", 99.5,  3),
    ("C99",   99.0,  4),
    ("C95",   95.0,  5),
]


def chirality_phase(n_lat):
    x = math.log(n_lat / N_STAR) / math.log(D * N_GEN)
    th = math.atan(N_GEN ** (2 * x - 1))
    s2 = math.sin(th) ** 2
    return s2, ("PRE" if th < math.pi / 4 else "POST")


def _per_seed(xi_mat, psi, k_field, q_field, n_lat):
    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
    delta = per_node_relative_delta(prep, LAMBDA_T, LAMBDA_S)["delta_full"]
    xi_no = xi_mat.copy()
    np.fill_diagonal(xi_no, 0.0)
    safe = np.maximum(xi_no.sum(axis=1), 1e-12)
    eiphi = np.exp(1j * np.angle(psi))
    nbhd = (xi_no @ eiphi) / safe
    rho = np.real(np.conj(eiphi) * nbhd)
    return delta, rho


def _layer_indices(arr, percentile_or_max):
    if percentile_or_max == "max":
        return {int(np.argmax(arr))}
    thr = float(np.percentile(arr, percentile_or_max))
    return set(int(i) for i in np.nonzero(arr > thr)[0])


def _gather(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    if "snapshots" in p.name.lower():
        seeds = load_snapshots(p, n_lat)
    else:
        seeds = load_canonical(p, n_lat)
    return [_per_seed(xi, psi, k, q, n_lat) for (xi, psi, k, q) in seeds]


def _per_regime_layer_shifts(seeds_data):
    """Pool rho-on-layer over seeds; return per-layer shift dict."""
    layer_rho = {ln: [] for ln, _, _ in LAYER_DEFS}
    all_rho = []
    for delta, rho in seeds_data:
        all_rho.append(rho)
        for ln, pc, _ in LAYER_DEFS:
            S = _layer_indices(delta, pc)
            if S:
                idx = np.array(sorted(S), dtype=int)
                layer_rho[ln].append(rho[idx])
    all_rho = np.concatenate(all_rho)
    rho_mean = float(all_rho.mean())
    out = {"rho_lattice_mean": rho_mean,
           "rho_lattice_sigma": float(all_rho.std()),
           "shifts": {}}
    for ln, _, n_rank in LAYER_DEFS:
        if layer_rho[ln]:
            arr = np.concatenate(layer_rho[ln])
            shift = float(arr.mean()) - rho_mean
            out["shifts"][ln] = {
                "n_rank": n_rank,
                "pooled_size": int(arr.size),
                "rho_layer_mean": float(arr.mean()),
                "shift": shift,
                "abs_shift": abs(shift),
            }
    return out


def _ratio_test(per_regime, model_name, model_fn):
    """For each regime, compute c-fitted = mean(|shift| / f(n)) across
    layers; then CV of |shift_n|/f(n) across layers per regime, and
    cross-regime aggregate.

    A perfect model has all per-layer ratios equal => CV=0.
    """
    results = []
    for reg, info in per_regime.items():
        ratios = []
        ranks = []
        shifts = []
        for ln, _, n_rank in LAYER_DEFS:
            if ln not in info["shifts"]:
                continue
            ranks.append(n_rank)
            shifts.append(info["shifts"][ln]["abs_shift"])
            f_n = model_fn(n_rank)
            ratios.append(info["shifts"][ln]["abs_shift"] / max(f_n, 1e-12))
        ratios = np.array(ratios)
        if ratios.size < 3:
            continue
        cv = float(ratios.std() / max(abs(ratios.mean()), 1e-12))
        results.append({
            "regime": reg, "n_layers": int(ratios.size),
            "ratios_per_layer": ratios.tolist(),
            "ratios_mean": float(ratios.mean()),
            "ratios_std": float(ratios.std()),
            "ratios_cv": cv,
        })
    return results


def _loglog_fit(per_regime):
    """log|shift| ~ -alpha * log(n) per regime."""
    out = []
    for reg, info in per_regime.items():
        n_arr = []
        s_arr = []
        for ln, _, n_rank in LAYER_DEFS:
            if ln not in info["shifts"]:
                continue
            sh = info["shifts"][ln]["abs_shift"]
            if sh > 0 and n_rank > 0:
                n_arr.append(n_rank)
                s_arr.append(sh)
        if len(n_arr) < 3:
            continue
        x = np.log(np.array(n_arr, dtype=float))
        y = np.log(np.array(s_arr, dtype=float))
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out.append({
            "regime": reg,
            "alpha_loglog": float(-slope),
            "c_intercept": float(np.exp(intercept)),
            "r2": float(r2),
        })
    return out


def main() -> int:
    print("=" * 110)
    print("Stage 6h: shell-rank scaling test (1/n vs 1/n^2 vs 1/sqrt(n))")
    print("=" * 110)
    print()

    pre_regime = {}
    post_regime = {}
    for reg, n_lat in LADDER:
        s2, phase = chirality_phase(n_lat)
        seeds_data = _gather(reg, n_lat)
        if seeds_data is None:
            continue
        info = _per_regime_layer_shifts(seeds_data)
        info["N"] = n_lat
        info["sin2t"] = s2
        info["phase"] = phase
        info["n_seeds"] = len(seeds_data)
        if phase == "PRE":
            pre_regime[reg] = info
        else:
            post_regime[reg] = info
        sup = info["shifts"].get("Csup", {})
        c95 = info["shifts"].get("C95", {})
        if sup and c95:
            print(f"  {reg:<8s} N={n_lat:>4d} {phase}  "
                  f"|shift_Csup|={sup['abs_shift']:.4f}  "
                  f"|shift_C95|={c95['abs_shift']:.4f}")

    models = {
        "1_over_n":       lambda n: 1.0 / n,
        "1_over_n_sq":    lambda n: 1.0 / (n * n),
        "1_over_sqrt_n":  lambda n: 1.0 / math.sqrt(n),
    }

    print()
    print("=" * 110)
    print("Per-regime ratio test  |shift_n| / f(n)  cross-LAYER CV")
    print("(CV<0.20 => model holds for that regime; CV<0.10 => excellent)")
    print("=" * 110)

    summary = {}
    for phase_name, regime_dict in [("PRE", pre_regime), ("POST", post_regime)]:
        print(f"\n{phase_name} (n={len(regime_dict)} regimes):")
        summary[phase_name] = {}
        for model_name, model_fn in models.items():
            results = _ratio_test(regime_dict, model_name, model_fn)
            cvs = [r["ratios_cv"] for r in results]
            mean_cv = float(np.mean(cvs)) if cvs else float("nan")
            summary[phase_name][model_name] = {
                "per_regime_results": results,
                "mean_cv": mean_cv,
                "n_regimes_below_0_20": int(sum(1 for c in cvs if c < 0.20)),
                "n_regimes_total": len(cvs),
            }
            print(f"  Model {model_name:<18s}: per-regime CV "
                  f"mean={mean_cv:.3f},  "
                  f"{summary[phase_name][model_name]['n_regimes_below_0_20']}"
                  f"/{summary[phase_name][model_name]['n_regimes_total']} "
                  f"regimes with CV<0.20")
            for r in results:
                print(f"    {r['regime']:<8s}: ratios = "
                      f"[{', '.join(f'{x:.3f}' for x in r['ratios_per_layer'])}]  "
                      f"CV={r['ratios_cv']:.3f}")

    print()
    print("=" * 110)
    print("Per-regime log-log fit  |shift| ~ n^(-alpha)")
    print("  alpha=0.5 => 1/sqrt(n)")
    print("  alpha=1.0 => 1/n")
    print("  alpha=2.0 => 1/n^2")
    print("=" * 110)
    for phase_name, regime_dict in [("PRE", pre_regime), ("POST", post_regime)]:
        ll = _loglog_fit(regime_dict)
        summary[phase_name]["loglog_fits"] = ll
        print(f"\n{phase_name}:")
        if not ll:
            print("  (no fits)")
            continue
        for r in ll:
            print(f"  {r['regime']:<8s}  alpha = {r['alpha_loglog']:>+5.3f}  "
                  f"c = {r['c_intercept']:.3f}  R^2 = {r['r2']:.3f}")
        a_arr = np.array([r["alpha_loglog"] for r in ll])
        print(f"  cross-regime alpha:  mean = {a_arr.mean():+.3f},  "
              f"std = {a_arr.std():.3f}")

    bundle = {
        "method": "stage6h_shell_rank_scaling_test",
        "schema_version": "1.0.0",
        "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
        "n_flip": N_FLIP,
        "ladder_within_p5_only": [r for r, _ in LADDER],
        "shell_rank_definition": {
            "n_1": "Csup", "n_2": "C99.9", "n_3": "C99.5",
            "n_4": "C99", "n_5": "C95",
        },
        "pre_regimes": pre_regime,
        "post_regimes": post_regime,
        "summary_by_phase": summary,
    }
    out = REPO / "outputs" / "stage6h_shell_rank_scaling_test.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print()
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
