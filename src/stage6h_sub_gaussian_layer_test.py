"""Stage 6h: sub-Gaussian extreme-value test on the Delta-residual
layer-rho hierarchy.

Tests whether the per-layer rho-shift sequence on the within-P5
ladder is consistent with the Massart sub-Gaussian extreme-value
bound

    E[|tail-mean of ordered statistics|] ~ sigma * sqrt(2 log(N/k))

evaluated on the per-node phase coherence rho_a, where the layer
top-k is selected by Delta_a (cross-observable selection, not
self-selection of rho).

If the rho-distribution conditional on Delta_a-top-k follows the
Massart-style sub-Gaussian bound, then |Delta rho|^layer scales
as sqrt(log(N/k_layer)), and the empirical sequence over the
five layers C^(95), C^(99), C^(99.5), C^(99.9), C^(sup) should
satisfy

  ratio  |Delta rho|_layer / sqrt(log(N/k_layer)) ~ const

across layers, with the constant equal to the per-node std of
rho on the lattice times sqrt(2) up to a Talagrand correction.

If the empirical ratio is approximately constant (CV < 20%
across layers), the layer-rho hierarchy is identified as the
sub-Gaussian extreme-value-concentration manifestation of
Theorem 15.18(ii) on a cross-observable selection.

If the ratio varies systematically (e.g. growing with shell
rank), the hierarchy is heavier-tailed than sub-Gaussian and
the Massart identification fails.

Output: outputs/stage6h_sub_gaussian_layer_test.json
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
    ("P5N256", 256),
    ("P5N300", 300),
    ("P5N512", 512),
]

LAYER_DEFS = [
    ("Csup",  "max"),
    ("C99_9", 99.9),
    ("C99_5", 99.5),
    ("C99",   99.0),
    ("C95",   95.0),
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
    w_sum = xi_no.sum(axis=1)
    safe = np.maximum(w_sum, 1e-12)
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


def main() -> int:
    print("=" * 110)
    print("Stage 6h: sub-Gaussian extreme-value test on layer-rho hierarchy")
    print("=" * 110)
    print()

    rows = []
    for reg, n_lat in LADDER:
        s2, phase = chirality_phase(n_lat)
        seeds_data = _gather(reg, n_lat)
        if seeds_data is None:
            continue

        # Pool rho and delta over all seeds for this regime
        # Layer indices are determined per seed (using delta), then
        # rho on those indices is pooled.
        layer_rho = {ln: [] for ln, _ in LAYER_DEFS}
        layer_size_pool = {ln: 0 for ln, _ in LAYER_DEFS}
        all_rho = []
        n_total = 0

        for delta, rho in seeds_data:
            all_rho.append(rho)
            n_total += rho.size
            for ln, pc in LAYER_DEFS:
                S = _layer_indices(delta, pc)
                if S:
                    idx = np.array(sorted(S), dtype=int)
                    layer_rho[ln].append(rho[idx])
                    layer_size_pool[ln] += len(S)

        all_rho = np.concatenate(all_rho)
        rho_mean = float(all_rho.mean())
        rho_sigma = float(all_rho.std())

        layer_stats = {}
        for ln, _ in LAYER_DEFS:
            if layer_rho[ln]:
                arr = np.concatenate(layer_rho[ln])
                k = int(arr.size)
                if k == 0 or n_total == 0:
                    continue
                empirical_mean = float(arr.mean())
                empirical_shift = empirical_mean - rho_mean
                # Massart sub-Gaussian bound for the conditional
                # tail-mean: |E[X | Delta-top-k]| ~ sigma_rho *
                # sqrt(2 * log(N_total / k))
                massart_arg = max(n_total / max(k, 1), 1.0)
                massart_pred = rho_sigma * math.sqrt(2.0 * math.log(massart_arg))
                ratio = abs(empirical_shift) / max(massart_pred, 1e-12)
                layer_stats[ln] = {
                    "pooled_size": k,
                    "n_total": n_total,
                    "empirical_rho_mean": empirical_mean,
                    "empirical_shift": empirical_shift,
                    "rho_lattice_mean": rho_mean,
                    "rho_lattice_sigma": rho_sigma,
                    "massart_arg_log": float(math.log(massart_arg)),
                    "massart_prediction":
                        float(massart_pred),
                    "ratio_emp_over_massart": float(ratio),
                }

        rows.append({
            "regime": reg, "N": n_lat,
            "sin2t": s2, "phase": phase,
            "n_seeds": len(seeds_data),
            "n_total_pooled": n_total,
            "rho_lattice_mean": rho_mean,
            "rho_lattice_sigma": rho_sigma,
            "layers": layer_stats,
        })
        sup = layer_stats.get("Csup", {})
        if sup:
            print(f"  {reg:<8s} N={n_lat:>4d} {phase}  "
                  f"sigma_rho={rho_sigma:.3f}  "
                  f"|shift_C95|={abs(layer_stats['C95']['empirical_shift']):.3f} "
                  f"/Massart={layer_stats['C95']['ratio_emp_over_massart']:.2f}  "
                  f"|shift_Csup|={abs(sup['empirical_shift']):.3f} "
                  f"/Massart={sup['ratio_emp_over_massart']:.2f}")

    print()
    print("=" * 110)
    print("Cross-regime ratio empirical |shift| / Massart-bound per layer")
    print("=" * 110)
    print("  If ratio ~ const across layers (CV<0.20): sub-Gaussian")
    print("  identification holds; the layer-rho hierarchy is the")
    print("  Massart manifestation of Theorem 15.18(ii) on cross-")
    print("  observable selection.")
    print()

    summary = {}
    for phase in ("PRE", "POST"):
        grp = [r for r in rows if r["phase"] == phase]
        n_grp = len(grp)
        summary[phase] = {"n_regimes": n_grp, "layers": {}}
        print(f"\n{phase} (n={n_grp}):")
        print(f"  {'layer':>8s}  "
              f"{'mean_ratio':>12s} {'std_ratio':>10s} "
              f"{'CV':>6s}  {'verdict':>20s}")
        ratios_per_layer = []
        for ln, _ in LAYER_DEFS:
            ratios = [r["layers"][ln]["ratio_emp_over_massart"]
                      for r in grp if ln in r["layers"]]
            if not ratios:
                continue
            r_arr = np.array(ratios)
            mean = float(r_arr.mean())
            std = float(r_arr.std())
            cv = std / max(abs(mean), 1e-12)
            verdict = ("SUB_GAUSSIAN" if cv < 0.20
                        else "DEVIATES_FROM_MASSART")
            ratios_per_layer.append((ln, mean, std, cv, verdict))
            summary[phase]["layers"][ln] = {
                "mean_ratio": mean, "std_ratio": std, "cv": cv,
                "verdict": verdict,
            }
            print(f"  {ln:>8s}  {mean:>12.3f} {std:>10.3f} "
                  f"{cv:>6.2f}  {verdict:>20s}")

        # Cross-layer constancy check (CV across layer-means)
        if ratios_per_layer:
            layer_means = np.array([x[1] for x in ratios_per_layer])
            cl_cv = float(layer_means.std()
                            / max(abs(layer_means.mean()), 1e-12))
            verdict_cross = ("SUB_GAUSSIAN_CROSS_LAYER"
                              if cl_cv < 0.20
                              else "LAYER_HIERARCHY_NOT_MASSART")
            summary[phase]["cross_layer_cv"] = cl_cv
            summary[phase]["cross_layer_verdict"] = verdict_cross
            print(f"\n  Cross-layer CV of mean-ratios: {cl_cv:.3f}  "
                  f"=> {verdict_cross}")

    # Per-(N/k) log-log fit on POST: empirical |shift| vs log(N/k)
    print()
    print("=" * 110)
    print("Per-regime log-log fit |shift| ~ (log(N/k))^alpha on POST")
    print("=" * 110)
    print("  alpha=0.5: pure Massart sub-Gaussian extreme-value")
    print("  alpha=1.0: heavier-than-Gaussian (linear in log)")
    print("  alpha=0.0: layer-independent (no scaling)")
    print()

    post_fits = []
    for r in rows:
        if r["phase"] != "POST":
            continue
        log_args = []
        shifts = []
        for ln, _ in LAYER_DEFS:
            if ln in r["layers"]:
                la = r["layers"][ln]
                if la["massart_arg_log"] > 0 and abs(la["empirical_shift"]) > 0:
                    log_args.append(la["massart_arg_log"])
                    shifts.append(abs(la["empirical_shift"]))
        if len(log_args) < 3:
            continue
        log_args = np.array(log_args)
        shifts = np.array(shifts)
        # log(shift) = log(c) + alpha * log(log(N/k))
        valid = (log_args > 0) & (shifts > 0)
        if valid.sum() < 3:
            continue
        x = np.log(log_args[valid])
        y = np.log(shifts[valid])
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        post_fits.append({
            "regime": r["regime"], "N": r["N"],
            "alpha_loglog": float(slope),
            "c_intercept": float(np.exp(intercept)),
            "r2": float(r2),
        })
        print(f"  {r['regime']:<8s} N={r['N']:>4d}  "
              f"alpha = {slope:>+5.3f}  c = {np.exp(intercept):.3f}  R^2 = {r2:.3f}")

    bundle = {
        "method": "stage6h_sub_gaussian_layer_test",
        "schema_version": "1.0.0",
        "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
        "n_flip": N_FLIP,
        "ladder_within_p5_only": [r for r, _ in LADDER],
        "regimes": rows,
        "phase_summary": summary,
        "post_loglog_fits": post_fits,
    }
    out = REPO / "outputs" / "stage6h_sub_gaussian_layer_test.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print()
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
