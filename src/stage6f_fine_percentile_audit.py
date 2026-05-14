"""Stage 6f fine-percentile audit: bulk-vs-matter-core transition diagnostic.

Same data pipeline as stage6f_full_tensor_norm_audit.py (per-node relative
Frobenius residual delta_full pooled over seeds per regime), but evaluated
on a fine-grained percentile grid p90, p91, p92, ..., p99, p99.5, p99.9.

For each percentile we compute:
  - per-regime value across the cleaned 11-regime canonical-physics ladder
    (P5, P6, P5N64, P7, P5N72, P8, P5N84, P5N100, P5N200, P5N300, P5N512;
    P5N128 K/Q persistence bug skipped)
  - power-law fit y(N) = c * N^(-alpha) with bootstrap 95% CI on alpha
  - Symanzik-2 fit y(N) = y_inf + b/N with bootstrap 95% CI on y_inf

The transition between bulk-bound (alpha > 0 strict, y_inf -> 0) and
matter-core saturation (y_inf finite, alpha small) is read off the
percentile sweep directly.

Output: outputs/stage6f_fine_percentile_audit.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np


class _BlockCupy:
    """Block any cupy import (forces numpy fallback in upstream modules)."""

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
    LADDER, LAMBDA_T, LAMBDA_S,
    load_canonical, load_snapshots, per_node_relative_delta,
)
from verify_galerkin_runner_A_hessian_ricci import per_seed_galerkin  # noqa: E402


PERCENTILES = [
    "median", "mean",
    "p90", "p91", "p92", "p93", "p94", "p95",
    "p96", "p97", "p98", "p99",
    "p99_5", "p99_9", "sup",
]


def _gather_regime(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    if "snapshots" in p.name.lower():
        seeds = load_snapshots(p, n_lat)
    else:
        seeds = load_canonical(p, n_lat)
    pool = []
    for xi_mat, psi, k_field, q_field in seeds:
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        comp = per_node_relative_delta(prep, LAMBDA_T, LAMBDA_S)
        pool.append(comp["delta_full"])
    return {
        "regime": reg, "N": n_lat, "n_seeds": len(seeds),
        "delta_full": np.concatenate(pool),
        "source_path": str(p),
    }


def _percentile(arr, name):
    if name == "median":
        return float(np.median(arr))
    if name == "mean":
        return float(arr.mean())
    if name == "sup":
        return float(arr.max())
    if name == "p99_5":
        return float(np.percentile(arr, 99.5))
    if name == "p99_9":
        return float(np.percentile(arr, 99.9))
    # pXX where XX in 90..99
    pct = float(name[1:])
    return float(np.percentile(arr, pct))


def _powerlaw_alpha(n_arr, y_arr):
    n_arr = np.asarray(n_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)
    mask = y_arr > 0
    if mask.sum() < 3:
        return float("nan"), float("nan")
    log_n = np.log(n_arr[mask])
    log_y = np.log(y_arr[mask])
    slope, intercept = np.polyfit(log_n, log_y, 1)
    pred = slope * log_n + intercept
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(-slope), float(r2)


def _symanzik_2(n_arr, y_arr):
    n_arr = np.asarray(n_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)
    if len(n_arr) < 3:
        return float("nan"), float("nan"), float("nan")
    x_mat = np.column_stack([np.ones_like(n_arr), 1.0 / n_arr])
    coef, *_ = np.linalg.lstsq(x_mat, y_arr, rcond=None)
    y_inf, b = coef
    pred = y_inf + b / n_arr
    ss_res = np.sum((y_arr - pred) ** 2)
    ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(y_inf), float(b), float(r2)


def _bootstrap(rng, n_arr, y_arr, fit_fn, n_boot=2000):
    n = len(n_arr)
    samples = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        try:
            res = fit_fn([n_arr[i] for i in idx], [y_arr[i] for i in idx])
            samples.append(res)
        except (ValueError, np.linalg.LinAlgError):
            continue
    return samples


def _ci(samples, lo=0.025, hi=0.975):
    if not samples:
        return None, None
    s = sorted(samples)
    nlo = int(lo * len(s))
    nhi = int(hi * len(s))
    return s[nlo], s[min(nhi, len(s) - 1)]


def main() -> int:
    print("=" * 110)
    print("Stage 6f fine-percentile audit: bulk-vs-matter-core transition")
    print("=" * 110)
    print(f"  Lambda_t = {LAMBDA_T}, Lambda_s = {LAMBDA_S}")
    print(f"  Percentiles: {PERCENTILES}")
    print()

    regimes = []
    for reg, n_lat in LADDER:
        r = _gather_regime(reg, n_lat)
        if r is None:
            print(f"  {reg} N={n_lat}: NPZ not found, skipping")
            continue
        spec = {pc: _percentile(r["delta_full"], pc) for pc in PERCENTILES}
        regimes.append({
            "regime": reg, "N": n_lat, "n_seeds": r["n_seeds"],
            "n_node": int(r["delta_full"].size),
            "percentiles": spec,
        })
        print(f"  {reg:<8} N={n_lat:>4}  ns={r['n_seeds']:>2}  "
              f"p90={spec['p90']:.4f} p95={spec['p95']:.4f} "
              f"p99={spec['p99']:.4f} sup={spec['sup']:.4f}")

    if len(regimes) < 3:
        print("  Not enough regimes for fits.")
        return 0

    print()
    print("=" * 110)
    print("Per-percentile fits across the cleaned ladder "
          f"({len(regimes)} regimes)")
    print("=" * 110)
    header = (f'{"pct":>8s}  '
              f'{"alpha":>9s} {"alpha_lo":>10s} {"alpha_hi":>10s} '
              f'{"R2_pl":>6s}  '
              f'{"y_inf":>10s} {"y_inf_lo":>10s} {"y_inf_hi":>10s} '
              f'{"R2_sym":>6s}')
    print(header)
    print("-" * len(header))

    n_arr = [r["N"] for r in regimes]
    rng = random.Random(0xC0FFEE)
    fit_results = {}
    for pc in PERCENTILES:
        y_arr = [r["percentiles"][pc] for r in regimes]
        alpha, r2_pl = _powerlaw_alpha(n_arr, y_arr)
        boot_alpha = _bootstrap(rng, n_arr, y_arr,
                                lambda nn, yy: _powerlaw_alpha(nn, yy)[0])
        a_lo, a_hi = _ci(boot_alpha)
        y_inf, b, r2_sy = _symanzik_2(n_arr, y_arr)
        boot_yinf = _bootstrap(rng, n_arr, y_arr,
                                lambda nn, yy: _symanzik_2(nn, yy)[0])
        y_lo, y_hi = _ci(boot_yinf)
        fit_results[pc] = {
            "alpha": alpha, "alpha_95_lo": a_lo, "alpha_95_hi": a_hi,
            "r2_powerlaw": r2_pl,
            "y_inf": y_inf, "y_inf_95_lo": y_lo, "y_inf_95_hi": y_hi,
            "b_symanzik": b, "r2_symanzik": r2_sy,
        }
        a_lo_s = f"{a_lo:>+10.4f}" if a_lo is not None else f"{'n/a':>10s}"
        a_hi_s = f"{a_hi:>+10.4f}" if a_hi is not None else f"{'n/a':>10s}"
        y_lo_s = f"{y_lo:>+10.4f}" if y_lo is not None else f"{'n/a':>10s}"
        y_hi_s = f"{y_hi:>+10.4f}" if y_hi is not None else f"{'n/a':>10s}"
        print(f"{pc:>8s}  "
              f"{alpha:>+9.4f} {a_lo_s} {a_hi_s} {r2_pl:>6.3f}  "
              f"{y_inf:>+10.4f} {y_lo_s} {y_hi_s} {r2_sy:>6.3f}")

    print()
    print("=" * 110)
    print("Bulk-vs-matter-core transition diagnostic:")
    print("  - bulk-bound: alpha_95_lo > 0  AND  y_inf_95_hi < 0.05")
    print("  - matter-core:  y_inf_95_lo > 0  (saturation at finite value)")
    print("=" * 110)
    transitions = []
    for pc in PERCENTILES:
        v = fit_results[pc]
        a_lo = v["alpha_95_lo"]
        y_lo = v["y_inf_95_lo"]
        y_hi = v["y_inf_95_hi"]
        bulk_bound = (a_lo is not None and a_lo > 0
                      and y_hi is not None and y_hi < 0.05)
        matter_core = y_lo is not None and y_lo > 0
        cls = "BULK_BOUND" if bulk_bound else (
            "MATTER_CORE" if matter_core else "TRANSITION")
        transitions.append((pc, cls))
        print(f"  {pc:>8s}: {cls}")

    bundle = {
        "method": "stage6f_fine_percentile_audit",
        "schema_version": "1.0.0",
        "lambda_t": LAMBDA_T,
        "lambda_s": LAMBDA_S,
        "percentiles": PERCENTILES,
        "ladder_skipped": ["P5N128 (K/Q persistence bug)"],
        "regimes": regimes,
        "fits": fit_results,
        "classification": {pc: cls for pc, cls in transitions},
    }
    out = REPO / "outputs" / "stage6f_fine_percentile_audit.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print()
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
