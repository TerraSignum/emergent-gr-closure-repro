"""(A3) Per-channel Symanzik 2+4 with Bootstrap CI on each asymptote.

Channels:
  - Λ_t = mean(T_00 - G_00) per regime
  - 3·|Λ_s| = mean over 3 axes of |λ_i - G_(ii)| in T-eigenframe
  - shear = mean ||G_off||_F in T-eigenframe
  - anisotropy_index = max - min Λ_s within node, mean

Bootstrap CI on:
  - Asymptote of each channel (Symanzik 2+4)
  - Power-law decay rate
  - Distance to System-R prediction (α_ξ², 3γ²/2, 0)

Output: outputs/per_channel_bootstrap_CI_audit.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

audit_path = REPO / "outputs" / "per_channel_symanzik_anisotropy_audit.json"
with open(audit_path) as f:
    aud = json.load(f)
rows = aud["per_regime"]
print(f"Loaded {len(rows)} regimes; running bootstrap CI on per-channel Symanzik...")

ALPHA_XI = 9.0/10.0
GAMMA = 1.0/10.0
LAMBDA_T_PRED = ALPHA_XI**2          # 0.810
LAMBDA_S_3SUM_PRED = 3*GAMMA**2/2    # 0.015

N_arr = np.array([r["N"] for r in rows], dtype=float)
Lt = np.array([r["Lambda_t_mean"] for r in rows])
L3 = np.array([r["Lambda_s_3sum_abs_mean"] for r in rows])
Sh = np.array([r["shear_mean"] for r in rows])
An = np.array([r["anisotropy_index_mean"] for r in rows])


def symanzik_24(N_arr, y_arr):
    A = np.column_stack([np.ones_like(N_arr), 1.0/N_arr**2, 1.0/N_arr**4])
    coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
    return float(coef[0])


def power_law(N_arr, y_arr):
    mask = (np.abs(y_arr) > 1e-12) & (N_arr > 0)
    if mask.sum() < 3: return float('nan')
    coefs = np.polyfit(np.log(N_arr[mask]), np.log(np.abs(y_arr[mask])), 1)
    return -float(coefs[0])


def bootstrap_resample(N_arr, y_arr, n_boot=2000, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    asymptotes, exponents = [], []
    n = len(N_arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            asymptotes.append(symanzik_24(N_arr[idx], y_arr[idx]))
            exponents.append(power_law(N_arr[idx], y_arr[idx]))
        except Exception:
            continue
    return np.array(asymptotes), np.array(exponents)


def main() -> int:
    print("="*100)
    print("(A3) Per-channel Symanzik 2+4 with Bootstrap CI (n_boot=2000)")
    print("="*100)
    rng = np.random.default_rng(42)

    out = {"method": "per_channel_bootstrap_CI", "n_boot": 2000, "channels": {}}
    print(f"\n{'Channel':<14} | {'asymp_med':>10} {'asymp_95%CI':>22} | {'α_med':>7} {'α_95%CI':>14} | prediction match?")
    print("-" * 110)

    for name, y, pred, pred_name in [
        ("Lambda_t",       Lt,   LAMBDA_T_PRED,    "α_ξ² = 0.810"),
        ("3|Lambda_s|",    L3,   LAMBDA_S_3SUM_PRED,"3γ²/2 = 0.015"),
        ("shear",          Sh,   0.0,              "0"),
        ("anisotropy",     An,   0.0,              "0"),
    ]:
        asym_b, exp_b = bootstrap_resample(N_arr, y, n_boot=2000, rng=rng)
        mask = np.isfinite(asym_b) & np.isfinite(exp_b) & (np.abs(asym_b) < 5.0)
        asym_b = asym_b[mask]; exp_b = exp_b[mask]
        if len(asym_b) < 100:
            print(f"{name:<14} insufficient: {len(asym_b)}"); continue
        asym_med = float(np.median(asym_b))
        asym_lo, asym_hi = np.percentile(asym_b, [2.5, 97.5])
        exp_med = float(np.median(exp_b))
        exp_lo, exp_hi = np.percentile(exp_b, [2.5, 97.5])
        in_CI = (asym_lo <= pred <= asym_hi)

        print(f"{name:<14} | {asym_med:>+10.4f} [{asym_lo:>+8.4f}, {asym_hi:>+8.4f}] | "
              f"{exp_med:>+7.2f} [{exp_lo:>+5.2f}, {exp_hi:>+5.2f}] | "
              f"{pred_name} {'IN CI' if in_CI else 'OUT'}")

        out["channels"][name] = {
            "n_resamples": int(len(asym_b)),
            "asymptote_median": asym_med,
            "asymptote_CI95": [float(asym_lo), float(asym_hi)],
            "decay_exponent_median": exp_med,
            "decay_exponent_CI95": [float(exp_lo), float(exp_hi)],
            "system_R_prediction": pred,
            "prediction_in_CI": bool(in_CI),
        }

    print()
    print("=== Combined verdict ===")
    chs = out["channels"]
    all_in = all(ch.get("prediction_in_CI", False) for ch in chs.values())
    if all_in:
        verdict = "ALL_FOUR_CHANNELS_BOOTSTRAP_CONSISTENT_WITH_SYSTEM_R"
    else:
        which_in = [n for n, ch in chs.items() if ch.get("prediction_in_CI", False)]
        verdict = f"PARTIAL_CONSISTENT: {which_in}"
    print(f"  {verdict}")
    out["verdict"] = verdict

    out_path = REPO / "outputs" / "per_channel_bootstrap_CI_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
