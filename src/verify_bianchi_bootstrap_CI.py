"""(A1) Bootstrap CI on the discrete Bianchi recovery.

We have ||∇_μ G^μν||_F per regime on 14 regimes. Bootstrap over regimes:
  - 1000 resamples with replacement of the 14-regime ladder
  - Per resample: Symanzik 2+4 fit to extract asymptote + power-law exponent
  - Report 95% CI on (B_asymptote, α_decay)

This converts the empirical N^(-1.5) from "single-fit observation" into
a statistically defensible bootstrap-CI claim.

Output: outputs/bianchi_bootstrap_CI_audit.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


# Load existing per-regime Bianchi data
audit_path = REPO / "outputs" / "discrete_bianchi_scaling_audit.json"
with open(audit_path) as f:
    aud = json.load(f)
rows = aud["per_regime"]
N_arr = np.array([r["N"] for r in rows], dtype=float)
Bt = np.array([r["B_time_med"] for r in rows])
Bs = np.array([r["B_spat_med"] for r in rows])
Bf = np.array([r["B_full_med"] for r in rows])

print(f"Loaded {len(rows)} regimes; running bootstrap CI...")


def symanzik_24(N_arr, y_arr):
    A = np.column_stack([np.ones_like(N_arr), 1.0/N_arr**2, 1.0/N_arr**4])
    coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
    return float(coef[0])


def power_law(N_arr, y_arr):
    mask = (y_arr > 1e-12) & (N_arr > 0)
    if mask.sum() < 3: return float('nan')
    coefs = np.polyfit(np.log(N_arr[mask]), np.log(y_arr[mask]), 1)
    return -float(coefs[0])


def bootstrap_resample_fits(N_arr, y_arr, n_boot=2000, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    asymptotes, exponents = [], []
    n = len(N_arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        N_b = N_arr[idx]
        y_b = y_arr[idx]
        try:
            asymptotes.append(symanzik_24(N_b, y_b))
            exponents.append(power_law(N_b, y_b))
        except Exception:
            continue
    return np.array(asymptotes), np.array(exponents)


def main() -> int:
    print("=" * 100)
    print("(A1) Bootstrap CI on Discrete Bianchi recovery (n_boot=2000)")
    print("=" * 100)
    rng = np.random.default_rng(42)

    print(f"\n{'Channel':<14} {'N':>3} | {'asymp_med':>10} {'asymp_95%CI':>22} | {'α_med':>7} {'α_95%CI':>16}")
    print("-" * 95)

    out = {"method": "bianchi_bootstrap_CI", "n_boot": 2000, "n_regimes": int(len(rows)),
           "channels": {}}

    for name, y in [("B_time", Bt), ("B_spat", Bs), ("B_full", Bf)]:
        asym_b, exp_b = bootstrap_resample_fits(N_arr, y, n_boot=2000, rng=rng)
        # Filter NaN / divergent
        mask = np.isfinite(asym_b) & np.isfinite(exp_b) & (np.abs(asym_b) < 1.0) & (np.abs(exp_b) < 5.0)
        asym_b = asym_b[mask]; exp_b = exp_b[mask]
        if len(asym_b) < 100:
            print(f"{name:<14} insufficient bootstraps after filter ({len(asym_b)})"); continue

        asym_med = float(np.median(asym_b))
        asym_lo, asym_hi = np.percentile(asym_b, [2.5, 97.5])
        exp_med = float(np.median(exp_b))
        exp_lo, exp_hi = np.percentile(exp_b, [2.5, 97.5])

        print(f"{name:<14} {len(rows):>3} | {asym_med:>+10.5f} [{asym_lo:>+8.5f}, {asym_hi:>+8.5f}] |"
              f" {exp_med:>7.3f} [{exp_lo:>5.2f}, {exp_hi:>5.2f}]")

        # Test: is asymptote consistent with 0?
        zero_in_CI = (asym_lo <= 0.0 <= asym_hi)
        # Test: is exponent significantly > 0?
        positive_decay = (exp_lo > 0)

        out["channels"][name] = {
            "n_resamples": int(len(asym_b)),
            "asymptote_median": asym_med,
            "asymptote_CI95": [float(asym_lo), float(asym_hi)],
            "asymptote_zero_in_CI": bool(zero_in_CI),
            "decay_exponent_median": exp_med,
            "decay_exponent_CI95": [float(exp_lo), float(exp_hi)],
            "decay_significantly_positive": bool(positive_decay),
        }

    print()
    print("=== Interpretation ===")
    for name, ch in out["channels"].items():
        print(f"  {name}:")
        print(f"    Asymptote 95%CI = [{ch['asymptote_CI95'][0]:+.5f}, {ch['asymptote_CI95'][1]:+.5f}]"
              f"  → 0 in CI: {ch['asymptote_zero_in_CI']}")
        print(f"    Power-law α 95%CI = [{ch['decay_exponent_CI95'][0]:.2f}, {ch['decay_exponent_CI95'][1]:.2f}]"
              f"  → α > 0 significant: {ch['decay_significantly_positive']}")

    # Verdict for full norm
    full = out["channels"].get("B_full", {})
    if full.get("asymptote_zero_in_CI") and full.get("decay_significantly_positive"):
        verdict = "BIANCHI_RECOVERY_BOOTSTRAP_CONFIRMED"
    elif full.get("decay_significantly_positive"):
        verdict = "BIANCHI_DECAY_CONFIRMED_ASYMPTOTE_BOUNDED"
    else:
        verdict = "BIANCHI_RECOVERY_NEEDS_MORE_DATA"
    print(f"\nVERDICT: {verdict}")
    out["verdict"] = verdict

    out_path = REPO / "outputs" / "bianchi_bootstrap_CI_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
