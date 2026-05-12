"""Reproducer: CLP closure-threshold sensitivity scan.

For each of the four CLP families (A, B/B4, C, D), reports:
  (i) per-family margin against the family-level closure-domain
      threshold (0.70 for A, C, D; 0.50 heuristic for B/B4 sub-components),
  (ii) the sensitivity of the verdict to threshold tightening:
      maximum threshold the family value would still satisfy.

Reads data/clp_scores.json. No NPZ access.

Usage:
    python ./src/audit_clp_threshold_sensitivity.py
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"


def main() -> int:
    bundle = json.loads((DATA / "clp_scores.json").read_text(encoding="utf-8"))
    axes = bundle["axes"]
    print("=" * 68)
    print("CLP threshold-sensitivity scan")
    print("=" * 68)
    print(f"{'family':<10} {'value':>8} {'thresh':>8} {'margin':>9} {'max_thr_pass':>14} status")
    rows = []
    for name, ax in axes.items():
        val = float(ax["value"])
        thr = float(ax["threshold"])
        margin = val - thr
        max_pass = val
        status = ax.get("status", "")
        print(f"{name:<10} {val:>8.4f} {thr:>8.4f} {margin:>+9.4f} {max_pass:>14.4f} {status}")
        rows.append({
            "family": name, "value": val, "threshold": thr,
            "margin": margin, "max_threshold_pass": max_pass,
            "status": status,
        })
    print()
    sub_b = axes["CLP-B"]["sub_components"]
    print("CLP-B/B4 sub-component sensitivity (heuristic threshold 0.50):")
    sub_rows = []
    for sub_name, sub in sub_b.items():
        a = float(sub["asymptote"])
        ci_lo, ci_hi = sub["bootstrap_CI95"]
        margin = a - 0.50
        ci_lo_margin = ci_lo - 0.50
        print(f"  {sub_name:<12} asymp={a:.4f} CI=[{ci_lo:.4f}, {ci_hi:.4f}] margin={margin:+.4f} CI95-low margin={ci_lo_margin:+.4f}")
        sub_rows.append({
            "name": sub_name, "asymptote": a,
            "ci95": [ci_lo, ci_hi],
            "margin_vs_0p50": margin,
            "ci95_low_margin_vs_0p50": ci_lo_margin,
            "ci95_low_above_0p50": ci_lo >= 0.50,
        })
    print()
    sub_c = axes["CLP-C"]["sub_components"]
    print("CLP-C Gamma-convergence sub-component scan (Dal Maso 1993, threshold 0.50):")
    for sub_name, sub in sub_c.items():
        v = float(sub["value"])
        margin = v - 0.50
        print(f"  {sub_name:<28} value={v:.4f} margin vs 0.50: {margin:+.4f}")

    out = {
        "method": "clp_threshold_sensitivity_scan",
        "per_family": rows,
        "clp_b_sub_components": sub_rows,
        "all_families_above_threshold": all(r["margin"] >= 0 for r in rows),
        "lowest_family": min(rows, key=lambda r: r["margin"])["family"],
        "lowest_family_margin": min(r["margin"] for r in rows),
    }
    out_path = REPO / "outputs" / "audit_clp_threshold_sensitivity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
