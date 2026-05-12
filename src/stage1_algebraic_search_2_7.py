"""Stage 1: systematic algebraic search for 2/7 (= f_neg asymptote)
in System-R rationals + integers + pi.

Targets:
  T1 = 2/7 = 0.2857142857  (f_neg asymptote)
  T2 = ln(5/2) = 0.9162907  (alternative form)
  T3 = -pi/200 = -0.01570796  (asym asymptote)
  T4 = -1/64 = -0.015625    (alternative asym candidate)
  T5 = 5/2 = 2.5            (ratio f_pos/f_neg)

Algebraic alphabet (System-R rationals + π + small integers):
  alpha_xi = 9/10 = 0.9
  beta_pi  = 15/16 = 0.9375
  gamma    = 1/10 = 0.1
  D        = 67/80 = 0.8375
  eps2     = 1/20 = 0.05
  N_gen    = 3
  pi
  small integers 1..10

Search strategy:
  - One- and two-element combinations (sum, diff, product, ratio, etc)
  - Three-element combinations
  - Integer linear combinations of rationals
  - Trigonometric (sin, cos, tan, arctan)
  - Special: 2/(2 N_gen + 1), Klein-quartic / hyperbolic triangle hints

Tolerance: ratio match to within 1e-4 relative.

Output: outputs/stage1_algebraic_search_2_7.json
"""
from __future__ import annotations
import json
import sys
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# System-R rationals
ALPHA_XI = Fraction(9, 10)
BETA_PI = Fraction(15, 16)
GAMMA = Fraction(1, 10)
D_R = Fraction(67, 80)
EPS2 = Fraction(1, 20)
N_GEN = Fraction(3, 1)

R_DICT = {
    "alpha_xi": (float(ALPHA_XI), ALPHA_XI),
    "beta_pi": (float(BETA_PI), BETA_PI),
    "gamma": (float(GAMMA), GAMMA),
    "D": (float(D_R), D_R),
    "eps2": (float(EPS2), EPS2),
    "N_gen": (3.0, N_GEN),
    "1": (1.0, Fraction(1, 1)),
    "2": (2.0, Fraction(2, 1)),
    "3": (3.0, Fraction(3, 1)),
    "4": (4.0, Fraction(4, 1)),
    "5": (5.0, Fraction(5, 1)),
    "7": (7.0, Fraction(7, 1)),
    "8": (8.0, Fraction(8, 1)),
}

PI = float(np.pi)

TARGETS = {
    "2/7 (f_neg)": (2.0 / 7.0, Fraction(2, 7)),
    "5/7 (f_pos)": (5.0 / 7.0, Fraction(5, 7)),
    "5/2 (ratio)": (5.0 / 2.0, Fraction(5, 2)),
    "ln(5/2)": (float(np.log(5 / 2)), None),
    "-pi/200 (asym)": (-PI / 200.0, None),
    "-1/64": (-1.0 / 64.0, Fraction(-1, 64)),
}

TOL_RELATIVE = 1e-3   # 0.1% tolerance
TOL_ABSOLUTE = 1e-3   # for near-zero values


def relative_error(predicted, target):
    if abs(target) < 1e-12:
        return abs(predicted - target)
    return abs(predicted - target) / abs(target)


def search_pure_rational_combinations():
    """Search ratios a/b, products a*b, sums a+b, etc on the rational
    alphabet for matches to rational targets."""
    matches = []
    keys = list(R_DICT.keys())
    for k1, k2 in combinations(keys, 2):
        v1, f1 = R_DICT[k1]
        v2, f2 = R_DICT[k2]
        # Operations
        ops = [
            (f"{k1}+{k2}", v1 + v2, f1 + f2),
            (f"{k1}-{k2}", v1 - v2, f1 - f2),
            (f"{k2}-{k1}", v2 - v1, f2 - f1),
            (f"{k1}*{k2}", v1 * v2, f1 * f2),
            (f"{k1}/{k2}", v1 / v2 if v2 != 0 else float("inf"),
             f1 / f2 if f2 != 0 else None),
            (f"{k2}/{k1}", v2 / v1 if v1 != 0 else float("inf"),
             f2 / f1 if f1 != 0 else None),
        ]
        for name, val, frac in ops:
            for tname, (tval, tfrac) in TARGETS.items():
                if abs(tval) > 1e-9:
                    err = relative_error(val, tval)
                    if err < TOL_RELATIVE:
                        matches.append({
                            "expression": name,
                            "value": val,
                            "target": tname,
                            "target_value": tval,
                            "rel_error": err,
                            "fraction": str(frac) if frac is not None else None,
                        })
    return matches


def search_three_element_rational():
    """Three-element rational combinations of form a+b*c, a*b+c, a*b*c."""
    matches = []
    keys = list(R_DICT.keys())
    for k1, k2, k3 in combinations(keys, 3):
        v1, _ = R_DICT[k1]
        v2, _ = R_DICT[k2]
        v3, _ = R_DICT[k3]
        forms = [
            (f"{k1}*{k2}+{k3}", v1 * v2 + v3),
            (f"{k1}*{k2}-{k3}", v1 * v2 - v3),
            (f"{k1}+{k2}*{k3}", v1 + v2 * v3),
            (f"{k1}-{k2}*{k3}", v1 - v2 * v3),
            (f"{k1}*{k2}*{k3}", v1 * v2 * v3),
            (f"({k1}+{k2})/{k3}", (v1 + v2) / v3 if v3 != 0 else 0),
            (f"({k1}-{k2})/{k3}", (v1 - v2) / v3 if v3 != 0 else 0),
            (f"{k1}/({k2}+{k3})", v1 / (v2 + v3) if v2 + v3 != 0 else 0),
        ]
        for name, val in forms:
            for tname, (tval, tfrac) in TARGETS.items():
                if abs(tval) > 1e-9:
                    err = relative_error(val, tval)
                    if err < TOL_RELATIVE / 2:  # tighter for 3-elem
                        matches.append({
                            "expression": name,
                            "value": val,
                            "target": tname,
                            "target_value": tval,
                            "rel_error": err,
                        })
    return matches


def search_with_pi():
    """Search expressions involving pi with rational alphabet."""
    matches = []
    keys = list(R_DICT.keys())
    for k1 in keys:
        v1, _ = R_DICT[k1]
        for k2 in keys:
            if k1 == k2:
                continue
            v2, _ = R_DICT[k2]
            forms = [
                (f"pi*{k1}*{k2}", PI * v1 * v2),
                (f"{k1}*{k2}/pi", v1 * v2 / PI),
                (f"pi*{k1}-{k2}", PI * v1 - v2),
                (f"pi*{k1}+{k2}", PI * v1 + v2),
                (f"{k1}*pi/2", v1 * PI / 2),
                (f"{k1}*pi/4", v1 * PI / 4),
                (f"{k1}/pi", v1 / PI),
                (f"{k1}*pi/200", v1 * PI / 200),
            ]
            for name, val in forms:
                for tname, (tval, _) in TARGETS.items():
                    if abs(tval) > 1e-9:
                        err = relative_error(val, tval)
                        if err < TOL_RELATIVE / 2:
                            matches.append({
                                "expression": name,
                                "value": val,
                                "target": tname,
                                "target_value": tval,
                                "rel_error": err,
                            })
    return matches


def search_special_forms():
    """Specifically test natural special forms:
      2/(2*N_gen + 1) = 2/7
      Klein quartic / (2,3,7) hyperbolic triangle hints
      Dimensional combinations
    """
    specials = {
        "2/(2*N_gen + 1)": 2.0 / (2 * 3 + 1),
        "1 - 5/7": 1.0 - 5 / 7,
        "1 - 5/(2*N_gen+1)": 1.0 - 5 / 7,
        "(N_gen-1)/(2*N_gen+1)": 2.0 / 7,
        "2/(N_gen + 4)": 2 / 7,
        "(2,3,7) hyperbolic-area / (2*pi)":
            np.pi * (1 - 1/2 - 1/3 - 1/7) / (2 * np.pi),
        "1/(2*N_gen + 1)": 1.0 / 7,
        "alpha_xi*N_gen / (2*N_gen+1)": float(ALPHA_XI) * 3 / 7,
        "N_gen/(N_gen + 4)": 3 / 7,  # related to triangle filling
        "5/(2*N_gen + 1)": 5 / 7,
        "1 - 2/(2*N_gen + 1)": 5 / 7,
        "alpha_xi*beta_pi": float(ALPHA_XI) * float(BETA_PI),
        "(1 - beta_pi) * 2*N_gen": (1 - float(BETA_PI)) * 6,
        "(1 - beta_pi) / gamma": (1 - float(BETA_PI)) / float(GAMMA),
        "(1 - beta_pi) * pi*N_gen":
            (1 - float(BETA_PI)) * float(np.pi) * 3,
        "16*gamma": 16 * float(GAMMA),
    }
    matches = []
    for name, val in specials.items():
        for tname, (tval, _) in TARGETS.items():
            if abs(tval) > 1e-9:
                err = relative_error(val, tval)
                if err < TOL_RELATIVE * 5:  # broader for specials
                    matches.append({
                        "expression": name,
                        "value": val,
                        "target": tname,
                        "target_value": tval,
                        "rel_error": err,
                    })
    return matches


def main():
    print("=" * 80)
    print("Stage 1: algebraic search for 2/7 (f_neg) and -pi/200 (asym)")
    print("=" * 80)
    print()
    print("Targets:")
    for tname, (tval, tfrac) in TARGETS.items():
        print(f"  {tname:<20s} = {tval:+.10f}"
              + (f"  ({tfrac})" if tfrac else ""))
    print()

    print("--- Pure rational two-element combinations ---")
    m1 = search_pure_rational_combinations()
    for m in sorted(m1, key=lambda x: x["rel_error"])[:30]:
        print(f"  {m['expression']:<25s} = {m['value']:+.6f}  "
              f"-> {m['target']:<20s}  "
              f"err = {m['rel_error']:.2e}"
              + (f"  [{m['fraction']}]" if m.get("fraction") else ""))

    print()
    print("--- Three-element rational combinations ---")
    m2 = search_three_element_rational()
    for m in sorted(m2, key=lambda x: x["rel_error"])[:30]:
        print(f"  {m['expression']:<28s} = {m['value']:+.6f}  "
              f"-> {m['target']:<20s}  err = {m['rel_error']:.2e}")

    print()
    print("--- Combinations with pi ---")
    m3 = search_with_pi()
    for m in sorted(m3, key=lambda x: x["rel_error"])[:30]:
        print(f"  {m['expression']:<25s} = {m['value']:+.6f}  "
              f"-> {m['target']:<20s}  err = {m['rel_error']:.2e}")

    print()
    print("--- Special forms ---")
    m4 = search_special_forms()
    for m in sorted(m4, key=lambda x: x["rel_error"])[:30]:
        print(f"  {m['expression']:<35s} = {m['value']:+.6f}  "
              f"-> {m['target']:<20s}  err = {m['rel_error']:.2e}")

    bundle = {
        "method": "stage1_algebraic_search_for_2_over_7",
        "targets": {k: v[0] for k, v in TARGETS.items()},
        "tolerance_relative": TOL_RELATIVE,
        "matches_pure_rational_2elem": m1,
        "matches_3elem": m2,
        "matches_with_pi": m3,
        "matches_specials": m4,
    }
    out = REPO / "outputs" / "stage1_algebraic_search_2_7.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
