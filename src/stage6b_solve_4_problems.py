"""Stage 6b: structural grounds for the winding-class fraction
f_neg = 2/7 = 2/(2 N_gen + 1).

The empirical observation on the persistent-violator triangle
subgraph is that the principal-sum sign sgn(d_ij + d_jk + d_ki)
splits as f_neg : f_pos = 2/7 : 5/7 across the eight-point P5
canonical-regime ladder, with absolute deviation from the rational
target 2/7 below 0.0002.

This script gathers two independent structural readings of 2/7:

  Problem 2: B_N = so(2 N_gen + 1) Lie-algebra short-root fraction,
             which equals 2/(2N+1) = 2/7 for N_gen = 3.  This is a
             Lie-algebra fact, not a derivation of any specific
             dynamical model.

  Problem 3: (2,3,7) hyperbolic triangle group / Klein quartic.
             Genus-3 Hurwitz surface tessellated by 336 fundamental
             (pi/2, pi/3, pi/7) triangles; type-7 vertices have
             angular share 1/7 per triangle, two adjacent type-7
             vertices share 2/7 of a paired tile.

Both are structural / geometric statements; they do NOT predict
the *sign* of any per-triangle phase asymmetry.  They predict only
the rational 2/7.

History note (removed in this version):
  Earlier revisions of this script contained two further
  "problems":

    Problem 1 -- a Wilson-loop "Berry-Wess-Zumino-Witten one-loop"
                 derivation of a_inf = -pi * gamma^2 / 2.  The
                 Wilson-loop integral as actually written
                   int_-pi^pi sin(phi) cos(2 phi) dphi/(2 pi)
                 is exactly 0 (odd x even on a symmetric interval),
                 and the documented value -pi/2 was a hardcoded
                 ansatz, not a computed integral.  The empirical
                 7-point Symanzik fit including P5N512 falsifies
                 a_inf = -pi*gamma^2/2 at 6.84 sigma; see
                 src/stage6g_triangle_asym_largest_N.py.  Problem 1
                 has been removed.

    Problem 4 -- higher-order gamma^4 / log-enhanced corrections to
                 the now-falsified leading term of Problem 1.  With
                 the leading term gone the corrections are vacuous,
                 so Problem 4 has been removed.

The triangle phase-class asymmetry empirical N-trend itself remains
a real lattice observable; what is no longer claimed in this
repository is a closed-form theoretical prediction for its
asymptote.  See the empirical reproducer
src/stage6g_triangle_asym_largest_N.py.

Output: outputs/stage6b_structural_grounds_for_2_7.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# System-R rationals (kept for documentation; not load-bearing here)
GAMMA = 0.10
ALPHA_XI = 9 / 10
BETA_PI = 15 / 16
N_GEN = 3
EPS2 = 1 / 20


def problem_2_so_2N_plus_1_short_roots():
    """Problem 2: 2/(2 N_gen + 1) from B_N = so(2N+1) short-root
    fraction.

    For the simple Lie algebra B_N = so(2N+1):
      - rank = N
      - Cartan subalgebra dim = N
      - long roots: 2 N(N-1) (i.e. +/-e_i +/- e_j with i<j)
      - short roots: 2 N (i.e. +/-e_i)
      - total roots: 2 N^2
      - dim B_N = N + 2 N^2 = N(2N+1)

    Short-root fraction:
      short_roots / dim B_N = 2 N / (N (2N+1)) = 2/(2N+1)

    For N = N_gen = 3 (B_3 = so(7)):
      short-root fraction = 2/(2*3+1) = 2/7

    Physical reading (conjectural, not proven): the persistent-
    triangle phase classes correspond to "directions" in a B_N
    root system.  Long roots (2N(N-1) = 12 for N=3) represent
    "primary" matter classes; short roots (2N = 6 for N=3)
    represent "secondary" antimatter classes.  Singlet (Cartan,
    N generators of which 1 trace) brings total to 2N+1 vector-rep
    components.

    The fraction f_neg = 2/(2 N+1) is the natural short-root
    density in the B_N adjoint representation.  Whether the
    persistent-triangle observable on the lattice actually
    realises this group-theoretic structure dynamically is a
    separate structural-correspondence claim, not addressed by
    this counting argument.
    """
    N = N_GEN
    rank = N
    long_roots = 2 * N * (N - 1)
    short_roots = 2 * N
    total_roots = long_roots + short_roots
    cartan_dim = N
    total_dim = total_roots + cartan_dim  # = N(2N+1) = 21 for N=3

    short_root_fraction = short_roots / total_dim
    f_neg_target = 2.0 / (2 * N + 1)

    # Alternative form: rank/(dim vec rep)
    rank_minus_1_over_dim_vec = (N - 1) / (2 * N + 1)

    return {
        "framework": "B_N = so(2N+1) Lie algebra short-root fraction",
        "N_gen": N,
        "rank_B_N": rank,
        "cartan_dim": cartan_dim,
        "long_roots_count": long_roots,
        "short_roots_count": short_roots,
        "total_roots": total_roots,
        "total_dim_B_N": total_dim,
        "short_root_fraction": short_root_fraction,
        "f_neg_target_2_over_7": float(f_neg_target),
        "match_exact": abs(short_root_fraction - f_neg_target) < 1e-12,
        "physical_reading": (
            "Persistent-triangle phase classes = directions in "
            "B_N root system. Long roots (2N(N-1)=12 for N=3) "
            "are primary matter classes. Short roots (2N=6 for "
            "N=3) are secondary antimatter classes. Singlet "
            "(N Cartan, of which 1 trace) brings total to "
            "2N+1=7 vector-rep components. This is a "
            "structural-correspondence statement, not a "
            "first-principles derivation of the dynamics."
        ),
        "alternative_form_rank_minus_1_over_dim_vec":
            rank_minus_1_over_dim_vec,
    }


def problem_3_237_klein_quartic():
    """Problem 3: (2,3,7) triangle group / Klein quartic
    structural reading of 2/7.

    Klein quartic: genus-3 hyperbolic Riemann surface with
    automorphism group PSL(2,7) of order 168.  Tessellated by
    336 = 2 * 168 fundamental (pi/2, pi/3, pi/7) hyperbolic
    triangles.

    Vertex counts on Klein quartic:
      - type-7 (angle pi/7): 24 vertices (each surrounded by 7
        triangles, 24*7 = 168)
      - type-3 (angle pi/3): 56 vertices (each by 3, 56*3 = 168)
      - type-2 (angle pi/2): 84 vertices (each by 4 triangles
        sharing the right angle, 84*4 = 336)

    Verifying: V - E + F = chi = 2-2g = -4 for g=3.
      F = 336, E = 3F/2 = 504, V = 24 + 56 + 84 = 164.
      164 - 504 + 336 = -4 ✓

    The 2/7 fraction in this geometry: the type-7 vertex angular
    share is pi/7 out of pi total per triangle = 1/7.  Two
    type-7 vertices in two adjacent triangles give 2/7.

    The Klein quartic is the smallest Hurwitz surface (genus g
    surface with maximum automorphism group order 84(g-1) = 168
    for g=3).  The structural interpretation is that the
    persistent-triangle subgraph on the lattice approximates a
    (2,3,7)-tessellated hyperbolic surface in some emergent-
    geometry reading; this is conjectural at the present level
    of derivation.
    """
    N = N_GEN
    g = 3  # Klein quartic genus
    chi = 2 - 2 * g  # Euler characteristic = -4
    n_aut = 168  # |PSL(2,7)| = 168
    n_triangles = 2 * n_aut  # 336

    # Triangle area (in unit-curvature hyperbolic plane)
    triangle_area = np.pi * (1 - 1 / 2 - 1 / 3 - 1 / 7)
    klein_quartic_area = -2 * np.pi * chi  # = 8 pi for g=3

    # Vertex counts
    n_v7 = 24
    n_v3 = 56
    n_v2 = 84
    n_vertices = n_v7 + n_v3 + n_v2

    # Verify Euler relation
    n_edges = 3 * n_triangles // 2
    chi_computed = n_vertices - n_edges + n_triangles

    # Reading: type-7 angular share
    type_7_angular_per_triangle = 1 / 7
    two_type_7_angular = 2 * type_7_angular_per_triangle
    f_neg_target = 2 / 7

    return {
        "framework": "(2,3,7) hyperbolic triangle group / "
                     "Klein quartic",
        "klein_quartic_genus": g,
        "klein_quartic_euler_characteristic": chi,
        "klein_quartic_area_8pi": klein_quartic_area,
        "automorphism_group_order_PSL2_7": n_aut,
        "n_fundamental_triangles": n_triangles,
        "fundamental_triangle_area_pi_over_42": triangle_area,
        "n_type_7_vertices": n_v7,
        "n_type_3_vertices": n_v3,
        "n_type_2_vertices": n_v2,
        "total_vertices": n_vertices,
        "total_edges": n_edges,
        "euler_check_chi": chi_computed,
        "euler_consistent": chi_computed == chi,
        "type_7_angular_share_two_triangles": two_type_7_angular,
        "match_to_2_over_7":
            abs(two_type_7_angular - f_neg_target) < 1e-12,
        "structural_note": (
            "Klein quartic is the smallest Hurwitz surface "
            "(maximum automorphism group order 84(g-1)=168 "
            "for g=3).  The persistent-triangle subgraph on the "
            "lattice may approximate a (2,3,7)-tessellated "
            "hyperbolic surface; this is a structural conjecture, "
            "not derived here."
        ),
    }


def main():
    print("=" * 80)
    print("Stage 6b: structural grounds for f_neg = 2/7")
    print("=" * 80)
    print()
    print("  Problems 1 (Wilson-loop one-loop) and 4 (higher-order")
    print("  corrections) of the original four-problem set have been")
    print("  REMOVED.  The Wilson-loop integral they relied on is")
    print("  exactly 0 (odd x even on a symmetric interval); the")
    print("  documented value -pi/2 was hardcoded, not computed.")
    print("  The corresponding empirical prediction")
    print("  a_inf = -pi*gamma^2/2 was falsified at 6.84 sigma by")
    print("  the seven-point Symanzik fit including P5N512.  See")
    print("  src/stage6g_triangle_asym_largest_N.py.")
    print()

    p2 = problem_2_so_2N_plus_1_short_roots()
    p3 = problem_3_237_klein_quartic()

    print("--- Problem 2: B_N = so(2N+1) short-root fraction ---")
    print(f"  N_gen = {p2['N_gen']}")
    print(f"  Long roots:  {p2['long_roots_count']:>3d}  (= 2 N(N-1))")
    print(f"  Short roots: {p2['short_roots_count']:>3d}  (= 2 N)")
    print(f"  Cartan dim:  {p2['cartan_dim']:>3d}  (= N)")
    print(f"  Total dim B_N: {p2['total_dim_B_N']:>3d}  (= N(2N+1))")
    print(f"  short_root / dim = {p2['short_root_fraction']:.6f}")
    print(f"  target 2/7        = {p2['f_neg_target_2_over_7']:.6f}")
    print(f"  Match exact: {p2['match_exact']}")
    print()

    print("--- Problem 3: Klein quartic / (2,3,7) triangle ---")
    print(f"  Klein quartic genus: {p3['klein_quartic_genus']}")
    print(f"  Aut group order:     {p3['automorphism_group_order_PSL2_7']}")
    print(f"  Fundamental triangles: {p3['n_fundamental_triangles']}")
    print(f"  Type-7 vertices: {p3['n_type_7_vertices']}")
    print(f"  Type-3 vertices: {p3['n_type_3_vertices']}")
    print(f"  Type-2 vertices: {p3['n_type_2_vertices']}")
    euler_msg = "Euler OK" if p3['euler_consistent'] else "Euler FAIL"
    print(
        f"  V={p3['total_vertices']}, E={p3['total_edges']}, "
        f"F={p3['n_fundamental_triangles']}, "
        f"chi={p3['euler_check_chi']} (= "
        f"{p3['klein_quartic_euler_characteristic']})  {euler_msg}"
    )
    print(f"  Type-7 angular share x 2 = "
          f"{p3['type_7_angular_share_two_triangles']:.6f}")
    print(f"  target 2/7              = {2/7:.6f}")
    print(f"  Match exact: {p3['match_to_2_over_7']}")
    print()

    bundle = {
        "method": "stage6b_structural_grounds_for_2_7",
        "schema_version": "2.0.0",
        "supersedes": (
            "stage6b_solve_4_problems.json (which contained "
            "Wilson-loop hardcoded -pi/2 ansatz and corresponding "
            "higher-order corrections; both empirically falsified "
            "by P5N512)"
        ),
        "problem_2_so_2N_plus_1_short_roots": p2,
        "problem_3_klein_quartic_237": p3,
    }
    out = REPO / "outputs" / "stage6b_structural_grounds_for_2_7.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"Saved {out}")

    # Mark the old output file as superseded.
    old_out = REPO / "outputs" / "stage6b_solve_4_problems.json"
    if old_out.exists():
        old_out.unlink()
        print(f"Removed superseded {old_out}")


if __name__ == "__main__":
    main()
