r"""
Recompute the Lorentzian 3+1 causal-ordering layer of Section 3 of the
manuscript.

The discrete-to-continuum bridge of the manuscript carries a Lorentzian
3+1 spacetime structure: a clean causal partial order on the underlying
graph, an emergent time direction in addition to the three spatial
dimensions, an existence statement for a Lorentzian metric in both
regimes, and a quantitative scalar that measures the residual
deviation of the lattice light-cone from a perfect Minkowski cone.
This last quantity, the "light-cone fuzziness" f_LC, is identified with
the spectral-dimension residual d_eff_residual = d_eff - 3.0 and is
the right 3+1-aware metric-quality complement to the strictly spatial
Kruskal-Shepard MDS-stress sigma(N).

This script:
  1. loads the bundled Lorentzian-3+1 data file;
  2. surfaces the existence statements (lorentzian_metric_exists,
     proper_time_well_defined, causal_structure_clean) in both regimes;
  3. surfaces the arrow-of-time strength via the bounce action S_bounce
     and the time-asymmetry ratio;
  4. surfaces the proper-time scalars (lattice time step, core-crossing
     time, derived Planck time vs measured Planck time);
  5. surfaces the thermodynamic-time scalars (beta_effective, Tolman
     consistency);
  6. assembles the three complementary metric-quality observables
     (sigma, f_LC, Delta_E) into a compact comparison block.

Usage:
    python ./src/recompute_lorentzian_3plus1.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def load_lorentzian_3plus1():
    with open(DATA / "lorentzian_3plus1.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    d = load_lorentzian_3plus1()
    print("=" * 72)
    print("Lorentzian 3+1 causal-ordering layer recompute (Section 3)")
    print("=" * 72)
    print()

    co = d["causal_ordering"]
    print("--- Causal ordering and Lorentzian-metric existence ---")
    print(f"  d_spacetime              = {co['d_spacetime']}")
    print(f"  d_spatial                = {co['d_spatial']}")
    print(f"  decomposition            = {co['decomposition']}")
    print(f"  causal_structure_clean   = {co['causal_structure_clean']}")
    print(f"  causal_modes_present     = {co['causal_modes_present']}")
    print(f"  causal_mode_mass         = {co['causal_mode_mass']}")
    print(f"  Lorentzian metric exists (canonical):  "
          f"{co['lorentzian_metric_exists_canonical']}")
    print(f"  Lorentzian metric exists (extended):   "
          f"{co['lorentzian_metric_exists_extended']}")
    print(f"  proper time well defined (canonical):  "
          f"{co['proper_time_well_defined_canonical']}")
    print(f"  proper time well defined (extended):   "
          f"{co['proper_time_well_defined_extended']}")
    print(f"  light-cone fuzziness f_LC (canonical):  "
          f"{co['light_cone_fuzziness_canonical']:.6f}")
    print(f"  light-cone fuzziness f_LC (extended):   "
          f"{co['light_cone_fuzziness_extended']:.6f}")
    print()

    aot = d["arrow_of_time"]
    print("--- Arrow of time ---")
    print(f"  Bounce action S_bounce (canonical):    "
          f"{aot['S_bounce_canonical']:.4f}")
    print(f"  Bounce action S_bounce (extended):     "
          f"{aot['S_bounce_extended']:.4f}")
    print(f"  time-asymmetry ratio (canonical):      "
          f"{aot['time_asymmetry_ratio_canonical']:.4e}")
    print(f"  time-asymmetry ratio (extended):       "
          f"{aot['time_asymmetry_ratio_extended']:.4e}")
    print(f"  arrow strength (canonical):            "
          f"{aot['arrow_strength_canonical']}")
    print(f"  arrow strength (extended):             "
          f"{aot['arrow_strength_extended']}")
    print()

    pt = d["proper_time"]
    print("--- Proper time ---")
    print(f"  dt_lattice (canonical, s):    "
          f"{pt['dt_lattice_s_canonical']:.4e}")
    print(f"  dt_lattice (extended,  s):    "
          f"{pt['dt_lattice_s_extended']:.4e}")
    print(f"  Planck time derived (s):      "
          f"{pt['t_Planck_derived_s']:.4e}")
    print(f"  Planck time measured (s):     "
          f"{pt['t_Planck_measured_s']:.4e}")
    print(f"  Planck time ratio:            "
          f"{pt['t_Planck_ratio']:.6f}")
    print()

    tt = d["thermodynamic_time"]
    print("--- Thermodynamic time ---")
    print(f"  beta_effective (canonical):   "
          f"{tt['beta_effective_canonical']:.4f}")
    print(f"  beta_effective (extended):    "
          f"{tt['beta_effective_extended']:.4f}")
    print(f"  Tolman-redshift factor (canonical):  "
          f"{tt['grav_redshift_factor_canonical']:.6f}")
    print(f"  Tolman-redshift factor (extended):   "
          f"{tt['grav_redshift_factor_extended']:.6f}")
    print(f"  Tolman consistency (canonical):      "
          f"{tt['tolman_consistency_canonical']}")
    print(f"  Tolman consistency (extended):       "
          f"{tt['tolman_consistency_extended']}")
    print()

    summ = d["metric_quality_axes_summary"]
    print("--- Three complementary metric-quality observables ---")
    print(f"  {'Observable':<48} {'canonical':>12} {'extended':>12}")
    print("  " + "-" * 75)
    for axis in summ["axes"]:
        name = axis["name"]
        v1 = axis["value_canonical"]
        v2 = axis["value_extended"]
        print(f"  {name:<48} {v1:>12.6f} {v2:>12.6f}")
    print()
    diffs = summ["differences"]
    print(f"  sigma - f_LC at P1:                   "
          f"{diffs['sigma_minus_lightcone_canonical']:>+.6f}")
    print(f"  sigma - f_LC at P2_prime:             "
          f"{diffs['sigma_minus_lightcone_extended']:>+.6f}")
    print(f"  f_LC - Delta_E at P1:                 "
          f"{diffs['lightcone_minus_einstein_gap_canonical']:>+.6f}")
    print(f"  f_LC - Delta_E at P2_prime:           "
          f"{diffs['lightcone_minus_einstein_gap_extended']:>+.6f}")
    print()
    print("  Interpretation: sigma decomposes (heuristically) into a")
    print("  3-D / 3+1 dimension/signature-mismatch contribution")
    print("  (sigma - f_LC ~ 0.11) plus the 3+1-Lorentzian intrinsic")
    print("  light-cone fuzziness (f_LC ~ 0.17). The Einstein-identity")
    print("  gap sits in the same range as f_LC and converges; sigma")
    print("  does not. The light-cone fuzziness is the right")
    print("  3+1-aware metric-quality complement.")
    print()

    out = {
        "criterion": "Lorentzian 3+1 causal-ordering recompute",
        "lorentzian_metric_exists": {
            "canonical": co["lorentzian_metric_exists_canonical"],
            "extended": co["lorentzian_metric_exists_extended"],
        },
        "proper_time_well_defined": {
            "canonical": co["proper_time_well_defined_canonical"],
            "extended": co["proper_time_well_defined_extended"],
        },
        "decomposition": co["decomposition"],
        "d_spacetime": co["d_spacetime"],
        "d_spatial": co["d_spatial"],
        "light_cone_fuzziness": {
            "canonical": co["light_cone_fuzziness_canonical"],
            "extended": co["light_cone_fuzziness_extended"],
        },
        "arrow_of_time": {
            "canonical": aot["arrow_of_time_canonical"],
            "extended": aot["arrow_of_time_extended"],
            "S_bounce_canonical": aot["S_bounce_canonical"],
            "S_bounce_extended": aot["S_bounce_extended"],
            "strength_canonical": aot["arrow_strength_canonical"],
            "strength_extended": aot["arrow_strength_extended"],
        },
        "planck_time_ratio": pt["t_Planck_ratio"],
        "tolman_consistency": {
            "canonical": tt["tolman_consistency_canonical"],
            "extended": tt["tolman_consistency_extended"],
        },
        "metric_quality_table": summ,
    }
    out_path = OUTPUTS / "lorentzian_3plus1_recompute.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
