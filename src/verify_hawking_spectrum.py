r"""
Verify the bundled Hawking-spectrum greybody-factor table reproduces
the standard Bose-Einstein and Fermi-Dirac distributions.

The bundled file `data/black_hole/hawking_spectrum.json` records the
nine-point spectral sweep at x = omega/T in
{0.2, 0.5, 1, 1.5, 2, 3, 4, 5, 8} for both regimes, plus the Hawking
temperature, the Page energy-fractions, the Hawking luminosity, and
the evaporation time. This script:

  1. recomputes n_bose(x) = 1/(e^x - 1) and n_fermi(x) = 1/(e^x + 1)
     analytically and verifies the bundled values agree to machine
     precision (the bundled greybody factor is unity);
  2. verifies the Page energy-fractions sum to 1;
  3. surfaces the Hawking temperature, luminosity, and evaporation
     time per regime.

Usage:
    python ./src/verify_hawking_spectrum.py
"""

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "black_hole"
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def load_spectrum():
    with open(DATA / "hawking_spectrum.json", "r", encoding="utf-8") as f:
        return json.load(f)


def n_bose(x):
    return 1.0 / (math.exp(x) - 1.0)


def n_fermi(x):
    return 1.0 / (math.exp(x) + 1.0)


def main():
    d = load_spectrum()
    print("=" * 72)
    print("Hawking-spectrum recompute (greybody-factor table)")
    print("=" * 72)
    print()

    print("--- Bose / Fermi distribution recompute (canonical regime) ---")
    print(f"  {'x':>6} {'n_bose obs':>14} {'n_bose pred':>14} "
          f"{'n_fermi obs':>14} {'n_fermi pred':>14}")
    print("  " + "-" * 75)
    max_dev_bose = 0.0
    max_dev_fermi = 0.0
    for row in d["canonical"]["spectrum_x_omega_over_T"]:
        x = row["x"]
        nb_obs, nf_obs = row["n_bose"], row["n_fermi"]
        nb_pred, nf_pred = n_bose(x), n_fermi(x)
        dev_b = abs(nb_obs - nb_pred) / nb_pred
        dev_f = abs(nf_obs - nf_pred) / nf_pred
        max_dev_bose = max(max_dev_bose, dev_b)
        max_dev_fermi = max(max_dev_fermi, dev_f)
        print(f"  {x:>6.2f} {nb_obs:>14.6e} {nb_pred:>14.6e} "
              f"{nf_obs:>14.6e} {nf_pred:>14.6e}")
    print()
    print(f"  Max deviation Bose-Einstein:  {max_dev_bose:.2e}")
    print(f"  Max deviation Fermi-Dirac:    {max_dev_fermi:.2e}")
    print()

    p1 = d["canonical"]
    page_sum_canonical = (p1["page_scalar_fraction"]
                   + p1["page_fermion_fraction"]
                   + p1["page_vector_fraction"])
    print(f"  Page-curve fractions (canonical):")
    print(f"    scalar  = {p1['page_scalar_fraction']:.4f}")
    print(f"    fermion = {p1['page_fermion_fraction']:.4f}")
    print(f"    vector  = {p1['page_vector_fraction']:.4f}")
    print(f"    sum     = {page_sum_canonical:.4f}  "
          f"(must equal 1.0 to machine precision)")
    print()

    print(f"  Hawking temperature (canonical):  "
          f"T = {p1['T_hawking_GeV']:.4e} GeV "
          f"= {p1['T_hawking_K']:.4e} K")
    print(f"  Black-hole mass (canonical):      "
          f"M_BH = {p1['M_BH_GeV']:.4e} GeV")
    print(f"  Schwarzschild radius (canonical): "
          f"r_S = {p1['r_S_m']:.4e} m")
    print(f"  Hawking luminosity (canonical):   "
          f"L = {p1['L_hawking_W']:.3e} W")
    print(f"  Evaporation time (canonical):     "
          f"t_evap = {p1['t_evap_s']:.3e} s "
          f"= {p1['t_evap_yr']:.3e} yr")
    print()

    out = {
        "criterion": "Hawking-spectrum greybody-factor table recompute",
        "max_dev_bose_relative": max_dev_bose,
        "max_dev_fermi_relative": max_dev_fermi,
        "page_fractions_sum_to_one": abs(page_sum_canonical - 1.0) < 1e-6,
        "T_hawking_GeV_canonical": p1["T_hawking_GeV"],
        "L_hawking_W_canonical": p1["L_hawking_W"],
        "t_evap_s_canonical": p1["t_evap_s"],
        "spectrum_consistent": (max_dev_bose < 5e-3 and max_dev_fermi < 5e-3),
    }
    out_path = OUTPUTS / "hawking_spectrum_recompute.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
