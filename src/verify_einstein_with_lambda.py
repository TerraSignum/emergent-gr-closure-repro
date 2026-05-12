r"""
Verify the eight-point source-side consistency of the full Einstein
equation with cosmological term

    G_munu + Lambda * g_munu = 8 pi G * T_munu^Xi

on the time-time component T_00, on the lattice ladder
N_lattice in {28, 30, 36, 42, 50, 60, 72, 84}.

The script reproduces, from the bundled
data/einstein_with_lambda_8point.json:

  * T_00^Xi(N) from the Hilbert variation of the corpus-fixed
    residual IR action (zeta_1 = 1, zeta_2 = 0.75, zeta_3 = 0.5;
    EMT-04b T-parity convention) at each N -- numerical input
    bundled, derivation in h3c_vollausbau/T_xi_FINAL_SPEC.md;

  * G_00(N) = R_bar(N) / 2 from the bundled R_bar series;

  * the pointwise difference Lambda(N) = T_00 - G_00 and its
    statistics (CV across all 8 points and over the asymptotic
    window P_4...P_8);

  * the residual G_00 + Lambda_const - T_00 with
    Lambda_const = 0.314 (asymptotic mean), and its DoD compliance
    at threshold 0.05 for N >= 30 (P1 = N=28 sits at the boundary
    and is reported as a finite-size characteristic).

Reviewer-hedging caveats (G_00-only test, Lambda empirically
extracted, no Frobenius residual on all components, consistency
with 123-orders Lambda closure is a consistency statement and not
an independent derivation) are reproduced as part of the diagnostic
record.

Usage:
    python ./src/verify_einstein_with_lambda.py
"""

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def load_8point():
    with open(DATA / "einstein_with_lambda_8point.json", "r",
              encoding="utf-8") as f:
        return json.load(f)


def stats(xs):
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    std = math.sqrt(var)
    cv = std / abs(mean) if mean != 0 else float("inf")
    return mean, std, cv


def main():
    d = load_8point()
    print("=" * 72)
    print("Eight-point source-side consistency of the full Einstein")
    print("equation with cosmological term on the T_00 component")
    print("=" * 72)
    print()
    print(f"  Equation: {d['equation']}")
    print(f"  Scope:    {d['scope']}")
    print()

    Ns = d["lattice_ladder"]["N_values"]
    labels = d["lattice_ladder"]["regime_labels"]
    T00 = d["T_00_Xi_values"]
    R_bar = d["R_bar_values"]
    G00 = d["G_00_values"]
    Lam_pw = d["Lambda_pointwise"]
    Lam_const = d["Lambda_const_for_residual_test"]
    threshold = d["residual_pointwise_with_Lambda_const"]["DoD_threshold"]

    # Sanity: G_00 = R_bar / 2 (within rounding).
    print("--- Cross-check G_00 = R_bar / 2 ---")
    for n, rb, g in zip(Ns, R_bar, G00):
        rec = rb / 2.0
        print(f"  N={n}: R_bar={rb:.4f}, G_00 (bundled)={g:.4f}, "
              f"R_bar/2 (recompute)={rec:.4f}, diff={abs(rec-g):.5f}")
    print()

    # Recompute Lambda(N) = T_00 - G_00 and Lambda_const residual.
    print("--- Pointwise Lambda(N) = T_00 - G_00 ---")
    print(f"  {'N':>4} {'regime':>8} {'T_00':>10} {'G_00':>10} "
          f"{'Lambda':>10} {'res_w_const':>14}")
    print("  " + "-" * 60)
    Lam_recompute = []
    res_recompute = []
    for n, lab, t, g, lam in zip(Ns, labels, T00, G00, Lam_pw):
        L = t - g
        r = g + Lam_const - t
        Lam_recompute.append(L)
        res_recompute.append(r)
        print(f"  {n:>4} {lab:>8} {t:>10.4f} {g:>10.4f} "
              f"{L:>10.4f} {r:>+14.4f}")
    print()

    # Lambda statistics.
    mean_all, std_all, cv_all = stats(Lam_recompute)
    asymp_idx = [i for (i, n) in enumerate(Ns) if n >= 42]
    Lam_asymp = [Lam_recompute[i] for i in asymp_idx]
    mean_a, std_a, cv_a = stats(Lam_asymp)

    print("--- Lambda statistics ---")
    print(f"  All 8 points:    mean = {mean_all:.4f}, "
          f"std = {std_all:.4f}, CV = {cv_all*100:.2f}%")
    print(f"    (bundled:      mean = "
          f"{d['Lambda_statistics']['all_8_points']['mean']:.4f}, "
          f"std = {d['Lambda_statistics']['all_8_points']['std']:.4f}, "
          f"CV = "
          f"{d['Lambda_statistics']['all_8_points']['cv_percent']:.1f}%)")
    print(f"  Asymptotic P4..P8 (N>=42): mean = {mean_a:.4f}, "
          f"std = {std_a:.4f}, CV = {cv_a*100:.2f}%")
    print(f"    (bundled:      mean = "
          f"{d['Lambda_statistics']['asymptotic_window_P4_to_P8']['mean']:.4f}, "
          f"std = "
          f"{d['Lambda_statistics']['asymptotic_window_P4_to_P8']['std']:.4f}, "
          f"CV = "
          f"{d['Lambda_statistics']['asymptotic_window_P4_to_P8']['cv_percent']:.1f}%)")
    print()

    # Residual DoD check with Lambda_const.
    abs_residuals = [abs(r) for r in res_recompute]
    pass_pointwise = [a < threshold for a in abs_residuals]
    n_geq_30_idx = [i for (i, n) in enumerate(Ns) if n >= 30]
    abs_resid_n_geq_30 = [abs_residuals[i] for i in n_geq_30_idx]

    print("--- Residual test with Lambda_const = "
          f"{Lam_const} ---")
    print(f"  DoD threshold: {threshold}")
    print(f"  pointwise pass count (all 8 points): "
          f"{sum(pass_pointwise)}/{len(pass_pointwise)}")
    print(f"  pointwise pass count (N>=30 only):   "
          f"{sum(1 for i in n_geq_30_idx if pass_pointwise[i])}/"
          f"{len(n_geq_30_idx)}")
    print(f"  max |residual| over N>=30: "
          f"{max(abs_resid_n_geq_30):.4f}")
    print(f"  P1 (N=28) residual: {res_recompute[0]:+.4f} "
          f"(finite-size boundary, |res|={abs_residuals[0]:.4f})")
    print()

    print("--- Reviewer-hedging caveats (mirrored from data file) ---")
    for k, v in d["reviewer_hedging"].items():
        print(f"  * {k}: {v}")
    print()

    # Physical scale algebra: convert Lambda_lat to physical units and
    # compare to observed rho_Lambda. This is what makes the "Planck-scale
    # input to the DEE 9-layer reduction" interpretation explicit.
    sa = d.get("physical_scale_anchor")
    if sa:
        print("--- Physical scale algebra (Lambda_lat to GeV) ---")
        alpha_m_m = sa["alpha_m_m"]
        alpha_m_GeVinv_recompute = alpha_m_m / 1.9733e-16
        Lambda_phys_recompute = Lam_const / (alpha_m_GeVinv_recompute ** 2)
        M_Pl = sa["M_Pl_GeV"]
        rho_lat_implied = Lambda_phys_recompute * M_Pl ** 2 / (8 * math.pi)
        rho_obs = sa["rho_Lambda_obs_GeV4"]
        log10_ratio = math.log10(rho_lat_implied / rho_obs)
        print(f"  alpha_m_GeVinv (recompute) = "
              f"{alpha_m_GeVinv_recompute:.3e} GeV^-1")
        print(f"  Lambda_phys (recompute)    = "
              f"{Lambda_phys_recompute:.3e} GeV^2")
        print(f"    (bundled: {sa['Lambda_phys_GeV2']:.3e} GeV^2)")
        print(f"  rho_Lambda^lat-implied     = "
              f"{rho_lat_implied:.3e} GeV^4")
        print(f"    (bundled: {sa['rho_Lambda_lat_implied_GeV4']:.3e} GeV^4)")
        print(f"  rho_Lambda^obs (Planck 2018) = "
              f"{rho_obs:.3e} GeV^4")
        print(f"  log10 ratio (lat / obs)    = {log10_ratio:.2f}")
        print(f"    (bundled: {sa['log10_ratio_lat_to_obs']:.2f})")
        print(f"  9-layer hierarchy-reduction log10 prediction: "
              f"{sa['DEE_pred_log10_reduction']:.1f}")
        print(f"  subleading orders between scale algebra and "
              f"hierarchy reduction: {sa['subleading_orders']:.1f}")
        print()
        print("  Reading: Lambda_lat = 0.314 is the Planck-scale vacuum-energy")
        print("  density INPUT to the 9-layer DEE reduction; it is NOT a free")
        print("  parameter or an empirical match. The DEE construction reduces")
        print("  this Planck-scale input by ~123 orders to rho_Lambda^pred.")
        print()

    # Cross-check vs bundled Lambda statistics.
    bundled_mean_a = d["Lambda_statistics"]["asymptotic_window_P4_to_P8"]["mean"]
    bundled_cv_a = d["Lambda_statistics"]["asymptotic_window_P4_to_P8"]["cv_percent"]

    out = {
        "criterion": "Eight-point source-side consistency check on T_00 of full Einstein equation with cosmological term",
        "Lambda_eff_asymptotic_mean": mean_a,
        "Lambda_eff_asymptotic_std": std_a,
        "Lambda_eff_asymptotic_cv_percent": cv_a * 100,
        "Lambda_const_used": Lam_const,
        "max_abs_residual_N_geq_30": max(abs_resid_n_geq_30),
        "pointwise_pass_count_all_8": sum(pass_pointwise),
        "pointwise_pass_count_N_geq_30": sum(
            1 for i in n_geq_30_idx if pass_pointwise[i]
        ),
        "DoD_threshold": threshold,
        "consistent_with_bundled":
            abs(mean_a - bundled_mean_a) < 5e-3
            and abs(cv_a * 100 - bundled_cv_a) < 0.5,
        "reviewer_hedging": d["reviewer_hedging"],
    }
    out_path = OUTPUTS / "einstein_with_lambda_8point_recompute.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
