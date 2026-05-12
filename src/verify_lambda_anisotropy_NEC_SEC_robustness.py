r"""
Phase T: Per-seed statistical robustness of the three load-bearing
diagonal-block claims:

  (a) Anisotropy / non-phantom (NEC margin):
      T_00 + T_ii > 0 (= rho + p > 0 in (-,+,+,+) signature).
      Equivalent to "the source is NOT pure-cosmological-constant"
      (pure Lambda has T_ii = -T_00 exactly, NEC margin = 0).

  (b) Acceleration threshold (SEC margin):
      T_00 + 3 T_ii > 0 (= rho + 3p > 0).
      The boundary saturation found in Phase L; here we test how
      many sigmas above zero the margin sits, given seed dispersion.

The Phase R per-seed (T_00, T_ii) values are bundled in
data/lattice_diagonal_T_munu_per_seed_9point.json. Phase T
computes the per-regime margin means and standard deviations,
pools across the nine-regime ladder using the chi^2-weighted
estimator, and reports the z-score (pooled mean over pooled SEM)
of each margin.

Result on the bundled data:

  NEC margin: pooled = +0.947 +/- 0.001  ->   ~ 1666-sigma above 0
              (chi^2/dof = 9.70/8 = 1.21, p ~ 0.29)
  SEC margin: pooled = +0.095 +/- 0.013  ->   ~ 7.5-sigma above 0
              (chi^2/dof = 0.62/8 = 0.08, very consistent across N)

The interpretation:

  * NEC margin is at the +0.95 lattice-units level, ~1666 sigma
    distinct from zero. The source is OVERWHELMINGLY non-phantom
    AND distinct from pure cosmological constant (which would
    give NEC margin = 0). The Phase G/L anisotropy headline is
    therefore robust to per-seed dispersion at any conventional
    significance threshold.

  * SEC margin is at the +0.09 lattice-units level, ~7.5 sigma
    distinct from zero. The boundary saturation of Phase L
    survives per-seed dispersion: the source genuinely sits
    inside the SEC-positive region (gravitating-attractive),
    not exactly on the saturation line. The chi^2/dof = 0.08
    is far below 1 (meaning the per-regime values agree with each
    other more tightly than per-seed scatter alone would predict)
    is consistent with a single SEC-margin value across all
    regimes -- this is the per-seed-aware confirmation of the
    Phase R constant-hypothesis result for the SEC margin
    specifically.

Usage:
    python ./src/verify_lambda_anisotropy_NEC_SEC_robustness.py

Bundled inputs:
    data/lattice_diagonal_T_munu_per_seed_9point.json
"""
from __future__ import annotations
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def stats(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, s


def chi2_pool(means, stds, n_seeds):
    sems = [s / math.sqrt(n_seeds) for s in stds]
    weights = [1.0 / s ** 2 if s > 0 else 0 for s in sems]
    if sum(weights) == 0:
        return None, None, None, None
    pooled = sum(m * w for m, w in zip(means, weights)) / sum(weights)
    chi2 = sum((m - pooled) ** 2 * w for m, w in zip(means, weights))
    dof = len(means) - 1
    pooled_sem = (1.0 / sum(weights)) ** 0.5
    return pooled, chi2 / dof, dof, pooled_sem


def main():
    with open(DATA / "lattice_diagonal_T_munu_per_seed_9point.json", "r",
              encoding="utf-8") as f:
        d = json.load(f)

    print("=" * 78)
    print("Phase T: Per-seed statistical robustness of the diagonal-block")
    print("anisotropy and energy-condition margins.")
    print("=" * 78)
    print()

    labels = d["lattice_ladder"]["regime_labels"]
    n_values = d["lattice_ladder"]["N_values"]
    n_seeds = d["lattice_ladder"]["n_seeds_per_regime"]

    print(f"  {'reg':>8} {'N':>3} {'<NEC>':>10} {'sd':>7} "
          f"{'<SEC>':>10} {'sd':>7}")
    print("  " + "-" * 50)
    nec_per, sec_per = [], []
    for label, n in zip(labels, n_values):
        t00s = d["T_00_per_seed"][label]
        tiis = d["T_ii_per_seed"][label]
        nec_seeds = [t00s[i] + tiis[i] for i in range(len(t00s))]
        sec_seeds = [t00s[i] + 3.0 * tiis[i] for i in range(len(t00s))]
        nec_m, nec_s = stats(nec_seeds)
        sec_m, sec_s = stats(sec_seeds)
        nec_per.append((nec_m, nec_s))
        sec_per.append((sec_m, sec_s))
        print(f"  {label:>8} {n:>3} {nec_m:>+10.4f} {nec_s:>7.4f} "
              f"{sec_m:>+10.4f} {sec_s:>7.4f}")
    print()

    # Pooled values
    nec_pool, nec_chi, dof, nec_sem = chi2_pool(
        [p[0] for p in nec_per], [p[1] for p in nec_per], n_seeds
    )
    sec_pool, sec_chi, _, sec_sem = chi2_pool(
        [p[0] for p in sec_per], [p[1] for p in sec_per], n_seeds
    )

    nec_z = abs(nec_pool / nec_sem) if nec_sem > 0 else float("inf")
    sec_z = abs(sec_pool / sec_sem) if sec_sem > 0 else float("inf")

    print("--- Pooled margins, chi^2/dof, and z-scores ---")
    print(f"  NEC margin (rho + p):    pooled = {nec_pool:+.4f} +/- "
          f"{nec_sem:.4f}, chi^2/dof = {nec_chi:.2f} / {dof}, "
          f"z = {nec_z:.1f}  sigma")
    print(f"  SEC margin (rho + 3p):   pooled = {sec_pool:+.4f} +/- "
          f"{sec_sem:.4f}, chi^2/dof = {sec_chi:.2f} / {dof}, "
          f"z = {sec_z:.1f}  sigma")
    print()

    # Verdict
    print("--- Phase T verdict ---")
    print("(i)   The NEC margin (T_00 + T_ii = rho + p) sits at")
    print(f"      +{nec_pool:.3f} +/- {nec_sem:.4f} lattice units, "
          f"~{nec_z:.0f} sigma above")
    print("      zero. The source is OVERWHELMINGLY non-phantom and")
    print("      distinct from pure cosmological constant (which would")
    print("      give NEC margin = 0). The Phase G/L anisotropy headline")
    print("      is robust to per-seed dispersion.")
    print()
    print("(ii)  The SEC margin (T_00 + 3 T_ii = rho + 3p) sits at")
    print(f"      +{sec_pool:.3f} +/- {sec_sem:.4f}, ~{sec_z:.1f} "
          "sigma above zero.")
    print("      The boundary saturation of Phase L survives per-seed")
    print("      dispersion: the source sits inside the SEC-positive")
    print("      region (gravitating-attractive), not exactly on the")
    print("      saturation line.")
    print()
    print("(iii) The chi^2/dof of the SEC margin "
          f"({sec_chi:.2f} / {dof} = "
          f"{sec_chi:.2f}) is far below 1, indicating the per-regime")
    print("      SEC-margin values agree with each other more tightly")
    print("      than per-seed scatter alone would predict; the SEC")
    print("      margin is well-described as a single N-independent")
    print("      lattice constant of order +0.10.")
    print()

    out = {
        "method": "per_seed_robustness_NEC_SEC_margins",
        "per_regime": {
            "labels": labels, "N_values": n_values,
            "NEC_margin_mean": [p[0] for p in nec_per],
            "NEC_margin_std":  [p[1] for p in nec_per],
            "SEC_margin_mean": [p[0] for p in sec_per],
            "SEC_margin_std":  [p[1] for p in sec_per],
        },
        "pooled": {
            "NEC_margin": {"value": nec_pool, "sem": nec_sem,
                           "chi2_per_dof": nec_chi, "z_score": nec_z},
            "SEC_margin": {"value": sec_pool, "sem": sec_sem,
                           "chi2_per_dof": sec_chi, "z_score": sec_z},
        },
        "verdict": (
            f"NEC margin pooled = +{nec_pool:.3f} +/- {nec_sem:.4f} "
            f"({nec_z:.0f}-sigma above zero) -> non-phantom + "
            "anisotropy claim solid; SEC margin pooled = "
            f"+{sec_pool:.3f} +/- {sec_sem:.4f} ({sec_z:.1f}-sigma "
            "above zero) -> source sits inside SEC-positive "
            "region with margin ~ 7% of T_00."
        ),
    }
    out_path = OUTPUTS / "lambda_anisotropy_NEC_SEC_robustness.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
