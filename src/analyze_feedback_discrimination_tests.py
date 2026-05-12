"""Two discriminating tests for the feedback-loop hypothesis:

TEST 7 (perturbed-IC discriminator via K/Q-bug regimes):
  P6N128 and P8N128 ran with default fallback K=0.55, Q=0.45 instead
  of lattice-correct K~1.33, Q~0.29. This is effectively a 'perturbed
  initial condition' on the K/Q sector. If feedback determines kappa_t
  = Lambda_t/T_00 = 0.987, the RATIO should survive the K/Q
  perturbation (same kappa_t ~ 0.987 even with halved T_00).
  If kappa_t shifts under K/Q perturbation, the static-rational
  reading is preferred.

TEST 8 (sound-wave inspired spectral wavelength decomposition):
  IR/UV separation: long wavelengths ('pass through walls') should be
  N-independent; short wavelengths ('bounce off walls') should drift
  with lattice cutoff. Decompose Xi-Gram eigenvalues into IR
  (smallest 3 non-trivial) and UV (largest 3) bands per N. The 4
  walls = 3+1 D analogue: each spatial dimension has a forward/
  backward boundary, plus the time dimension makes 4 wall pairs.

  IR-stability test:
    sigma_eigval_IR(N) across N -> small (preserved across N)
    sigma_eigval_UV(N) across N -> large (drifts with cutoff)

  If the rational identification (alpha_xi, gamma, ...) lives in
  the IR sector (long wavelengths), the values should be UV-cutoff
  independent and the finite-N drift is purely UV-noise.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from _d1_npz_discovery import find_d1_npz


def find_d1_npz_canonical(regime: str) -> Path | None:
    """Direct lookup augmenting find_d1_npz with results_d1_fix17/16
    locations, where the canonical-regime NPZ files actually live."""
    p = find_d1_npz(regime, REPO)
    if p is not None and p.exists():
        return p
    ext = REPO.parent
    cand = []
    rl = regime.lower()
    if rl in ("p0", "p1", "p2prime", "p3", "p4", "p5"):
        cand.append(ext / "results_d1_fix17" / f"d1_{rl}.npz")
    if rl in ("p6", "p7", "p8"):
        cand.append(ext / "results_d1_fix16" / rl / f"d1_{rl}.npz")
    for c in cand:
        if c.exists():
            return c
    return None


def gram_eigenvalues(xi):
    """Trace-normalised Gram eigenvalues, descending."""
    xi_clean = np.where(np.isfinite(xi), xi, 0.0)
    if not np.any(np.abs(xi_clean) > 1e-15):
        return np.zeros(xi.shape[0])
    try:
        sv = np.linalg.svd(xi_clean, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.zeros(xi.shape[0])
    eigvals = sv ** 2
    s = float(np.sum(eigvals))
    if s < 1e-12:
        return np.zeros(xi.shape[0])
    return np.sort(eigvals / s)[::-1]


# ============================================================
# TEST 7: K/Q-bug regimes as effective perturbed-IC discriminator
# ============================================================
def test7_kappa_invariance():
    print("=" * 78)
    print("TEST 7: kappa_t = Lambda_t / T_00 invariance under K/Q perturbation")
    print("=" * 78)
    pr = json.loads(
        (REPO / "outputs" / "per_regime_lambda_t_universal_audit.json"
         ).read_text())
    print(f"{'regime':>10} {'N':>4} {'T_00':>8} {'Lambda_t':>10} {'kappa_t':>10} {'class':>16}")
    canonical = []
    perturbed = []
    for r in pr["per_regime"]:
        kt = r["Lambda_t_over_T_00_ratio"]
        cls = "K/Q-perturbed" if r["regime"].endswith("N128") else "canonical"
        print(f"{r['regime']:>10} {r['N']:>4} {r['T_00_med']:>8.4f} "
              f"{r['Lambda_t_optimal']:>10.4f} {kt:>10.5f} {cls:>16}")
        if r["regime"].endswith("N128"):
            perturbed.append(kt)
        else:
            canonical.append(kt)
    can_arr = np.asarray(canonical)
    per_arr = np.asarray(perturbed)
    print()
    print(f"Canonical regimes (N!=128): "
          f"kappa_t = {float(np.mean(can_arr)):.5f} +- {float(np.std(can_arr)):.5f} "
          f"(n={can_arr.size})")
    print(f"K/Q-perturbed regimes (N=128): "
          f"kappa_t = {float(np.mean(per_arr)):.5f} +- {float(np.std(per_arr)):.5f} "
          f"(n={per_arr.size})")
    print()
    diff_in_sigma = ((float(np.mean(per_arr)) - float(np.mean(can_arr)))
                     / max(float(np.std(can_arr)), 1e-9))
    print(f"  Distance perturbed vs canonical: "
          f"{abs(float(np.mean(per_arr)) - float(np.mean(can_arr))):.5f}")
    print(f"  In units of canonical std: {diff_in_sigma:+.2f}")
    if abs(diff_in_sigma) < 1.5:
        verdict = ("kappa_t is INVARIANT under K/Q perturbation. The ratio "
                   "Lambda_t/T_00 = 0.987 survives effective perturbed-IC, "
                   "supporting a feedback-attractor reading of kappa_t.")
    else:
        verdict = ("kappa_t SHIFTS under K/Q perturbation; ratio is K/Q-"
                   "dependent rather than feedback-determined.")
    print(f"\n  VERDICT: {verdict}")
    return {
        "test": "kappa_t_invariance_under_KQ_perturbation",
        "canonical_kappa": {"mean": float(np.mean(can_arr)),
                             "std": float(np.std(can_arr)),
                             "n": int(can_arr.size)},
        "perturbed_kappa": {"mean": float(np.mean(per_arr)),
                             "std": float(np.std(per_arr)),
                             "n": int(per_arr.size)},
        "absolute_difference": float(abs(np.mean(per_arr)
                                          - np.mean(can_arr))),
        "z_distance": float(diff_in_sigma),
        "verdict": verdict,
    }


# ============================================================
# TEST 8: Sound-wave-inspired IR/UV spectral decomposition
# ============================================================
def test8_ir_uv_decomposition():
    print()
    print("=" * 78)
    print("TEST 8: IR/UV spectral decomposition (sound-wave analogy)")
    print("=" * 78)
    print("Long wavelengths (small eigvals/k) <-> IR <-> 'pass through walls'")
    print("Short wavelengths (large eigvals/k) <-> UV <-> 'bounce off walls'")
    print()
    LADDER = [
        ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
        ("P4", 42), ("P5", 50), ("P6", 60), ("P7", 72), ("P8", 84),
    ]
    rows = []
    for reg, n_lat in LADDER:
        p = find_d1_npz_canonical(reg)
        if p is None or not p.exists():
            print(f"  [skip] no d1 npz for {reg}")
            continue
        z = np.load(p, allow_pickle=True)
        seed_keys = [k for k in z.files
                     if k.startswith("xi_seed") and k[7:].isdigit()]
        seed_keys = sorted(seed_keys, key=lambda k: int(k[7:]))[:4]
        if not seed_keys:
            print(f"  [skip] no xi_seed* in {reg}")
            continue
        ir_top3, uv_top3, mid_top3 = [], [], []
        for sk in seed_keys:
            xi = np.asarray(z[sk], dtype=float)
            if xi.ndim != 2 or xi.shape[0] != xi.shape[1]:
                continue
            eigs = gram_eigenvalues(xi)
            # Eigvals descending: index 0 = largest = UV (in sense
            # of strongest mode; in a Laplacian sense we'd flip).
            # Here we identify:
            #   IR = lowest 3 non-trivial = indices [-3:]
            #   UV = highest 3 = indices [0:3]
            #   MID = middle 3 around N/2
            if eigs.size < 6:
                continue
            uv = eigs[:3]
            ir = eigs[-3:]
            mid_start = max(0, n_lat // 2 - 1)
            mid = eigs[mid_start:mid_start + 3]
            uv_top3.append(uv)
            ir_top3.append(ir)
            mid_top3.append(mid)
        if not uv_top3:
            continue
        uv_arr = np.array(uv_top3)
        ir_arr = np.array(ir_top3)
        mid_arr = np.array(mid_top3)
        # Per-band mean over seeds
        rows.append({
            "regime": reg, "N": n_lat,
            "uv_mean": uv_arr.mean(axis=0).tolist(),
            "ir_mean": ir_arr.mean(axis=0).tolist(),
            "mid_mean": mid_arr.mean(axis=0).tolist(),
            "uv_norm": float(uv_arr.mean()),
            "ir_norm": float(ir_arr.mean()),
            "mid_norm": float(mid_arr.mean()),
        })

    print(f"{'regime':>10} {'N':>4} {'IR (lowest 3)':>27} "
          f"{'UV (highest 3)':>27} {'MID (3)':>22}")
    for r in rows:
        ir_str = "[" + ", ".join(f"{v:.5f}" for v in r["ir_mean"]) + "]"
        uv_str = "[" + ", ".join(f"{v:.4f}" for v in r["uv_mean"]) + "]"
        mid_str = "[" + ", ".join(f"{v:.5f}" for v in r["mid_mean"]) + "]"
        print(f"{r['regime']:>10} {r['N']:>4} {ir_str:>27} "
              f"{uv_str:>27} {mid_str:>22}")

    # IR-stability across N: relative std of IR mean
    ir_norms = np.array([r["ir_norm"] for r in rows])
    uv_norms = np.array([r["uv_norm"] for r in rows])
    mid_norms = np.array([r["mid_norm"] for r in rows])
    cv_ir = float(ir_norms.std() / ir_norms.mean()) if ir_norms.mean() > 0 else float("nan")
    cv_uv = float(uv_norms.std() / uv_norms.mean()) if uv_norms.mean() > 0 else float("nan")
    cv_mid = float(mid_norms.std() / mid_norms.mean()) if mid_norms.mean() > 0 else float("nan")
    print()
    print(f"  IR  mean across regimes: {float(ir_norms.mean()):.5f}, "
          f"CV = {cv_ir:.3f}")
    print(f"  MID mean across regimes: {float(mid_norms.mean()):.5f}, "
          f"CV = {cv_mid:.3f}")
    print(f"  UV  mean across regimes: {float(uv_norms.mean()):.5f}, "
          f"CV = {cv_uv:.3f}")
    print()
    print("Sound-wave hypothesis prediction:")
    print("  IR-band CV << UV-band CV  (IR passes through walls,")
    print("                              UV bounces / drifts with cutoff)")
    print()
    if cv_ir < cv_uv * 0.5:
        verdict = ("IR-band IS substantially more N-stable than UV "
                   f"(CV_IR/CV_UV = {cv_ir/cv_uv:.2f} << 1). Sound-wave "
                   "prediction VERIFIED: long-wavelength modes are "
                   "lattice-cutoff insensitive while short-wavelength "
                   "modes drift with N.")
    elif cv_ir < cv_uv:
        verdict = (f"IR-band more stable than UV (CV_IR/CV_UV = "
                   f"{cv_ir/cv_uv:.2f}) but not dramatically so; "
                   "sound-wave prediction PARTIALLY supported.")
    else:
        verdict = (f"IR-band NOT more stable than UV "
                   f"(CV_IR/CV_UV = {cv_ir/cv_uv:.2f}). Sound-wave "
                   "prediction NOT supported on Xi-Gram spectrum.")
    print(f"  VERDICT: {verdict}")

    # Test correlation: does the per-regime alpha_xi_eff drift correlate
    # with UV-band mean? If yes, the alpha_xi_eff drift is UV-noise.
    pr = json.loads(
        (REPO / "outputs" / "per_regime_lambda_t_universal_audit.json"
         ).read_text())
    alpha_per_reg = {}
    for r in pr["per_regime"]:
        if not r["regime"].endswith("N128"):
            alpha_per_reg[r["regime"]] = math.sqrt(max(r["Lambda_t_optimal"], 0))
    paired_alpha = []
    paired_uv = []
    paired_ir = []
    for r in rows:
        if r["regime"] in alpha_per_reg:
            paired_alpha.append(alpha_per_reg[r["regime"]])
            paired_uv.append(r["uv_norm"])
            paired_ir.append(r["ir_norm"])
    print()
    print("Cross-regime correlation alpha_xi_eff vs IR/UV:")
    if len(paired_alpha) >= 4:
        r_uv = float(np.corrcoef(paired_alpha, paired_uv)[0, 1])
        r_ir = float(np.corrcoef(paired_alpha, paired_ir)[0, 1])
        print(f"  Pearson r(alpha_xi_eff, UV-band) = {r_uv:+.3f}")
        print(f"  Pearson r(alpha_xi_eff, IR-band) = {r_ir:+.3f}")
        if abs(r_uv) > abs(r_ir) * 1.5:
            corr_verdict = ("alpha_xi_eff drift correlates more with UV "
                            "than IR -> drift is UV-cutoff noise, not IR "
                            "physics. This supports a 'rationals live "
                            "in IR' reading.")
        elif abs(r_ir) > abs(r_uv) * 1.5:
            corr_verdict = ("alpha_xi_eff drift correlates more with IR "
                            "than UV -> drift is IR-physics, not UV-noise.")
        else:
            corr_verdict = ("alpha_xi_eff drift correlates similarly with "
                            "both bands -> ambiguous.")
        print(f"  -> {corr_verdict}")
    else:
        r_uv = r_ir = None
        corr_verdict = "Insufficient regime overlap for correlation test."
        print(f"  {corr_verdict}")

    return {
        "test": "ir_uv_spectral_decomposition_sound_wave_analogy",
        "per_regime": rows,
        "ir_mean_across_regimes": float(ir_norms.mean()),
        "uv_mean_across_regimes": float(uv_norms.mean()),
        "mid_mean_across_regimes": float(mid_norms.mean()),
        "CV_IR": cv_ir, "CV_MID": cv_mid, "CV_UV": cv_uv,
        "ratio_CV_IR_over_CV_UV": cv_ir / cv_uv if cv_uv > 0 else None,
        "alpha_xi_correlation": {
            "pearson_r_alpha_vs_UV": r_uv,
            "pearson_r_alpha_vs_IR": r_ir,
        },
        "verdict_ir_stability": verdict,
        "verdict_correlation": corr_verdict,
    }


def main() -> int:
    bundle = {}
    bundle["test_7"] = test7_kappa_invariance()
    bundle["test_8"] = test8_ir_uv_decomposition()

    print()
    print("=" * 78)
    print("CRITICAL SYNTHESIS")
    print("=" * 78)
    t7 = bundle["test_7"]
    t8 = bundle["test_8"]
    print()
    print(f"TEST 7 (kappa_t under K/Q perturbation): "
          f"|delta_kappa| = {t7['absolute_difference']:.5f} "
          f"({t7['z_distance']:+.2f} canonical-sigma)")
    print(f"  -> {t7['verdict']}")
    print()
    ratio = t8.get("ratio_CV_IR_over_CV_UV")
    if ratio is None or not math.isfinite(ratio):
        print("TEST 8 (IR/UV stability ratio): CV_IR/CV_UV = N/A "
              "(insufficient regimes)")
    else:
        print(f"TEST 8 (IR/UV stability ratio): "
              f"CV_IR/CV_UV = {ratio:.3f}")
    print(f"  -> {t8['verdict_ir_stability']}")
    print(f"  -> {t8['verdict_correlation']}")

    out = REPO / "outputs" / "feedback_discrimination_tests.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
