"""Detailed follow-up to T7/T8: stronger discriminators between
feedback-attractor and structural-co-scaling readings of
kappa_t = Lambda_t/T_00 = 0.987.

Three new tests:

T9: K/Q-independent Xi-reference test
  Compute Var(Xi_full) and mean(|Xi|) per regime as K/Q-independent
  Xi-field-physics references. Then form ratios:
    R_Lambda_Xi = Lambda_t / Var(Xi)
    R_T_Xi      = T_00 / Var(Xi)
  Compare canonical to K/Q-perturbed at N=128.
    - If feedback: kappa_t preserved, but R_Lambda_Xi and R_T_Xi may
      both shift together with K/Q.
    - If Lambda_t is sourced by Xi-physics directly (not K/Q):
      R_Lambda_Xi same at canonical and K/Q-perturbed -> sharp
      discriminator. R_T_Xi would still differ.
    - If T_00 is K/Q-coupled but Lambda_t isn't, R_Lambda_Xi/R_T_Xi
      would differ at K/Q-perturbed.

T10: C1 constraint residual at K/Q-perturbed
  alpha_xi_eff(N) + gamma vs 1, where gamma is held fixed at the
  canonical single-point readout. The K/Q-bug HALVES Lambda_t to
  0.418 -> alpha_xi_eff = sqrt(0.418) = 0.647.
    - If alpha_xi = sqrt(Lambda_t) is universal, C1 = 0.647 + 0.10
      - 1 = -0.253 (massive failure at K/Q-perturbed).
    - If feedback determines kappa_t but not alpha_xi separately,
      sqrt(Lambda_t) is NOT alpha_xi_eff at K/Q-perturbed; we need
      a renormalised reading like alpha_xi_eff = sqrt(Lambda_t/T_00 *
      <T_00>_canonical). Test this renormalised reading too.

T11: True FFT wavelength decomposition (sound-wave proper test)
  For each canonical regime's seed-averaged Xi matrix:
    - Take 2D FFT of Xi(i,j)
    - Bin |F[Xi]|^2 by radial wavenumber k_phys = (2 pi / N) * m
    - Compute power(k_phys) per bin
  Compare across regimes:
    - For each k_phys bin, CV across N
    - Long-wavelength (low k_phys) should be N-stable -> small CV
    - Short-wavelength (high k_phys) should drift with cutoff -> large CV

DO NOT modify the manuscript; report only.
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


def find_d1_npz_canonical(regime: str):
    p = find_d1_npz(regime, REPO)
    if p is not None and p.exists():
        return p
    ext = REPO.parent
    rl = regime.lower()
    cand = []
    if rl in ("p0", "p1", "p2prime", "p3", "p4", "p5"):
        cand.append(ext / "results_d1_fix17" / f"d1_{rl}.npz")
    if rl in ("p6", "p7", "p8"):
        cand.append(ext / "results_d1_fix16" / rl / f"d1_{rl}.npz")
    if rl == "p6n128":
        cand.append(ext / "results_d1_p6n128_canonical" / "d1_p6n128.npz")
    if rl == "p8n128":
        cand.append(ext / "results_d1_p8n128_canonical" / "d1_p8n128.npz")
    if rl == "p5n128":
        cand.append(ext / "results_d1_p5n128" / "d1_p5n128.npz")
    if rl == "p5n100":
        cand.append(ext / "results_d1_p5n100" / "d1_p5n100.npz")
    if rl == "p5n64":
        cand.append(ext / "results_d1_p5n64" / "d1_p5n64.npz")
    for c in cand:
        if c.exists():
            return c
    return None


def get_xi_matrices(z, n_lat, max_seeds=32):
    """Return list of N x N Xi matrices, accommodating either xi_seed*
    keys (canonical regimes) or dense_cell_edge_xi_values (K/Q-bug
    regimes)."""
    if "dense_cell_edge_xi_values" in z.files:
        edge = z["dense_cell_edge_xi_values"]
        if edge.ndim == 3 and edge.shape[1] == n_lat == edge.shape[2]:
            ns = min(int(edge.shape[0]), max_seeds)
            return [edge[s] for s in range(ns)]
    seed_keys = sorted([k for k in z.files
                        if k.startswith("xi_seed") and k[7:].isdigit()],
                       key=lambda k: int(k[7:]))
    out = []
    for sk in seed_keys[:max_seeds]:
        m = np.asarray(z[sk], dtype=float)
        if m.ndim == 2 and m.shape[0] == m.shape[1] == n_lat:
            out.append(m)
    return out


def compute_xi_stats(xi_list):
    var_full = []
    mean_abs = []
    for xi in xi_list:
        x = np.asarray(xi, dtype=float)
        x = np.where(np.isfinite(x), x, 0.0)
        var_full.append(float(np.var(x)))
        mean_abs.append(float(np.mean(np.abs(x))))
    return {
        "var_mean": float(np.mean(var_full)),
        "var_std": float(np.std(var_full)),
        "mabs_mean": float(np.mean(mean_abs)),
        "mabs_std": float(np.std(mean_abs)),
        "n_seeds": len(var_full),
    }


# ============================================================
# T9: K/Q-independent Xi-reference test
# ============================================================
def t9_xi_reference():
    print("=" * 78)
    print("T9: K/Q-independent Xi-reference (Var(Xi), mean(|Xi|))")
    print("=" * 78)
    pr = json.loads(
        (REPO / "outputs" / "per_regime_lambda_t_universal_audit.json"
         ).read_text())
    by_reg = {r["regime"]: r for r in pr["per_regime"]}

    LADDER = [
        ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
        ("P4", 42), ("P5", 50), ("P6", 60), ("P5N64", 64),
        ("P7", 72), ("P8", 84), ("P5N100", 100),
        ("P6N128", 128), ("P8N128", 128),
    ]
    rows = []
    for reg, n_lat in LADDER:
        p = find_d1_npz_canonical(reg)
        if p is None or not p.exists():
            print(f"  [skip] no d1 npz for {reg}")
            continue
        z = np.load(p, allow_pickle=True)
        xis = get_xi_matrices(z, n_lat, max_seeds=32)
        if not xis:
            print(f"  [skip] no Xi matrices in {reg}")
            continue
        s = compute_xi_stats(xis)
        meta = by_reg.get(reg)
        if meta is None:
            continue
        is_perturbed = reg.endswith("N128") and reg in ("P6N128", "P8N128")
        rows.append({
            "regime": reg, "N": n_lat,
            "var_xi": s["var_mean"], "var_xi_std": s["var_std"],
            "mabs_xi": s["mabs_mean"],
            "Lambda_t": meta["Lambda_t_optimal"],
            "T_00": meta["T_00_med"],
            "kappa_t": meta["Lambda_t_over_T_00_ratio"],
            "perturbed": is_perturbed,
        })

    print(f"{'regime':>8} {'N':>4} {'Var(Xi)':>10} {'mabs':>8} "
          f"{'Lt':>8} {'T00':>8} {'kappa':>8} "
          f"{'Lt/Var':>8} {'T00/Var':>8} {'cls':>5}")
    for r in rows:
        cls = "PERT" if r["perturbed"] else "can"
        r_l_v = r["Lambda_t"] / r["var_xi"] if r["var_xi"] > 0 else float("nan")
        r_t_v = r["T_00"] / r["var_xi"] if r["var_xi"] > 0 else float("nan")
        r["R_Lambda_Xi"] = r_l_v
        r["R_T_Xi"] = r_t_v
        print(f"{r['regime']:>8} {r['N']:>4} "
              f"{r['var_xi']:>10.5f} {r['mabs_xi']:>8.4f} "
              f"{r['Lambda_t']:>8.4f} {r['T_00']:>8.4f} "
              f"{r['kappa_t']:>8.4f} "
              f"{r_l_v:>8.2f} {r_t_v:>8.2f} {cls:>5}")

    can_rows = [r for r in rows if not r["perturbed"]]
    pert_rows = [r for r in rows if r["perturbed"]]
    print()
    if can_rows and pert_rows:
        # Compare canonical at large-N (last 3) vs K/Q-perturbed
        can_last = can_rows[-3:]
        can_var = float(np.mean([r["var_xi"] for r in can_last]))
        can_RL = float(np.mean([r["R_Lambda_Xi"] for r in can_last]))
        can_RT = float(np.mean([r["R_T_Xi"] for r in can_last]))
        per_var = float(np.mean([r["var_xi"] for r in pert_rows]))
        per_RL = float(np.mean([r["R_Lambda_Xi"] for r in pert_rows]))
        per_RT = float(np.mean([r["R_T_Xi"] for r in pert_rows]))
        print(f"  Canonical (N=84..100): Var(Xi) = {can_var:.5f}, "
              f"Lambda/Var = {can_RL:.2f}, T_00/Var = {can_RT:.2f}")
        print(f"  K/Q-perturbed (N=128): Var(Xi) = {per_var:.5f}, "
              f"Lambda/Var = {per_RL:.2f}, T_00/Var = {per_RT:.2f}")
        print()
        print(f"  Var(Xi) ratio perturbed/canonical: "
              f"{per_var / can_var:.4f}")
        print(f"  Lambda_t/Var(Xi) ratio perturbed/canonical: "
              f"{per_RL / can_RL:.4f}")
        print(f"  T_00/Var(Xi) ratio perturbed/canonical: "
              f"{per_RT / can_RT:.4f}")
        print()
        print("  Interpretation:")
        print("    - If Var(Xi) is K/Q-independent (Xi-physics upstream),")
        print("      ratio perturbed/canonical Var = 1.")
        print("    - If Lambda_t/Var(Xi) drops by ~factor 2 at perturbed,")
        print("      Lambda_t scales with K/Q (NOT pure Xi-physics).")
        print("    - If Lambda_t/Var(Xi) is preserved, Lambda_t is")
        print("      sourced by Xi-physics independent of K/Q.")
        rL_change = abs(per_RL / can_RL - 1.0) if can_RL > 0 else float("nan")
        rT_change = abs(per_RT / can_RT - 1.0) if can_RT > 0 else float("nan")
        print()
        if rL_change < 0.10 and rT_change > 0.30:
            verdict = ("Lambda_t/Var(Xi) PRESERVED, T_00/Var(Xi) "
                       "DROPS -> Lambda_t is K/Q-independent (sourced "
                       "by Xi-physics), T_00 is K/Q-coupled. Strong "
                       "discriminator AGAINST simple co-scaling.")
        elif rL_change > 0.30 and rT_change > 0.30:
            verdict = ("BOTH Lambda_t/Var and T_00/Var DROP at perturbed "
                       "-> structural co-scaling: Lambda_t and T_00 both "
                       "depend on K/Q in the same way. Consistent with "
                       "Lambda_t = T_00^rec mediator-form.")
        else:
            verdict = ("Mixed signal: Lambda_t/Var change "
                       f"{rL_change:.2%}, T_00/Var change {rT_change:.2%}.")
        print(f"  VERDICT: {verdict}")
    else:
        verdict = "Insufficient regime overlap"
        per_var = per_RL = per_RT = can_var = can_RL = can_RT = float("nan")
    return {
        "test": "T9_xi_reference_KQ_independence",
        "rows": rows,
        "verdict": verdict,
    }


# ============================================================
# T10: C1 constraint residual at K/Q-perturbed regimes
# ============================================================
def t10_c1_at_perturbed():
    print()
    print("=" * 78)
    print("T10: C1 constraint residual at K/Q-perturbed regimes")
    print("=" * 78)
    GAMMA = 0.100206
    pr = json.loads(
        (REPO / "outputs" / "per_regime_lambda_t_universal_audit.json"
         ).read_text())
    rows = []
    for r in pr["per_regime"]:
        Lt = r["Lambda_t_optimal"]
        T00 = r["T_00_med"]
        kappa = r["Lambda_t_over_T_00_ratio"]
        # Reading A: alpha_xi = sqrt(Lambda_t)
        a_A = math.sqrt(max(Lt, 0.0))
        c1_A = a_A + GAMMA - 1.0
        # Reading B (renormalised feedback):
        # alpha_xi_eff = sqrt(kappa * <T_00>_can) ~ sqrt(0.987 * 0.85)
        # but evaluated at this regime's kappa and a fixed T_00 baseline
        T00_BASELINE = 0.85   # canonical large-N T_00 mean
        a_B = math.sqrt(kappa * T00_BASELINE)
        c1_B = a_B + GAMMA - 1.0
        # Reading C (feedback locks kappa_t, alpha_xi from kappa alone):
        # alpha_xi^2 ~= kappa  (drop normalisation; check shape)
        a_C = math.sqrt(kappa)
        c1_C = a_C + GAMMA - 1.0
        rows.append({
            "regime": r["regime"], "N": r["N"],
            "Lambda_t": Lt, "T_00": T00, "kappa": kappa,
            "alpha_xi_A": a_A, "C1_A": c1_A,
            "alpha_xi_B": a_B, "C1_B": c1_B,
            "alpha_xi_C": a_C, "C1_C": c1_C,
            "perturbed": r["regime"] in ("P6N128", "P8N128"),
        })

    print(f"  Reading A: alpha_xi = sqrt(Lambda_t)")
    print(f"  Reading B: alpha_xi = sqrt(kappa * <T_00>_canonical=0.85)  "
          f"(renormalised feedback)")
    print(f"  Reading C: alpha_xi = sqrt(kappa)  (pure-ratio feedback)")
    print()
    print(f"{'regime':>8} {'N':>4} {'kappa':>8} {'a_A':>7} {'C1_A':>9} "
          f"{'a_B':>7} {'C1_B':>9} {'a_C':>7} {'C1_C':>9} {'cls':>5}")
    for r in rows:
        cls = "PERT" if r["perturbed"] else "can"
        print(f"{r['regime']:>8} {r['N']:>4} {r['kappa']:>8.4f} "
              f"{r['alpha_xi_A']:>7.4f} {r['C1_A']:>+9.4f} "
              f"{r['alpha_xi_B']:>7.4f} {r['C1_B']:>+9.4f} "
              f"{r['alpha_xi_C']:>7.4f} {r['C1_C']:>+9.4f} {cls:>5}")

    can = [r for r in rows if not r["perturbed"]]
    pert = [r for r in rows if r["perturbed"]]
    print()
    print("  Canonical large-N (N=72..100, n=4) C1 |residual| means:")
    can_lateN = [r for r in can if r["N"] >= 72]
    if can_lateN:
        for k in ["A", "B", "C"]:
            mu = float(np.mean(np.abs([r[f"C1_{k}"] for r in can_lateN])))
            print(f"    Reading {k}: <|C1|> = {mu:.4f}")
    print()
    print("  K/Q-perturbed N=128 C1 |residual|:")
    if pert:
        for k in ["A", "B", "C"]:
            mu = float(np.mean(np.abs([r[f"C1_{k}"] for r in pert])))
            print(f"    Reading {k}: <|C1|> = {mu:.4f}")

    print()
    print("  Discrimination logic:")
    print("    Reading A passes at canonical ~0 BUT FAILS at perturbed")
    print("        (~0.25)  -> static-rational reading rejected at K/Q-pert.")
    print("    Reading B passes at canonical AND at perturbed if feedback")
    print("        determines kappa, then alpha_xi reconstitutes via T_00.")
    print("    Reading C tests kappa-only feedback divorced from T_00")
    print("        magnitude.")
    print()

    # Pick the reading that wins:
    a_dev = float(np.mean(np.abs([r["C1_A"] for r in can_lateN + pert])))
    b_dev = float(np.mean(np.abs([r["C1_B"] for r in can_lateN + pert])))
    c_dev = float(np.mean(np.abs([r["C1_C"] for r in can_lateN + pert])))
    print(f"  Total |C1| over (canonical-lateN + perturbed):")
    print(f"    Reading A: {a_dev:.4f}")
    print(f"    Reading B: {b_dev:.4f}")
    print(f"    Reading C: {c_dev:.4f}")
    best = min([("A", a_dev), ("B", b_dev), ("C", c_dev)],
               key=lambda x: x[1])
    print(f"  -> Best: Reading {best[0]} with mean residual {best[1]:.4f}")
    return {
        "test": "T10_C1_at_perturbed",
        "rows": rows,
        "summary_means_lateN_plus_pert": {
            "A": a_dev, "B": b_dev, "C": c_dev,
        },
        "best_reading": best[0],
    }


# ============================================================
# T11: True FFT wavelength decomposition
# ============================================================
def t11_fft_wavelength():
    print()
    print("=" * 78)
    print("T11: True FFT wavelength decomposition (sound-wave proper)")
    print("=" * 78)
    LADDER = [
        ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
        ("P4", 42), ("P5", 50), ("P6", 60), ("P7", 72), ("P8", 84),
    ]
    K_BINS = np.linspace(0, math.pi, 6 + 1)  # 6 radial k-bins, [0, pi]
    BIN_LABELS = [f"[{K_BINS[i]:.2f}, {K_BINS[i+1]:.2f}]"
                  for i in range(len(K_BINS) - 1)]
    rows = []
    for reg, n_lat in LADDER:
        p = find_d1_npz_canonical(reg)
        if p is None or not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        xis = get_xi_matrices(z, n_lat, max_seeds=32)
        if not xis:
            continue
        # Average over seeds
        xi_mean = np.mean([np.where(np.isfinite(x), x, 0.0)
                            for x in xis], axis=0)
        # Subtract per-row mean to remove DC
        xi_mean = xi_mean - np.mean(xi_mean)
        # 2D FFT
        F = np.fft.fft2(xi_mean)
        P = np.abs(F) ** 2
        # Build radial k_phys grid
        n = n_lat
        k1 = 2 * math.pi * np.fft.fftfreq(n)  # in [-pi, pi)
        k2 = 2 * math.pi * np.fft.fftfreq(n)
        K1, K2 = np.meshgrid(k1, k2, indexing="ij")
        K_RAD = np.sqrt(K1 ** 2 + K2 ** 2)
        # Bin
        bin_powers = []
        for i in range(len(K_BINS) - 1):
            mask = (K_RAD >= K_BINS[i]) & (K_RAD < K_BINS[i + 1])
            if not np.any(mask):
                bin_powers.append(0.0)
                continue
            bin_powers.append(float(np.mean(P[mask])))
        total = sum(bin_powers)
        if total > 0:
            bin_powers_norm = [x / total for x in bin_powers]
        else:
            bin_powers_norm = bin_powers
        rows.append({
            "regime": reg, "N": n_lat,
            "bin_powers": bin_powers,
            "bin_powers_norm": bin_powers_norm,
        })

    if not rows:
        print("  [skip] no rows")
        return {"test": "T11_fft_wavelength", "rows": [], "verdict": "skip"}

    # Print per-bin powers (normalised to total) per regime
    print(f"  Bins (k_phys in [0, pi]):")
    for i, lab in enumerate(BIN_LABELS):
        print(f"    Bin {i}: {lab}")
    print()
    print(f"{'regime':>8} {'N':>4} | "
          + "  ".join(f"bin{i}".rjust(7) for i in range(len(K_BINS) - 1)))
    for r in rows:
        cells = "  ".join(f"{v:>7.4f}" for v in r["bin_powers_norm"])
        print(f"{r['regime']:>8} {r['N']:>4} | {cells}")

    # CV per bin across regimes
    print()
    print("  CV (std/mean) per k-bin across regimes:")
    print(f"  {'bin':>4} {'k-range':>14} {'mean':>9} {'std':>9} {'CV':>7}")
    cvs = []
    for i in range(len(K_BINS) - 1):
        col = [r["bin_powers_norm"][i] for r in rows]
        mu = float(np.mean(col))
        sd = float(np.std(col))
        cv = sd / mu if mu > 0 else float("nan")
        cvs.append(cv)
        print(f"  {i:>4} {BIN_LABELS[i]:>14} {mu:>9.5f} "
              f"{sd:>9.5f} {cv:>7.3f}")

    # Sound-wave: low-k CV << high-k CV
    print()
    valid = [(i, c) for i, c in enumerate(cvs)
             if c is not None and math.isfinite(c)]
    if len(valid) >= 4:
        low_k_cv = float(np.mean([c for i, c in valid[:2]]))
        high_k_cv = float(np.mean([c for i, c in valid[-2:]]))
        ratio = low_k_cv / high_k_cv if high_k_cv > 0 else float("nan")
        print(f"  Low-k mean CV (bins 0-1):  {low_k_cv:.3f}")
        print(f"  High-k mean CV (last 2):   {high_k_cv:.3f}")
        print(f"  Ratio low_k_CV / high_k_CV: {ratio:.3f}")
        if ratio < 0.5:
            verdict = ("SUPPORTED: low-k modes more N-stable than high-k "
                       f"(ratio {ratio:.2f}). Sound-wave analogy holds: "
                       "long wavelengths cutoff-insensitive, short "
                       "wavelengths drift with cutoff.")
        elif ratio < 1.0:
            verdict = (f"PARTIAL: low-k more stable than high-k (ratio "
                       f"{ratio:.2f}) but not dramatically.")
        elif ratio < 2.0:
            verdict = (f"NEUTRAL: low-k and high-k similarly stable "
                       f"(ratio {ratio:.2f}).")
        else:
            verdict = (f"REJECTED: low-k LESS stable than high-k "
                       f"(ratio {ratio:.2f}). Sound-wave analogy "
                       "INVERTED on Xi-Fourier spectrum.")
    else:
        ratio = None
        verdict = "Insufficient bins"
    print(f"  VERDICT: {verdict}")
    return {
        "test": "T11_fft_wavelength",
        "k_bins": K_BINS.tolist(),
        "rows": rows,
        "cv_per_bin": cvs,
        "low_high_cv_ratio": ratio,
        "verdict": verdict,
    }


def main() -> int:
    bundle = {}
    bundle["T9"] = t9_xi_reference()
    bundle["T10"] = t10_c1_at_perturbed()
    bundle["T11"] = t11_fft_wavelength()

    print()
    print("=" * 78)
    print("CRITICAL SYNTHESIS T9-T11")
    print("=" * 78)
    print(f"T9 (K/Q-independent Xi-reference): {bundle['T9']['verdict']}")
    print()
    print(f"T10 (C1 at K/Q-perturbed): best reading "
          f"{bundle['T10']['best_reading']}; mean residuals "
          f"A={bundle['T10']['summary_means_lateN_plus_pert']['A']:.4f}, "
          f"B={bundle['T10']['summary_means_lateN_plus_pert']['B']:.4f}, "
          f"C={bundle['T10']['summary_means_lateN_plus_pert']['C']:.4f}")
    print()
    print(f"T11 (FFT wavelength): {bundle['T11']['verdict']}")

    out = REPO / "outputs" / "feedback_detailed_T9_T11.json"
    out.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
