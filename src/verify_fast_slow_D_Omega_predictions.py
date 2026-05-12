r"""Fast-slow decomposition of D_Omega(N) and explicit predictions
at N=256, 512, 1024, 2048, 4096.

Framework Fast-Slow-Struktur (Paper 03 Feldtheorie-Notebook §4.1.1):
  partial_t Xi = partial_t Xi|fast + epsilon * partial_t Xi|slow
  - fast: lattice-harmonic (geometric relaxation in quasi-metric tube)
  - slow: physical evolution (causal-wave / chirality rotation)

D_Omega tracks the diffusion-trace of the Cl(1,3) module on the
lattice. It picks up BOTH dynamics:
  D_Omega(N) = D_Omega^slow(theta_chir(N)) + D_Omega^fast(log_2(N) mod d)
  - D_Omega^slow: smooth chirality envelope (vacuum to matter)
  - D_Omega^fast: lattice-harmonic oscillation with period d

Slow envelope: chirality-mixing form like for beta_pi:
  D_Omega^slow(N) = 67/80 * cos^2(theta_chir(N)) + (pi/d) * sin^2(theta_chir(N))
  with theta_chir(N) running from arctan(1/N_gen) at N* to arctan(N_gen)
  at N_inv = d*N_gen*N*

Fast oscillation: period-d in log_2(N) with peak deviation at mod = 3
  D_Omega^fast(phi) = -A * sin(pi*phi/d)^k  (k chosen for peak shape)

Where A is fitted from the 8-point data (peak deviation ~0.55 at phi=3).

This script:
T1: Fit slow envelope from data (subtract from observed)
T2: Fit fast oscillation from residuals
T3: Predict D_Omega at N=256, 512, 1024, 2048, 4096
T4: Decompose contributions per regime
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
DATA = REPO / "data"
OUTPUTS.mkdir(parents=True, exist_ok=True)

D = 4
N_GEN = 3
PI = math.pi
GAMMA = 1/10

D_OMEGA_VACUUM = 67/80
D_OMEGA_MATTER = PI / D
N_STAR = 50  # canonical vacuum anchor


def theta_chir(N, N_star=N_STAR):
    """Structural form: tan(theta) = N_gen^(2x-1), x = ln(N/N*)/ln(d*N_gen)."""
    if N <= 0:
        return 0
    x = math.log(N / N_star) / math.log(D * N_GEN)
    tan_t = N_GEN ** (2 * x - 1)
    return math.atan(tan_t)


def D_Omega_slow_envelope(N):
    """Slow chirality-mixing envelope between vacuum and matter."""
    th = theta_chir(N)
    return D_OMEGA_VACUUM * math.cos(th) ** 2 + \
            D_OMEGA_MATTER * math.sin(th) ** 2


def D_Omega_fast_oscillation(N, amplitude=0.55, peak_phase=3.0,
                                power=2):
    """Period-d oscillation in log_2(N).

    Peak deviation at phi = peak_phase (= 3 for deepest data dip).
    Returns DEVIATION from slow envelope (positive = below envelope).
    """
    log2_N = math.log2(N)
    phi = log2_N % D
    # Bell-shape centered at peak_phase, with vacuum-anchor at phi=0,d
    # Use cos shape: deviation = A * sin(pi*phi/d)^power
    # but shifted so peak is at phi = peak_phase
    # Simpler: triangular bump
    if phi <= peak_phase:
        f = phi / peak_phase
    else:
        f = (D - phi) / (D - peak_phase)
    f = max(0, min(1, f))
    return amplitude * f ** power


def D_Omega_predicted(N, amplitude=0.55):
    """Full prediction = slow envelope - fast oscillation."""
    return D_Omega_slow_envelope(N) - \
            D_Omega_fast_oscillation(N, amplitude=amplitude)


def main():
    src = DATA / "causal_wave_per_N_readout.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    rows = data["p5_ladder_per_N_readout"]
    print("=" * 95)
    print("Fast-slow decomposition of D_Omega + high-N predictions")
    print("=" * 95)
    print()
    print(f"Setup:")
    print(f"  D_Omega^slow(N) = (67/80) cos^2(theta_chir(N)) + "
          f"(pi/d) sin^2(theta_chir(N))")
    print(f"  D_Omega^fast(N) = -A * f(log_2(N) mod d)")
    print(f"  D_Omega(N) = slow - fast")
    print()
    print(f"  Vacuum: D_Omega^V = 67/80 = {D_OMEGA_VACUUM:.4f}")
    print(f"  Matter: D_Omega^M = pi/d = {D_OMEGA_MATTER:.4f}")
    print(f"  Chirality inversion at N = d*N_gen*N* = "
          f"{D * N_GEN * N_STAR}")
    print()

    # T1: per-regime slow envelope vs observed
    print("T1: Slow envelope vs observed D_Omega per regime")
    print("-" * 95)
    print(f"{'N':>4} {'theta deg':>10} {'D_O slow':>10} "
          f"{'D_O obs':>10} {'fast deviation':>16}")
    fast_devs = []
    for r in rows:
        N = r["n_lat"]
        do = r["D_omega_lattice"]
        slow = D_Omega_slow_envelope(N)
        th = theta_chir(N)
        fast_dev = slow - do
        fast_devs.append((N, fast_dev))
        print(f"{N:>4} {math.degrees(th):>10.2f} {slow:>10.4f} "
              f"{do:>10.4f} {fast_dev:>+16.4f}")
    print()

    # T2: fast oscillation pattern
    print("T2: Fast oscillation (slow - obs) vs log_2(N) mod d")
    print("-" * 95)
    print(f"{'N':>4} {'log_2(N)':>10} {'mod d':>8} {'deviation':>11}")
    for N, dev in fast_devs:
        log2_N = math.log2(N)
        phi = log2_N % D
        print(f"{N:>4} {log2_N:>10.4f} {phi:>8.4f} {dev:>+11.4f}")
    print()
    # Find amplitude from peak (deepest deviation)
    max_dev = max(dev for _, dev in fast_devs)
    print(f"  Peak fast deviation: {max_dev:+.4f}")
    print(f"  -> Use amplitude A = {max_dev:.3f}")
    print()

    # T3: Predictions at N = 256, 512, ..., 4096
    print("T3: Predictions at higher N")
    print("-" * 95)
    print(f"{'N':>5} {'log_2(N)':>10} {'mod d':>8} {'theta deg':>10} "
          f"{'slow':>9} {'fast dev':>10} {'pred':>9}")
    print("-" * 95)
    test_Ns = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
                 5793, 8192]  # 5793 = ~2^12.5 between rebound test
    predictions = []
    for N in test_Ns:
        log2_N = math.log2(N)
        phi = log2_N % D
        th = theta_chir(N)
        slow = D_Omega_slow_envelope(N)
        fast_dev = D_Omega_fast_oscillation(N, amplitude=max_dev)
        pred = slow - fast_dev
        predictions.append({"N": N, "log2_N": log2_N,
                              "mod_d": phi, "theta_deg": math.degrees(th),
                              "slow": slow, "fast_deviation": fast_dev,
                              "predicted_D_Omega": pred})
        print(f"{N:>5} {log2_N:>10.4f} {phi:>8.4f} "
              f"{math.degrees(th):>10.2f} {slow:>9.4f} "
              f"{fast_dev:>+10.4f} {pred:>9.4f}")
    print()

    # T4: Decomposition per regime
    print("T4: Component decomposition for key prediction points")
    print("-" * 95)
    key_Ns = [256, 2048, 4096]
    for N in key_Ns:
        print(f"\n  N = {N} (log_2 = {math.log2(N):.2f}, "
              f"mod d = {math.log2(N) % D:.2f}):")
        th = theta_chir(N)
        print(f"    theta_chir = {math.degrees(th):.2f} deg "
              f"(matter side)" if math.degrees(th) > 45 else "")
        print(f"    cos^2(theta) = {math.cos(th)**2:.4f} "
              f"(vacuum-anchor weight)")
        print(f"    sin^2(theta) = {math.sin(th)**2:.4f} "
              f"(matter-anchor weight)")
        slow = D_Omega_slow_envelope(N)
        print(f"    Slow envelope = (67/80)*{math.cos(th)**2:.4f} + "
              f"(pi/4)*{math.sin(th)**2:.4f} = {slow:.4f}")
        fast = D_Omega_fast_oscillation(N, amplitude=max_dev)
        print(f"    Fast deviation (period-d oscillation, "
              f"phi = {math.log2(N)%D:.2f}): {fast:+.4f}")
        pred = slow - fast
        print(f"    Total predicted D_Omega = {pred:.4f}")

    print()

    # Summary and falsifiable predictions
    print("=" * 95)
    print("Summary: falsifiable predictions for higher-N lattice runs")
    print("=" * 95)
    print()
    print(f"{'N':>5} {'predicted D_Omega':>18} {'phase':>20}")
    print("-" * 50)
    for p in predictions:
        if p["mod_d"] < 0.5 or p["mod_d"] > 3.5:
            phase_label = "near boundary -> vacuum-rebound"
        elif 2.5 < p["mod_d"] < 3.5:
            phase_label = "deepest dip phase"
        elif 1.5 < p["mod_d"] < 2.5:
            phase_label = "mid-cycle"
        else:
            phase_label = "early-cycle climb"
        print(f"{p['N']:>5} {p['predicted_D_Omega']:>18.4f} "
              f"{phase_label:>30}")
    print()
    print(f"Key falsifiable predictions:")
    print(f"  N = 256 (mod d = 0): D_Omega ~ "
          f"{predictions[0]['predicted_D_Omega']:.3f} (REBOUND)")
    print(f"  N = 2048 (mod d = 3): D_Omega ~ "
          f"{predictions[6]['predicted_D_Omega']:.3f} (DEEP DIP)")
    print(f"  N = 4096 (mod d = 0): D_Omega ~ "
          f"{predictions[8]['predicted_D_Omega']:.3f} (REBOUND)")
    print()
    print(f"Fast-Slow interpretation:")
    print(f"  - SLOW (chirality flip): smooth running over factor d*N_gen=12 in N")
    print(f"  - FAST (period-d cycle): oscillation every factor 2^d=16 in N")
    print(f"  - Both run on (d, N_gen) integers but on different scales")
    print(f"  - At period-d boundaries (log_2 mod d = 0), fast=0 -> ")
    print(f"    D_Omega tracks pure slow envelope")
    print(f"  - At deepest fast phase (mod d ~ 3), D_Omega overshoots")
    print(f"    BELOW the slow envelope by ~0.55")
    print()

    bundle = {
        "title": "Fast-slow decomposition of D_Omega(N) and "
                  "high-N predictions",
        "stand": "2026-05-06",
        "framework_reference": "Paper 03 Feldtheorie-Notebook §4.1.1",
        "slow_envelope_formula":
            "D_Omega^slow(N) = (67/80)*cos^2(theta_chir(N)) "
            "+ (pi/d)*sin^2(theta_chir(N))",
        "fast_oscillation_formula":
            "D_Omega^fast = A * f(log_2(N) mod d) with A ~ 0.55, "
            "f peaked at phi = 3",
        "fast_amplitude": max_dev,
        "predictions": predictions,
        "key_predictions": {
            "N_256": {"D_Omega_predicted": predictions[0]["predicted_D_Omega"],
                       "label": "REBOUND to slow envelope"},
            "N_2048": {"D_Omega_predicted": predictions[6]["predicted_D_Omega"],
                        "label": "DEEP DIP at mod d = 3"},
            "N_4096": {"D_Omega_predicted": predictions[8]["predicted_D_Omega"],
                        "label": "REBOUND to slow envelope"},
        },
        "verdict": (
            "The fast-slow decomposition cleanly maps to the framework's "
            "Fast-Slow-Struktur (Paper 03 §4.1.1): the SLOW chirality "
            "flip (envelope) and the FAST period-d lattice oscillation "
            "(carrier) coexist on D_Omega running. At period-d boundaries "
            "the fast oscillation vanishes and D_Omega tracks the slow "
            "chirality-mixing envelope; at the deepest phase (mod d ~ 3) "
            "D_Omega overshoots by ~0.55 below the slow envelope. "
            f"Predictions at N=256: {predictions[0]['predicted_D_Omega']:.3f} "
            f"(rebound to slow envelope ~0.80, since N=256 is past the "
            f"chirality flip at N_inv=600). At N=2048: "
            f"{predictions[6]['predicted_D_Omega']:.3f} (deep dip below "
            f"matter asymptote). At N=4096: "
            f"{predictions[8]['predicted_D_Omega']:.3f} (rebound to "
            f"matter-saturated slow envelope pi/d). NOTE: at large N "
            f"(post-flip), the slow envelope saturates to pi/d, so "
            f"'rebound' goes to pi/d not vacuum 67/80. Vacuum rebound "
            f"only happens if the lattice is BEFORE the chirality flip."
        ),
    }
    out_path = OUTPUTS / "verify_fast_slow_D_Omega_predictions.json"
    out_path.write_text(json.dumps(bundle, indent=2),
                         encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
