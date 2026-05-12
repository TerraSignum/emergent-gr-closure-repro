"""GPU-accelerated minimal lattice simulator for triangle phase-class
asym N-trend search.

This is a SIMPLIFIED model of the canonical fast-slow lattice dynamics,
designed to capture the ESSENTIAL physics that produces the persistent-
violator triangle phase asymmetry:

  1. Xi matrix evolves toward the M3-closed state via Floyd-Warshall
     iterations in log-space:    d_ik <- min(d_ik, d_ij + d_jk)
     where d_ij = -log Xi_ij. This enforces sub-multiplicativity.
  2. Phase psi(x) relaxes under XY-model coupling weighted by Xi:
        E = -sum_{ij} Xi_ij cos(phi_i - phi_j) + epsilon_CP * sum_i sin(phi_i)
     The epsilon_CP term is a structural CP-breaking from the framework
     (gamma^2-suppressed in System-R).
  3. K, Q fields evolve with their own dynamics (here: random walk
     with small drift), used as anti-symmetric "DCA" weights.
  4. Snapshots taken every M iterations to allow persistent-edge ID.

VALIDATION REQUIRED: this minimal model must reproduce the asym N-trend
seen in the full simulation. Validation step: run at N in {64, 100, 200}
and compare per-seed asym mean to the existing measured values:
   N=64:  asym_meas = +0.0251 +/- 0.0093
   N=100: asym_meas = +0.0027 +/- 0.0060
   N=200: asym_meas = -0.0076 +/- 0.0053

If the minimal model agrees within 2 sigma per regime, run at N=512,
N=1024, N=2048 to PIN the asymptote and distinguish -pi/200 from -1/64.

Tunable parameters:
  --epsilon_cp 0.01  (CP breaking scale, default gamma^2 = 0.01)
  --n_iter 100       (relaxation steps)
  --n_snaps 10       (number of snapshots over trajectory)
  --alpha_phase 0.05 (phase relaxation rate)
  --xi_init_max 0.5  (initial Xi max)

Usage:
  python src/gpu_minimal_lattice_search.py --N 512 --n_seeds 12

Output: outputs/gpu_minimal_lattice_search_N<N>.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def get_xp():
    try:
        import cupy as cp
        cp.cuda.Device(0).synchronize()
        return cp, True
    except (ImportError, Exception):
        return np, False


def m3_closure_step(d, xp):
    """One Floyd-Warshall iteration in log-space:
       d_ik = min(d_ik, d_ij + d_jk).
    For partial relaxation, do K random j's per step.
    """
    n = d.shape[-1]
    rng = xp.random.default_rng(int(xp.random.randint(0, 10**9)))
    j_choices = rng.integers(0, n, size=8).tolist()
    for j in j_choices:
        d_ij = d[..., :, j:j+1]      # (..., N, 1)
        d_jk = d[..., j:j+1, :]      # (..., 1, N)
        d = xp.minimum(d, d_ij + d_jk)
    return d


def phase_relax_step(phi, xi, alpha, eps_cp, xp):
    """XY-model phase relaxation weighted by Xi, with epsilon_CP tilt.
    phi: (n_seeds, N), xi: (n_seeds, N, N) symmetric
    Returns updated phi.
    """
    # Gradient of E = -sum_j Xi_ij cos(phi_i - phi_j)
    # dE/dphi_i = sum_j Xi_ij sin(phi_i - phi_j)
    cos_phi = xp.cos(phi)
    sin_phi = xp.sin(phi)
    # cos_phi_i - cos_phi_j: shape (s, N, N)
    sin_diff_ij = (sin_phi[:, :, None] * cos_phi[:, None, :]
                    - cos_phi[:, :, None] * sin_phi[:, None, :])
    grad = (xi * sin_diff_ij).sum(axis=-1)
    # Add epsilon_CP CP-breaking term: dE_extra/dphi_i = eps_cp * cos(phi_i)
    grad += eps_cp * cos_phi
    return phi - alpha * grad


def run_minimal_simulation(n_lat, n_seeds, n_iter, n_snaps,
                              alpha_phase, eps_cp, xi_init_max,
                              noise_amp=0.05, decay_rate=0.02,
                              alpha_xi=0.15, seed=0, xp=np):
    """Run the minimal lattice simulation. Captures:
       - M3-closure pull (sub-multiplicative growth)
       - Stochastic noise (simulates lattice fluctuations)
       - Decay toward bulk Xi (prevents saturation)
       - Phase coupling with epsilon_CP CP-breaking term

    The trade-off is set such that some edges cross the c_info threshold
    persistently (becoming 'persistent-violator' edges in audit terms),
    while others fluctuate in the bulk.
    """
    rng = xp.random.default_rng(seed)
    # Initialize Xi symmetric in [0.05, xi_init_max]
    Xi = rng.uniform(0.05, xi_init_max,
                       size=(n_seeds, n_lat, n_lat)).astype(xp.float32)
    Xi = 0.5 * (Xi + xp.transpose(Xi, (0, 2, 1)))
    diag_idx = xp.arange(n_lat)
    Xi[:, diag_idx, diag_idx] = 1.0
    # Phi uniform
    phi = rng.uniform(-xp.pi, xp.pi,
                        size=(n_seeds, n_lat)).astype(xp.float32)
    # Bulk reference Xi (mean, decays toward this)
    xi_bulk = float(0.3)
    snap_every = max(1, n_iter // n_snaps)
    xi_snaps = []
    snap_steps = []
    for it in range(n_iter):
        # M3-closure pull (in log-space)
        d = -xp.log(xp.maximum(Xi, 1e-9))
        d = xp.minimum(d, 50.0)
        d = m3_closure_step(d, xp)
        Xi_target = xp.exp(-d)
        # Update: alpha_xi pull toward M3 target, decay_rate toward bulk
        Xi = (1 - alpha_xi - decay_rate) * Xi \
             + alpha_xi * Xi_target \
             + decay_rate * xi_bulk
        # Noise injection (lattice fluctuations)
        noise = (rng.standard_normal(size=Xi.shape).astype(xp.float32)
                  * noise_amp)
        # symmetric noise
        noise = 0.5 * (noise + xp.transpose(noise, (0, 2, 1)))
        Xi = Xi + noise * Xi * (1 - Xi)
        # Clip + diagonal + symmetrize
        Xi = xp.clip(Xi, 0.001, 0.999)
        Xi[:, diag_idx, diag_idx] = 1.0
        Xi = 0.5 * (Xi + xp.transpose(Xi, (0, 2, 1)))
        # Phase relaxation
        phi = phase_relax_step(phi, Xi, alpha_phase, eps_cp, xp)
        if (it + 1) % snap_every == 0:
            xi_snaps.append(Xi.copy())
            snap_steps.append(it + 1)
    if len(xi_snaps) < n_snaps:
        xi_snaps.append(Xi.copy())
        snap_steps.append(n_iter)
    xi_traj = xp.stack(xi_snaps, axis=1)
    return xi_traj, phi


def per_seed_asym(xi_traj_seed, phi_seed, n_lat, threshold, xp):
    """Compute asym for one seed."""
    d_xi = xp.abs(xp.diff(xi_traj_seed, axis=0))
    eye = xp.eye(n_lat, dtype=bool)
    d_off = d_xi[:, ~eye]
    pos = d_off > 0
    if pos.any():
        v_med = float(xp.median(d_off[pos]))
    else:
        v_med = 1e-6
    c_info = threshold * v_med
    crossed = (d_xi > c_info)
    pers_full = crossed.mean(axis=0) > 0.5
    pers_full = pers_full & ~eye
    if not bool(pers_full.any()):
        return None
    # Triangle enumeration
    p = pers_full
    i_idx = xp.arange(n_lat).reshape(-1, 1, 1)
    j_idx = xp.arange(n_lat).reshape(1, -1, 1)
    k_idx = xp.arange(n_lat).reshape(1, 1, -1)
    order_mask = (i_idx < j_idx) & (j_idx < k_idx)
    triangle_mask = p[:, :, None] & p[None, :, :] & p[:, None, :]
    triangle_mask = triangle_mask & order_mask
    # phase signs
    phi = phi_seed
    d_ij = xp.angle(xp.exp(1j * (phi.reshape(1, -1, 1) - phi.reshape(-1, 1, 1)))).real
    d_jk = xp.angle(xp.exp(1j * (phi.reshape(1, 1, -1) - phi.reshape(1, -1, 1)))).real
    d_ki = xp.angle(xp.exp(1j * (phi.reshape(-1, 1, 1) - phi.reshape(1, 1, -1)))).real
    s_ij = xp.sign(d_ij)
    s_jk = xp.sign(d_jk)
    s_ki = xp.sign(d_ki)
    nonzero = (s_ij != 0) & (s_jk != 0) & (s_ki != 0)
    valid = triangle_mask & nonzero
    n_pos = ((s_ij > 0).astype(xp.int32) + (s_jk > 0).astype(xp.int32)
              + (s_ki > 0).astype(xp.int32))
    n_PPN = int((valid & (n_pos == 2)).sum())
    n_PNN = int((valid & (n_pos == 1)).sum())
    if n_PPN + n_PNN == 0:
        return None
    return (n_PPN - n_PNN) / (n_PPN + n_PNN)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--n_seeds", type=int, default=12)
    p.add_argument("--n_iter", type=int, default=100)
    p.add_argument("--n_snaps", type=int, default=10)
    p.add_argument("--alpha_phase", type=float, default=0.05)
    p.add_argument("--epsilon_cp", type=float, default=0.01)
    p.add_argument("--xi_init_max", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    xp, gpu_ok = get_xp()
    print("=" * 80)
    print(f"GPU minimal lattice search — N={args.N}, n_seeds={args.n_seeds}")
    print(f"  Backend: {'CuPy(GPU)' if gpu_ok else 'NumPy(CPU)'}")
    print(f"  Iter={args.n_iter}, snaps={args.n_snaps}, "
          f"alpha={args.alpha_phase}, eps_cp={args.epsilon_cp}, "
          f"xi_init_max={args.xi_init_max}")
    print("=" * 80)
    t0 = time.perf_counter()
    xi_traj, phi = run_minimal_simulation(
        args.N, args.n_seeds, args.n_iter, args.n_snaps,
        args.alpha_phase, args.epsilon_cp, args.xi_init_max,
        seed=args.seed, xp=xp)
    if gpu_ok:
        xp.cuda.Device(0).synchronize()
    sim_time = time.perf_counter() - t0
    print(f"\nSimulation done in {sim_time:.1f}s "
          f"({sim_time/args.n_seeds:.2f}s per seed)")
    # Per-seed asym
    asym_list = []
    for s in range(args.n_seeds):
        a = per_seed_asym(xi_traj[s], phi[s], args.N,
                            args.threshold, xp)
        if a is not None:
            asym_list.append(a)
    if not asym_list:
        print("No asym values (no persistent triangles found)")
        return
    arr = np.array(asym_list)
    mu = float(arr.mean())
    sd = float(arr.std())
    unc = sd / np.sqrt(len(arr))
    print(f"\nAsym(PPN-PNN) per-seed: n={len(arr)}")
    print(f"  Per-seed values: {[f'{a:+.4f}' for a in arr]}")
    print(f"  Mean: {mu:+.5f}")
    print(f"  Std:  {sd:+.5f}")
    print(f"  Unc of mean: {unc:.5f}")
    print(f"  Sigma vs 0: {mu/unc:+.2f}")
    print(f"\nReference values (full simulation):")
    print(f"  N=64  asym_meas = +0.0251 +/- 0.0093")
    print(f"  N=100 asym_meas = +0.0027 +/- 0.0060")
    print(f"  N=200 asym_meas = -0.0076 +/- 0.0053")
    print(f"  N=300 asym_meas = -0.0098 +/- 0.0061")
    bundle = {
        "method": "gpu_minimal_lattice_search",
        "params": {
            "N": args.N, "n_seeds": args.n_seeds,
            "n_iter": args.n_iter, "n_snaps": args.n_snaps,
            "alpha_phase": args.alpha_phase,
            "epsilon_cp": args.epsilon_cp,
            "xi_init_max": args.xi_init_max,
            "threshold": args.threshold,
        },
        "backend": "CuPy" if gpu_ok else "NumPy",
        "sim_time_seconds": sim_time,
        "asym_per_seed": arr.tolist(),
        "asym_mean": mu,
        "asym_std": sd,
        "asym_unc": unc,
    }
    out = REPO / "outputs" / f"gpu_minimal_lattice_search_N{args.N}.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
