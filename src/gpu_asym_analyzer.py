"""GPU-accelerated triangle phase-class asym analyzer (CuPy).

For each existing P5N* snapshots.npz, computes per-seed:
  - persistent-edge mask (delta_xi > c_info, > 50% of timesteps)
  - persistent-triangle enumeration (3-cycles)
  - PPN/PNN class counts (via psi-phase signs)
  - asym = (n_PPN - n_PNN) / (n_PPN + n_PNN)
on the GPU. Falls back to numpy if cupy unavailable.

CuPy advantage: enumerate triangles via N^3 vectorised mask-and operation
instead of Python adjacency loops. At N=1024 the analysis runs in ~5-10s
on a modern GPU vs minutes on CPU.

Usage:
  python src/gpu_asym_analyzer.py [--regime NAME] [--threshold 2.0]
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
PARENT = REPO.parent

LADDER = [
    ("P5N64",  64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz"),
    ("P5N72",  72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz"),
    ("P5N84",  84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz"),
    ("P5N100",100,  "results_d1_p5n100_24seeds/P5N100.snapshots.npz"),
    ("P5N128",128,  "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz"),
    ("P5N200",200,  "results_d1_p5n200_8seeds/P5N200.snapshots.npz"),
    ("P5N256",256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz"),
    ("P5N300",300,  "results_d1_p5n300_12seeds/P5N300.snapshots.npz"),
    ("P5N512", 512,  "results_d1_p5n512_12seeds/P5N512.snapshots.npz"),
]


def get_xp():
    try:
        import cupy as cp
        cp.cuda.Device(0).synchronize()
        return cp, True
    except (ImportError, Exception):
        return np, False


def per_seed_asym_gpu(xi_traj, psi_real, psi_imag, n_lat, threshold,
                        xp):
    """Compute per-seed asym(PPN-PNN) entirely on GPU.

    xi_traj: (n_snap, N, N) float
    psi_real, psi_imag: (n_snap, N) float
    Returns asym = (n_PPN - n_PNN) / (n_PPN + n_PNN).
    """
    xi = xp.asarray(xi_traj, dtype=xp.float32)
    psi_r = xp.asarray(psi_real[-1], dtype=xp.float32)
    psi_i = xp.asarray(psi_imag[-1], dtype=xp.float32)
    n_snap = xi.shape[0]
    # Off-diagonal d_xi
    d_xi = xp.abs(xp.diff(xi, axis=0))   # (n_snap-1, N, N)
    # off-diagonal mask
    eye = xp.eye(n_lat, dtype=bool)
    d_off = d_xi[:, ~eye]
    pos = d_off > 0
    if pos.any():
        v_med = float(xp.median(d_off[pos]))
    else:
        v_med = 1e-6
    c_info = threshold * v_med
    # persistent edge mask: > 50% of (n_snap-1) timesteps
    crossed = (d_xi > c_info)             # (n_snap-1, N, N)
    pers_full = crossed.mean(axis=0) > 0.5  # (N, N)
    pers_full = pers_full & ~eye
    # Enumerate triangles via fully-vectorised N^3 boolean operation
    # adj[i,j,k] = pers[i,j] & pers[j,k] & pers[i,k]
    p = pers_full
    # i<j<k strict ordering: triangle if p[i,j] & p[j,k] & p[i,k]
    # generate all (i<j<k) via index masks - we use full N^3 then filter
    # Memory: N^3 booleans = N^3/8 bytes. At N=1024, 134MB. OK.
    # At N=2048, 1GB - need chunking.
    if n_lat <= 1024:
        i_idx = xp.arange(n_lat).reshape(-1, 1, 1)
        j_idx = xp.arange(n_lat).reshape(1, -1, 1)
        k_idx = xp.arange(n_lat).reshape(1, 1, -1)
        order_mask = (i_idx < j_idx) & (j_idx < k_idx)
        triangle_mask = p[:, :, None] & p[None, :, :] & p[:, None, :]
        triangle_mask = triangle_mask & order_mask
        # Count and gather phase signs
        # psi-phase per node:
        phi = xp.arctan2(psi_i, psi_r)
        # for each triangle (i,j,k), edge phases:
        # d_ij = wrapped(phi_j - phi_i), d_jk, d_ki
        # Use einsum-like broadcasting
        d_ij = phi.reshape(1, -1, 1) - phi.reshape(-1, 1, 1)
        d_jk = phi.reshape(1, 1, -1) - phi.reshape(1, -1, 1)
        d_ki = phi.reshape(-1, 1, 1) - phi.reshape(1, 1, -1)
        # Wrap to (-pi, pi]
        d_ij = xp.angle(xp.exp(1j * d_ij)).real
        d_jk = xp.angle(xp.exp(1j * d_jk)).real
        d_ki = xp.angle(xp.exp(1j * d_ki)).real
        s_ij = xp.sign(d_ij)
        s_jk = xp.sign(d_jk)
        s_ki = xp.sign(d_ki)
        # Count positives per triangle
        n_pos = (s_ij > 0).astype(xp.int32) + (s_jk > 0).astype(xp.int32) + (s_ki > 0).astype(xp.int32)
        # Skip triangles with any zero sign
        nonzero = (s_ij != 0) & (s_jk != 0) & (s_ki != 0)
        valid = triangle_mask & nonzero
        ppn = valid & (n_pos == 2)
        pnn = valid & (n_pos == 1)
        n_PPN = int(ppn.sum())
        n_PNN = int(pnn.sum())
    else:
        # Chunked enumeration for N > 1024
        n_PPN, n_PNN = _chunked_triangle_classify(
            p, psi_r, psi_i, n_lat, xp)
    if n_PPN + n_PNN == 0:
        return None
    return (n_PPN - n_PNN) / (n_PPN + n_PNN)


def _chunked_triangle_classify(p, psi_r, psi_i, n_lat, xp,
                                  chunk=256):
    """Memory-saving chunked triangle enumeration for very large N."""
    phi = xp.arctan2(psi_i, psi_r)
    n_PPN = 0
    n_PNN = 0
    for i_start in range(0, n_lat, chunk):
        i_end = min(i_start + chunk, n_lat)
        i_idx = xp.arange(i_start, i_end)
        for j_start in range(i_start + 1, n_lat, chunk):
            j_end = min(j_start + chunk, n_lat)
            j_idx = xp.arange(j_start, j_end)
            # k indices: j_end .. n_lat
            if j_end >= n_lat:
                continue
            k_idx = xp.arange(j_end, n_lat)
            # Build sub-tensor i in [i_start, i_end), j in [j_start, j_end), k in [j_end, n_lat)
            p_ij = p[i_idx][:, j_idx]    # (Ci, Cj)
            p_jk = p[j_idx][:, k_idx]    # (Cj, Ck)
            p_ik = p[i_idx][:, k_idx]    # (Ci, Ck)
            triangle_mask = (p_ij[:, :, None] & p_jk[None, :, :]
                              & p_ik[:, None, :])
            phi_i = phi[i_idx].reshape(-1, 1, 1)
            phi_j = phi[j_idx].reshape(1, -1, 1)
            phi_k = phi[k_idx].reshape(1, 1, -1)
            d_ij = xp.angle(xp.exp(1j * (phi_j - phi_i))).real
            d_jk = xp.angle(xp.exp(1j * (phi_k - phi_j))).real
            d_ki = xp.angle(xp.exp(1j * (phi_i - phi_k))).real
            s_ij = xp.sign(d_ij)
            s_jk = xp.sign(d_jk)
            s_ki = xp.sign(d_ki)
            nonzero = (s_ij != 0) & (s_jk != 0) & (s_ki != 0)
            valid = triangle_mask & nonzero
            n_pos = ((s_ij > 0).astype(xp.int32)
                      + (s_jk > 0).astype(xp.int32)
                      + (s_ki > 0).astype(xp.int32))
            n_PPN += int((valid & (n_pos == 2)).sum())
            n_PNN += int((valid & (n_pos == 1)).sum())
    return n_PPN, n_PNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", default=None,
                        help="single regime to run (default: all)")
    parser.add_argument("--threshold", type=float, default=2.0)
    args = parser.parse_args()
    xp, gpu_ok = get_xp()
    print("=" * 80)
    print(f"GPU asym analyzer — {'CuPy(GPU)' if gpu_ok else 'NumPy(CPU)'} backend")
    print(f"  threshold factor c_info = {args.threshold} x median |Delta Xi|")
    print("=" * 80)
    rows = []
    targets = [(r, n, p) for (r, n, p) in LADDER
               if args.regime is None or r == args.regime]
    for regime, n_lat, rel in targets:
        fp = PARENT / rel
        if not fp.exists():
            print(f"  {regime}: missing")
            continue
        z = np.load(fp, allow_pickle=True)
        snaps = z["edge_xi_snapshots"]
        psi_r = z["psi_real_snapshots"]
        psi_i = z["psi_imag_snapshots"]
        n_seeds = int(snaps.shape[0])
        per_seed_asym = []
        t0 = time.perf_counter()
        for s in range(n_seeds):
            a = per_seed_asym_gpu(snaps[s], psi_r[s], psi_i[s],
                                     n_lat, args.threshold, xp)
            if a is not None:
                per_seed_asym.append(a)
        if gpu_ok:
            xp.cuda.Device(0).synchronize()
        elapsed = time.perf_counter() - t0
        if not per_seed_asym:
            continue
        arr = np.array(per_seed_asym)
        mu = float(arr.mean())
        sd = float(arr.std())
        unc = sd / np.sqrt(len(arr))
        sigma_vs_0 = mu / unc if unc > 1e-12 else float("nan")
        print(f"  {regime:<7} N={n_lat:>4d} n_s={len(arr):>3d}  "
              f"asym={mu:+.5f}+-{unc:.5f} ({sigma_vs_0:+.2f}sigma)  "
              f"elapsed {elapsed:.2f}s")
        rows.append({
            "regime": regime, "N": n_lat,
            "n_seeds": len(arr),
            "asym_mean": mu, "asym_std": sd, "asym_unc": unc,
            "sigma_vs_0": sigma_vs_0,
            "elapsed_seconds": elapsed,
            "asym_per_seed": arr.tolist(),
        })
    # Symanzik fit
    if len(rows) >= 3:
        N = np.array([r["N"] for r in rows], dtype=float)
        y = np.array([r["asym_mean"] for r in rows])
        u = np.array([r["asym_unc"] for r in rows])
        x = 1.0 / N
        w = 1.0 / np.maximum(u, 1e-6) ** 2
        A = np.column_stack([np.ones_like(x), x])
        AtWA = A.T @ (w[:, None] * A)
        AtWy = A.T @ (w * y)
        coef = np.linalg.solve(AtWA, AtWy)
        a_inf, b = coef
        cov = np.linalg.inv(AtWA)
        a_unc = float(np.sqrt(cov[0, 0]))
        target = -np.pi / 200
        print(f"\n  Asymptote a_inf = {a_inf:+.5f} +/- {a_unc:.5f}")
        print(f"  Target -pi/200 = {target:+.5f}, diff = {a_inf - target:+.5f}")
    bundle = {
        "method": "gpu_asym_analyzer",
        "backend": "CuPy" if gpu_ok else "NumPy",
        "threshold_factor": args.threshold,
        "rows": rows,
    }
    out = REPO / "outputs" / "gpu_asym_analyzer.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
