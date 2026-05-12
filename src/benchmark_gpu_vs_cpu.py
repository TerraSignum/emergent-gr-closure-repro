"""(Opt-6) GPU vs CPU benchmark for the Galerkin Hessian-Ricci pipeline.

Tests `per_seed_galerkin` on a P5N200 snapshot with both numpy and cupy
backends. Reports throughput, identifies bottlenecks.
"""
from __future__ import annotations
import time
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def benchmark_numpy(xi_mat, psi, k_field, q_field, n_lat):
    from verify_galerkin_runner_A_hessian_ricci import per_seed_galerkin
    t0 = time.perf_counter()
    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
    t1 = time.perf_counter()
    g00 = np.asarray(prep["g_00_h"])
    t00 = np.asarray(prep["t00"])
    return t1 - t0, g00, t00


def benchmark_cupy(xi_mat_np, psi_np, k_field_np, q_field_np, n_lat):
    try:
        import cupy as cp
    except ImportError:
        return None, None, None
    from verify_galerkin_runner_A_hessian_ricci import per_seed_galerkin
    # Warmup
    cp.cuda.Device(0).synchronize()
    t0 = time.perf_counter()
    prep = per_seed_galerkin(xi_mat_np, psi_np, k_field_np, q_field_np, n_lat, cp)
    cp.cuda.Device(0).synchronize()
    t1 = time.perf_counter()
    g00 = cp.asnumpy(prep["g_00_h"])
    t00 = cp.asnumpy(prep["t00"])
    return t1 - t0, g00, t00


def main() -> int:
    print("="*80)
    print("(Opt-6) GPU (cupy) vs CPU (numpy) benchmark — per_seed_galerkin")
    print("="*80)

    snap = REPO.parent / "results_d1_p5n200_v2" / "P5N200.snapshots.npz"
    if not snap.exists():
        print(f"Snapshot file not found: {snap}"); return 1
    d = np.load(snap, allow_pickle=True)
    n_lat = int(d["n_lat"][0])
    xi_mat = d["edge_xi_snapshots"][0, -1, :, :].copy()
    np.fill_diagonal(xi_mat, 1.0)
    psi = d["psi_real_snapshots"][0, -1, :] + 1j*d["psi_imag_snapshots"][0, -1, :]
    k_field = d["ff_K_seed0"]
    q_field = d["ff_Q_seed0"]
    print(f"\nLoaded P5N200 snapshot, N={n_lat}")

    # CPU
    print("\nNumPy (CPU):")
    times_cpu = []
    for i in range(3):
        t, g00, t00 = benchmark_numpy(xi_mat, psi, k_field, q_field, n_lat)
        times_cpu.append(t)
        print(f"  run {i+1}: {t*1000:.1f} ms")
    print(f"  best: {min(times_cpu)*1000:.1f} ms")

    # GPU
    print("\nCuPy (GPU):")
    try:
        import cupy as cp
        # warmup
        _ = cp.array([1, 2, 3]).sum()
        cp.cuda.Device(0).synchronize()
        times_gpu = []
        for i in range(3):
            t, g00_gpu, t00_gpu = benchmark_cupy(xi_mat, psi, k_field, q_field, n_lat)
            times_gpu.append(t)
            print(f"  run {i+1}: {t*1000:.1f} ms")
        print(f"  best: {min(times_gpu)*1000:.1f} ms")
        speedup = min(times_cpu) / min(times_gpu)
        print(f"\nSpeedup (GPU/CPU): {speedup:.2f}×")
        # Validate identical results
        rel_err_g = float(np.max(np.abs(g00_gpu - g00) / (np.abs(g00) + 1e-12)))
        rel_err_t = float(np.max(np.abs(t00_gpu - t00) / (np.abs(t00) + 1e-12)))
        print(f"Numerical agreement: max rel err g_00 = {rel_err_g:.2e}, t_00 = {rel_err_t:.2e}")
    except Exception as e:
        print(f"  GPU error: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
