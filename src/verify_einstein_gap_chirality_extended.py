"""Extended chirality-balance witness fit on the within-P5 ladder.

The original five-point chirality fit
(data/einstein_gap_5point_fit.json) sits across regimes P1..P5 at small
lattice sizes N ∈ {28, 30, 36, 42, 50}. With within-regime large-N data
now available at the canonical regime (P5N72, P5N84, P5N128, P5N200,
P5N256, P5N300), the chirality-balance deviation
1 - <chirality_balance>_N can be fit at fixed regime physics, removing
cross-regime confounds from the exponent extraction.

Reads:
  results_d1_fix17/d1_p5.npz                            (N=50)
  results_d1_p5n72_24seeds/P5N72.snapshots.npz               (N=72)
  results_d1_p5n84_24seeds/P5N84.snapshots.npz               (N=84)
  results_d1_p5n128_kq_fixed/P5N128.snapshots.npz                       (N=128)
  results_d1_p5n200_8seeds/P5N200.snapshots.npz                (N=200)
  results_d1_p5n256_12seeds/P5N256.snapshots.npz                (N=256)
  results_d1_p5n300_12seeds/P5N300.snapshots.npz                (N=300)

Writes:
  data/einstein_gap_chirality_within_p5.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_DATA = Path(__file__).resolve().parent.parent / "data"


def reconstruct_xi(edge_xi_flat: np.ndarray, n_nodes: int) -> np.ndarray:
    """Lift packed upper-triangle edge_xi to a symmetric N×N matrix."""
    arr = np.asarray(edge_xi_flat, dtype=float).ravel()
    M = np.zeros((n_nodes, n_nodes), dtype=float)
    iu = np.triu_indices(n_nodes, k=1)
    expected = iu[0].size
    if arr.size == expected:
        M[iu] = arr
        M = M + M.T
    elif arr.size == n_nodes * n_nodes:
        M = arr.reshape(n_nodes, n_nodes)
        M = 0.5 * (M + M.T)
    else:
        return np.zeros((n_nodes, n_nodes))
    np.fill_diagonal(M, 1.0)
    return M


def chirality_balance(xi: np.ndarray, phase: np.ndarray) -> float:
    xi = np.asarray(xi, dtype=float)
    phase = np.asarray(phase, dtype=float).ravel()
    if xi.shape[0] != phase.shape[0]:
        return float("nan")
    G = xi @ xi.T
    G = 0.5 * (G + G.T)
    tr = float(np.trace(G))
    if tr < 1e-12:
        return float("nan")
    G = G / tr
    try:
        _, V = np.linalg.eigh(G)
    except np.linalg.LinAlgError:
        return float("nan")
    V = V[:, ::-1]
    n_modes = min(6, V.shape[1])
    pa = np.cos(phase)
    n = float(np.linalg.norm(pa))
    if n < 1e-12:
        return float("nan")
    pa = pa / n
    proj = V[:, :n_modes].T @ pa
    nq = n_modes // 2
    cq = float(np.mean(np.abs(proj[:nq])))
    cl = float(np.mean(np.abs(proj[nq:n_modes])))
    return float(np.clip(1.0 - abs(cq - cl), 0.0, 1.0))


def _devs_from_d1(npz_path: Path, n_lat: int) -> tuple[float, int]:
    z = np.load(npz_path, allow_pickle=True)
    if "dense_cell_edge_xi_values" not in z.files:
        return float("nan"), 0
    if "dense_cell_node_phase_values" not in z.files:
        return float("nan"), 0
    edge = z["dense_cell_edge_xi_values"]
    phase = z["dense_cell_node_phase_values"]
    n_seeds = edge.shape[0]
    bals = []
    for s in range(n_seeds):
        xi = reconstruct_xi(edge[s], n_lat)
        b = chirality_balance(xi, phase[s])
        if np.isfinite(b):
            bals.append(b)
    if not bals:
        return float("nan"), 0
    return 1.0 - float(np.mean(bals)), len(bals)


def _devs_from_snapshot(npz_path: Path) -> tuple[float, int]:
    z = np.load(npz_path, allow_pickle=True)
    if "edge_xi_snapshots" not in z.files:
        return float("nan"), 0
    edge = z["edge_xi_snapshots"]
    psi_r = z["psi_real_snapshots"]
    psi_i = z["psi_imag_snapshots"]
    n_seeds = int(edge.shape[0])
    n_snap = int(edge.shape[1])
    n_lat = int(edge.shape[2])
    bals = []
    for s in range(n_seeds):
        xi = edge[s, n_snap - 1]
        xi = 0.5 * (xi + xi.T)
        np.fill_diagonal(xi, 1.0)
        phase = np.arctan2(psi_i[s, n_snap - 1].ravel(),
                           psi_r[s, n_snap - 1].ravel())
        if phase.shape[0] != n_lat:
            phase = phase[:n_lat]
        b = chirality_balance(xi, phase)
        if np.isfinite(b):
            bals.append(b)
    if not bals:
        return float("nan"), 0
    return 1.0 - float(np.mean(bals)), len(bals)


def loglog_fit(N: list[float], dev: list[float]) -> dict:
    Ns = np.array(N, dtype=float)
    Ds = np.array(dev, dtype=float)
    mask = Ds > 0
    if mask.sum() < 3:
        return {}
    ln_N = np.log(Ns[mask])
    ln_D = np.log(Ds[mask])
    slope, intercept = np.polyfit(ln_N, ln_D, 1)
    pred = intercept + slope * ln_N
    ss_res = float(np.sum((ln_D - pred) ** 2))
    ss_tot = float(np.sum((ln_D - np.mean(ln_D)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    alpha = -float(slope)
    return {
        "alpha_fit": alpha,
        "C_fit": float(np.exp(intercept)),
        "r2_loglog": r2,
        "alpha_target": 2.0 / 3.0,
        "ratio": alpha / (2.0 / 3.0),
        "deviation_2over3_relative": abs(alpha / (2.0 / 3.0) - 1.0),
        "n_points": int(mask.sum()),
    }


def fixed_alpha_fit(N: list[float], dev: list[float],
                    alpha: float = 2.0 / 3.0) -> dict:
    Ns = np.array(N, dtype=float)
    Ds = np.array(dev, dtype=float)
    A = np.column_stack([np.ones_like(Ns), Ns ** (-alpha)])
    coef, *_ = np.linalg.lstsq(A, Ds, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((Ds - pred) ** 2))
    ss_tot = float(np.sum((Ds - np.mean(Ds)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "alpha_fixed": alpha,
        "Delta_inf": float(coef[0]),
        "C_2over3": float(coef[1]),
        "r2": r2,
    }


def main() -> int:
    print("=" * 80)
    print("Within-P5 chirality-balance ladder for the Einstein-gap exponent")
    print("=" * 80)

    rows = []
    # N=50 from the legacy d1 dense-cell format (P5 baseline anchor)
    p5_50 = REPO_ROOT / "results_d1_fix17" / "d1_p5.npz"
    if p5_50.exists():
        d, k = _devs_from_d1(p5_50, 50)
        rows.append(("P5N50", 50, d, k, "d1"))

    # All within-P5 multi-seed runs at N >= 64 use the snapshot npz format
    # (edge_xi_snapshots + psi_real/imag_snapshots), so we route them
    # uniformly through _devs_from_snapshot. Previously N=72/84/128 were
    # mis-routed through _devs_from_d1 (legacy dense-cell keys), which
    # silently produced n_seeds=0 / deviation=None.
    snapshot_paths = [
        (64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz"),
        (72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz"),
        (84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz"),
        (100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz"),
        (128, "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz"),
        (200, "results_d1_p5n200_8seeds/P5N200.snapshots.npz"),
        (256, "results_d1_p5n256_12seeds/P5N256.snapshots.npz"),
        (300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz"),
        (512, "results_d1_p5n512_12seeds/P5N512.snapshots.npz"),
    ]
    for n, sub in snapshot_paths:
        p = REPO_ROOT / sub
        if p.exists():
            d, k = _devs_from_snapshot(p)
            rows.append((f"P5N{n}", n, d, k, "snapshot"))

    print(f"{'tag':10} {'N':>5} {'n_seeds':>8} {'1-<bal>':>12} {'source':>10}")
    Ns, Ds = [], []
    for tag, N, dev, ns, src in rows:
        if np.isfinite(dev):
            print(f"{tag:10} {N:>5} {ns:>8} {dev:>12.6f} {src:>10}")
            Ns.append(N)
            Ds.append(dev)
        else:
            print(f"{tag:10} {N:>5} {ns:>8} {'nan':>12} {src:>10}")

    print()
    fit_free = loglog_fit(Ns, Ds)
    print("Free-alpha log-log fit:")
    for k, v in fit_free.items():
        print(f"  {k} = {v}")

    print()
    fit_fix = fixed_alpha_fit(Ns, Ds, 2.0 / 3.0)
    print("Fixed alpha = 2/3 fit:  Delta(N) = Delta_inf + C * N^(-2/3)")
    for k, v in fit_fix.items():
        print(f"  {k} = {v}")

    bundle = {
        "title": ("Extended chirality-balance witness on the within-P5 "
                  "lattice ladder; recomputes the 5-point fit at fixed "
                  "regime physics."),
        "N_values": Ns,
        "deviation_means": Ds,
        "rows": [{"tag": t, "N": N, "n_seeds": ns,
                  "deviation": (None if not np.isfinite(d) else d),
                  "source": s}
                 for (t, N, d, ns, s) in rows],
        "free_alpha_fit": fit_free,
        "fixed_alpha_2over3_fit": fit_fix,
        "comparison_to_5point_fit": {
            "5point_alpha_fit": 0.6355052661819749,
            "5point_r2": 0.7791603590019913,
            "5point_relative_to_2over3": 0.04674210073,
        },
    }
    out = PAPER_DATA / "einstein_gap_chirality_within_p5.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
