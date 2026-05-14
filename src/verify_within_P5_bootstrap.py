"""(E1) Within-P5 sequence Bootstrap CI on Λ_t and per-direction closure.

Within-regime sequence eliminates regime-physics drift and gives the
cleanest test of N→∞ convergence to α_ξ² = 0.81.

P5 family with stored ff_K_seed:
  N=50 (P5), 64 (P5N64), 72 (P5N72_v2), 84 (P5N84_v2),
  100 (P5N100), 200 (P5N200_v2)

Bootstrap CI on Symanzik 2+4 asymptote and power-law decay.

Output: outputs/within_P5_bootstrap_audit.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


class _BlockCupy:
    def find_module(self, name, path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self
    def load_module(self, name):
        raise ImportError("cupy disabled")
sys.meta_path.insert(0, _BlockCupy())

from _d1_npz_discovery import find_d1_npz
from verify_galerkin_runner_A_hessian_ricci import (
    edge_to_matrix, per_seed_galerkin)

PARENT = REPO.parent

# Within-P5 sequence with cleaned ff_K_seed
P5_LADDER = [
    ("P5",      50, "canonical"),
    ("P5N64",   64, "canonical"),
    ("P5N72",   72, "snapshot_v2"),
    ("P5N84",   84, "snapshot_v2"),
    ("P5N100", 100, "canonical"),
    ("P5N200", 200, "snapshot_v2"),
    ("P5N256", 256, "canonical"),
    ("P5N512", 512, "canonical"),
    ("P5N300", 300, "snapshot"),  # new: results_d1_p5n300_12seeds/P5N300.snapshots.npz
]
ALPHA_XI = 9.0/10.0
GAMMA = 1.0/10.0


def get_path(reg, src):
    # Always defer to find_d1_npz which checks
    # _kq_fixed / _24seeds / _12seeds / _8seeds / _v2 / canonical
    # in precedence order — the src tag is informational.
    return find_d1_npz(reg, REPO)


def lambda_t_per_seed(reg, n_lat, p):
    """Returns list of seed-level Λ_t = mean(T_00 - G_00) values."""
    if not p or not p.exists(): return None
    d = np.load(p, allow_pickle=True)
    if "dense_cell_edge_xi_values" in d.keys():
        e = d["dense_cell_edge_xi_values"]
        a = d["dense_cell_node_amplitude_values"]
        ph = d["dense_cell_node_phase_values"]
        n_seeds = min(e.shape[0], 32)
        xi_seed = lambda s: edge_to_matrix(e[s], n_lat)
        psi_seed = lambda s: a[s] * np.exp(1j*ph[s])
    elif "edge_xi_snapshots" in d.keys():
        n_seeds = int(d["edge_xi_snapshots"].shape[0])
        xi_seed = lambda s: d["edge_xi_snapshots"][s, -1, :, :].copy()
        psi_seed = lambda s: d["psi_real_snapshots"][s, -1, :] + 1j*d["psi_imag_snapshots"][s, -1, :]
    else:
        return None

    seed_means = []
    for s in range(n_seeds):
        xi_mat = xi_seed(s); np.fill_diagonal(xi_mat, 1.0)
        psi = psi_seed(s)
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        g00 = np.asarray(prep["g_00_h"])
        t00 = np.asarray(prep["t00"])
        mask = (t00 > 0.05) & np.isfinite(t00) & np.isfinite(g00)
        if not np.any(mask): continue
        seed_means.append(float(np.mean(t00[mask] - g00[mask])))
    return seed_means


def symanzik_24(N_arr, y_arr):
    A = np.column_stack([np.ones_like(N_arr), 1.0/N_arr**2, 1.0/N_arr**4])
    coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
    return float(coef[0])


def main() -> int:
    print("="*100)
    print("(E1) Within-P5 sequence Bootstrap-CI on Λ_t convergence to α_ξ²")
    print("="*100)
    print(f"\nLadder: P5/P5N64/P5N72/P5N84/P5N100/P5N200")
    print()

    seed_data = []  # list of (N, [seed1, seed2, ...])
    for reg, n_lat, src in P5_LADDER:
        path = get_path(reg, src)
        Lt_seeds = lambda_t_per_seed(reg, n_lat, path)
        if Lt_seeds is None or not Lt_seeds:
            print(f"  {reg} N={n_lat}: file missing"); continue
        seed_data.append((n_lat, Lt_seeds))
        mean_v = float(np.mean(Lt_seeds))
        std_v = float(np.std(Lt_seeds))
        print(f"  {reg:<10} N={n_lat:>4}: per-seed Λ_t = {Lt_seeds} → mean={mean_v:.4f}, std={std_v:.4f}")

    # Bootstrap: sample seeds with replacement within each N, refit Symanzik
    print()
    print("Bootstrap (n_boot=2000): sample seeds with replacement at each N, refit Symanzik 2+4")
    rng = np.random.default_rng(42)
    n_boot = 2000
    asymptotes = []
    for _ in range(n_boot):
        N_arr, y_arr = [], []
        for n_lat, seeds in seed_data:
            idx = rng.integers(0, len(seeds), size=len(seeds))
            y_resampled = np.mean([seeds[i] for i in idx])
            N_arr.append(n_lat); y_arr.append(y_resampled)
        try:
            asymp = symanzik_24(np.array(N_arr, dtype=float), np.array(y_arr))
            asymptotes.append(asymp)
        except Exception:
            continue
    asymptotes = np.array(asymptotes)
    asymptotes = asymptotes[np.isfinite(asymptotes) & (np.abs(asymptotes) < 5.0)]

    asym_med = float(np.median(asymptotes))
    asym_lo, asym_hi = np.percentile(asymptotes, [2.5, 97.5])

    print()
    print(f"Within-P5 Symanzik^∞ asymptote:")
    print(f"  median = {asym_med:.4f}")
    print(f"  95% CI = [{asym_lo:.4f}, {asym_hi:.4f}]")
    print(f"  α_ξ² = 0.810 in CI? {asym_lo <= 0.810 <= asym_hi}")
    print(f"  α_ξ² + γ² = 0.820 in CI? {asym_lo <= 0.820 <= asym_hi}")
    print(f"  α_ξ² - γ²/2 = 0.805 in CI? {asym_lo <= 0.805 <= asym_hi}")

    # Also: at largest N (P5N200), what's the seed-spread?
    if seed_data:
        last_N, last_seeds = seed_data[-1]
        print()
        print(f"Largest-N (N={last_N}) seed-spread: {last_seeds}, mean={np.mean(last_seeds):.4f}, std={np.std(last_seeds):.4f}")
        print(f"  Distance to α_ξ²=0.810: {abs(np.mean(last_seeds) - 0.810):.4f}")

    out = {
        "method": "within_P5_bootstrap",
        "n_boot": int(n_boot),
        "ladder": [(n_lat, len(seeds)) for n_lat, seeds in seed_data],
        "asymptote_median": asym_med,
        "asymptote_CI95": [float(asym_lo), float(asym_hi)],
        "alpha_xi_sq_in_CI": bool(asym_lo <= 0.810 <= asym_hi),
        "predictions_in_CI": {
            "alpha_xi_sq_0.810": bool(asym_lo <= 0.810 <= asym_hi),
            "alpha_xi_sq_plus_gamma_sq_0.820": bool(asym_lo <= 0.820 <= asym_hi),
            "alpha_xi_sq_minus_half_gamma_sq_0.805": bool(asym_lo <= 0.805 <= asym_hi),
        },
        "per_regime_seeds": {f"N={n}": seeds for n, seeds in seed_data},
    }
    out_path = REPO / "outputs" / "within_P5_bootstrap_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
