"""Within-P5 ladder Runner A: per-node 4x4 Galerkin Frobenius residual
on the seven canonical P5-physics sizes
N in {50, 64, 72, 84, 100, 200, 300}.

The within-P5 ladder uses two source classes:
  - canonical d1 npz for N in {50, 64, 100} (these have ff_K/ff_Q
    persisted)
  - snapshot v2 npz for N in {72, 84, 200, 300} (these also have
    ff_K/ff_Q persisted in the v2 schema)

The non-v2 canonical P5N72 / P5N84 npz lack ff_K/ff_Q and would
fall back to the constant default (k=0.55, q=0.45), which
suppresses T_00^rec by a factor of two and returns an artefactual
bimodal result. This script avoids that source.

Writes:
  outputs/within_p5_runner_A_hessian_ricci.json
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
    edge_to_matrix, per_seed_galerkin, frob_residual_three_variants)

PARENT = REPO.parent

LADDER = [
    ("P5",     50,  "canonical"),
    ("P5N64",  64,  "canonical"),
    ("P5N72",  72,  "snapshot_v2"),
    ("P5N84",  84,  "snapshot_v2"),
    ("P5N100", 100, "canonical"),
    ("P5N200", 200, "snapshot_v2"),
    ("P5N300", 300, "snapshot"),
    ("P5N512", 512, "canonical"),
    ("P5N256", 256, "canonical"),
]


def get_path(reg: str, src: str) -> Path | None:
    if src == "canonical":
        return find_d1_npz(reg, REPO)
    if src == "snapshot":
        return PARENT / f"results_d1_{reg.lower()}" / f"{reg}.snapshots.npz"
    return PARENT / f"results_d1_{reg.lower()}_v2" / f"{reg}.snapshots.npz"


def load_seeds(p: Path, n_lat: int):
    """Returns list of (xi_mat, psi, k_field, q_field) per seed."""
    d = np.load(p, allow_pickle=True)
    seeds = []
    if "dense_cell_edge_xi_values" in d.files:
        edge = d["dense_cell_edge_xi_values"]
        amp = d["dense_cell_node_amplitude_values"]
        phase = d["dense_cell_node_phase_values"]
        n_seeds = min(int(edge.shape[0]), 32)
        for s in range(n_seeds):
            xi = edge_to_matrix(edge[s], n_lat)
            np.fill_diagonal(xi, 1.0)
            psi = amp[s] * np.exp(1j * phase[s])
            k = d.get(f"ff_K_seed{s}",
                      np.full((n_lat, n_lat), 0.55))
            q = d.get(f"ff_Q_seed{s}",
                      np.full((n_lat, n_lat), 0.45))
            seeds.append((xi, psi, np.asarray(k), np.asarray(q)))
    elif "edge_xi_snapshots" in d.files:
        edge = d["edge_xi_snapshots"]
        psi_r = d["psi_real_snapshots"]
        psi_i = d["psi_imag_snapshots"]
        n_seeds = min(int(edge.shape[0]), 32)
        for s in range(n_seeds):
            xi = edge[s, -1, :, :].copy()
            xi = 0.5 * (xi + xi.T)
            np.fill_diagonal(xi, 1.0)
            psi = psi_r[s, -1, :].astype(np.complex128).reshape(-1)
            psi = psi + 1j * psi_i[s, -1, :].reshape(-1)
            if psi.shape[0] != n_lat:
                psi = psi[:n_lat]
            k = d.get(f"ff_K_seed{s}",
                      np.full((n_lat, n_lat), 0.55))
            q = d.get(f"ff_Q_seed{s}",
                      np.full((n_lat, n_lat), 0.45))
            seeds.append((xi, psi, np.asarray(k), np.asarray(q)))
    return seeds


def main() -> int:
    print("=" * 78)
    print("Within-P5 Runner A: per-node 4x4 Galerkin Frobenius across")
    print("N in {50, 64, 72, 84, 100, 200, 300} at fixed P5 physics")
    print("=" * 78)
    rows = []
    for reg, n_lat, src in LADDER:
        p = get_path(reg, src)
        if p is None or not p.exists():
            print(f"  {reg} N={n_lat}: file missing")
            continue
        seeds = load_seeds(p, n_lat)
        if not seeds:
            print(f"  {reg} N={n_lat}: no seeds")
            continue
        per_seed = []
        for s, (xi, psi, k, q) in enumerate(seeds):
            prep = per_seed_galerkin(xi, psi, k, q, n_lat, np)
            res = frob_residual_three_variants(prep, np)
            per_seed.append(res)
        seed_blind_med = [float(r["blind_frob_median"]) for r in per_seed]
        seed_blind_mean = [float(r["blind_frob_mean"]) for r in per_seed]
        seed_struct_med = [float(r["struct_frob_median"]) for r in per_seed]
        seed_struct_mean = [float(r["struct_frob_mean"]) for r in per_seed]
        med = float(np.mean(seed_blind_med))
        mn = float(np.mean(seed_blind_mean))
        med_struct = float(np.mean(seed_struct_med))
        mn_struct = float(np.mean(seed_struct_mean))
        print(f"  {reg:<8} N={n_lat:>4}: blind med={med:.4f} mean={mn:.4f}"
              f" | struct med={med_struct:.4f}")
        rows.append({
            "regime": reg, "N": n_lat, "source": src,
            "n_seeds": len(seeds),
            "blind_frob_median": med, "blind_frob_mean": mn,
            "struct_frob_median": med_struct,
            "struct_frob_mean": mn_struct,
            "per_seed_blind_median": seed_blind_med,
            "per_seed_blind_mean": seed_blind_mean,
            "per_seed_struct_median": seed_struct_med,
            "per_seed_struct_mean": seed_struct_mean,
        })

    bundle = {
        "method": "within_p5_runner_A_per_node_galerkin",
        "title": ("Within-P5 ladder per-node 4x4 Galerkin Frobenius "
                  "residual at seven lattice sizes."),
        "trend": rows,
    }
    out = REPO / "outputs" / "within_p5_runner_A_hessian_ricci.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
