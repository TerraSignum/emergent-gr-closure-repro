"""Convert a snapshot NPZ (from run_d1_snapshot) to the canonical
D1 NPZ format consumed by verify_galerkin_runner_A_hessian_ricci.

Snapshot NPZ has:
  edge_xi_snapshots: (n_seeds, n_snapshots, N, N)
  psi_real_snapshots: (n_seeds, n_snapshots, N)
  psi_imag_snapshots: (n_seeds, n_snapshots, N)

Canonical D1 NPZ has:
  dense_cell_edge_xi_values: (n_seeds, N, N)
  dense_cell_node_amplitude_values: (n_seeds, N)
  dense_cell_node_phase_values: (n_seeds, N)

We pick the LAST snapshot per seed (steady-state lattice state).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np


def convert(snapshot_path: Path, out_path: Path):
    d = np.load(snapshot_path, allow_pickle=True)
    edge_snaps = d["edge_xi_snapshots"]      # (n_seeds, n_snaps, N, N)
    psi_r = d["psi_real_snapshots"]          # (n_seeds, n_snaps, N)
    psi_i = d["psi_imag_snapshots"]          # (n_seeds, n_snaps, N)

    # Take last snapshot per seed
    edges = edge_snaps[:, -1, :, :]          # (n_seeds, N, N)
    psi_r_end = psi_r[:, -1, :]              # (n_seeds, N)
    psi_i_end = psi_i[:, -1, :]              # (n_seeds, N)
    amp = np.sqrt(psi_r_end ** 2 + psi_i_end ** 2)
    phase = np.arctan2(psi_i_end, psi_r_end)

    np.savez(
        out_path,
        dense_cell_edge_xi_values=edges,
        dense_cell_node_amplitude_values=amp,
        dense_cell_node_phase_values=phase,
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python _convert_snapshot_to_d1.py <snap.npz> <out.npz>")
        raise SystemExit(2)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"OK -> {sys.argv[2]}")
