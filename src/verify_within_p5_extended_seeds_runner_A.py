"""Within-P5 Runner A on the extended-seed snapshot files.

Picks the seed-richest snapshot file per regime (preferring the
fresh _24seeds / _8seeds runs over the older _v2 / canonical
files). Runs the per-node 4x4 Galerkin Frobenius computation on
each, dumps per-seed values for downstream bootstrap.

Selection precedence per regime:
  1. results_d1_<reg>_24seeds/<REG>.snapshots.npz (latest)
  2. results_d1_<reg>_8seeds/<REG>.snapshots.npz
  3. results_d1_<reg>_v2/<REG>.snapshots.npz
  4. canonical d1 npz (via _d1_npz_discovery)

For each regime the file is required to have ff_K_seed* /
ff_Q_seed* persisted; if not, the regime is skipped (avoids the
default-K/Q artefact).

Writes:
  outputs/within_p5_extended_seeds_runner_A.json
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
    ("P5",     50),
    ("P5N64",  64),
    ("P5N72",  72),
    ("P5N84",  84),
    ("P5N100", 100),
    ("P5N200", 200),
    ("P5N300", 300),
    ("P5N256", 256),
    ("P5N512", 512),
]


def find_best_npz(reg: str, n_lat: int) -> tuple[Path | None, str]:
    for suffix, label in [("_kq_fixed", "kq_fixed"),
                          ("_24seeds", "24seeds"),
                          ("_16seeds", "16seeds"),
                          ("_12seeds", "12seeds"),
                          ("_8seeds", "8seeds"),
                          ("_v2", "v2"),
                          ("", "canonical_dir")]:
        p = PARENT / f"results_d1_{reg.lower()}{suffix}"
        if not p.is_dir():
            continue
        snap = p / f"{reg}.snapshots.npz"
        if snap.exists():
            return snap, label
        d1 = p / f"d1_{reg.lower()}.npz"
        if d1.exists():
            return d1, label
    p = find_d1_npz(reg, REPO)
    if p is not None and p.exists():
        return p, "discovery"
    return None, "none"


def has_kq_persisted(npz_path: Path) -> bool:
    z = np.load(npz_path, allow_pickle=True)
    return any("ff_K_seed" in k for k in z.files)


def load_seeds(p: Path, n_lat: int):
    d = np.load(p, allow_pickle=True)
    seeds = []
    if "dense_cell_edge_xi_values" in d.files:
        edge = d["dense_cell_edge_xi_values"]
        amp = d["dense_cell_node_amplitude_values"]
        phase = d["dense_cell_node_phase_values"]
        n_seeds = int(edge.shape[0])
        for s in range(n_seeds):
            xi = edge_to_matrix(edge[s], n_lat)
            np.fill_diagonal(xi, 1.0)
            psi = amp[s] * np.exp(1j * phase[s])
            k = d.get(f"ff_K_seed{s}", None)
            q = d.get(f"ff_Q_seed{s}", None)
            if k is None or q is None:
                continue
            seeds.append((xi, psi, np.asarray(k), np.asarray(q)))
    elif "edge_xi_snapshots" in d.files:
        edge = d["edge_xi_snapshots"]
        psi_r = d["psi_real_snapshots"]
        psi_i = d["psi_imag_snapshots"]
        n_seeds = int(edge.shape[0])
        for s in range(n_seeds):
            xi = edge[s, -1, :, :].copy()
            xi = 0.5 * (xi + xi.T)
            np.fill_diagonal(xi, 1.0)
            psi = (psi_r[s, -1, :].astype(np.complex128).reshape(-1)
                   + 1j * psi_i[s, -1, :].reshape(-1))
            if psi.shape[0] != n_lat:
                psi = psi[:n_lat]
            k = d.get(f"ff_K_seed{s}", None)
            q = d.get(f"ff_Q_seed{s}", None)
            if k is None or q is None:
                continue
            seeds.append((xi, psi, np.asarray(k), np.asarray(q)))
    return seeds


def main() -> int:
    print("=" * 78)
    print("Within-P5 extended-seed Runner A")
    print("=" * 78)
    rows = []
    for reg, n_lat in LADDER:
        p, label = find_best_npz(reg, n_lat)
        if p is None:
            print(f"  {reg} N={n_lat}: NO FILE")
            continue
        if not has_kq_persisted(p):
            print(f"  {reg} N={n_lat}: SKIP ({p.name}: no ff_K_seed*)")
            continue
        seeds = load_seeds(p, n_lat)
        if not seeds:
            print(f"  {reg} N={n_lat}: SKIP (no usable seeds)")
            continue
        per_seed_blind_med = []
        per_seed_blind_mean = []
        per_seed_struct_med = []
        per_seed_struct_mean = []
        for xi, psi, k, q in seeds:
            prep = per_seed_galerkin(xi, psi, k, q, n_lat, np)
            res = frob_residual_three_variants(prep, np)
            per_seed_blind_med.append(float(res["blind_frob_median"]))
            per_seed_blind_mean.append(float(res["blind_frob_mean"]))
            per_seed_struct_med.append(float(res["struct_frob_median"]))
            per_seed_struct_mean.append(float(res["struct_frob_mean"]))
        med = float(np.mean(per_seed_blind_med))
        sd  = float(np.std(per_seed_blind_med))
        mn  = float(np.mean(per_seed_blind_mean))
        print(f"  {reg:<8} N={n_lat:>4} ({label:>9}): "
              f"n_seeds={len(seeds):>2} blind_med={med:.4f} +- {sd:.4f}")
        rows.append({
            "regime": reg, "N": n_lat, "n_seeds": len(seeds),
            "source_label": label,
            "source_file": str(p.relative_to(PARENT)),
            "blind_frob_median": med,
            "blind_frob_median_std": sd,
            "blind_frob_mean": mn,
            "per_seed_blind_median": per_seed_blind_med,
            "per_seed_blind_mean": per_seed_blind_mean,
            "per_seed_struct_median": per_seed_struct_med,
            "per_seed_struct_mean": per_seed_struct_mean,
        })

    bundle = {
        "method": "within_p5_extended_seeds_runner_A",
        "title": ("Within-P5 ladder per-node 4x4 Galerkin Frobenius "
                  "residual using the seed-richest available file per "
                  "regime."),
        "trend": rows,
    }
    out = REPO / "outputs" / "within_p5_extended_seeds_runner_A.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
