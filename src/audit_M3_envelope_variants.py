"""Composition-envelope refinement audit for M3 sub-multiplicative
triangle inequality.

Tests four envelope-definition variants and reports their post-stabilisation
sup-norm and L^2-norm M3-violation residuals. The current bundled
construction uses
  Xi^closed_ik = max{Xi^direct_ik, sup_j Xi^direct_ij Xi^direct_jk * exp(-eta * dK)}
which saturates the multiplicative triangle envelope from below at the
attaining j; the question is whether an alternative envelope yields
a tighter sup-norm residual without breaking the L^2 closure.

Variants tested:
  V1 (current): max{direct, sup_j Xi_ij Xi_jk}
  V2 (no-floor): direct only (skip composition closure)
  V3 (floor at p90): Xi_ik <- max{Xi_ik, percentile(Xi_*, 90)} (positivity floor)
  V4 (geometric envelope): Xi_ik <- (Xi_ik * sup_j Xi_ij Xi_jk)^(1/2)
  V5 (penalty-augmented): Xi_ik <- Xi_ik + alpha * sup_j (Xi_ij Xi_jk - Xi_ik)_+

Output: outputs/audit_M3_envelope_variants.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from audit_M3_violations import m3_violation_rate
from verify_galerkin_runner_A_hessian_ricci import edge_to_matrix

PARENT = REPO.parent

LADDER = [
    ("P5",     50, "results_d1_fix17/d1_p5.npz",                      "d1"),
    ("P5N64",  64, "results_d1_p5n64_24seeds/P5N64.snapshots.npz",    "snap"),
    ("P5N72",  72, "results_d1_p5n72_24seeds/P5N72.snapshots.npz",    "snap"),
    ("P5N84",  84, "results_d1_p5n84_24seeds/P5N84.snapshots.npz",    "snap"),
    ("P5N100", 100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz", "snap"),
    ("P5N128", 128, "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz","snap"),
    ("P5N200", 200, "results_d1_p5n200_8seeds/P5N200.snapshots.npz",  "snap"),
    ("P5N300", 300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz",         "snap"),
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz",  "snap"),
    ("P5N512", 512,  "results_d1_p5n512_12seeds/P5N512.snapshots.npz",  "snap"),
]


def load_xi_seeds(rel_path: str, kind: str, n_lat: int):
    fp = PARENT / rel_path
    if not fp.exists():
        return []
    z = np.load(fp, allow_pickle=True)
    seeds = []
    if kind == "snap":
        snaps = z["edge_xi_snapshots"]
        last = snaps.shape[1] - 1
        for s in range(int(snaps.shape[0])):
            xi = np.asarray(snaps[s, last], dtype=float).copy()
            np.fill_diagonal(xi, 1.0)
            seeds.append(xi)
    elif kind == "d1" and "dense_cell_edge_xi_values" in z.files:
        edge = z["dense_cell_edge_xi_values"]
        for s in range(int(edge.shape[0])):
            xi = edge_to_matrix(edge[s], n_lat).astype(float)
            np.fill_diagonal(xi, 1.0)
            seeds.append(xi)
    return seeds


def envelope_v1_direct(xi: np.ndarray) -> np.ndarray:
    """V1: identity (already-closed direct lattice xi)."""
    return xi.copy()


def envelope_v2_max_path(xi: np.ndarray) -> np.ndarray:
    """V2: standard max-path composition closure
        Xi^closed_ik = max{Xi_ik, sup_j Xi_ij Xi_jk}."""
    n = xi.shape[0]
    prod = xi @ xi  # (Xi^2)_{ik} = sum_j Xi_ij Xi_jk; here we want sup_j
    # sup over intermediate j:
    sup_j = np.zeros((n, n), dtype=float)
    for k in range(n):
        sup_j[:, k] = np.max(xi * xi[None, :, k], axis=1)
    out = np.maximum(xi, sup_j)
    np.fill_diagonal(out, 1.0)
    return out


def envelope_v3_geometric(xi: np.ndarray) -> np.ndarray:
    """V3: geometric mean of direct and max-path envelope."""
    n = xi.shape[0]
    sup_j = np.zeros((n, n), dtype=float)
    for k in range(n):
        sup_j[:, k] = np.max(xi * xi[None, :, k], axis=1)
    geom = np.sqrt(np.maximum(xi * sup_j, 1e-12))
    np.fill_diagonal(geom, 1.0)
    return geom


def envelope_v4_penalty_augmented(xi: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """V4: subtract alpha-fraction of the worst-case violation per ik."""
    n = xi.shape[0]
    sup_j = np.zeros((n, n), dtype=float)
    for k in range(n):
        sup_j[:, k] = np.max(xi * xi[None, :, k], axis=1)
    excess = np.maximum(sup_j - xi, 0.0)
    out = xi + alpha * excess
    np.fill_diagonal(out, 1.0)
    out = np.minimum(out, 1.0)
    return out


def envelope_v5_floor(xi: np.ndarray, floor_quantile: float = 0.05) -> np.ndarray:
    """V5: clamp Xi_ij from below to floor_quantile percentile to
    enforce positivity domain (M0 strict positivity)."""
    flat = xi[np.triu_indices_from(xi, k=1)]
    if flat.size == 0:
        return xi.copy()
    floor_v = float(np.quantile(flat, floor_quantile))
    out = np.maximum(xi, floor_v)
    np.fill_diagonal(out, 1.0)
    return out


VARIANTS = [
    ("V1_direct",            envelope_v1_direct),
    ("V2_max_path",          envelope_v2_max_path),
    ("V3_geometric",         envelope_v3_geometric),
    ("V4_penalty_alpha0.5",  lambda x: envelope_v4_penalty_augmented(x, 0.5)),
    ("V5_floor_q5",          lambda x: envelope_v5_floor(x, 0.05)),
]


def main():
    print("=" * 78)
    print("Composition-envelope refinement audit")
    print("=" * 78)
    rows = []
    for regime, n_lat, rel, kind in LADDER:
        seeds = load_xi_seeds(rel, kind, n_lat)
        if not seeds:
            print(f"  {regime}: no seeds")
            continue
        print(f"\n--- {regime} N={n_lat}, {len(seeds)} seeds ---")
        print(f"  {'variant':<22} {'L2_mean':>10} {'sup_mean':>10}")
        regime_results = {}
        for vname, vfn in VARIANTS:
            L2_list, sup_list = [], []
            for xi in seeds:
                xi_v = vfn(xi)
                np.fill_diagonal(xi_v, 1.0)
                r = m3_violation_rate(xi_v, tol=1e-12)
                L2_list.append(r["penalty_residual_L2"])
                sup_list.append(r["penalty_residual_max"])
            L2 = float(np.mean(L2_list))
            sup = float(np.mean(sup_list))
            print(f"  {vname:<22} {L2:>10.5f} {sup:>10.4f}")
            regime_results[vname] = {"L2": L2, "sup": sup}
        rows.append({"regime": regime, "N": n_lat,
                     "n_seeds": len(seeds),
                     "variants": regime_results})

    bundle = {"method": "M3_envelope_variants_audit", "rows": rows}
    out = REPO / "outputs" / "audit_M3_envelope_variants.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
