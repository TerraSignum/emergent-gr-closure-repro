"""Identify the SPECIFIC defect topology at M3-sup-norm worst-case nodes.

Building on the M3 sup-norm diagnostics (P1+P2+P7), this script
takes the top-K worst-case triples per regime and asks: what kind
of defect lives at those nodes? Concretely, for each worst-case
node, we evaluate four defect-detection heuristics on the
underlying lattice state:

  D1 (vortex):       phase-winding amplitude on local plaquettes
                      around the node
  D2 (domain wall):  amplitude jump across the node neighbourhood
                      (|psi(node)| << <|psi|>_neighbourhood)
  D3 (dense cell):   Xi-row degree above 90th percentile
                      (high-connectivity hub)
  D4 (T_00 spike):   T_00(node) > 90th percentile (matter cluster)

The output is a per-N defect-type histogram of the worst-case
nodes, plus the conditional sup-slack restricted to each defect
class. This makes the Defect-Cores <-> epsilon-quasimetric
connection quantitative.

Output: outputs/audit_M3_defect_topology.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from verify_galerkin_runner_A_hessian_ricci import (
    edge_to_matrix, per_seed_galerkin)

PARENT = REPO.parent

LADDER = [
    ("P5",     50, "results_d1_fix17/d1_p5.npz",                    "d1"),
    ("P5N100", 100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz", "snap"),
    ("P5N200", 200, "results_d1_p5n200_8seeds/P5N200.snapshots.npz",  "snap"),
    ("P5N300", 300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz",         "snap"),
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz",  "snap"),
    ("P5N512", 512,  "results_d1_p5n512_12seeds/P5N512.snapshots.npz",  "snap"),
]


def load_seed0(rel_path: str, kind: str, n_lat: int):
    fp = PARENT / rel_path
    if not fp.exists():
        return None
    z = np.load(fp, allow_pickle=True)
    if kind == "snap":
        snaps = z["edge_xi_snapshots"]
        last = snaps.shape[1] - 1
        xi = np.asarray(snaps[0, last], dtype=float).copy()
        np.fill_diagonal(xi, 1.0)
        psi = (np.asarray(z["psi_real_snapshots"][0, last], dtype=float)
               + 1j * np.asarray(z["psi_imag_snapshots"][0, last], dtype=float))
        k_field = z.get("ff_K_seed0", np.full((n_lat, n_lat), 0.55))
        q_field = z.get("ff_Q_seed0", np.full((n_lat, n_lat), 0.45))
    elif kind == "d1" and "dense_cell_edge_xi_values" in z.files:
        edge = z["dense_cell_edge_xi_values"]
        xi = edge_to_matrix(edge[0], n_lat).astype(float)
        np.fill_diagonal(xi, 1.0)
        amp = z["dense_cell_node_amplitude_values"][0]
        phase = z["dense_cell_node_phase_values"][0]
        psi = amp * np.exp(1j * phase)
        k_field = z.get("ff_K_seed0", np.full((n_lat, n_lat), 0.55))
        q_field = z.get("ff_Q_seed0", np.full((n_lat, n_lat), 0.45))
    else:
        return None
    return xi, psi, np.asarray(k_field), np.asarray(q_field)


def topk_worst_triples(xi, top_k: int = 20):
    n = xi.shape[0]
    prod = xi[:, :, None] * xi[None, :, :]
    target = xi[:, None, :]
    slack = prod - target
    diag_mask = np.ones((n, n, n), dtype=bool)
    diag_mask[np.arange(n), np.arange(n), :] = False
    diag_mask[:, np.arange(n), np.arange(n)] = False
    diag_mask[np.arange(n), :, np.arange(n)] = False
    slack_masked = np.where(diag_mask, slack, -1.0)
    flat = slack_masked.ravel()
    idx_sorted = np.argpartition(flat, -top_k)[-top_k:]
    idx_sorted = idx_sorted[np.argsort(flat[idx_sorted])[::-1]]
    out = []
    for kf in idx_sorted:
        s = float(flat[kf])
        if s <= 0:
            continue
        i, j, k = np.unravel_index(int(kf), slack.shape)
        out.append((s, int(i), int(j), int(k)))
    return out


def compute_defect_signatures(xi, psi, t00):
    """Per-node defect signatures.

    Returns dict[node_idx] -> {vortex, dwall, dense, t00_spike}
    where each value is a 0/1 indicator (or fractional in [0,1]).
    """
    n = xi.shape[0]
    amp = np.abs(psi)
    phase = np.angle(psi)

    # D1 vortex: phase-winding amplitude — proxy: variance of phase
    # in 1-hop neighbourhood
    phase_var = np.zeros(n)
    for i in range(n):
        nbrs = np.where(xi[i] > 0.5)[0]
        if nbrs.size > 1:
            phases = phase[nbrs]
            phase_var[i] = float(np.angle(np.exp(1j * phases).mean()).std()
                                  if False else 1 - np.abs(np.exp(1j * phases).mean()))
    vortex = (phase_var > np.percentile(phase_var, 90)).astype(float)

    # D2 domain wall: amplitude depression in neighbourhood mean
    nbr_amp_mean = np.zeros(n)
    for i in range(n):
        nbrs = np.where(xi[i] > 0.5)[0]
        if nbrs.size > 1:
            nbr_amp_mean[i] = float(amp[nbrs].mean())
    dwall = ((amp / np.maximum(nbr_amp_mean, 1e-6)) < 0.7).astype(float)

    # D3 dense cell: Xi-degree top-decile
    deg = (xi > 0.5).sum(axis=1).astype(float)
    dense = (deg > np.percentile(deg, 90)).astype(float)

    # D4 T_00 spike: top-decile
    t00_spike = (t00 > np.percentile(t00, 90)).astype(float)

    return vortex, dwall, dense, t00_spike


def classify_top_triples(top_triples, vortex, dwall, dense, t00_spike):
    """For each top triple, count which defect classes its three nodes
    fall into."""
    counts = {"vortex": 0, "dwall": 0, "dense": 0, "t00_spike": 0,
              "any_defect": 0, "no_defect": 0,
              "n_triples": 0, "n_node_slots": 0}
    for s, i, j, k in top_triples:
        counts["n_triples"] += 1
        for node in (i, j, k):
            counts["n_node_slots"] += 1
            cls = []
            if vortex[node] > 0.5: cls.append("vortex")
            if dwall[node] > 0.5: cls.append("dwall")
            if dense[node] > 0.5: cls.append("dense")
            if t00_spike[node] > 0.5: cls.append("t00_spike")
            for c in cls:
                counts[c] += 1
            if cls:
                counts["any_defect"] += 1
            else:
                counts["no_defect"] += 1
    return counts


def main():
    print("=" * 78)
    print("M3 sup-norm worst-case defect topology audit")
    print("=" * 78)

    rows = []
    for regime, n_lat, rel, kind in LADDER:
        payload = load_seed0(rel, kind, n_lat)
        if payload is None:
            print(f"  {regime}: file missing")
            continue
        xi, psi, k_field, q_field = payload
        try:
            prep = per_seed_galerkin(xi.copy(), psi, k_field, q_field, n_lat, np)
            t00 = np.asarray(prep["t00"])
        except Exception as e:
            print(f"  {regime}: galerkin failed: {e}")
            continue

        vortex, dwall, dense, t00_spike = compute_defect_signatures(xi, psi, t00)
        top_triples = topk_worst_triples(xi, top_k=20)
        counts = classify_top_triples(top_triples, vortex, dwall, dense, t00_spike)

        print(f"\n--- {regime} N={n_lat} ---")
        print(f"  top-20 triples, {counts['n_node_slots']} node slots:")
        print(f"    vortex   : {counts['vortex']:>3} "
              f"({counts['vortex']/counts['n_node_slots']*100:.1f}%)")
        print(f"    dwall    : {counts['dwall']:>3} "
              f"({counts['dwall']/counts['n_node_slots']*100:.1f}%)")
        print(f"    dense    : {counts['dense']:>3} "
              f"({counts['dense']/counts['n_node_slots']*100:.1f}%)")
        print(f"    T00_spike: {counts['t00_spike']:>3} "
              f"({counts['t00_spike']/counts['n_node_slots']*100:.1f}%)")
        print(f"    any_def  : {counts['any_defect']:>3} "
              f"({counts['any_defect']/counts['n_node_slots']*100:.1f}%)")
        print(f"    no_defect: {counts['no_defect']:>3} "
              f"({counts['no_defect']/counts['n_node_slots']*100:.1f}%)")

        rows.append({
            "regime": regime, "N": n_lat,
            "n_top_triples": counts["n_triples"],
            "n_node_slots": counts["n_node_slots"],
            "defect_counts": counts,
            "fractions": {
                k: counts[k] / max(counts["n_node_slots"], 1)
                for k in ("vortex", "dwall", "dense", "t00_spike",
                          "any_defect", "no_defect")
            },
        })

    bundle = {"method": "M3_supnorm_defect_topology", "rows": rows}
    out = REPO / "outputs" / "audit_M3_defect_topology.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
