"""Lemma B Step-4 HSD factorization test (2026-05-15).

Tests sub-claim A of the Harmonic-Sum-Decomposition (HSD) analytical
sketch (notes/lemma_B_step4_cavity_sketch_2026_05_15.md):

    P(k, k_Δ) approximately factorizes into independent
    (degree, triangle-count) components as N -> infinity?

Diagnostic: joint cumulant kappa_{1,1}(N) = <k * k_Δ> - <k> * <k_Δ>.
If kappa_{1,1}(N) decays to zero as N -> infinity (under N^{-alpha}
power-law fit), sub-claim A is empirically supported and the HSD
analytical route is viable.

If kappa_{1,1}(N) does NOT decay to zero, the factorization hypothesis
fails and Step-4 reverts to the full PPM + SM cavity machinery without
the simplifying tensor-product structure.

Output: outputs/verify_lemma_B_step4_hsd_factorization.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import networkx as nx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from _d1_npz_discovery import find_d1_npz  # noqa: E402

CARRIER_LADDER = [
    ("P5",     50),
    ("P5N64",  64),
    ("P5N72",  72),
    ("P5N84",  84),
    ("P5N100", 100),
    ("P5N128", 128),
    ("P5N200", 200),
    ("P5N256", 256),
    ("P5N300", 300),
    ("P5N512", 512),
    ("P5N600", 600),
]
TAU = 0.10
TARGET_SKEL = 7.0 / 24.0


MAX_CARRIER_SEEDS = 24


def load_xi_snapshots(regime: str, n_lat: int):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return []
    d = np.load(p, allow_pickle=True)
    snaps = []
    if "edge_xi_snapshots" in d.files:
        arr = d["edge_xi_snapshots"]
        if arr.ndim == 4:
            for s in range(min(MAX_CARRIER_SEEDS, arr.shape[0])):
                xi = np.asarray(arr[s, -1], dtype=float)
                if xi.shape == (n_lat, n_lat):
                    snaps.append(xi)
        return snaps
    for s in range(MAX_CARRIER_SEEDS):
        key = f"xi_seed{s}"
        if key not in d.files:
            break
        xi = np.asarray(d[key], dtype=float)
        if xi.shape == (n_lat, n_lat):
            snaps.append(xi)
    return snaps


def skeleton_from_xi(xi):
    w = 0.5 * (xi + xi.T)
    np.fill_diagonal(w, 0.0)
    w = np.maximum(w, 0.0)
    return (w > TAU).astype(np.float64)


def per_node_pairs(giant):
    """Return list of (k_i, k_Δ_i) per node of giant component."""
    degs = dict(giant.degree())
    tris = nx.triangles(giant)
    return [(degs[v], tris[v]) for v in giant.nodes()]


def joint_moments(pairs):
    """Return (<k>, <k_Δ>, <k * k_Δ>, kappa_{1,1}, sample_size)."""
    if not pairs:
        return None
    arr = np.asarray(pairs, dtype=np.float64)
    k = arr[:, 0]
    kd = arr[:, 1]
    mean_k = float(np.mean(k))
    mean_kd = float(np.mean(kd))
    mean_kkd = float(np.mean(k * kd))
    kappa11 = mean_kkd - mean_k * mean_kd
    return {
        "n_nodes": int(len(pairs)),
        "mean_k": mean_k,
        "mean_kd": mean_kd,
        "mean_k_kd": mean_kkd,
        "kappa_1_1": kappa11,
        "var_k": float(np.var(k)),
        "var_kd": float(np.var(kd)),
        "pearson_r": (float(kappa11 / np.sqrt(np.var(k) * np.var(kd)))
                      if np.var(k) * np.var(kd) > 0 else None),
    }


def power_law_fit(xs, ys):
    """Fit y = a + b * x^{-alpha} on (xs, ys); return (a_inf, b, alpha, r2)."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    # Two-parameter Symanzik fit y = a + b/x (alpha = 1)
    A = np.column_stack([np.ones_like(xs), 1.0 / xs])
    sol, *_ = np.linalg.lstsq(A, ys, rcond=None)
    a_inf, b = float(sol[0]), float(sol[1])
    yp = a_inf + b / xs
    ss_res = float(np.sum((ys - yp) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return a_inf, b, r2


def main():
    print("=" * 72)
    print("Lemma B Step-4 HSD factorization test")
    print("Sub-claim A: kappa_{1,1}(N) -> 0  as N -> infinity?")
    print(f"Target skeleton gap 7/24 = {TARGET_SKEL:.6f}")
    print("=" * 72)
    per_regime = []
    for regime, n_lat in CARRIER_LADDER:
        snaps = load_xi_snapshots(regime, n_lat)
        if not snaps:
            print(f"  [skip {regime}: no snapshots]")
            continue
        pairs_all = []
        n_seeds_used = 0
        for xi in snaps:
            skel = skeleton_from_xi(xi)
            G = nx.from_numpy_array(skel)
            G.remove_edges_from(nx.selfloop_edges(G))
            if G.number_of_edges() == 0:
                continue
            ccomps = list(nx.connected_components(G))
            if not ccomps:
                continue
            giant = G.subgraph(max(ccomps, key=len)).copy()
            if giant.number_of_nodes() < 4:
                continue
            pairs_all.extend(per_node_pairs(giant))
            n_seeds_used += 1
        moms = joint_moments(pairs_all)
        if moms is None:
            print(f"  [skip {regime}: no usable seeds]")
            continue
        moms["regime"] = regime
        moms["N"] = n_lat
        moms["n_seeds"] = n_seeds_used
        per_regime.append(moms)
        print(f"  {regime:8s} N={n_lat:4d} n={n_seeds_used:2d}  "
              f"<k>={moms['mean_k']:6.2f}  "
              f"<k_d>={moms['mean_kd']:7.2f}  "
              f"kappa_11={moms['kappa_1_1']:+9.3f}  "
              f"r={moms['pearson_r']:+.3f}")

    if len(per_regime) < 3:
        print("ERROR: insufficient ladder coverage for Symanzik fit")
        sys.exit(1)

    Ns = [r["N"] for r in per_regime]
    kappas = [r["kappa_1_1"] for r in per_regime]
    a_inf, b, r2 = power_law_fit(Ns, kappas)

    print()
    print("-" * 72)
    print("Symanzik-1 fit kappa_{1,1}(N) = a_inf + b/N:")
    print(f"  a_inf = {a_inf:+.4f}      (HSD predicts a_inf = 0)")
    print(f"  b     = {b:+.4f}")
    print(f"  R^2   = {r2:.4f}")

    if abs(a_inf) < 0.5:
        verdict = ("HSD_FACTORIZATION_SUPPORTED: kappa_{1,1}(N) "
                   "asymptote consistent with zero (|a_inf| < 0.5); "
                   "the tensor-product factorization hypothesis "
                   "(sub-claim A) is empirically supported.")
    elif abs(a_inf) < 2.0:
        verdict = (f"HSD_FACTORIZATION_INCONCLUSIVE: kappa_{{1,1}} "
                   f"a_inf = {a_inf:+.3f} not consistent with zero "
                   "but small; finite-N corrections or higher-order "
                   "joint cumulants may dominate.")
    else:
        verdict = (f"HSD_FACTORIZATION_FALSIFIED: kappa_{{1,1}} "
                   f"a_inf = {a_inf:+.3f} significantly non-zero; "
                   "the tensor-product factorization hypothesis "
                   "(sub-claim A) fails empirically. Step-4 must "
                   "use full PPM + Silva-Metz cavity machinery "
                   "without simplifying factorization.")

    print()
    print(verdict)

    out = {
        "method": "verify_lemma_B_step4_hsd_factorization",
        "stand": "2026-05-15",
        "question": ("Does kappa_{1,1}(N) = <k*k_d> - <k>*<k_d> "
                     "decay to 0 as N -> infinity? "
                     "(tests HSD sub-claim A: tensor-product "
                     "factorization of P(k, k_d))"),
        "tau": TAU,
        "target_skel_7_24": TARGET_SKEL,
        "per_regime": per_regime,
        "symanzik1_kappa_1_1": {
            "a_inf": a_inf,
            "b": b,
            "r2": r2,
        },
        "verdict": verdict,
    }
    out_path = REPO / "outputs" / \
        "verify_lemma_B_step4_hsd_factorization.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
