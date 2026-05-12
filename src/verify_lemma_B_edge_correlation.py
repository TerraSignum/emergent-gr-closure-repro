"""Lemma B Phase-2 Step 4a pre-flight: edge-correlation audit.

Step 3d locked in the Friedman-regime classification of the
tau=0.10 skeleton. Friedman's theorem requires the edge-
formation rule to produce a random-regular-like graph, which
in turn requires the edge inclusions to be approximately
*independent* (the Erdős-Rényi limit) or to deviate from
independence in a controlled, structured way. This audit
measures the deviation from edge-independence.

For each snapshot:
  - p_edge = empirical edge-inclusion probability for the
    tau=0.10 skeleton.
  - C_3 = observed triangle count.
  - C_3^ER = expected triangle count under ER independence,
    = C(N, 3) * p_edge^3 (or equivalently N^3 * p_edge^3 / 6).
  - triangle excess ratio = C_3 / C_3^ER.
  - global clustering coefficient
    = 3 * #triangles / #connected-vertex-triples.

For Friedman-type random regular graphs the clustering
coefficient -> 0 as N -> infinity (sparse, locally tree-like).
The triangle-excess ratio -> 1 as N -> infinity in the ER
limit. Persistent triangle excess > 1 indicates correlated
edge formation; persistent ratio == 1 supports the structured-
random hypothesis.

Output: outputs/verify_lemma_B_edge_correlation.json
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REPO_ROOT = REPO.parent
OUT = REPO / "outputs" / "verify_lemma_B_edge_correlation.json"

LADDER = [
    ("P5",     50,  "results_d1_fix17/d1_p5.npz",            "xi_seedK"),
    ("P5N64",  64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz", "edge_xi_snapshots"),
    ("P5N72",  72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz", "edge_xi_snapshots"),
    ("P5N84",  84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz", "edge_xi_snapshots"),
    ("P5N100", 100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz", "edge_xi_snapshots"),
    ("P5N128", 128, "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz", "edge_xi_snapshots"),
    ("P5N200", 200, "results_d1_p5n200_8seeds/P5N200.snapshots.npz", "edge_xi_snapshots"),
    ("P5N256", 256, "results_d1_p5n256_12seeds/P5N256.snapshots.npz", "edge_xi_snapshots"),
    ("P5N300", 300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz", "edge_xi_snapshots"),
    ("P5N512", 512, "results_d1_p5n512_12seeds/P5N512.snapshots.npz", "edge_xi_snapshots"),
]

TAU_SKEL = 0.10


def load_all_xi(npz_path: Path, hint: str) -> list[np.ndarray]:
    if not npz_path.exists():
        return []
    z = np.load(npz_path, allow_pickle=True)
    matrices: list[np.ndarray] = []
    if hint == "edge_xi_snapshots" and "edge_xi_snapshots" in z.files:
        snaps = np.asarray(z["edge_xi_snapshots"])
        last = snaps.shape[1] - 1
        for s in range(snaps.shape[0]):
            xi = np.asarray(snaps[s, last], dtype=float).copy()
            np.fill_diagonal(xi, 1.0)
            matrices.append(xi)
        return matrices
    if hint == "xi_seedK":
        n_seeds = sum(1 for k in z.files if k.startswith("xi_seed"))
        for s in range(n_seeds):
            key = f"xi_seed{s}"
            if key not in z.files:
                continue
            xi = np.asarray(z[key], dtype=float).copy()
            np.fill_diagonal(xi, 1.0)
            matrices.append(xi)
        return matrices
    return matrices


def edge_correlation_diagnostics(xi: np.ndarray, tau: float
                                 ) -> dict[str, Any] | None:
    """Edge-formation rule diagnostics on the tau-skeleton."""
    n = xi.shape[0]
    a_skel = ((xi - np.eye(n)) > tau).astype(np.int8)
    np.fill_diagonal(a_skel, 0)
    a_skel = ((a_skel + a_skel.T) > 0).astype(np.int8)

    # Empirical edge probability
    edges_count = int(a_skel.sum() / 2)
    pairs = n * (n - 1) // 2
    p_edge = edges_count / pairs if pairs > 0 else 0.0

    # Triangle count = trace(A^3) / 6
    a_float = a_skel.astype(float)
    a2 = a_float @ a_float
    a3 = a2 @ a_float
    triangle_count = int(np.trace(a3) / 6.0)

    # Connected vertex triples (paths of length 2 between i and j)
    # = sum_i C(deg(i), 2) (undirected)
    deg = a_skel.sum(axis=1).astype(np.int64)
    triples = int(np.sum(deg * (deg - 1) // 2))

    # ER expected triangles = C(N,3) * p_edge^3
    n_choose_3 = n * (n - 1) * (n - 2) / 6.0
    expected_triangles_er = n_choose_3 * p_edge ** 3

    # Global clustering coefficient = 3 * triangles / triples
    clustering = 3.0 * triangle_count / triples if triples > 0 else 0.0

    # Excess ratio
    excess = (triangle_count / expected_triangles_er
              if expected_triangles_er > 0 else 0.0)

    return {
        "p_edge": p_edge,
        "edges_count": edges_count,
        "triangles_observed": triangle_count,
        "triangles_expected_ER": float(expected_triangles_er),
        "triangle_excess_ratio": float(excess),
        "global_clustering_coeff": float(clustering),
        "connected_triples": triples,
    }


def audit_regime(regime, n_lat, rel, hint):
    npz = REPO_ROOT / rel
    xis = load_all_xi(npz, hint)
    if not xis:
        return {"regime": regime, "N": n_lat,
                "n_seeds_loaded": 0,
                "status": "SNAPSHOT_NOT_AVAILABLE"}
    diags = [edge_correlation_diagnostics(xi, TAU_SKEL) for xi in xis]
    diags = [d for d in diags if d is not None]
    if not diags:
        return {"regime": regime, "N": n_lat,
                "n_seeds_loaded": len(xis),
                "status": "ALL_SEEDS_FAILED"}
    out = {"regime": regime, "N": n_lat,
           "n_seeds_loaded": len(xis),
           "n_seeds_valid": len(diags),
           "status": "OK"}
    for key in ("p_edge", "triangle_excess_ratio",
                "global_clustering_coeff",
                "triangles_observed", "triangles_expected_ER"):
        vals = [d[key] for d in diags]
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def _aicc(sse, n, k):
    return n * np.log(sse / n + 1e-30) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)


def fit_scaling(per_regime, key):
    valid = [(r["N"], r[f"{key}_mean"]) for r in per_regime
             if r.get("status") == "OK"]
    if len(valid) < 3:
        return None
    n_arr = np.array([v[0] for v in valid], dtype=float)
    y = np.array([v[1] for v in valid], dtype=float)
    n_pts = len(valid)
    # const
    c = float(y.mean())
    sse_c = float(((y - c) ** 2).sum())
    # Symanzik-1
    a_mat = np.column_stack([np.ones_like(n_arr), 1.0 / n_arr])
    sol, *_ = np.linalg.lstsq(a_mat, y, rcond=None)
    c_inf, a_sym = float(sol[0]), float(sol[1])
    sse_s = float(((y - (c_inf + a_sym / n_arr)) ** 2).sum())
    a_c = _aicc(sse_c, n_pts, 1)
    a_s = _aicc(sse_s, n_pts, 2)
    preferred = "const" if a_c < a_s else "symanzik_1"
    asym = c if preferred == "const" else c_inf
    return {"const": {"c": c, "AICc": a_c},
            "symanzik_1": {"c_inf": c_inf, "a": a_sym, "AICc": a_s},
            "preferred": preferred, "asymptote": asym}


def main():
    per_regime = [audit_regime(*row) for row in LADDER]
    fits = {
        "p_edge": fit_scaling(per_regime, "p_edge"),
        "triangle_excess_ratio": fit_scaling(per_regime, "triangle_excess_ratio"),
        "global_clustering_coeff": fit_scaling(per_regime,
                                              "global_clustering_coeff"),
    }
    # Verdict: ER-like iff
    #   triangle_excess_ratio -> 1
    #   global_clustering_coeff -> 0
    excess_asym = fits["triangle_excess_ratio"]["asymptote"] if fits["triangle_excess_ratio"] else None
    cluster_asym = fits["global_clustering_coeff"]["asymptote"] if fits["global_clustering_coeff"] else None

    if excess_asym is not None and cluster_asym is not None:
        if abs(excess_asym - 1.0) < 0.5 and cluster_asym < 0.1:
            verdict = ("ER_LIKE — edge formation is approximately "
                       "independent (triangle excess ~1, clustering -> 0)")
        elif excess_asym > 2.0 or cluster_asym > 0.2:
            verdict = (f"CORRELATED — strong triangle excess "
                       f"({excess_asym:.2f}) or clustering "
                       f"({cluster_asym:.3f}); structured edge formation")
        else:
            verdict = (f"WEAKLY_CORRELATED — triangle excess "
                       f"{excess_asym:.2f}, clustering "
                       f"{cluster_asym:.3f}; mild departure from ER")
    else:
        verdict = "INSUFFICIENT_DATA"

    out = {
        "headline": ("Lemma B Phase-2 Step 4a pre-flight: edge-"
                     "formation rule audit on the tau=0.10 skeleton. "
                     "Tests whether the carrier-action produces "
                     "Erdős-Rényi-independent edge inclusions or "
                     "correlated structured edges via triangle "
                     "excess and global clustering coefficient."),
        "method": (
            "For each snapshot: skeleton A_skel = 1[Xi > 0.10]; "
            "p_edge = #edges / C(N,2); triangle count = "
            "trace(A^3)/6; expected_ER = C(N,3) p_edge^3; "
            "excess = obs / ER; clustering coefficient = "
            "3 * triangles / triples. Per regime: cross-seed "
            "mean. AICc model selection (const vs Symanzik-1) "
            "for N-scaling."),
        "tau_skeleton": TAU_SKEL,
        "per_regime": per_regime,
        "scaling_fits": fits,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_summary(per_regime, fits, verdict)
    return 0


def print_summary(per_regime, fits, verdict):
    print("=" * 78)
    print("Lemma B Phase-2 Step 4a pre-flight: edge-correlation audit")
    print("=" * 78)
    print(f"{'Regime':<8} {'N':>4} {'seeds':>6} "
          f"{'p_edge':>10} {'triangles':>11} {'tri_ER':>11} "
          f"{'excess':>8} {'cluster':>9}")
    print("-" * 78)
    for r in per_regime:
        if r["status"] != "OK":
            print(f"{r['regime']:<8} {r['N']:>4}  {r['status']}")
            continue
        print(f"{r['regime']:<8} {r['N']:>4} "
              f"{r['n_seeds_valid']:>6} "
              f"{r['p_edge_mean']:>10.5f} "
              f"{r['triangles_observed_mean']:>11.1f} "
              f"{r['triangles_expected_ER_mean']:>11.1f} "
              f"{r['triangle_excess_ratio_mean']:>8.3f} "
              f"{r['global_clustering_coeff_mean']:>9.4f}")
    print()
    print("N-scaling fits (preferred per AICc):")
    for k, f in fits.items():
        if f is None:
            continue
        best = f["preferred"]
        if best == "const":
            print(f"  {k:<28s} const c = {f['const']['c']:.5f}")
        else:
            p = f["symanzik_1"]
            print(f"  {k:<28s} symanzik_1: c_inf = {p['c_inf']:.5f}, "
                  f"a = {p['a']:.3f}")
    print()
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    raise SystemExit(main())
