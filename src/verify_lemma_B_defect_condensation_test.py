"""Lemma B Defect-Condensation Hypothesis Test.

Tests whether the weighted-Laplacian spectral gap lambda_2 is
structurally carried by defect / matter-core nodes — i.e.\ whether
the uniform spectral gap (Lemma B) is a *consequence* of defect
condensation rather than a separate axiom.

Hypothesis (Defect-Condensation): defect nodes (matter-core) form
a percolating network whose removal collapses lambda_2; random
removal of the same number of nodes does NOT collapse lambda_2.
This is a candidate dynamical origin for the (SG) axiom of Lemma B.

Audit design:
  1. For each regime in the 10-point canonical ladder, for each
     seed, compute lambda_2 of the full weighted Laplacian
     (baseline).
  2. Identify matter-core nodes via row-variance(Xi) (corpus-
     validated proxy, AUC 0.83-0.90 LORO 8/8).
  3. For each percentile threshold p in {1%, 5%, 10%, 25%, 50%}:
     a. REMOVE the top-p% matter-core nodes; compute lambda_2
        of the resulting sub-Laplacian.
     b. NULL: 20 random permutations of "matter-core" labels;
        remove same number of random nodes; compute lambda_2.
  4. Aggregate: per-regime mean lambda_2 vs random null,
     bootstrap CI on the difference.
  5. Branch-resolved: vacuum (N <= 100) vs matter (N >= 256).

Verdict:
  - SG_DEFECT_DRIVEN: matter-core removal collapses lambda_2
    significantly more than random removal across regimes.
  - SG_GRAPH_WIDE: no significant difference between matter-core
    and random removal -> SG is a graph-wide property, not
    defect-driven.

Output: outputs/verify_lemma_B_defect_condensation_test.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REPO_ROOT = REPO.parent
OUT = REPO / "outputs" / "verify_lemma_B_defect_condensation_test.json"

# Canonical-physics ladder
LADDER = [
    ("P5",     50,  "results_d1_fix17/d1_p5.npz",                       "xi_seedK"),
    ("P5N64",  64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz",     "edge_xi_snapshots"),
    ("P5N72",  72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz",     "edge_xi_snapshots"),
    ("P5N84",  84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz",     "edge_xi_snapshots"),
    ("P5N100", 100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz",   "edge_xi_snapshots"),
    ("P5N128", 128, "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz",  "edge_xi_snapshots"),
    ("P5N200", 200, "results_d1_p5n200_8seeds/P5N200.snapshots.npz",    "edge_xi_snapshots"),
    ("P5N256", 256, "results_d1_p5n256_12seeds/P5N256.snapshots.npz",   "edge_xi_snapshots"),
    ("P5N300", 300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz",   "edge_xi_snapshots"),
    ("P5N512", 512, "results_d1_p5n512_12seeds/P5N512.snapshots.npz",   "edge_xi_snapshots"),
]

PERCENTILES = [1, 5, 10, 25, 50]      # remove top-p% matter-core nodes
N_RANDOM_PERM = 20                     # random-removal null permutations
MAX_SEEDS_PER_REGIME = 8               # throttle for cost
RNG_SEED = 42


def load_xi_matrices(path: Path, hint: str, max_seeds: int):
    """Yield Xi matrices at the last timestep for each seed."""
    if not path.exists():
        return
    z = np.load(path, allow_pickle=True)
    if hint == "xi_seedK":
        for s in range(max_seeds):
            key = f"xi_seed{s}"
            if key not in z.files:
                break
            xi = np.asarray(z[key], dtype=float).copy()
            np.fill_diagonal(xi, 1.0)
            yield xi
    else:
        snaps = np.asarray(z["edge_xi_snapshots"])
        n_seeds = min(max_seeds, snaps.shape[0])
        last = snaps.shape[1] - 1
        for s in range(n_seeds):
            xi = np.asarray(snaps[s, last], dtype=float).copy()
            np.fill_diagonal(xi, 1.0)
            yield xi


def weighted_laplacian_lambda2(xi: np.ndarray) -> float | None:
    """lambda_2 of normalised weighted Laplacian on Xi."""
    n = xi.shape[0]
    if n < 3:
        return None
    w = xi - np.eye(n)
    w = np.maximum(w, 0.0)
    deg = w.sum(axis=1)
    if np.any(deg <= 1e-12):
        return None
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    norm = w * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    lap = np.eye(n) - norm
    lap = 0.5 * (lap + lap.T)
    eig = np.linalg.eigvalsh(lap)
    return float(eig[1])  # smallest non-zero


def matter_core_ranking(xi: np.ndarray) -> np.ndarray:
    """Row-variance(Xi) proxy: AUC 0.83-0.90 LORO 8/8 in the corpus."""
    n = xi.shape[0]
    w = xi - np.eye(n)
    w = np.maximum(w, 0.0)
    return np.array([np.var(w[i, :]) for i in range(n)])


def fiedler_halo_ranking(xi: np.ndarray) -> np.ndarray:
    """|Fiedler vector amplitude| ranking. The top entries form the
    'halo' set identified earlier as BFS distance 1-2 from matter-
    core cusps in verify_lemma_B_fiedler_halo_test.py."""
    n = xi.shape[0]
    w = xi - np.eye(n)
    w = np.maximum(w, 0.0)
    deg = w.sum(axis=1)
    if np.any(deg <= 1e-12):
        return np.zeros(n)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    norm = w * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    lap = np.eye(n) - norm
    lap = 0.5 * (lap + lap.T)
    _, vec = np.linalg.eigh(lap)
    return np.abs(vec[:, 1])


def remove_nodes_and_lambda2(xi: np.ndarray, indices_to_remove: np.ndarray) -> float | None:
    """Return lambda_2 after removing the given indices from Xi."""
    n = xi.shape[0]
    keep = np.array([i for i in range(n) if i not in set(indices_to_remove)])
    if keep.size < 3:
        return None
    xi_sub = xi[np.ix_(keep, keep)]
    return weighted_laplacian_lambda2(xi_sub)


def per_snapshot_test(xi: np.ndarray, rng: np.random.Generator) -> dict | None:
    """Two parallel ranking schemes:
       - matter-core via row-variance (cusp-like, NFW)
       - Fiedler-halo via |v_2(i)| (halo identified earlier as
         BFS dist 1-2 from cusp in fiedler_halo_test).
    For each, remove top-p% nodes and compare lambda_2 vs random
    null at same p.
    """
    n = xi.shape[0]
    lam2_full = weighted_laplacian_lambda2(xi)
    if lam2_full is None:
        return None
    rank_core = matter_core_ranking(xi)
    rank_halo = fiedler_halo_ranking(xi)
    sorted_core = np.argsort(-rank_core)
    sorted_halo = np.argsort(-rank_halo)
    out = {"n": n, "lambda2_full": lam2_full,
           "per_percentile_core": {}, "per_percentile_halo": {}}

    for p in PERCENTILES:
        k = max(1, int(np.round(p / 100.0 * n)))

        # null: random-removal shared across the two ranking tests
        lam2_random_perms = []
        for _ in range(N_RANDOM_PERM):
            idx_random = rng.choice(n, size=k, replace=False)
            v = remove_nodes_and_lambda2(xi, idx_random)
            if v is not None:
                lam2_random_perms.append(v)
        if not lam2_random_perms:
            continue
        lam2_random_arr = np.array(lam2_random_perms)
        rand_mean = float(lam2_random_arr.mean())
        rand_std = float(lam2_random_arr.std())

        for label, sorted_idx, bucket in (
                ("core", sorted_core, out["per_percentile_core"]),
                ("halo", sorted_halo, out["per_percentile_halo"])):
            idx_remove = sorted_idx[:k]
            lam2_rem = remove_nodes_and_lambda2(xi, idx_remove)
            bucket[f"p{p}"] = {
                "k_removed": int(k),
                "lam2_after_removal": lam2_rem,
                "lam2_after_random_removal_mean": rand_mean,
                "lam2_after_random_removal_std": rand_std,
                "diff_minus_random": (
                    None if lam2_rem is None
                    else float(lam2_rem - rand_mean)),
                "z_score": (
                    None if lam2_rem is None or rand_std < 1e-9
                    else float((lam2_rem - rand_mean) / rand_std)),
            }
    return out


def audit_regime(regime: str, n_lat: int, rel: str, hint: str) -> dict:
    npz = REPO_ROOT / rel
    rng = np.random.default_rng(RNG_SEED)
    snapshots = list(load_xi_matrices(npz, hint, MAX_SEEDS_PER_REGIME))
    if not snapshots:
        return {"regime": regime, "N": n_lat, "status": "NO_DATA"}
    snap_results = []
    for xi in snapshots:
        r = per_snapshot_test(xi, rng)
        if r is not None:
            snap_results.append(r)
    if not snap_results:
        return {"regime": regime, "N": n_lat, "status": "ALL_DEGENERATE"}

    # Aggregate per-percentile across seeds, separately for the
    # two ranking schemes (core = row-variance / NFW cusp,
    # halo = |Fiedler v_2|).
    agg = {"lambda2_full_mean": float(np.mean(
        [r["lambda2_full"] for r in snap_results]))}
    for ranking in ("core", "halo"):
        bucket_key = f"per_percentile_{ranking}"
        agg[ranking] = {}
        for p in PERCENTILES:
            key = f"p{p}"
            vals_rem  = [r[bucket_key][key]["lam2_after_removal"]
                         for r in snap_results
                         if key in r[bucket_key]
                         and r[bucket_key][key]["lam2_after_removal"] is not None]
            vals_rand = [r[bucket_key][key]["lam2_after_random_removal_mean"]
                         for r in snap_results
                         if key in r[bucket_key]]
            vals_z    = [r[bucket_key][key]["z_score"]
                         for r in snap_results
                         if key in r[bucket_key]
                         and r[bucket_key][key]["z_score"] is not None]
            if not vals_rem or not vals_rand:
                continue
            agg[ranking][f"p{p}"] = {
                "n_seeds": len(vals_rem),
                "lam2_removal_mean": float(np.mean(vals_rem)),
                "lam2_random_removal_mean": float(np.mean(vals_rand)),
                "diff_mean": float(np.mean(vals_rem) - np.mean(vals_rand)),
                "z_score_mean": float(np.mean(vals_z)) if vals_z else None,
                "z_score_count": len(vals_z),
            }
    return {"regime": regime, "N": n_lat, "n_seeds": len(snap_results),
            "status": "OK", "aggregate": agg}


def main() -> int:
    per_regime = [audit_regime(*row) for row in LADDER]
    ok = [r for r in per_regime if r.get("status") == "OK"]
    if not ok:
        print("ERROR: no usable regimes")
        return 1

    # Branch-resolved aggregation
    vac = [r for r in ok if r["N"] <= 100]
    mat = [r for r in ok if r["N"] >= 256]

    def branch_summary(regs: list, label: str) -> dict:
        summary = {"label": label, "regimes": [r["regime"] for r in regs]}
        for ranking in ("core", "halo"):
            summary[ranking] = {}
            for p in PERCENTILES:
                key = f"p{p}"
                rem_vals, rand_vals, z_vals = [], [], []
                for r in regs:
                    a = r.get("aggregate", {}).get(ranking, {}).get(key)
                    if a is None:
                        continue
                    rem_vals.append(a["lam2_removal_mean"])
                    rand_vals.append(a["lam2_random_removal_mean"])
                    if a.get("z_score_mean") is not None:
                        z_vals.append(a["z_score_mean"])
                if not rem_vals:
                    continue
                summary[ranking][key] = {
                    "removal_mean": float(np.mean(rem_vals)),
                    "random_mean": float(np.mean(rand_vals)),
                    "delta": float(np.mean(rem_vals) - np.mean(rand_vals)),
                    "z_mean": float(np.mean(z_vals)) if z_vals else None,
                    "delta_fraction": (
                        float((np.mean(rem_vals) - np.mean(rand_vals))
                              / max(abs(np.mean(rand_vals)), 1e-9))),
                }
        return summary

    vac_summary = branch_summary(vac, "vacuum_N_leq_100")
    mat_summary = branch_summary(mat, "matter_N_geq_256")

    # Verdict per ranking: defect-driven means z < -1 (removing
    # nodes collapses lambda_2 below random) at deep percentiles.
    def verdict_for(summary: dict, ranking: str) -> dict:
        scores = []
        bucket = summary.get(ranking, {})
        for p in PERCENTILES:
            a = bucket.get(f"p{p}")
            if a is None:
                continue
            scores.append({"p_pct": p,
                           "z": a.get("z_mean"),
                           "delta_fraction": a.get("delta_fraction")})
        return {
            "ranking": ranking,
            "per_percentile_scores": scores,
            "drives_SG": bool(any(s["z"] is not None and s["z"] < -1.0
                                   for s in scores if s["p_pct"] in (10, 25, 50))),
        }

    out = {
        "headline": (
            "Lemma B defect-condensation hypothesis test: does removing "
            "matter-core nodes (row-variance proxy) collapse lambda_2 more "
            "than removing the same number of random nodes? Tested across "
            "5 percentile thresholds (1%/5%/10%/25%/50%) and 10-regime "
            "canonical ladder; branch-resolved vacuum vs matter."),
        "method": {
            "ladder": [r[0] for r in LADDER],
            "matter_core_proxy": "row_variance_Xi_AUC_0.83-0.90_LORO",
            "percentiles_top": PERCENTILES,
            "n_random_permutations": N_RANDOM_PERM,
            "max_seeds_per_regime": MAX_SEEDS_PER_REGIME,
            "rng_seed": RNG_SEED,
        },
        "per_regime": per_regime,
        "branch_summaries": {
            "vacuum_N_leq_100": vac_summary,
            "matter_N_geq_256": mat_summary,
        },
        "verdicts": {
            "vacuum_core": verdict_for(vac_summary, "core"),
            "vacuum_halo": verdict_for(vac_summary, "halo"),
            "matter_core": verdict_for(mat_summary, "core"),
            "matter_halo": verdict_for(mat_summary, "halo"),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_summary(out)
    return 0


def print_summary(out: dict) -> None:
    print("=" * 110)
    print("Lemma B defect-condensation hypothesis test")
    print("=" * 110)
    print()
    print("Branch-resolved summary (two rankings: core = row-variance, halo = |Fiedler v_2|):")
    for branch_label, branch in out["branch_summaries"].items():
        print(f"\n=== {branch_label} (regimes: {branch.get('regimes', [])}) ===")
        for ranking in ("core", "halo"):
            print(f"\n  Ranking: {ranking}")
            print(f"  {'p%':<5} {'remove_lam2':>13} {'random_lam2':>13} "
                  f"{'delta':>10} {'delta/random':>13} {'z_mean':>9}")
            for p in PERCENTILES:
                a = branch.get(ranking, {}).get(f"p{p}")
                if a is None:
                    continue
                z_str = f"{a['z_mean']:.2f}" if a['z_mean'] is not None else "NA"
                print(f"  top-{p:>2}% {a['removal_mean']:>13.4f} {a['random_mean']:>13.4f} "
                      f"{a['delta']:>+10.4f} {a['delta_fraction']*100:>+12.2f}% "
                      f"{z_str:>9}")
    print()
    print("Verdicts:")
    for label, v in out["verdicts"].items():
        flag = ("**DRIVES_SG**" if v["drives_SG"]
                else "no significant SG advantage over random")
        print(f"  {label} ({v['ranking']}): {flag}")
    print()
    print(f"Output: {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    raise SystemExit(main())
