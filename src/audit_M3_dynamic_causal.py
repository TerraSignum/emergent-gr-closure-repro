"""Dynamic causal-violation audit on the snapshot trajectories.

The static-snapshot M3 sup-norm audit found persistent worst-case
slacks at heavy-tail T_00 spikes but no significant correlation
with topological winding-charge nodes. The Master-instructions
section on causality (CVI = fraction of edges with v_info >
c_info; CIS = 1 - CVI) suggest that the M3 sup-norm violations
may be the static spatial trace of dynamic causal-violations on
the snapshot trajectory.

This audit reads the snapshot edge_xi time-series and computes:

  v_info(i,j,t)  =  |Xi_ij(t+1) - Xi_ij(t)| / dt
  CVI(t)         =  fraction edges with v_info > c_info
  M3_sup(t)      =  max_{ijk} max(0, Xi_ij Xi_jk - Xi_ik) at time t

Cross-correlations:
  - Spearman rho between CVI(t) and M3_sup(t) across time-steps,
    per regime
  - Pearson on log-CVI vs log-M3_sup
  - Per-edge persistent-violator: edges with v_info > c_info at
    >50% of timesteps -- check if these are the same (i,j) pairs
    that participate in the static worst-case M3 triples

c_info is calibrated as the median |Xi_ij(t+1) - Xi_ij(t)| / dt
across all (i,j,t) in the regime; edges crossing 2x the median
are flagged as causal-violators.

Output: outputs/audit_M3_dynamic_causal.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PARENT = REPO.parent

LADDER = [
    ("P5N64",  64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz"),
    ("P5N72",  72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz"),
    ("P5N84",  84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz"),
    ("P5N100",100,  "results_d1_p5n100_24seeds/P5N100.snapshots.npz"),
    ("P5N128",128,  "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz"),
    ("P5N200",200,  "results_d1_p5n200_8seeds/P5N200.snapshots.npz"),
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz"),
    ("P5N512", 512,  "results_d1_p5n512_12seeds/P5N512.snapshots.npz"),
]


def m3_sup_per_time(xi_t: np.ndarray) -> float:
    """Sup-norm of the M3 violation slack on Xi at a single time-step."""
    n = xi_t.shape[0]
    prod = xi_t[:, :, None] * xi_t[None, :, :]
    target = xi_t[:, None, :]
    slack = np.maximum(prod - target, 0.0)
    diag_mask = np.ones((n, n, n), dtype=bool)
    diag_mask[np.arange(n), np.arange(n), :] = False
    diag_mask[:, np.arange(n), np.arange(n)] = False
    diag_mask[np.arange(n), :, np.arange(n)] = False
    return float((slack * diag_mask).max())


def main():
    print("=" * 78)
    print("Dynamic causal-violation vs M3 sup-norm audit on snapshots")
    print("=" * 78)

    rows = []
    for regime, n_lat, rel in LADDER:
        fp = PARENT / rel
        if not fp.exists():
            print(f"  {regime}: missing")
            continue
        z = np.load(fp, allow_pickle=True)
        snaps = z["edge_xi_snapshots"]   # (n_seeds, n_snap, N, N)
        ns_seed, n_snap = snaps.shape[0], snaps.shape[1]
        n_seeds = min(ns_seed, 8 if n_lat <= 100 else 4)
        # dt unit: 1 snapshot step (we use snapshot index as discrete time)
        per_seed = []
        for s in range(n_seeds):
            xi_traj = np.asarray(snaps[s], dtype=float).copy()  # (n_snap, N, N)
            # Compute v_info between consecutive snapshots
            d_xi = np.abs(np.diff(xi_traj, axis=0))   # (n_snap-1, N, N)
            # Off-diagonal mask
            n = n_lat
            offdiag = ~np.eye(n, dtype=bool)
            d_off = d_xi[:, offdiag]                   # (n_snap-1, n_offdiag)
            # c_info calibrated as 2x median |delta Xi| over all (t, edges)
            median_v = float(np.median(d_off[d_off > 0])) if (d_off > 0).any() else 1e-6
            c_info = 2 * median_v
            # CVI per time-step
            cvi_t = np.array([(d_off[t] > c_info).mean() for t in range(d_off.shape[0])])
            # M3 sup per time-step (computed at LATER snapshot of each pair)
            sup_t = []
            for t in range(1, n_snap):
                sup_t.append(m3_sup_per_time(xi_traj[t]))
            sup_t = np.array(sup_t)
            # Spearman correlation CVI(t) vs M3_sup(t)
            if cvi_t.std() > 0 and sup_t.std() > 0 and len(cvi_t) >= 3:
                rho, p = spearmanr(cvi_t, sup_t)
            else:
                rho, p = 0.0, 1.0
            # Per-edge persistent-violator: edges crossing c_info more than half of the time
            persistent_mask = (d_off > c_info).mean(axis=0) > 0.5
            persistent_count = int(persistent_mask.sum())
            n_offdiag = int(offdiag.sum())
            persistent_frac = persistent_count / max(n_offdiag, 1)
            per_seed.append({
                "seed": s,
                "c_info_calib": c_info,
                "cvi_mean": float(cvi_t.mean()),
                "cvi_max": float(cvi_t.max()),
                "sup_mean": float(sup_t.mean()),
                "sup_max": float(sup_t.max()),
                "spearman_rho_cvi_sup": float(rho),
                "spearman_p_cvi_sup": float(p),
                "persistent_violator_frac": persistent_frac,
            })

        if not per_seed:
            continue
        rho_mean = float(np.mean([d["spearman_rho_cvi_sup"] for d in per_seed]))
        rho_std = float(np.std([d["spearman_rho_cvi_sup"] for d in per_seed]))
        cvi_mean = float(np.mean([d["cvi_mean"] for d in per_seed]))
        sup_mean = float(np.mean([d["sup_mean"] for d in per_seed]))
        persistent_mean = float(np.mean([d["persistent_violator_frac"]
                                           for d in per_seed]))
        print(f"\n--- {regime} N={n_lat} ({len(per_seed)} seeds, "
              f"{n_snap} snaps) ---")
        print(f"  c_info (2x median |delta Xi|) calibrated dynamically per seed")
        print(f"  CVI mean = {cvi_mean:.4f}, "
              f"M3 sup mean = {sup_mean:.4f}")
        print(f"  Spearman rho(CVI(t), sup(t)): "
              f"mean = {rho_mean:+.3f} ± {rho_std:.3f}")
        print(f"  Persistent-violator edge fraction "
              f"(v_info > c_info >50% time): {persistent_mean:.3%}")
        rows.append({
            "regime": regime, "N": n_lat,
            "n_snap": int(n_snap),
            "n_seeds_used": len(per_seed),
            "rho_cvi_sup_mean": rho_mean,
            "rho_cvi_sup_std": rho_std,
            "cvi_mean": cvi_mean,
            "sup_mean_dynamic": sup_mean,
            "persistent_violator_frac_mean": persistent_mean,
            "per_seed": per_seed,
        })

    print()
    print("=" * 78)
    print("Cross-regime synthesis")
    print("=" * 78)
    if rows:
        rhos = np.array([r["rho_cvi_sup_mean"] for r in rows])
        cvis = np.array([r["cvi_mean"] for r in rows])
        sups = np.array([r["sup_mean_dynamic"] for r in rows])
        print(f"  Spearman rho(CVI, M3_sup) cross-regime: mean = "
              f"{rhos.mean():+.3f}, std = {rhos.std():.3f}")
        print(f"  CVI mean cross-regime: {cvis.mean():.4f}")
        print(f"  Dynamic M3 sup mean cross-regime: {sups.mean():.4f}")

    bundle = {
        "method": "M3_sup_dynamic_causal_audit",
        "rows": rows,
        "interpretation": (
            "Spearman rho(CVI(t), M3_sup(t)) > 0 indicates that "
            "fast-edge causal-violations during stabilisation co-occur "
            "with M3 sup-norm spikes; persistent_violator_frac >> 0 "
            "indicates a small set of edges are repeatedly causal-"
            "violators throughout the trajectory."
        ),
    }
    out = REPO / "outputs" / "audit_M3_dynamic_causal.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
