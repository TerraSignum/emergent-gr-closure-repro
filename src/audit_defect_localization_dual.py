"""Test two defect-localization hypotheses for the asym(PPN-PNN) signal:

Test U (user 2026-05-02): minimal M3-distance defect as classifier
  For each persistent triangle (i,j,k):
    d_ij = -log Xi_ij, etc.
    Three M3-deficits:
      delta_1 = d_ij + d_jk - d_ik  (negative = M3 violation; positive = strict triangle)
      delta_2 = d_jk + d_ki - d_ij
      delta_3 = d_ki + d_ij - d_jk
    Δ_min = the deficit closest to zero in absolute value
    Sign(Δ_min) classifies triangle into "almost-strict" vs "almost-violating"
    Test: does sign correlate with PPN/PNN class?

Test M (mine): phase singularity colocation
  Identify phase singularity nodes: |psi_i|^2 < 5th percentile (vortex-core analog)
  For each persistent triangle, count how many of the 3 corners are singularities
  Subset triangles by singularity-corner count {0, 1, 2, 3}
  Test: does asym(PPN-PNN) per subset differ from bulk?

Joint test: do BOTH classifications track the asymmetry sign?

Output: outputs/audit_defect_localization_dual.json
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

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
    ("P5N256",256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz"),
    ("P5N300",300,  "results_d1_p5n300_12seeds/P5N300.snapshots.npz"),
    ("P5N512", 512,  "results_d1_p5n512_12seeds/P5N512.snapshots.npz"),
]


def find_persistent_triangles(pers_edges):
    edge_set = set()
    for a, b in pers_edges:
        a, b = int(a), int(b)
        if a == b:
            continue
        edge_set.add((min(a, b), max(a, b)))
    adj = defaultdict(set)
    for a, b in edge_set:
        adj[a].add(b)
        adj[b].add(a)
    triangles = []
    for (i, j) in edge_set:
        common = adj[i] & adj[j]
        for k in common:
            if k <= j:
                continue
            triangles.append((i, j, k))
    return triangles


def per_triangle_features(triangles, xi_last, psi_last):
    """For each triangle, compute:
    - delta_min = signed min M3 deficit
    - corner_singularity flags (3-tuple bool)
    - PPN/PNN class
    Returns list of (delta_min, n_singularity, class) tuples.
    """
    if not triangles:
        return []
    psi_abs2 = np.real(psi_last) ** 2 + np.imag(psi_last) ** 2
    sing_threshold = float(np.percentile(psi_abs2, 5))
    sing_mask = psi_abs2 < sing_threshold
    phi = np.angle(psi_last)
    out = []
    for i, j, k in triangles:
        # M3 deficits in log-space
        d_ij = -np.log(max(xi_last[i, j], 1e-9))
        d_jk = -np.log(max(xi_last[j, k], 1e-9))
        d_ki = -np.log(max(xi_last[k, i], 1e-9))
        delta_1 = d_ij + d_jk - d_ki
        delta_2 = d_jk + d_ki - d_ij
        delta_3 = d_ki + d_ij - d_jk
        # signed minimum (smallest in absolute value)
        deltas = [delta_1, delta_2, delta_3]
        delta_min = min(deltas, key=abs)
        # Singularity count
        n_sing = (int(sing_mask[i]) + int(sing_mask[j])
                   + int(sing_mask[k]))
        # Phase signs
        d_ij_p = np.angle(np.exp(1j * (phi[j] - phi[i])))
        d_jk_p = np.angle(np.exp(1j * (phi[k] - phi[j])))
        d_ki_p = np.angle(np.exp(1j * (phi[i] - phi[k])))
        if (abs(d_ij_p) < 1e-9 or abs(d_jk_p) < 1e-9
            or abs(d_ki_p) < 1e-9):
            tri_class = "ambiguous"
        else:
            n_pos = ((1 if d_ij_p > 0 else 0)
                      + (1 if d_jk_p > 0 else 0)
                      + (1 if d_ki_p > 0 else 0))
            if n_pos == 3:
                tri_class = "PPP"
            elif n_pos == 2:
                tri_class = "PPN"
            elif n_pos == 1:
                tri_class = "PNN"
            else:
                tri_class = "NNN"
        out.append({
            "delta_min": delta_min,
            "n_singularity_corners": n_sing,
            "class": tri_class,
        })
    return out


def asym_in_subset(features, subset_filter):
    """asym = (n_PPN - n_PNN) / (n_PPN + n_PNN) on subset matching filter."""
    n_PPN = sum(1 for f in features if subset_filter(f) and f["class"] == "PPN")
    n_PNN = sum(1 for f in features if subset_filter(f) and f["class"] == "PNN")
    if n_PPN + n_PNN == 0:
        return float("nan"), 0, 0
    return (n_PPN - n_PNN) / (n_PPN + n_PNN), n_PPN, n_PNN


def main():
    print("=" * 80)
    print("Defect-Localization Dual Audit: Test U (M3-min-defect) + Test M (psi-singularity)")
    print("=" * 80)
    print()
    rows = []
    for regime, n_lat, rel in LADDER:
        fp = PARENT / rel
        if not fp.exists():
            continue
        z = np.load(fp, allow_pickle=True)
        snaps = z["edge_xi_snapshots"]
        psi_r = z["psi_real_snapshots"]
        psi_i = z["psi_imag_snapshots"]
        n_seeds = int(snaps.shape[0])
        per_seed = []
        for s in range(n_seeds):
            xi_traj = np.asarray(snaps[s], dtype=float).copy()
            xi_last = xi_traj[-1].copy()
            psi_last = (psi_r[s, -1].astype(float)
                        + 1j * psi_i[s, -1].astype(float))
            d_xi = np.abs(np.diff(xi_traj, axis=0))
            offdiag = ~np.eye(n_lat, dtype=bool)
            d_off = d_xi[:, offdiag]
            v_med = (float(np.median(d_off[d_off > 0]))
                      if (d_off > 0).any() else 1e-6)
            c_info = 2 * v_med
            persistent_mask_off = (d_off > c_info).mean(axis=0) > 0.5
            ij_idx = np.argwhere(offdiag)
            pers_edges = ij_idx[persistent_mask_off]
            triangles = find_persistent_triangles(pers_edges)
            feats = per_triangle_features(triangles, xi_last, psi_last)
            if not feats:
                continue
            # Test U: by sign of delta_min
            asym_u_pos, npp, npn = asym_in_subset(
                feats, lambda f: f["delta_min"] > 0)
            asym_u_neg, npp_n, npn_n = asym_in_subset(
                feats, lambda f: f["delta_min"] < 0)
            asym_u_all, _, _ = asym_in_subset(feats, lambda f: True)
            # Test M: by singularity corners count
            asym_m_0, _, _ = asym_in_subset(
                feats, lambda f: f["n_singularity_corners"] == 0)
            asym_m_1, _, _ = asym_in_subset(
                feats, lambda f: f["n_singularity_corners"] == 1)
            asym_m_2plus, _, _ = asym_in_subset(
                feats, lambda f: f["n_singularity_corners"] >= 2)
            n_total = len(feats)
            n_sing_0 = sum(1 for f in feats
                            if f["n_singularity_corners"] == 0)
            n_sing_1 = sum(1 for f in feats
                            if f["n_singularity_corners"] == 1)
            n_sing_2 = sum(1 for f in feats
                            if f["n_singularity_corners"] >= 2)
            n_dpos = sum(1 for f in feats if f["delta_min"] > 0)
            n_dneg = sum(1 for f in feats if f["delta_min"] < 0)
            per_seed.append({
                "seed": s,
                "n_triangles": n_total,
                "n_delta_pos": n_dpos, "n_delta_neg": n_dneg,
                "n_sing_0": n_sing_0, "n_sing_1": n_sing_1,
                "n_sing_2plus": n_sing_2,
                "asym_all": asym_u_all,
                "asym_U_delta_pos": asym_u_pos,
                "asym_U_delta_neg": asym_u_neg,
                "asym_M_sing_0": asym_m_0,
                "asym_M_sing_1": asym_m_1,
                "asym_M_sing_2plus": asym_m_2plus,
            })
        if not per_seed:
            continue
        # Aggregate (mean of seeds, ignoring nan)
        def mn(key):
            vals = [d[key] for d in per_seed
                     if not (isinstance(d[key], float) and np.isnan(d[key]))]
            return float(np.mean(vals)) if vals else float("nan")
        def std(key):
            vals = [d[key] for d in per_seed
                     if not (isinstance(d[key], float) and np.isnan(d[key]))]
            return (float(np.std(vals)) / np.sqrt(max(len(vals), 1))
                     if vals else float("nan"))
        asym_all_m = mn("asym_all")
        asym_U_pos = mn("asym_U_delta_pos")
        asym_U_neg = mn("asym_U_delta_neg")
        asym_M_0 = mn("asym_M_sing_0")
        asym_M_1 = mn("asym_M_sing_1")
        asym_M_2 = mn("asym_M_sing_2plus")
        n_dpos = mn("n_delta_pos")
        n_dneg = mn("n_delta_neg")
        n_sing0 = mn("n_sing_0")
        n_sing1 = mn("n_sing_1")
        n_sing2 = mn("n_sing_2plus")
        unc_U_pos = std("asym_U_delta_pos")
        unc_U_neg = std("asym_U_delta_neg")
        unc_M_2 = std("asym_M_sing_2plus")
        print(f"--- {regime} N={n_lat} (n_seeds={len(per_seed)}) ---")
        print(f"  Test U (M3 delta_min sign):")
        print(f"    Δ>0 triangles n_avg={n_dpos:.0f}  "
              f"asym={asym_U_pos:+.4f} +/- {unc_U_pos:.4f}")
        print(f"    Δ<0 triangles n_avg={n_dneg:.0f}  "
              f"asym={asym_U_neg:+.4f} +/- {unc_U_neg:.4f}")
        print(f"    Diff (Δ>0 - Δ<0) = "
              f"{asym_U_pos - asym_U_neg:+.4f}")
        print(f"  Test M (psi-singularity corners):")
        print(f"    0 sing-corners (n_avg={n_sing0:.0f}):     "
              f"asym={asym_M_0:+.4f}")
        print(f"    1 sing-corner  (n_avg={n_sing1:.0f}):     "
              f"asym={asym_M_1:+.4f}")
        print(f"    2+ sing-corners (n_avg={n_sing2:.0f}):    "
              f"asym={asym_M_2:+.4f} +/- {unc_M_2:.4f}")
        print(f"  asym_all = {asym_all_m:+.4f}")
        rows.append({
            "regime": regime, "N": n_lat, "n_seeds": len(per_seed),
            "asym_all_mean": asym_all_m,
            "asym_U_delta_pos_mean": asym_U_pos,
            "asym_U_delta_neg_mean": asym_U_neg,
            "asym_U_pos_unc": unc_U_pos,
            "asym_U_neg_unc": unc_U_neg,
            "asym_M_sing_0_mean": asym_M_0,
            "asym_M_sing_1_mean": asym_M_1,
            "asym_M_sing_2plus_mean": asym_M_2,
            "asym_M_2plus_unc": unc_M_2,
            "n_delta_pos_mean": n_dpos,
            "n_delta_neg_mean": n_dneg,
            "n_sing_0_mean": n_sing0,
            "n_sing_1_mean": n_sing1,
            "n_sing_2plus_mean": n_sing2,
            "per_seed": per_seed,
        })
    print()
    print("=" * 80)
    print("Cross-regime synthesis")
    print("=" * 80)
    if rows:
        # Test U
        u_pos = np.array([r["asym_U_delta_pos_mean"] for r in rows
                            if not np.isnan(r["asym_U_delta_pos_mean"])])
        u_neg = np.array([r["asym_U_delta_neg_mean"] for r in rows
                            if not np.isnan(r["asym_U_delta_neg_mean"])])
        u_all = np.array([r["asym_all_mean"] for r in rows
                            if not np.isnan(r["asym_all_mean"])])
        if u_pos.size:
            print(f"  Test U: asym (Δ_min > 0) = {u_pos.mean():+.5f} "
                  f"+/- {u_pos.std()/np.sqrt(len(u_pos)):.5f}")
        if u_neg.size:
            print(f"  Test U: asym (Δ_min < 0) = {u_neg.mean():+.5f} "
                  f"+/- {u_neg.std()/np.sqrt(len(u_neg)):.5f}")
        if u_pos.size and u_neg.size:
            diff = u_pos.mean() - u_neg.mean()
            diff_unc = np.sqrt((u_pos.std()/np.sqrt(len(u_pos)))**2
                                + (u_neg.std()/np.sqrt(len(u_neg)))**2)
            print(f"  Test U: Diff (>0)-(<0) = {diff:+.5f} +/- {diff_unc:.5f} "
                  f"({abs(diff)/max(diff_unc,1e-9):.2f}σ)")
        # Test M
        m_0 = np.array([r["asym_M_sing_0_mean"] for r in rows
                          if not np.isnan(r["asym_M_sing_0_mean"])])
        m_2 = np.array([r["asym_M_sing_2plus_mean"] for r in rows
                          if not np.isnan(r["asym_M_sing_2plus_mean"])])
        if m_0.size:
            print(f"  Test M: asym (0 sing) = {m_0.mean():+.5f} "
                  f"+/- {m_0.std()/np.sqrt(len(m_0)):.5f}")
        if m_2.size:
            print(f"  Test M: asym (≥2 sing) = {m_2.mean():+.5f} "
                  f"+/- {m_2.std()/np.sqrt(len(m_2)):.5f}")
        if m_0.size and m_2.size:
            diff = m_2.mean() - m_0.mean()
            diff_unc = np.sqrt((m_0.std()/np.sqrt(len(m_0)))**2
                                + (m_2.std()/np.sqrt(len(m_2)))**2)
            print(f"  Test M: Diff (≥2)-(0) = {diff:+.5f} +/- {diff_unc:.5f} "
                  f"({abs(diff)/max(diff_unc,1e-9):.2f}σ)")
        if u_all.size:
            print(f"  asym_all (baseline) = {u_all.mean():+.5f} "
                  f"+/- {u_all.std()/np.sqrt(len(u_all)):.5f}")

    bundle = {
        "method": "defect_localization_dual_audit",
        "rows": rows,
    }
    out = REPO / "outputs" / "audit_defect_localization_dual.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
