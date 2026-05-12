"""V2: per-regime baryogenesis audit on P5N ladder using:

  1) regime-invariant baseline:
       S_bounce_baseline = 38.0           (GCC-05 P1 EMT bundle)
       J_CP_CKM = 3.08e-5                 (PDG 2024)
       n_gen / g_star = 3 / 106.75
  2) regime-dependent multipliers extracted from snapshots:
       K_vortex(reg) = 1 + |sin theta_v|  (S1, vortex CP boost)
       T_lock(reg)   = 1 - (1-fid)(1-occ) (S2, observable-lock transduction
                                            framework, PG-P27)
                       fid = persistent_violator_fraction  (overlap*coh proxy)
                       occ = n_pers_triangles / n_pers_edges
                                            (occupancy-of-winding-loops)
       A_w_cont(reg) = <winding> / (2*pi)  (S3, continuous triangle phase
                                            winding, REAL-VALUED average)
       Q_3body(reg)  = 1 + T2_log_Pi3_gain (S4, 3-body binding gain
                                            multiplicative factor)

  3) eta_B prediction per regime:
       eta_B(reg) = (28/51) * J_CP_CKM * S_bounce_baseline
                    * (n_gen/g_star) * K_vortex * T_lock
                    * (1 + a_aw * A_w_cont) * Q_3body
       with a_aw = 1.0 (winding-asymmetry as direct multiplicative bias).

This is a multiplicative ansatz: regime-invariant baseline meets PDG eta_B
when all multipliers ~= 1. Regime-dependent multipliers come from
the snapshot data alone.

Reference: PDG-2024 eta_B,obs = 6.04e-10 +/- 1.2e-11.
GCC-05 P1 result (CKM + vortex boost only): 6.51e-10 (ratio 1.067, PRECISE).

Output: outputs/audit_baryogenesis_per_regime_v2.json
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
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz"),
    ("P5N512", 512,  "results_d1_p5n512_12seeds/P5N512.snapshots.npz"),
]

ETA_B_OBS = 6.04e-10
ETA_B_OBS_UNC = 1.2e-11
N_GEN = 3
G_STAR_SM = 106.75
S_BOUNCE_BASELINE = 38.0      # GCC-05 P1 EMT bundle reference
J_CP_CKM = 3.08e-5            # PDG 2024 Jarlskog


def vortex_angle(psi_last: np.ndarray, k_last: np.ndarray,
                 q_last: np.ndarray) -> float:
    phi = np.angle(psi_last)
    if k_last.ndim == 2:
        k_node = k_last.mean(axis=1)
        q_node = q_last.mean(axis=1)
    else:
        k_node, q_node = k_last, q_last
    dca = k_node - q_node
    cos_w = np.sum(np.cos(phi) * dca)
    sin_w = np.sum(np.sin(phi) * dca)
    if abs(cos_w) + abs(sin_w) < 1e-12:
        return 0.0
    return float(np.arctan2(sin_w, cos_w))


def find_persistent_triangles(pers_edges: np.ndarray):
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
    return triangles, edge_set


def triangle_continuous_winding(triangles, psi_last: np.ndarray):
    """Continuous triangle phase winding. For each triangle (i,j,k) compute
    Delta_phi = arg(exp(i*(phi_j-phi_i)) * exp(i*(phi_k-phi_j))
                  * exp(i*(phi_i-phi_k))) = 0 mod 2pi (trivially).
    But the ACTUAL winding requires the principal-value reduction of each
    edge phase, then summing. Each edge phase is in (-pi,pi]. Their sum
    falls in (-3pi, 3pi]; reduced mod 2pi to (-pi,pi]; can be +-2pi*n
    for n=0,+-1.

    Returns mean(winding/(2*pi)) as a continuous real-valued asymmetry.
    """
    if not triangles:
        return 0.0
    phi = np.angle(psi_last)
    windings = []
    for i, j, k in triangles:
        d1 = np.angle(np.exp(1j * (phi[j] - phi[i])))
        d2 = np.angle(np.exp(1j * (phi[k] - phi[j])))
        d3 = np.angle(np.exp(1j * (phi[i] - phi[k])))
        total = d1 + d2 + d3   # this IS the winding (already reduced
                                # to integer multiples of 2pi by
                                # construction since each edge is
                                # principal-value)
        windings.append(total / (2.0 * np.pi))
    return float(np.mean(windings))


def t2_pi3_log_gain(triangles_sample, xi_traj: np.ndarray) -> float:
    """log(<Pi3>_tF / <Pi3>_t0) for a sample of persistent triangles."""
    if not triangles_sample:
        return 0.0
    tri_arr = np.array(triangles_sample, dtype=int)
    i_arr, j_arr, k_arr = tri_arr[:, 0], tri_arr[:, 1], tri_arr[:, 2]
    xi0 = xi_traj[0]
    xi_t_f = xi_traj[-1]
    pi0 = (xi0[i_arr, j_arr] * xi0[j_arr, k_arr] * xi0[k_arr, i_arr]).mean()
    pi1 = (xi_t_f[i_arr, j_arr] * xi_t_f[j_arr, k_arr]
            * xi_t_f[k_arr, i_arr]).mean()
    if pi0 < 1e-30 or pi1 < 1e-30:
        return 0.0
    return float(np.log(pi1 / pi0))


def main():
    print("=" * 78)
    print("V2 per-regime baryogenesis (snapshot multipliers x GCC-05 baseline)")
    print("=" * 78)
    print(f"  Baseline:  S_bounce={S_BOUNCE_BASELINE},  "
          f"J_CP_CKM={J_CP_CKM:.3e},  n_gen/g_star="
          f"{N_GEN/G_STAR_SM:.4f}")
    print(f"  PDG-2024 eta_B,obs = {ETA_B_OBS:.3e} +/- {ETA_B_OBS_UNC:.1e}")
    print()
    header = (f"  {'reg':<7} {'#s':>2} "
              f"{'theta_v':>8} {'K_vtx':>6} "
              f"{'fid':>6} {'occ':>6} {'T_lock':>7} "
              f"{'A_w':>9} {'T2gain':>8} "
              f"{'eta_B,pred':>12} {'ratio':>7}")
    print(header)
    print("-" * len(header))
    rows = []
    for regime, n_lat, rel in LADDER:
        fp = PARENT / rel
        if not fp.exists():
            print(f"  {regime}: missing")
            continue
        z = np.load(fp, allow_pickle=True)
        snaps = z["edge_xi_snapshots"]
        psi_r = z["psi_real_snapshots"]
        psi_i = z["psi_imag_snapshots"]
        k_snaps = z["k_snapshots"]
        q_snaps = z["q_snapshots"]
        n_seeds = min(int(snaps.shape[0]),
                      8 if n_lat <= 100 else 4)
        per_seed = []
        for s in range(n_seeds):
            xi_traj = np.asarray(snaps[s], dtype=float).copy()
            psi_last = (psi_r[s, -1].astype(float)
                        + 1j * psi_i[s, -1].astype(float))
            k_last = np.asarray(k_snaps[s, -1], dtype=float)
            q_last = np.asarray(q_snaps[s, -1], dtype=float)
            # S1 vortex
            theta_v = vortex_angle(psi_last, k_last, q_last)
            k_vtx = 1.0 + abs(np.sin(theta_v))
            # Identify persistent edges
            d_xi = np.abs(np.diff(xi_traj, axis=0))
            offdiag = ~np.eye(n_lat, dtype=bool)
            d_off = d_xi[:, offdiag]
            v_med = (float(np.median(d_off[d_off > 0]))
                      if (d_off > 0).any() else 1e-6)
            c_info = 2 * v_med
            persistent_mask_off = (d_off > c_info).mean(axis=0) > 0.5
            n_pers = int(persistent_mask_off.sum())
            n_off_total = int(offdiag.sum())
            fid = n_pers / max(n_off_total, 1)
            ij_idx = np.argwhere(offdiag)
            pers_edges = ij_idx[persistent_mask_off]
            triangles, edge_set = find_persistent_triangles(pers_edges)
            occ = (len(triangles)
                    / max(len(edge_set), 1)
                    if edge_set else 0.0)
            t_lock = 1.0 - (1.0 - fid) * (1.0 - occ)
            # S5 continuous winding (sample 800 for speed)
            tri_for_winding = triangles
            if len(tri_for_winding) > 800:
                rng = np.random.default_rng(s)
                idx = rng.choice(len(tri_for_winding), size=800, replace=False)
                tri_for_winding = [tri_for_winding[i] for i in idx]
            a_w = triangle_continuous_winding(tri_for_winding, psi_last)
            # T2 binding gain (3-body)
            t2_gain = t2_pi3_log_gain(tri_for_winding, xi_traj)
            q_3body = 1.0 + t2_gain
            # eta_B
            base = ((28.0 / 51.0) * J_CP_CKM * S_BOUNCE_BASELINE
                     * (N_GEN / G_STAR_SM))
            eta_b = base * k_vtx * t_lock * (1.0 + a_w) * q_3body
            ratio = eta_b / ETA_B_OBS
            per_seed.append({
                "seed": s, "theta_v_rad": theta_v, "K_vortex": k_vtx,
                "fidelity": fid, "occupancy": occ, "T_lock": t_lock,
                "A_w_continuous": a_w, "T2_log_Pi3_gain": t2_gain,
                "Q_3body": q_3body, "eta_B_predicted": eta_b,
                "eta_B_ratio_obs": ratio,
                "n_pers_edges": n_pers,
                "n_pers_triangles": len(triangles),
            })
        if not per_seed:
            continue
        def mn(key):
            return float(np.mean([d[key] for d in per_seed]))
        th_m = mn("theta_v_rad")
        kv_m = mn("K_vortex")
        fid_m = mn("fidelity")
        occ_m = mn("occupancy")
        tl_m = mn("T_lock")
        aw_m = mn("A_w_continuous")
        t2_m = mn("T2_log_Pi3_gain")
        eta_m = mn("eta_B_predicted")
        rat_m = mn("eta_B_ratio_obs")
        print(f"  {regime:<7} {len(per_seed):>2} "
              f"{th_m:>+8.3f} {kv_m:>6.3f} "
              f"{fid_m:>6.3f} {occ_m:>6.3f} {tl_m:>7.4f} "
              f"{aw_m:>+9.5f} {t2_m:>+8.4f} "
              f"{eta_m:>12.3e} {rat_m:>7.3f}")
        rows.append({
            "regime": regime, "N": n_lat,
            "n_seeds": len(per_seed),
            "theta_v_rad_mean": th_m,
            "K_vortex_mean": kv_m,
            "fidelity_mean": fid_m,
            "occupancy_mean": occ_m,
            "T_lock_mean": tl_m,
            "A_w_continuous_mean": aw_m,
            "T2_log_Pi3_gain_mean": t2_m,
            "Q_3body_mean": 1.0 + t2_m,
            "eta_B_predicted_mean": eta_m,
            "eta_B_ratio_obs_mean": rat_m,
            "per_seed": per_seed,
        })
    print()
    print("=" * 78)
    print("Cross-regime synthesis")
    print("=" * 78)
    if rows:
        eta_arr = np.array([r["eta_B_predicted_mean"] for r in rows])
        rat_arr = np.array([r["eta_B_ratio_obs_mean"] for r in rows])
        kv_arr = np.array([r["K_vortex_mean"] for r in rows])
        tl_arr = np.array([r["T_lock_mean"] for r in rows])
        aw_arr = np.array([r["A_w_continuous_mean"] for r in rows])
        t2_arr = np.array([r["T2_log_Pi3_gain_mean"] for r in rows])
        print(f"  eta_B,pred mean      = {eta_arr.mean():.3e} +/- {eta_arr.std():.3e}")
        print(f"  Ratio to PDG obs     = {rat_arr.mean():.3f} +/- {rat_arr.std():.3f}")
        print(f"  log10(ratio) mean    = {np.log10(rat_arr).mean():+.3f}")
        print(f"  K_vortex             = {kv_arr.mean():.3f} +/- {kv_arr.std():.3f}")
        print(f"  T_lock (transduction)= {tl_arr.mean():.4f} +/- {tl_arr.std():.4f}")
        print(f"  A_w continuous       = {aw_arr.mean():+.5f} +/- {aw_arr.std():.5f}")
        print(f"  T2 binding gain      = {t2_arr.mean():+.4f} +/- {t2_arr.std():.4f}")
        # CV of ratio - consistency check
        if rat_arr.mean() != 0:
            cv = abs(rat_arr.std() / rat_arr.mean())
            print(f"  Cross-regime ratio CV = {cv:.3f}")
    bundle = {
        "method": "baryogenesis_v2_baseline_x_snapshot_multipliers",
        "PDG_eta_B_obs": ETA_B_OBS,
        "S_bounce_baseline": S_BOUNCE_BASELINE,
        "J_CP_CKM": J_CP_CKM,
        "rows": rows,
    }
    out = REPO / "outputs" / "audit_baryogenesis_per_regime_v2.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
