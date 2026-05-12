"""Three falsification tests for the per-regime baryogenesis hypothesis.

Stufe A test (specificity of bi->tri binding gain):
  H0: T2 = log(<Pi3>_tF / <Pi3>_t0) is positive on PERSISTENT triangles AND
      on a random sample of NON-PERSISTENT triangles equally well.
  H1: T2 is significantly larger on persistent triangles.
  --> If T2_persistent > T2_random by at least 1 sigma cross-regime, the
      bi->tri binding gain is specific to persistent-violator structure,
      not a generic Xi-trajectory artefact.

Stufe B test (K_vortex CP-specificity):
  H0: K_vortex computed with the actual psi-phase phi(x) equals K_vortex
      computed with phi(x) randomly permuted across nodes (any K-Q
      anti-symmetry alone gives the boost).
  H1: K_vortex from real phases is significantly larger than the
      shuffled-phase null.
  --> If real K_vortex >> shuffled K_vortex cross-regime, the CP-boost
      is genuinely from psi-DCA alignment, not a residual K-Q-only effect.

Stufe C test (decay-asymmetry mass inversion at higher N):
  Re-run the triangle phase-class audit on the same 6 P5N regimes plus
  P5N256 and P5N300 (8 regimes total). Track:
     P2 m(PPN)/m(PNN) trend with N - does it move below 1 (SM 0.99862)?
     P4 PPN-PNN frequency asymmetry - does it sharpen with N?
  Hypothesis: at sufficient N, internal triangle structure differentiates
  Proton-class from Neutron-class. If mass ratios remain at 1.0 +/- 0.001
  even at N=300, the resolution argument fails.

Output: outputs/audit_baryogenesis_hypothesis_tests.json
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

N_SHUFFLE_TRIALS = 50  # for Stufe B null distribution
N_RANDOM_TRIANGLES = 1000  # for Stufe A non-persistent baseline


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


def t2_pi3_log_gain(triangles_sample, xi_traj):
    if not triangles_sample:
        return 0.0
    tri_arr = np.array(triangles_sample, dtype=int)
    i_arr, j_arr, k_arr = tri_arr[:, 0], tri_arr[:, 1], tri_arr[:, 2]
    xi0 = xi_traj[0]
    xi_t_f = xi_traj[-1]
    pi0 = (xi0[i_arr, j_arr] * xi0[j_arr, k_arr]
            * xi0[k_arr, i_arr]).mean()
    pi1 = (xi_t_f[i_arr, j_arr] * xi_t_f[j_arr, k_arr]
            * xi_t_f[k_arr, i_arr]).mean()
    if pi0 < 1e-30 or pi1 < 1e-30:
        return 0.0
    return float(np.log(pi1 / pi0))


def random_triangles(n_lat, n_target, seed):
    """Sample n_target triangles uniformly from all C(N,3) possibilities,
    independent of any persistence mask (Stufe A null sample)."""
    rng = np.random.default_rng(seed)
    out = []
    seen = set()
    while len(out) < n_target:
        i, j, k = sorted(rng.integers(0, n_lat, 3).tolist())
        if i == j or j == k or i == k:
            continue
        if (i, j, k) in seen:
            continue
        seen.add((i, j, k))
        out.append((i, j, k))
    return out


def vortex_angle_real(psi, k_field, q_field):
    phi = np.angle(psi)
    if k_field.ndim == 2:
        k_node = k_field.mean(axis=1)
        q_node = q_field.mean(axis=1)
    else:
        k_node, q_node = k_field, q_field
    dca = k_node - q_node
    cos_w = np.sum(np.cos(phi) * dca)
    sin_w = np.sum(np.sin(phi) * dca)
    if abs(cos_w) + abs(sin_w) < 1e-12:
        return 0.0
    return float(np.arctan2(sin_w, cos_w))


def vortex_angle_shuffled(psi, k_field, q_field, n_trials, seed):
    """Null distribution: K_vortex with phase shuffled across nodes."""
    n = psi.shape[0]
    if k_field.ndim == 2:
        k_node = k_field.mean(axis=1)
        q_node = q_field.mean(axis=1)
    else:
        k_node, q_node = k_field, q_field
    dca = k_node - q_node
    rng = np.random.default_rng(seed)
    phi_real = np.angle(psi)
    null_kvx = []
    for _ in range(n_trials):
        perm = rng.permutation(n)
        phi_perm = phi_real[perm]
        cos_w = np.sum(np.cos(phi_perm) * dca)
        sin_w = np.sum(np.sin(phi_perm) * dca)
        theta = (np.arctan2(sin_w, cos_w)
                 if abs(cos_w) + abs(sin_w) > 1e-12 else 0.0)
        null_kvx.append(1.0 + abs(np.sin(theta)))
    return null_kvx


def classify_triangle_signs(triangles, psi_last):
    out = []
    phi = np.angle(psi_last)
    for i, j, k in triangles:
        d_ij = np.angle(np.exp(1j * (phi[j] - phi[i])))
        d_jk = np.angle(np.exp(1j * (phi[k] - phi[j])))
        d_ki = np.angle(np.exp(1j * (phi[i] - phi[k])))
        s_ij = int(np.sign(d_ij)) if abs(d_ij) > 1e-9 else 0
        s_jk = int(np.sign(d_jk)) if abs(d_jk) > 1e-9 else 0
        s_ki = int(np.sign(d_ki)) if abs(d_ki) > 1e-9 else 0
        out.append((s_ij, s_jk, s_ki))
    return out


def class_label(sig):
    n_pos = sum(1 for s in sig if s > 0)
    n_neg = sum(1 for s in sig if s < 0)
    n_zero = sum(1 for s in sig if s == 0)
    if n_zero > 0:
        return "ambiguous"
    if n_pos == 3:
        return "PPP"
    if n_pos == 2:
        return "PPN"
    if n_pos == 1:
        return "PNN"
    return "NNN"


def main():
    print("=" * 80)
    print("Three falsification tests for the per-regime baryogenesis hypothesis")
    print("=" * 80)
    print()
    rows = []
    for regime, n_lat, rel in LADDER:
        fp = PARENT / rel
        if not fp.exists():
            print(f"{regime}: missing")
            continue
        z = np.load(fp, allow_pickle=True)
        snaps = z["edge_xi_snapshots"]
        psi_r = z["psi_real_snapshots"]
        psi_i = z["psi_imag_snapshots"]
        k_snaps = z["k_snapshots"]
        q_snaps = z["q_snapshots"]
        n_seeds = int(snaps.shape[0])  # use ALL available seeds
        per_seed_results = []
        cls_count_total = defaultdict(int)
        cls_mass_total = defaultdict(list)
        for s in range(n_seeds):
            xi_traj = np.asarray(snaps[s], dtype=float).copy()
            psi_last = (psi_r[s, -1].astype(float)
                        + 1j * psi_i[s, -1].astype(float))
            k_last = np.asarray(k_snaps[s, -1], dtype=float)
            q_last = np.asarray(q_snaps[s, -1], dtype=float)
            # Persistent edges/triangles
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
            if not triangles:
                continue
            # Stufe A: T2 on persistent triangles (sample 800 if too many)
            tri_for_t2 = triangles
            if len(tri_for_t2) > 800:
                rng = np.random.default_rng(s)
                idx = rng.choice(len(tri_for_t2), size=800, replace=False)
                tri_for_t2 = [tri_for_t2[i] for i in idx]
            t2_persistent = t2_pi3_log_gain(tri_for_t2, xi_traj)
            # Stufe A: T2 on random NON-persistent triangle sample
            random_tri = random_triangles(
                n_lat, N_RANDOM_TRIANGLES, seed=1000 + s)
            t2_random = t2_pi3_log_gain(random_tri, xi_traj)
            # Stufe B: K_vortex real vs shuffled null
            theta_real = vortex_angle_real(psi_last, k_last, q_last)
            k_vtx_real = 1.0 + abs(np.sin(theta_real))
            null_kvx = vortex_angle_shuffled(
                psi_last, k_last, q_last, N_SHUFFLE_TRIALS, seed=2000 + s)
            null_mean = float(np.mean(null_kvx))
            null_std = float(np.std(null_kvx))
            z_score = ((k_vtx_real - null_mean) / null_std
                       if null_std > 1e-9 else 0.0)
            # Stufe C: triangle phase classes (mass proxy)
            sigs = classify_triangle_signs(tri_for_t2, psi_last)
            psi_abs2 = (np.real(psi_last) ** 2
                        + np.imag(psi_last) ** 2)
            cls_count = defaultdict(int)
            cls_mass = defaultdict(list)
            for (i, j, k), sig in zip(tri_for_t2, sigs):
                lbl = class_label(sig)
                cls_count[lbl] += 1
                cls_mass[lbl].append(
                    (psi_abs2[i] + psi_abs2[j] + psi_abs2[k]) / 3.0)
            for lbl in ["PPP", "PPN", "PNN", "NNN"]:
                cls_count_total[lbl] += cls_count[lbl]
                cls_mass_total[lbl].extend(cls_mass[lbl])
            per_seed_results.append({
                "seed": s,
                "T2_persistent": t2_persistent,
                "T2_random_baseline": t2_random,
                "T2_specificity": t2_persistent - t2_random,
                "K_vortex_real": k_vtx_real,
                "K_vortex_null_mean": null_mean,
                "K_vortex_null_std": null_std,
                "K_vortex_z_score": z_score,
                "n_pers_triangles": len(triangles),
            })
        if not per_seed_results:
            continue
        def m(key):
            return float(np.mean([d[key] for d in per_seed_results]))
        def sd(key):
            return float(np.std([d[key] for d in per_seed_results]))
        # Stufe C aggregated
        total_cls = sum(cls_count_total.values())
        frac = {lbl: cls_count_total[lbl] / max(total_cls, 1)
                 for lbl in ["PPP", "PPN", "PNN", "NNN"]}
        mass_mean = {lbl: float(np.mean(cls_mass_total[lbl]))
                       if cls_mass_total[lbl] else float("nan")
                     for lbl in ["PPP", "PPN", "PNN", "NNN"]}
        ratio_pn = (mass_mean["PPN"] / mass_mean["PNN"]
                    if mass_mean["PNN"] > 0 else float("nan"))
        asym_2 = ((cls_count_total["PPN"] - cls_count_total["PNN"])
                   / max(cls_count_total["PPN"]
                           + cls_count_total["PNN"], 1))
        print(f"--- {regime} N={n_lat} (n_seeds={len(per_seed_results)}) ---")
        print(f"  Stufe A: T2_persistent={m('T2_persistent'):+.4f}+-{sd('T2_persistent'):.4f}  "
              f"T2_random={m('T2_random_baseline'):+.4f}+-{sd('T2_random_baseline'):.4f}  "
              f"specificity={m('T2_specificity'):+.4f}")
        print(f"  Stufe B: K_real={m('K_vortex_real'):.3f} "
              f"null_mean={m('K_vortex_null_mean'):.3f} "
              f"null_std={m('K_vortex_null_std'):.3f} "
              f"z={m('K_vortex_z_score'):+.2f}")
        print(f"  Stufe C: m(PPN)/m(PNN)={ratio_pn:.5f}  "
              f"asym(PPN-PNN)={asym_2:+.4f}  "
              f"freq[PPP,PPN,PNN,NNN]=[{frac['PPP']:.3f},{frac['PPN']:.3f},{frac['PNN']:.3f},{frac['NNN']:.3f}]")
        rows.append({
            "regime": regime, "N": n_lat, "n_seeds": len(per_seed_results),
            "stufe_A": {
                "T2_persistent_mean": m("T2_persistent"),
                "T2_persistent_std": sd("T2_persistent"),
                "T2_random_mean": m("T2_random_baseline"),
                "T2_random_std": sd("T2_random_baseline"),
                "T2_specificity_mean": m("T2_specificity"),
            },
            "stufe_B": {
                "K_vortex_real_mean": m("K_vortex_real"),
                "K_vortex_null_mean": m("K_vortex_null_mean"),
                "K_vortex_null_std": m("K_vortex_null_std"),
                "z_score_mean": m("K_vortex_z_score"),
            },
            "stufe_C": {
                "frequencies": frac,
                "mass_proxy_mean": mass_mean,
                "P2_mass_PPN_over_PNN": ratio_pn,
                "P4_asym_PPN_minus_PNN": asym_2,
                "n_triangles_total": total_cls,
            },
            "per_seed": per_seed_results,
        })
    print()
    print("=" * 80)
    print("Cross-regime synthesis (Stufe A, B, C falsification)")
    print("=" * 80)
    if rows:
        # Stufe A
        spec_arr = np.array([r["stufe_A"]["T2_specificity_mean"] for r in rows])
        pers_arr = np.array([r["stufe_A"]["T2_persistent_mean"] for r in rows])
        rand_arr = np.array([r["stufe_A"]["T2_random_mean"] for r in rows])
        print(f"  Stufe A: T2_persistent cross-regime = "
              f"{pers_arr.mean():+.4f} +/- {pers_arr.std():.4f}")
        print(f"           T2_random cross-regime     = "
              f"{rand_arr.mean():+.4f} +/- {rand_arr.std():.4f}")
        print(f"           Specificity (pers-random)  = "
              f"{spec_arr.mean():+.4f} +/- {spec_arr.std():.4f}")
        sig_a = (spec_arr.mean() / spec_arr.std()
                  if spec_arr.std() > 1e-9 else 0.0)
        print(f"           Significance (mean/std)    = {sig_a:+.2f} sigma")
        # Stufe B
        z_arr = np.array([r["stufe_B"]["z_score_mean"] for r in rows])
        kreal = np.array([r["stufe_B"]["K_vortex_real_mean"] for r in rows])
        knull = np.array([r["stufe_B"]["K_vortex_null_mean"] for r in rows])
        print(f"  Stufe B: K_vortex real mean = "
              f"{kreal.mean():.3f} +/- {kreal.std():.3f}")
        print(f"           K_vortex null mean = "
              f"{knull.mean():.3f} +/- {knull.std():.3f}")
        print(f"           Z-score cross-regime = "
              f"{z_arr.mean():+.2f} +/- {z_arr.std():.2f}")
        # Stufe C
        ratio_arr = np.array([r["stufe_C"]["P2_mass_PPN_over_PNN"]
                                for r in rows])
        asym_arr = np.array([r["stufe_C"]["P4_asym_PPN_minus_PNN"]
                                for r in rows])
        print(f"  Stufe C: m(PPN)/m(PNN) cross-regime = "
              f"{ratio_arr.mean():.5f} +/- {ratio_arr.std():.5f}  "
              f"(SM 0.99862)")
        print(f"           PPN-PNN asym cross-regime  = "
              f"{asym_arr.mean():+.4f} +/- {asym_arr.std():.4f}")
        # Stufe C N-trend
        N_arr = np.array([r["N"] for r in rows], dtype=float)
        print(f"\n           N-trend in m(PPN)/m(PNN):")
        for r in rows:
            print(f"             N={r['N']:>4d}  ratio={r['stufe_C']['P2_mass_PPN_over_PNN']:.5f}  "
                  f"asym={r['stufe_C']['P4_asym_PPN_minus_PNN']:+.4f}  "
                  f"n_tri={r['stufe_C']['n_triangles_total']}")

    bundle = {
        "method": "baryogenesis_hypothesis_tests_StufeA_B_C",
        "rows": rows,
    }
    out = REPO / "outputs" / "audit_baryogenesis_hypothesis_tests.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
