"""L6-L9: deeper localization audits beyond bulk-edge-phase-statistics.

L6 K-Q field third-moment skewness/kurtosis:
  K_vortex turned out to be ~ 1+2/pi (Bernoulli-random magnitude).
  But K and Q themselves are real fields with their own statistics.
  Test: skewness, kurtosis, third moment of (K-Q) per regime.
  Cross-regime: any reproducible non-zero skewness > sigma?

L7 T_00 hot-spot localization of persistent triangles:
  Per regime, identify T_00 top-decile nodes (heavy-tail matter).
  Per persistent triangle, count how many of 3 nodes are in top-decile.
  Categorize triangles into: 0/1/2/3 nodes in T_00 hot zone.
  Test: do INSIDE-T_00 triangles have different mass-proxy, edge sign
  asym, or Pi3 binding than OUTSIDE triangles?

L8 Direction-asymmetric edge activity:
  For each persistent edge (i,j): activity Delta_xi over the trajectory.
  Compare activity distribution in "forward" direction (i.e. paired
  with phase-gradient phi_j > phi_i) vs "backward" (phi_j < phi_i).
  Is the activity systematically different along forward vs backward
  edges? This would indicate a direction-asymmetric arrow in the
  field dynamics (Sakharov-2 carrier).

L9 Hopf density x triangle co-localization:
  Compute discrete Hopf-density at each node from psi-phase winding
  on a small spatial neighborhood. Per persistent triangle: mean Hopf
  density over its 3 nodes. Test: do high-Hopf triangles differ from
  low-Hopf triangles in mass, asym-class frequency, or edge activity?

Output: outputs/audit_L6_L9_field_localization.json
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
    return triangles, edge_set


def t00_proxy(xi_last, k_node, q_node, n_lat):
    """Cheap T_00 proxy: per-node energy density via sum of incident
    Xi-edge tightness * (k+q) field. Heavy-tail nodes are the
    matter-localized hot spots seen in M3-supnorm-diagnostics."""
    np.fill_diagonal(xi_last, 0.0)
    e_density = (-np.log(np.maximum(xi_last, 1e-9))).sum(axis=1) / n_lat
    return e_density * (np.abs(k_node) + np.abs(q_node))


def hopf_density(psi_last, n_lat, win=5):
    """Discrete proxy for Hopf-density: per node i, the local total
    phase winding over the win-sized neighborhood [i-win, i+win]
    (periodic), divided by 2pi."""
    phi = np.angle(psi_last)
    out = np.zeros(n_lat)
    for i in range(n_lat):
        idx = np.arange(i - win, i + win) % n_lat
        ph = phi[idx]
        # Differential phases
        dph = np.diff(ph, append=ph[0])
        dph = np.angle(np.exp(1j * dph))
        out[i] = float(dph.sum() / (2.0 * np.pi))
    return out


def l6_kq_field_moments(k_field, q_field):
    """L6: skewness / kurtosis / third moment of (K-Q) field."""
    if k_field.ndim == 2:
        k_node = k_field.mean(axis=1)
        q_node = q_field.mean(axis=1)
    else:
        k_node, q_node = k_field, q_field
    delta = k_node - q_node
    mu = float(delta.mean())
    sd = float(delta.std())
    if sd < 1e-12:
        return {
            "kq_mean": mu, "kq_std": sd,
            "kq_skewness": float("nan"),
            "kq_kurtosis": float("nan"),
            "kq_third_moment": 0.0,
        }
    delta_n = (delta - mu) / sd
    skew = float(np.mean(delta_n ** 3))
    kurt = float(np.mean(delta_n ** 4) - 3.0)
    third = float(np.mean((delta - mu) ** 3))
    return {
        "kq_mean": mu, "kq_std": sd,
        "kq_skewness": skew, "kq_kurtosis": kurt,
        "kq_third_moment": third,
    }


def l7_t00_localization(triangles, t00, psi_last, xi_last, n_lat,
                          n_max=2000):
    if not triangles:
        return {f"frac_{k}_inside": float("nan") for k in (0, 1, 2, 3)}
    psi_abs2 = np.real(psi_last) ** 2 + np.imag(psi_last) ** 2
    threshold = float(np.percentile(t00, 90))
    in_zone = t00 > threshold
    sample = (triangles
              if len(triangles) <= n_max
              else [triangles[i] for i in np.random.default_rng(0).choice(
                    len(triangles), n_max, replace=False)])
    counts = defaultdict(int)
    mass_per_count = defaultdict(list)
    pi3_per_count = defaultdict(list)
    asym_per_count = defaultdict(list)
    phi = np.angle(psi_last)
    for i, j, k in sample:
        n_in = int(in_zone[i]) + int(in_zone[j]) + int(in_zone[k])
        counts[n_in] += 1
        m_proxy = (psi_abs2[i] + psi_abs2[j] + psi_abs2[k]) / 3.0
        pi3 = xi_last[i, j] * xi_last[j, k] * xi_last[k, i]
        d_ij = np.angle(np.exp(1j * (phi[j] - phi[i])))
        d_jk = np.angle(np.exp(1j * (phi[k] - phi[j])))
        d_ki = np.angle(np.exp(1j * (phi[i] - phi[k])))
        signs = (int(np.sign(d_ij)), int(np.sign(d_jk)),
                  int(np.sign(d_ki)))
        n_pos = sum(1 for s_ in signs if s_ > 0)
        # Asym contribution: +1 if PPN, -1 if PNN, 0 else
        if n_pos == 2:
            asym_per_count[n_in].append(+1)
        elif n_pos == 1:
            asym_per_count[n_in].append(-1)
        mass_per_count[n_in].append(m_proxy)
        pi3_per_count[n_in].append(pi3)
    total = sum(counts.values())
    out = {}
    for c in (0, 1, 2, 3):
        out[f"frac_{c}_inside"] = (counts[c] / max(total, 1)
                                     if total > 0 else 0.0)
        out[f"mass_{c}_inside"] = (float(np.mean(mass_per_count[c]))
                                     if mass_per_count[c]
                                     else float("nan"))
        out[f"Pi3_{c}_inside"] = (float(np.mean(pi3_per_count[c]))
                                    if pi3_per_count[c]
                                    else float("nan"))
        # asym (PPN-PNN)/(PPN+PNN) per count category
        a_arr = asym_per_count[c]
        if a_arr:
            a_pos = sum(1 for x in a_arr if x > 0)
            a_neg = sum(1 for x in a_arr if x < 0)
            out[f"asym_{c}_inside"] = (a_pos - a_neg) / max(a_pos + a_neg, 1)
        else:
            out[f"asym_{c}_inside"] = float("nan")
    return out


def l8_direction_asymmetric_activity(pers_edges, xi_traj, psi_last,
                                       n_lat):
    """Per persistent edge (i,j) with i<j and phase phi_i, phi_j:
    forward = (phi_j > phi_i in modular sense), backward = otherwise.
    Activity = |Delta xi| trajectory mean. Compare forward vs backward."""
    if pers_edges.size == 0:
        return {
            "n_forward": 0, "n_backward": 0,
            "activity_forward": float("nan"),
            "activity_backward": float("nan"),
            "activity_ratio_fwd_bwd": float("nan"),
        }
    phi = np.angle(psi_last)
    d_xi = np.abs(np.diff(xi_traj, axis=0))
    fwd_act = []
    bwd_act = []
    for ie in range(pers_edges.shape[0]):
        i, j = pers_edges[ie]
        i, j = int(i), int(j)
        d_phi = np.angle(np.exp(1j * (phi[j] - phi[i])))
        act = float(d_xi[:, i, j].mean())
        if d_phi > 0:
            fwd_act.append(act)
        elif d_phi < 0:
            bwd_act.append(act)
    if not fwd_act or not bwd_act:
        return {
            "n_forward": len(fwd_act), "n_backward": len(bwd_act),
            "activity_forward": float("nan"),
            "activity_backward": float("nan"),
            "activity_ratio_fwd_bwd": float("nan"),
        }
    af = float(np.mean(fwd_act))
    ab = float(np.mean(bwd_act))
    return {
        "n_forward": len(fwd_act), "n_backward": len(bwd_act),
        "activity_forward": af,
        "activity_backward": ab,
        "activity_ratio_fwd_bwd": af / ab if ab > 0 else float("nan"),
    }


def l9_hopf_triangle_colocalization(triangles, hopf, psi_last, xi_last,
                                       n_max=2000):
    if not triangles:
        return {
            "hopf_corr_high_low_mass_ratio": float("nan"),
            "n_high_hopf": 0, "n_low_hopf": 0,
        }
    psi_abs2 = np.real(psi_last) ** 2 + np.imag(psi_last) ** 2
    sample = (triangles
              if len(triangles) <= n_max
              else [triangles[i] for i in np.random.default_rng(0).choice(
                    len(triangles), n_max, replace=False)])
    tri_hopf = []
    tri_mass = []
    tri_pi3 = []
    for i, j, k in sample:
        h = (hopf[i] + hopf[j] + hopf[k]) / 3.0
        m_proxy = (psi_abs2[i] + psi_abs2[j] + psi_abs2[k]) / 3.0
        pi3 = xi_last[i, j] * xi_last[j, k] * xi_last[k, i]
        tri_hopf.append(h)
        tri_mass.append(m_proxy)
        tri_pi3.append(pi3)
    tri_hopf = np.array(tri_hopf)
    tri_mass = np.array(tri_mass)
    tri_pi3 = np.array(tri_pi3)
    threshold = float(np.percentile(np.abs(tri_hopf), 80))
    high = np.abs(tri_hopf) > threshold
    low = ~high
    n_high = int(high.sum())
    n_low = int(low.sum())
    if n_high == 0 or n_low == 0:
        return {
            "hopf_corr_high_low_mass_ratio": float("nan"),
            "n_high_hopf": n_high, "n_low_hopf": n_low,
        }
    return {
        "n_high_hopf": n_high, "n_low_hopf": n_low,
        "mass_high_hopf": float(tri_mass[high].mean()),
        "mass_low_hopf": float(tri_mass[low].mean()),
        "Pi3_high_hopf": float(tri_pi3[high].mean()),
        "Pi3_low_hopf": float(tri_pi3[low].mean()),
        "hopf_corr_high_low_mass_ratio": float(tri_mass[high].mean()
                                                  / max(tri_mass[low].mean(), 1e-12)),
    }


def main():
    print("=" * 80)
    print("L6-L9 deeper-localization audits (full-seed)")
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
        k_snaps = z["k_snapshots"]
        q_snaps = z["q_snapshots"]
        n_seeds = int(snaps.shape[0])
        per_seed = []
        for s in range(n_seeds):
            xi_traj = np.asarray(snaps[s], dtype=float).copy()
            psi_last = (psi_r[s, -1].astype(float)
                        + 1j * psi_i[s, -1].astype(float))
            k_last = np.asarray(k_snaps[s, -1], dtype=float)
            q_last = np.asarray(q_snaps[s, -1], dtype=float)
            d_xi = np.abs(np.diff(xi_traj, axis=0))
            offdiag = ~np.eye(n_lat, dtype=bool)
            d_off = d_xi[:, offdiag]
            v_med = (float(np.median(d_off[d_off > 0]))
                      if (d_off > 0).any() else 1e-6)
            c_info = 2 * v_med
            persistent_mask_off = (d_off > c_info).mean(axis=0) > 0.5
            ij_idx = np.argwhere(offdiag)
            pers_edges = ij_idx[persistent_mask_off]
            triangles, edge_set = find_persistent_triangles(pers_edges)
            xi_last = xi_traj[-1].copy()
            # L6 K-Q moments
            l6 = l6_kq_field_moments(k_last, q_last)
            # T_00 proxy for L7
            if k_last.ndim == 2:
                k_node = k_last.mean(axis=1)
                q_node = q_last.mean(axis=1)
            else:
                k_node, q_node = k_last, q_last
            t00 = t00_proxy(xi_last.copy(), k_node, q_node, n_lat)
            # L7 T_00 localization
            l7 = l7_t00_localization(triangles, t00, psi_last,
                                          xi_traj[-1], n_lat)
            # L8 direction-asymmetric activity
            l8 = l8_direction_asymmetric_activity(pers_edges, xi_traj,
                                                       psi_last, n_lat)
            # L9 Hopf density correlation
            hopf = hopf_density(psi_last, n_lat, win=max(3, n_lat // 30))
            l9 = l9_hopf_triangle_colocalization(triangles, hopf,
                                                       psi_last, xi_traj[-1])
            per_seed.append({"seed": s,
                             "L6": l6, "L7": l7, "L8": l8, "L9": l9})
        if not per_seed:
            continue
        def mn(grp, key):
            vals = []
            for d in per_seed:
                v = d.get(grp, {}).get(key)
                if v is None:
                    continue
                if isinstance(v, float) and np.isnan(v):
                    continue
                vals.append(v)
            return float(np.mean(vals)) if vals else float("nan")
        def sd(grp, key):
            vals = []
            for d in per_seed:
                v = d.get(grp, {}).get(key)
                if v is None:
                    continue
                if isinstance(v, float) and np.isnan(v):
                    continue
                vals.append(v)
            return float(np.std(vals)) if vals else float("nan")
        l6_skew = mn("L6", "kq_skewness")
        l6_skew_unc = sd("L6", "kq_skewness") / np.sqrt(len(per_seed))
        l6_kurt = mn("L6", "kq_kurtosis")
        l6_third = mn("L6", "kq_third_moment")
        l7_asym_3 = mn("L7", "asym_3_inside")
        l7_asym_0 = mn("L7", "asym_0_inside")
        l7_mass_3 = mn("L7", "mass_3_inside")
        l7_mass_0 = mn("L7", "mass_0_inside")
        l7_frac_3 = mn("L7", "frac_3_inside")
        l7_frac_0 = mn("L7", "frac_0_inside")
        l8_ratio = mn("L8", "activity_ratio_fwd_bwd")
        l8_ratio_unc = sd("L8", "activity_ratio_fwd_bwd") / np.sqrt(len(per_seed))
        l9_ratio = mn("L9", "hopf_corr_high_low_mass_ratio")
        l9_ratio_unc = sd("L9", "hopf_corr_high_low_mass_ratio") / np.sqrt(len(per_seed))
        print(f"--- {regime} N={n_lat} (n_seeds={len(per_seed)}) ---")
        print(f"  L6 K-Q field: skew={l6_skew:+.4f} +/- {l6_skew_unc:.4f}  "
              f"kurt={l6_kurt:+.4f}  third={l6_third:+.5f}")
        print(f"  L7 T00 localization: frac_inside_all3={l7_frac_3:.3f}  "
              f"frac_outside_all={l7_frac_0:.3f}  "
              f"asym_inside3={l7_asym_3:+.4f}  asym_outside={l7_asym_0:+.4f}")
        print(f"  L8 direction asym: act_fwd/act_bwd = {l8_ratio:.5f} +/- {l8_ratio_unc:.5f}")
        print(f"  L9 Hopf-corr: mass_high/mass_low = {l9_ratio:.5f} +/- {l9_ratio_unc:.5f}")
        rows.append({
            "regime": regime, "N": n_lat,
            "n_seeds": len(per_seed),
            "L6_kq_skewness_mean": l6_skew,
            "L6_kq_skewness_unc": float(l6_skew_unc),
            "L6_kq_kurtosis_mean": l6_kurt,
            "L6_kq_third_moment_mean": l6_third,
            "L7_asym_inside_3": l7_asym_3,
            "L7_asym_outside_all": l7_asym_0,
            "L7_mass_inside_3": l7_mass_3,
            "L7_mass_outside_all": l7_mass_0,
            "L7_frac_inside_3": l7_frac_3,
            "L7_frac_outside_all": l7_frac_0,
            "L8_activity_ratio_fwd_bwd": l8_ratio,
            "L8_activity_ratio_unc": float(l8_ratio_unc),
            "L9_mass_ratio_high_low_hopf": l9_ratio,
            "L9_mass_ratio_high_low_unc": float(l9_ratio_unc),
            "per_seed": per_seed,
        })
    print()
    print("=" * 80)
    print("Cross-regime synthesis L6-L9 (full-seeds)")
    print("=" * 80)
    if rows:
        for tag, key, ref in [
            ("L6 K-Q skewness (=0: symmetric)",
             "L6_kq_skewness_mean", 0.0),
            ("L6 K-Q kurtosis (=0: gaussian)",
             "L6_kq_kurtosis_mean", 0.0),
            ("L7 asym INSIDE-T00-3 (PPN-PNN frac)",
             "L7_asym_inside_3", 0.0),
            ("L7 asym OUTSIDE-T00-0 (PPN-PNN frac)",
             "L7_asym_outside_all", 0.0),
            ("L7 mass_inside3/mass_outside0 (=1: no localization)",
             None, None),  # computed below
            ("L8 activity_ratio fwd/bwd (=1: symmetric)",
             "L8_activity_ratio_fwd_bwd", 1.0),
            ("L9 mass_high/low_Hopf (=1: no Hopf coupling)",
             "L9_mass_ratio_high_low_hopf", 1.0),
        ]:
            if key is None:
                # compute mass_inside_3 / mass_outside_0 cross-regime
                m3 = np.array([r["L7_mass_inside_3"] for r in rows
                                if not np.isnan(r["L7_mass_inside_3"])])
                m0 = np.array([r["L7_mass_outside_all"] for r in rows
                                if not np.isnan(r["L7_mass_outside_all"])])
                if m3.size and m0.size:
                    rat = m3 / m0
                    print(f"  {tag}: {rat.mean():.5f} +/- {rat.std():.5f}")
                continue
            arr = np.array([r[key] for r in rows
                             if not np.isnan(r[key])])
            if arr.size:
                ref_val = ref if ref is not None else 0.0
                sigma = (abs(arr.mean() - ref_val) / arr.std()
                          if arr.std() > 1e-12 else float("nan"))
                print(f"  {tag}: {arr.mean():+.5f} +/- {arr.std():.5f}"
                      f"  (sigma vs ref {ref_val}: {sigma:.2f})")

    bundle = {
        "method": "L6_L9_field_localization_audit",
        "rows": rows,
    }
    out = REPO / "outputs" / "audit_L6_L9_field_localization.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
