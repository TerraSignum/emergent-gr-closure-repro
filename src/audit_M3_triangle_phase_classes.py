"""Triangle phase-class audit: SM-baryon-analog classification.

User hypothesis 2026-05-02: each persistent triangle has 3 edges, each
carrying a phase-difference of definite sign (+ or -); the SM baryon
content emerges from the sign-signature of the triple:

  (+,+,+)         -> 3-positive-wave configuration  (Delta-baryon analog)
  (+,+,-) and perms -> 2 positive 1 negative       (Proton uud analog)
  (+,-,-) and perms -> 1 positive 2 negative       (Neutron udd analog)
  (-,-,-)         -> 3-negative-wave configuration  (anti-Delta analog)

If we interpret + waves as up-quark analog (charge +2/3) and - waves as
down-quark analog (charge -1/3), then triangle "charge" = sum of these:

  (+,+,+)  ->  +2     (Delta++)
  (+,+,-)  ->  +1     (Proton)
  (+,-,-)  ->   0     (Neutron)   <- via 2/3 + (-1/3) + (-1/3) = 0
  (-,-,-)  ->  -1     (anti-Proton)

For each persistent triangle we measure:
  - sign signature (one of (+++), (++-), (+--), (---) up to permutation)
  - count per class (frequency)
  - mass proxy: mean of (|psi_i|^2 + |psi_j|^2 + |psi_k|^2) on the
    three triangle nodes at end-snapshot
  - binding proxy: mean Xi-triple product (Pi_3) at end-snapshot
  - edge activity: mean Delta-Xi over trajectory on the 3 edges

Predictions to test:
  P1 (BBN ratio): n_proton-class / n_neutron-class ~= 7 at thermal
        freeze-out (kosmological reference). Lattice equivalent unclear,
        but check the ratio of class frequencies.
  P2 (mass inversion): mean mass-proxy of (++-) class < mean mass-proxy
        of (+--) class (Neutron heavier than Proton in SM).
        Ratio in SM: m_p/m_n = 938.272/939.565 = 0.99862.
  P3 (Delta-baryon hierarchy): mean mass-proxy (+++) > (++-) > (+--).
        SM Delta++ mass 1232 MeV > Proton 938 MeV (ratio 1.31).
  P4 (anti-baryon symmetry): n(+++) ~= n(---) and n(++-) ~= n(+--)
        if the lattice is CP-symmetric; deviations indicate baryogenesis.

Edge-sign convention: for triangle (i,j,k) with i<j<k we orient the
loop i->j->k->i and define edge_sign(i,j) = sign(arg(psi_j/psi_i)) where
arg returns principal value in (-pi, pi].

Output: outputs/audit_M3_triangle_phase_classes.json
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

# SM reference values
M_PROTON_MEV = 938.272
M_NEUTRON_MEV = 939.565
M_DELTA_PP_MEV = 1232.0
SM_M_RATIO_PN = M_PROTON_MEV / M_NEUTRON_MEV          # 0.99862
SM_M_RATIO_DELTAP = M_DELTA_PP_MEV / M_PROTON_MEV     # 1.313
BBN_NP_OVER_NN = 7.0  # at freeze-out


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


def classify_triangle_signs(triangles, psi_last):
    """Per triangle (i,j,k) with i<j<k, oriented loop i->j->k->i,
    return tuple (s_ij, s_jk, s_ki) of edge-phase-sign in {+1,0,-1}.
    s_ab = sign of principal-value arg(psi_b/psi_a).
    """
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
    """Map sign-triple to class label by counting positives/negatives.
    Excluding zero-signs (which we treat as discard to avoid mixing).
    """
    n_pos = sum(1 for s in sig if s > 0)
    n_neg = sum(1 for s in sig if s < 0)
    n_zero = sum(1 for s in sig if s == 0)
    if n_zero > 0:
        return "ambiguous"
    if n_pos == 3:
        return "PPP"     # +++
    if n_pos == 2:
        return "PPN"     # ++- (Proton uud analog)
    if n_pos == 1:
        return "PNN"     # +-- (Neutron udd analog)
    return "NNN"         # ---


def main():
    print("=" * 78)
    print("Triangle phase-class audit (SM-baryon-analog classification)")
    print("=" * 78)
    print(f"  SM reference:")
    print(f"    m_p/m_n = {SM_M_RATIO_PN:.5f}  (Neutron 1.3 MeV heavier)")
    print(f"    m_Delta++/m_p = {SM_M_RATIO_DELTAP:.3f}")
    print(f"    BBN n_p/n_n freezeout = {BBN_NP_OVER_NN}")
    print()
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
        n_seeds = min(int(snaps.shape[0]),
                      8 if n_lat <= 100 else 4)
        # Aggregate over seeds: per-class count + mass + binding
        cls_count = defaultdict(int)
        cls_mass_proxy = defaultdict(list)
        cls_pi3 = defaultdict(list)
        cls_edge_act = defaultdict(list)
        for s in range(n_seeds):
            xi_traj = np.asarray(snaps[s], dtype=float).copy()
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
            if len(triangles) > 2000:
                rng = np.random.default_rng(s)
                idx = rng.choice(len(triangles), size=2000, replace=False)
                triangles = [triangles[i] for i in idx]
            if not triangles:
                continue
            sigs = classify_triangle_signs(triangles, psi_last)
            xi_last = xi_traj[-1]
            psi_abs2 = (np.real(psi_last) ** 2
                         + np.imag(psi_last) ** 2)
            mean_act = d_xi.mean(axis=0)  # per-edge mean activity
            for (i, j, k), sig in zip(triangles, sigs):
                lbl = class_label(sig)
                cls_count[lbl] += 1
                m_proxy = (psi_abs2[i] + psi_abs2[j] + psi_abs2[k]) / 3.0
                cls_mass_proxy[lbl].append(m_proxy)
                pi3 = (xi_last[i, j] * xi_last[j, k]
                        * xi_last[k, i])
                cls_pi3[lbl].append(pi3)
                ea = (mean_act[i, j] + mean_act[j, k]
                       + mean_act[k, i]) / 3.0
                cls_edge_act[lbl].append(ea)
        total = sum(cls_count.values())
        if total == 0:
            print(f"  {regime}: no persistent triangles")
            continue
        frac = {lbl: cls_count[lbl] / total
                 for lbl in ["PPP", "PPN", "PNN", "NNN", "ambiguous"]}
        m_mean = {lbl: float(np.mean(cls_mass_proxy[lbl])
                              if cls_mass_proxy[lbl] else np.nan)
                  for lbl in ["PPP", "PPN", "PNN", "NNN"]}
        pi3_mean = {lbl: float(np.mean(cls_pi3[lbl])
                                 if cls_pi3[lbl] else np.nan)
                    for lbl in ["PPP", "PPN", "PNN", "NNN"]}
        ea_mean = {lbl: float(np.mean(cls_edge_act[lbl])
                                if cls_edge_act[lbl] else np.nan)
                   for lbl in ["PPP", "PPN", "PNN", "NNN"]}
        # Predictions
        # P2 mass inversion: m_PPN < m_PNN (Proton lighter than Neutron)?
        # Compute ratio with measurement uncertainty
        m_ratio_pn = (m_mean["PPN"] / m_mean["PNN"]
                      if m_mean["PNN"] > 0 else np.nan)
        # P3 Delta hierarchy: m_PPP / m_PPN
        m_ratio_dp = (m_mean["PPP"] / m_mean["PPN"]
                      if m_mean["PPN"] > 0 else np.nan)
        # P1 BBN: n_PPN / n_PNN
        n_ratio_pn = (cls_count["PPN"] / cls_count["PNN"]
                      if cls_count["PNN"] > 0 else np.nan)
        # P4 baryon-anti-baryon asymmetry
        bary_asym_3 = ((cls_count["PPP"] - cls_count["NNN"])
                        / max(cls_count["PPP"] + cls_count["NNN"], 1))
        bary_asym_2 = ((cls_count["PPN"] - cls_count["PNN"])
                        / max(cls_count["PPN"] + cls_count["PNN"], 1))
        print(f"\n--- {regime} N={n_lat}  (n_triangles_total={total}) ---")
        print(f"  Frequencies:  PPP={frac['PPP']:.3f}  PPN={frac['PPN']:.3f}  "
              f"PNN={frac['PNN']:.3f}  NNN={frac['NNN']:.3f}  "
              f"amb={frac['ambiguous']:.3f}")
        print(f"  Mass proxy <|psi|^2>: "
              f"PPP={m_mean['PPP']:.4f}  PPN={m_mean['PPN']:.4f}  "
              f"PNN={m_mean['PNN']:.4f}  NNN={m_mean['NNN']:.4f}")
        print(f"  Pi3 binding proxy:   "
              f"PPP={pi3_mean['PPP']:.4f}  PPN={pi3_mean['PPN']:.4f}  "
              f"PNN={pi3_mean['PNN']:.4f}  NNN={pi3_mean['NNN']:.4f}")
        print(f"  P1 BBN test n(PPN)/n(PNN) = {n_ratio_pn:.3f}  "
              f"(SM ref ~{BBN_NP_OVER_NN})")
        print(f"  P2 mass inv  m(PPN)/m(PNN) = {m_ratio_pn:.5f}  "
              f"(SM 0.99862, m_p < m_n => ratio < 1)")
        print(f"  P3 Delta     m(PPP)/m(PPN) = {m_ratio_dp:.3f}  "
              f"(SM 1.313)")
        print(f"  P4 asym 3    (PPP-NNN)/sum = {bary_asym_3:+.3f}")
        print(f"  P4 asym 2    (PPN-PNN)/sum = {bary_asym_2:+.3f}")
        rows.append({
            "regime": regime, "N": n_lat, "n_seeds": n_seeds,
            "n_triangles_total": total,
            "frequencies": frac,
            "mass_proxy_mean": m_mean,
            "Pi3_mean": pi3_mean,
            "edge_activity_mean": ea_mean,
            "P1_BBN_n_PPN_over_PNN": n_ratio_pn,
            "P2_mass_PPN_over_PNN": m_ratio_pn,
            "P3_mass_PPP_over_PPN": m_ratio_dp,
            "P4_baryon_asym_3wave": bary_asym_3,
            "P4_baryon_asym_2wave": bary_asym_2,
        })
    print()
    print("=" * 78)
    print("Cross-regime synthesis")
    print("=" * 78)
    if rows:
        m_pn = np.array([r["P2_mass_PPN_over_PNN"] for r in rows
                         if not np.isnan(r["P2_mass_PPN_over_PNN"])])
        n_pn = np.array([r["P1_BBN_n_PPN_over_PNN"] for r in rows
                         if not np.isnan(r["P1_BBN_n_PPN_over_PNN"])])
        m_dp = np.array([r["P3_mass_PPP_over_PPN"] for r in rows
                         if not np.isnan(r["P3_mass_PPP_over_PPN"])])
        ba3 = np.array([r["P4_baryon_asym_3wave"] for r in rows])
        ba2 = np.array([r["P4_baryon_asym_2wave"] for r in rows])
        if m_pn.size:
            print(f"  P2 m(PPN)/m(PNN): mean = {m_pn.mean():.5f} "
                  f"+/- {m_pn.std():.5f}  (SM 0.99862)")
        if n_pn.size:
            print(f"  P1 n(PPN)/n(PNN): mean = {n_pn.mean():.3f} "
                  f"+/- {n_pn.std():.3f}  (SM-BBN 7.0)")
        if m_dp.size:
            print(f"  P3 m(PPP)/m(PPN): mean = {m_dp.mean():.3f} "
                  f"+/- {m_dp.std():.3f}  (SM 1.313)")
        print(f"  P4 baryon-asym 3wave: {ba3.mean():+.4f} "
              f"+/- {ba3.std():.4f}")
        print(f"  P4 baryon-asym 2wave: {ba2.mean():+.4f} "
              f"+/- {ba2.std():.4f}")
    bundle = {
        "method": "M3_triangle_phase_classification_SM_baryon_analog",
        "SM_references": {
            "m_proton_MeV": M_PROTON_MEV,
            "m_neutron_MeV": M_NEUTRON_MEV,
            "m_Delta_pp_MeV": M_DELTA_PP_MEV,
            "m_p_over_m_n": SM_M_RATIO_PN,
            "m_Delta_over_m_p": SM_M_RATIO_DELTAP,
            "BBN_n_p_over_n_n_freezeout": BBN_NP_OVER_NN,
        },
        "rows": rows,
    }
    out = REPO / "outputs" / "audit_M3_triangle_phase_classes.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
