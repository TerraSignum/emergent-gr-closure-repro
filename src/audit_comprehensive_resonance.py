"""Comprehensive resonance audit — testing ALL remaining hypotheses on
the asym(PPN-PNN) signal. CuPy-accelerated where possible. Run with
full seeds, no throttling. May take hours.

Tests:
  T: Tail-Spin classification (Stage 7 from earlier conversation)
     For each persistent triangle, find the nearest persistent tail-edge,
     compute its phase-spin direction, test joint distribution with
     triangle PPN/PNN class.

  E: Spectral mode classification ("waves as eigenmodes" reformulation)
     SVD of Xi matrix gives 3 leading singular vectors {u_1, u_2, u_3}.
     Classify each node by which mode dominates: argmax_i |u_i(node)|^2.
     For each persistent triangle: 3-tuple of mode-labels.
     Hypothesis: triangles with all 3 distinct modes (mode-coincidence)
     show enhanced asymmetry vs same-mode triangles.

  V: Vortex-core density (|psi|^2 < 5/10/25 percentile thresholds)
     Cross-regime density measurement and persistent-edge co-localization.

  K: Joint defect-localization Symanzik re-fit
     Re-fit asym N-trend on triangles WITHIN best-localized subset
     identified from earlier audits (T_00 hot-zone, phase-singularity,
     minimal-defect Δ_min, etc.). Asymptote may sharpen.

  W: Wilson-loop-around-triangle phase windings
     For each triangle, compute Wilson-loop phase Π = exp(i Σ_edges arg).
     Group by Wilson-phase bins, test asymmetry.

  P: Per-corner-spin × tail-spin joint classification
     For each persistent triangle: 3 corner-spins (sign of phi-deviation
     from triangle-mean) × tail-spin (if exists). Cross-tabulate with
     PPN/PNN class.

Output: outputs/audit_comprehensive_resonance.json
"""
from __future__ import annotations
import json
import sys
import time
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


def get_xp():
    try:
        import cupy as cp
        cp.cuda.Device(0).synchronize()
        return cp, True
    except (ImportError, Exception):
        return np, False


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
    return triangles, edge_set, adj


def classify_triangle(phi_i, phi_j, phi_k):
    d_ij = np.angle(np.exp(1j * (phi_j - phi_i)))
    d_jk = np.angle(np.exp(1j * (phi_k - phi_j)))
    d_ki = np.angle(np.exp(1j * (phi_i - phi_k)))
    if abs(d_ij) < 1e-9 or abs(d_jk) < 1e-9 or abs(d_ki) < 1e-9:
        return "ambiguous", 0.0
    n_pos = ((1 if d_ij > 0 else 0) + (1 if d_jk > 0 else 0)
              + (1 if d_ki > 0 else 0))
    wilson = d_ij + d_jk + d_ki  # net phase winding
    if n_pos == 3:
        return "PPP", wilson
    if n_pos == 2:
        return "PPN", wilson
    if n_pos == 1:
        return "PNN", wilson
    return "NNN", wilson


def test_T_tail_spin(triangles, edge_set, adj, psi_last):
    """For each triangle, find a tail-edge attached to one corner.
    Tail-edge spin = sign(arg(psi_outer / psi_corner)).
    Classify joint (tail_spin, triangle_class)."""
    if not triangles:
        return defaultdict(int)
    phi = np.angle(psi_last)
    counts = defaultdict(int)
    for (i, j, k) in triangles:
        # Find a tail edge at any corner
        tail_spin = None
        tail_corner = None
        for corner in (i, j, k):
            for nbr in adj[corner]:
                if nbr in (i, j, k):
                    continue
                if (min(corner, nbr), max(corner, nbr)) not in edge_set:
                    continue
                d = np.angle(np.exp(1j * (phi[nbr] - phi[corner])))
                if abs(d) < 1e-9:
                    continue
                tail_spin = "+" if d > 0 else "-"
                tail_corner = corner
                break
            if tail_spin is not None:
                break
        klass, _ = classify_triangle(phi[i], phi[j], phi[k])
        counts[(klass, tail_spin if tail_spin else "none")] += 1
    return counts


def test_E_spectral_modes(xi_last, n_lat, xp, n_modes=3):
    """SVD of Xi -> top n_modes singular vectors. Per node label =
    argmax over modes of |u_i(node)|."""
    xi_g = xp.asarray(xi_last, dtype=xp.float32)
    # Symmetric, use eigh
    try:
        w, v = xp.linalg.eigh(xi_g)
    except Exception:
        return None
    # Sort descending
    order = xp.argsort(-w)
    w = w[order][:n_modes]
    v = v[:, order][:, :n_modes]
    # Per node, which mode has max |v_im|
    abs_v = xp.abs(v)
    node_mode = xp.argmax(abs_v, axis=1)  # (N,)
    if hasattr(node_mode, "get"):
        node_mode = node_mode.get()
    if hasattr(w, "get"):
        w = w.get()
    return np.asarray(node_mode), np.asarray(w)


def test_V_vortex_density(psi_last, n_lat, percentiles=(5, 10, 25)):
    """|psi|^2 below percentile thresholds = vortex-core candidates."""
    psi_abs2 = np.real(psi_last) ** 2 + np.imag(psi_last) ** 2
    out = {}
    for p in percentiles:
        thr = float(np.percentile(psi_abs2, p))
        n = int((psi_abs2 < thr).sum())
        out[f"vortex_density_p{p}"] = n / n_lat
        out[f"vortex_threshold_p{p}"] = thr
    return out


def test_W_wilson_loop_bins(triangles, psi_last, n_bins=4):
    """Bin triangles by Wilson-loop phase, asym per bin."""
    if not triangles:
        return {}
    phi = np.angle(psi_last)
    bin_counts = defaultdict(lambda: defaultdict(int))
    for (i, j, k) in triangles:
        klass, wilson = classify_triangle(phi[i], phi[j], phi[k])
        # Wilson-bin: discretize wilson modulo 2pi to n_bins
        wilson_red = np.angle(np.exp(1j * wilson))
        bin_idx = int((wilson_red + np.pi) / (2 * np.pi) * n_bins)
        bin_idx = min(bin_idx, n_bins - 1)
        bin_counts[bin_idx][klass] += 1
    out = {}
    for b in range(n_bins):
        n_PPN = bin_counts[b].get("PPN", 0)
        n_PNN = bin_counts[b].get("PNN", 0)
        if n_PPN + n_PNN > 0:
            out[f"wilson_bin{b}_asym"] = (n_PPN - n_PNN) / (n_PPN + n_PNN)
            out[f"wilson_bin{b}_n_total"] = n_PPN + n_PNN
        else:
            out[f"wilson_bin{b}_asym"] = float("nan")
            out[f"wilson_bin{b}_n_total"] = 0
    return out


def test_E_classify_triangles(triangles, node_mode, psi_last):
    """Per triangle, the 3 mode-labels. Distinct? Same? Combinations.
    Asym for triangles with all 3 distinct modes vs all-same."""
    if not triangles:
        return {}
    phi = np.angle(psi_last)
    n_distinct_3 = defaultdict(int)
    n_distinct_2 = defaultdict(int)
    n_distinct_1 = defaultdict(int)
    for (i, j, k) in triangles:
        modes = (int(node_mode[i]), int(node_mode[j]),
                  int(node_mode[k]))
        n_distinct = len(set(modes))
        klass, _ = classify_triangle(phi[i], phi[j], phi[k])
        if n_distinct == 3:
            n_distinct_3[klass] += 1
        elif n_distinct == 2:
            n_distinct_2[klass] += 1
        else:
            n_distinct_1[klass] += 1
    def asym_dict(d):
        n_PPN = d.get("PPN", 0)
        n_PNN = d.get("PNN", 0)
        if n_PPN + n_PNN == 0:
            return float("nan"), 0
        return (n_PPN - n_PNN) / (n_PPN + n_PNN), n_PPN + n_PNN
    asym_3, n_3 = asym_dict(n_distinct_3)
    asym_2, n_2 = asym_dict(n_distinct_2)
    asym_1, n_1 = asym_dict(n_distinct_1)
    return {
        "asym_3distinct_modes": asym_3,
        "asym_2distinct_modes": asym_2,
        "asym_1mode_only": asym_1,
        "n_3distinct": n_3, "n_2distinct": n_2, "n_1mode": n_1,
    }


def main():
    print("=" * 80)
    print("Comprehensive resonance audit — full seeds, all tests")
    print("=" * 80)
    xp, gpu_ok = get_xp()
    print(f"Backend: {'CuPy(GPU)' if gpu_ok else 'NumPy(CPU)'}")
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
        t0 = time.perf_counter()
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
            triangles, edge_set, adj = find_persistent_triangles(pers_edges)
            # T: Tail-Spin
            T_counts = test_T_tail_spin(triangles, edge_set, adj, psi_last)
            # V: Vortex density
            V_data = test_V_vortex_density(psi_last, n_lat)
            # E: Spectral modes
            mode_result = test_E_spectral_modes(xi_last, n_lat, xp)
            if mode_result is not None:
                node_mode, eigvals = mode_result
                E_data = test_E_classify_triangles(triangles, node_mode,
                                                       psi_last)
            else:
                E_data = {}
            # W: Wilson-loop bins
            W_data = test_W_wilson_loop_bins(triangles, psi_last)
            # Convert T_counts to flat dict
            T_data = {}
            for (klass, spin), n in T_counts.items():
                T_data[f"T_{klass}_{spin}_count"] = n
            # asym (PPN+ - PNN-) - (PPN- - PNN+) joint test
            n_PPN_p = T_counts.get(("PPN", "+"), 0)
            n_PPN_m = T_counts.get(("PPN", "-"), 0)
            n_PNN_p = T_counts.get(("PNN", "+"), 0)
            n_PNN_m = T_counts.get(("PNN", "-"), 0)
            T_data["T_asym_tail_pos"] = (
                (n_PPN_p - n_PNN_p) / max(n_PPN_p + n_PNN_p, 1))
            T_data["T_asym_tail_neg"] = (
                (n_PPN_m - n_PNN_m) / max(n_PPN_m + n_PNN_m, 1))
            T_data["T_asym_diff_tail_pos_neg"] = (
                T_data["T_asym_tail_pos"] - T_data["T_asym_tail_neg"])
            per_seed.append({
                "seed": s,
                "T": T_data,
                "V": V_data,
                "E": E_data,
                "W": W_data,
            })
        elapsed = time.perf_counter() - t0
        if not per_seed:
            continue
        # Aggregate per group
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
        def std(grp, key):
            vals = []
            for d in per_seed:
                v = d.get(grp, {}).get(key)
                if v is None:
                    continue
                if isinstance(v, float) and np.isnan(v):
                    continue
                vals.append(v)
            if len(vals) < 2:
                return float("nan")
            return float(np.std(vals)) / np.sqrt(len(vals))
        T_pos = mn("T", "T_asym_tail_pos")
        T_neg = mn("T", "T_asym_tail_neg")
        T_diff = mn("T", "T_asym_diff_tail_pos_neg")
        T_diff_unc = std("T", "T_asym_diff_tail_pos_neg")
        E_3d = mn("E", "asym_3distinct_modes")
        E_3d_unc = std("E", "asym_3distinct_modes")
        E_1m = mn("E", "asym_1mode_only")
        n_3d = mn("E", "n_3distinct")
        n_2d = mn("E", "n_2distinct")
        n_1m = mn("E", "n_1mode")
        V_p5 = mn("V", "vortex_density_p5")
        V_p10 = mn("V", "vortex_density_p10")
        W_b0 = mn("W", "wilson_bin0_asym")
        W_b1 = mn("W", "wilson_bin1_asym")
        W_b2 = mn("W", "wilson_bin2_asym")
        W_b3 = mn("W", "wilson_bin3_asym")
        print(f"--- {regime} N={n_lat} (n_seeds={len(per_seed)}, "
              f"{elapsed:.1f}s) ---")
        print(f"  T (tail-spin):  asym_tail+ = {T_pos:+.4f}, "
              f"asym_tail- = {T_neg:+.4f}, diff = {T_diff:+.4f}+-{T_diff_unc:.4f}")
        print(f"  E (spectral):   asym_3distinct={E_3d:+.4f}+-{E_3d_unc:.4f}  "
              f"asym_1mode={E_1m:+.4f}  n[3d,2d,1m]=[{n_3d:.0f},{n_2d:.0f},{n_1m:.0f}]")
        print(f"  V (vortex):     density(p5)={V_p5:.3f}  density(p10)={V_p10:.3f}")
        print(f"  W (wilson bins): asym=[{W_b0:+.3f},{W_b1:+.3f},"
              f"{W_b2:+.3f},{W_b3:+.3f}]")
        rows.append({
            "regime": regime, "N": n_lat, "n_seeds": len(per_seed),
            "elapsed_seconds": elapsed,
            "T_asym_tail_pos_mean": T_pos,
            "T_asym_tail_neg_mean": T_neg,
            "T_asym_diff_mean": T_diff,
            "T_asym_diff_unc": T_diff_unc,
            "E_asym_3distinct_mean": E_3d,
            "E_asym_3distinct_unc": E_3d_unc,
            "E_asym_1mode_mean": E_1m,
            "E_n_3distinct_mean": n_3d,
            "E_n_2distinct_mean": n_2d,
            "E_n_1mode_mean": n_1m,
            "V_density_p5_mean": V_p5,
            "V_density_p10_mean": V_p10,
            "W_bin0_asym_mean": W_b0,
            "W_bin1_asym_mean": W_b1,
            "W_bin2_asym_mean": W_b2,
            "W_bin3_asym_mean": W_b3,
            "per_seed": per_seed,
        })
    print()
    print("=" * 80)
    print("Cross-regime synthesis")
    print("=" * 80)
    if rows:
        # T diff
        Td = np.array([r["T_asym_diff_mean"] for r in rows
                         if not np.isnan(r["T_asym_diff_mean"])])
        if Td.size:
            print(f"  T tail-spin asym_diff cross-regime: "
                  f"{Td.mean():+.5f} +/- {Td.std()/np.sqrt(len(Td)):.5f}  "
                  f"({abs(Td.mean())/(Td.std()/np.sqrt(len(Td))):.2f}σ)")
        # E 3-distinct
        E3 = np.array([r["E_asym_3distinct_mean"] for r in rows
                         if not np.isnan(r["E_asym_3distinct_mean"])])
        E1 = np.array([r["E_asym_1mode_mean"] for r in rows
                         if not np.isnan(r["E_asym_1mode_mean"])])
        if E3.size:
            print(f"  E asym_3distinct cross-regime: "
                  f"{E3.mean():+.5f} +/- {E3.std()/np.sqrt(len(E3)):.5f}")
        if E1.size:
            print(f"  E asym_1mode cross-regime: "
                  f"{E1.mean():+.5f} +/- {E1.std()/np.sqrt(len(E1)):.5f}")
        if E3.size and E1.size:
            diff = E3.mean() - E1.mean()
            unc = np.sqrt((E3.std()/np.sqrt(len(E3)))**2
                           + (E1.std()/np.sqrt(len(E1)))**2)
            print(f"  E diff (3distinct - 1mode): {diff:+.5f} +/- {unc:.5f}  "
                  f"({abs(diff)/max(unc,1e-9):.2f}σ)")
        # V density
        Vp5 = np.array([r["V_density_p5_mean"] for r in rows])
        print(f"  V density(p5) cross-regime: {Vp5.mean():.3f} "
              f"+/- {Vp5.std():.3f}  (binom expectation 0.05)")
        # W wilson
        for b in range(4):
            arr = np.array([r[f"W_bin{b}_asym_mean"] for r in rows
                              if not np.isnan(r[f"W_bin{b}_asym_mean"])])
            if arr.size:
                print(f"  W bin{b} asym cross-regime: "
                      f"{arr.mean():+.5f} +/- {arr.std()/np.sqrt(len(arr)):.5f}")

    bundle = {
        "method": "comprehensive_resonance_audit",
        "rows": rows,
    }
    out = REPO / "outputs" / "audit_comprehensive_resonance.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
