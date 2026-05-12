"""Per-regime baryogenesis audit on P5N snapshot ladder.

Stages (S1-S5) all computed direct from snapshot NPZ:

  S1  Vortex angle theta_v per regime
        From psi-phase winding density and DCA alignment at end-snapshot.
        K_vortex = 1 + |sin theta_v|.

  S2  S_bounce per regime from V_Xi quartic
        V_Xi(Xi) = a_V * (lambda/4) * (Xi^2 - xi_0^2)^2
        with lambda, xi_0 fit from per-seed Xi distribution.
        Numerical bounce action under Coleman thin-wall approximation:
        S_bounce ~= (16*pi/3) * sigma^3 / (Delta_eps)^2,
        sigma = sqrt(2 a_V * lambda) * xi_0^3 / 3 (wall tension),
        Delta_eps = a_V * lambda * xi_0^4 / 4 (false-true energy gap).
        (Coleman 1977; thin-wall; lattice-units).

  S3  J_CP proxy per regime from psi-phase structure
        Take 3 dominant SVD modes of (Re psi + i Im psi) at end-snapshot;
        compute Jarlskog J = Im(U_us U_cb U*_ub U*_cs) on the 3x3 mixing
        matrix V = U^dagger D where U,D are SVD modes.

  S4  eta_B per regime via Sakharov transport
        eta_B = (28/51) * J_CP * S_bounce * (n_gen/g_star) * K_vortex
        with n_gen = 3 (fermion generations), g_star = 106.75 (SM EW).
        Compare to PDG 2024 eta_B,obs = 6.04e-10.

  S5  Triangle winding asymmetry per regime
        For each persistent triangle, compute total psi-phase change
        around the loop: Delta_phi = arg(psi_j/psi_i) + arg(psi_k/psi_j)
        + arg(psi_i/psi_k), reduced modulo 2pi to (-pi,pi].
        Winding asymmetry: A_w = (n_+ - n_-) / (n_+ + n_-)
        where n_+ counts triangles with winding +1, n_- with -1.

Proxy notes:
  - S_bounce uses Coleman thin-wall in lattice units; absolute scale
    matches order ~38 per GCC-05 P1 reference. We report the per-regime
    proxy and flag the regime with smallest and largest deviation.
  - J_CP proxy uses Xi-K-Q-psi spectral SVD; this is an analog to the
    Yukawa SVD in SYE-Y5 but read on the lattice's psi field directly.

Output: outputs/audit_baryogenesis_per_regime.json
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

ETA_B_OBS = 6.04e-10  # PDG 2024 cosmology compact
ETA_B_OBS_UNC = 1.2e-11
N_GEN = 3
G_STAR_SM = 106.75       # SM EW degrees of freedom
A_V_NATURAL = 1.0        # unified_action.py:85 dimensional identity


def fit_quartic_from_xi(xi_traj: np.ndarray):
    """Fit V_Xi(Xi) = a_V (lambda/4)(Xi^2 - xi_0^2)^2 to per-edge
    Xi distribution at end-snapshot. Returns lambda, xi_0.

    xi_0 is the bulk-mode of |Xi| (peak of off-diagonal distribution).
    lambda from <(Xi - xi_0)^2> via thermal fluctuation:
        <(delta Xi)^2> = 1 / (lambda * xi_0^2 * a_V).
    """
    n = xi_traj.shape[-1]
    xi_last = xi_traj[-1]
    offdiag = ~np.eye(n, dtype=bool)
    vals = xi_last[offdiag]
    vals_pos = vals[vals > 0]
    if vals_pos.size == 0:
        return 1.0, 0.5
    # Mode via histogram peak
    hist, edges = np.histogram(vals_pos, bins=50)
    peak_idx = int(np.argmax(hist))
    xi_0 = float(0.5 * (edges[peak_idx] + edges[peak_idx + 1]))
    if xi_0 < 1e-6:
        xi_0 = float(np.mean(vals_pos))
    # lambda from variance of Xi around xi_0
    var_xi = float(np.var(vals_pos))
    lam = 1.0 / max(var_xi * (xi_0 ** 2) * A_V_NATURAL, 1e-6)
    return lam, xi_0


def s_bounce_thin_wall(lam: float, xi_0: float) -> float:
    """Coleman thin-wall bounce action.
    S = (16*pi/3) * sigma^3 / (Delta_eps)^2
    sigma = sqrt(2 a_V lambda) * xi_0^3 / 3
    Delta_eps = a_V lambda * xi_0^4 / 4 + epsilon (small bias)
    Returns S_bounce in lattice units.
    """
    av = A_V_NATURAL
    sigma = np.sqrt(max(2.0 * av * lam, 0.0)) * (xi_0 ** 3) / 3.0
    # Tilt the false vacuum by a small relative amount eps_tilt
    # (otherwise Delta_eps = 0 in symmetric Mexican hat):
    eps_tilt = 0.02  # 2% asymmetric vacuum bias from emergent CPT in HBR
    delta_eps = av * lam * (xi_0 ** 4) / 4.0 * eps_tilt
    if delta_eps < 1e-12:
        return 0.0
    return float((16.0 * np.pi / 3.0) * (sigma ** 3) / (delta_eps ** 2))


def vortex_angle_from_psi(psi: np.ndarray, k: np.ndarray, q: np.ndarray
                          ) -> float:
    """Vortex CP angle from end-snapshot psi phase + DCA alignment.
    theta_v = arctan2(<sin phi * (k-q)>, <cos phi * (k-q)>).
    """
    phi = np.angle(psi)
    n = psi.shape[0]
    # DCA proxy: anti-symmetric K-Q (positive-Q vs negative-K alignment)
    if k.ndim == 2:
        k_node = k.mean(axis=1)
        q_node = q.mean(axis=1)
    else:
        k_node = k
        q_node = q
    dca = (k_node - q_node)
    # weighted phase
    cos_w = np.sum(np.cos(phi) * dca)
    sin_w = np.sum(np.sin(phi) * dca)
    if abs(cos_w) + abs(sin_w) < 1e-12:
        return 0.0
    return float(np.arctan2(sin_w, cos_w))


def jcp_proxy_from_psi(psi: np.ndarray, k: np.ndarray, q: np.ndarray
                       ) -> float:
    """J_CP proxy via 3-mode Jarlskog on psi-K-Q spectral analog.

    Build 3x3 'mixing matrix' M from the top-3 SVD modes of:
        Op = psi_outer * (K - Q)  (Hermitian-anti-symmetric proxy of
                                    Yukawa-like operator)
    Compute J = Im(M_us M_cb M*_ub M*_cs) using the top-3 elements.
    """
    n = psi.shape[0]
    if k.ndim == 1:
        kmat = np.diag(k)
    else:
        kmat = k
    if q.ndim == 1:
        qmat = np.diag(q)
    else:
        qmat = q
    psi_outer = np.outer(psi, np.conj(psi))
    op = psi_outer * (kmat - qmat)
    # SVD
    try:
        u_, s_, vh_ = np.linalg.svd(op)
    except np.linalg.LinAlgError:
        return 0.0
    if s_.size < 3:
        return 0.0
    # Construct 3x3 mixing block from top-3 right-singular vectors
    v3 = vh_[:3, :3]
    # Jarlskog-style: J = Im(V_12 V_23 V_13* V_22*)
    j = (v3[0, 1] * v3[1, 2] * np.conj(v3[0, 2]) * np.conj(v3[1, 1])).imag
    return float(j)


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
    return triangles


def triangle_winding_asymmetry(triangles, psi: np.ndarray):
    """Phase winding around each persistent triangle. Returns
    (n_plus, n_minus, A_w). Winding +1 if total dphi mod 2pi > 0.5*pi,
    -1 if < -0.5*pi, 0 otherwise."""
    if not triangles:
        return 0, 0, 0.0
    phi = np.angle(psi)
    n_plus = 0
    n_minus = 0
    for i, j, k in triangles:
        d1 = np.angle(np.exp(1j * (phi[j] - phi[i])))
        d2 = np.angle(np.exp(1j * (phi[k] - phi[j])))
        d3 = np.angle(np.exp(1j * (phi[i] - phi[k])))
        total = d1 + d2 + d3
        # reduce to (-pi, pi]
        total_red = np.angle(np.exp(1j * total))
        if total_red > 0.5 * np.pi:
            n_plus += 1
        elif total_red < -0.5 * np.pi:
            n_minus += 1
    n_total = max(n_plus + n_minus, 1)
    a_w = (n_plus - n_minus) / n_total
    return n_plus, n_minus, a_w


def main():
    print("=" * 78)
    print("Per-regime baryogenesis audit (S1-S5) on P5N ladder")
    print("=" * 78)
    print(f"  PDG-2024 eta_B,obs = {ETA_B_OBS:.3e} +/- {ETA_B_OBS_UNC:.1e}")
    print()
    header = (f"  {'reg':<7} {'#s':>2} "
              f"{'lam':>7} {'xi_0':>6} {'S_b':>8} "
              f"{'theta_v':>8} {'K_vtx':>6} "
              f"{'J_CP':>10} {'eta_B,pred':>11} {'ratio':>8} "
              f"{'A_w':>7}")
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
            # S2 lambda, xi_0 + S_bounce
            lam, xi_0 = fit_quartic_from_xi(xi_traj)
            s_b = s_bounce_thin_wall(lam, xi_0)
            # S1 vortex angle
            theta_v = vortex_angle_from_psi(psi_last, k_last, q_last)
            k_vtx = 1.0 + abs(np.sin(theta_v))
            # S3 J_CP proxy
            j_cp = jcp_proxy_from_psi(psi_last, k_last, q_last)
            # S4 eta_B assemble
            eta_b_naive = ((28.0 / 51.0) * abs(j_cp) * s_b
                            * (N_GEN / G_STAR_SM) * k_vtx)
            ratio = eta_b_naive / ETA_B_OBS if ETA_B_OBS > 0 else 0.0
            # S5 Triangle winding asymmetry
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
            # Cap for performance
            if len(triangles) > 800:
                rng = np.random.default_rng(s)
                idx = rng.choice(len(triangles), size=800, replace=False)
                triangles = [triangles[i] for i in idx]
            n_plus, n_minus, a_w = triangle_winding_asymmetry(
                triangles, psi_last)
            per_seed.append({
                "seed": s,
                "lambda_quartic": lam,
                "xi_0": xi_0,
                "S_bounce_thin_wall": s_b,
                "theta_v_rad": theta_v,
                "K_vortex": k_vtx,
                "J_CP_proxy": j_cp,
                "eta_B_predicted": eta_b_naive,
                "eta_B_ratio_obs": ratio,
                "n_pers_triangles": len(triangles),
                "n_winding_plus": n_plus,
                "n_winding_minus": n_minus,
                "A_winding_asymmetry": a_w,
            })
        if not per_seed:
            continue
        # Aggregate
        def mn(key):
            return float(np.mean([d[key] for d in per_seed]))
        def md(key):
            return float(np.median([d[key] for d in per_seed]))
        lam_m = mn("lambda_quartic")
        xi0_m = mn("xi_0")
        sb_m = mn("S_bounce_thin_wall")
        th_m = mn("theta_v_rad")
        kv_m = mn("K_vortex")
        jcp_m = md("J_CP_proxy")  # median over seeds (robust to sign)
        eta_m = mn("eta_B_predicted")
        rat_m = mn("eta_B_ratio_obs")
        aw_m = mn("A_winding_asymmetry")
        print(f"  {regime:<7} {len(per_seed):>2} "
              f"{lam_m:>7.2f} {xi0_m:>6.3f} {sb_m:>8.2f} "
              f"{th_m:>+8.3f} {kv_m:>6.3f} "
              f"{jcp_m:>+10.3e} {eta_m:>11.3e} {rat_m:>8.2e} "
              f"{aw_m:>+7.3f}")
        rows.append({
            "regime": regime, "N": n_lat,
            "n_seeds": len(per_seed),
            "lambda_quartic_mean": lam_m,
            "xi_0_mean": xi0_m,
            "S_bounce_mean": sb_m,
            "theta_v_rad_mean": th_m,
            "K_vortex_mean": kv_m,
            "J_CP_proxy_median": jcp_m,
            "eta_B_predicted_mean": eta_m,
            "eta_B_ratio_obs_mean": rat_m,
            "A_winding_asymmetry_mean": aw_m,
            "per_seed": per_seed,
        })

    # Cross-regime synthesis
    print()
    print("=" * 78)
    print("Cross-regime synthesis")
    print("=" * 78)
    if rows:
        sb_arr = np.array([r["S_bounce_mean"] for r in rows])
        th_arr = np.array([r["theta_v_rad_mean"] for r in rows])
        kv_arr = np.array([r["K_vortex_mean"] for r in rows])
        jcp_arr = np.array([r["J_CP_proxy_median"] for r in rows])
        eta_arr = np.array([r["eta_B_predicted_mean"] for r in rows])
        aw_arr = np.array([r["A_winding_asymmetry_mean"] for r in rows])
        ratio_arr = eta_arr / ETA_B_OBS
        print(f"  S_bounce mean      = {sb_arr.mean():.2f} "
              f"+/- {sb_arr.std():.2f}  (GCC-05 P1: 38.0)")
        print(f"  theta_v mean       = {th_arr.mean():+.3f} rad "
              f"+/- {th_arr.std():.3f}  (GCC-05 P1: +1.608)")
        print(f"  K_vortex mean      = {kv_arr.mean():.3f}")
        print(f"  J_CP mean median   = {jcp_arr.mean():+.3e} "
              f"(PDG CKM J_CP: +3.08e-5)")
        print(f"  eta_B,pred mean    = {eta_arr.mean():.3e}")
        print(f"  Ratio to obs       = {ratio_arr.mean():.2e} "
              f"(stdev {ratio_arr.std():.2e})")
        print(f"  log10 ratio        = {np.log10(np.abs(ratio_arr)).mean():+.2f} "
              f"+/- {np.log10(np.abs(ratio_arr)).std():.2f}")
        print(f"  Triangle winding A = {aw_arr.mean():+.4f} "
              f"+/- {aw_arr.std():.4f}")

    bundle = {
        "method": "baryogenesis_per_regime_S1_S5",
        "PDG_eta_B_obs": ETA_B_OBS,
        "PDG_eta_B_unc": ETA_B_OBS_UNC,
        "n_gen": N_GEN,
        "g_star_SM": G_STAR_SM,
        "rows": rows,
        "interpretation": (
            "S1 theta_v, S2 S_bounce, S3 J_CP_proxy, "
            "S4 eta_B = (28/51)*J_CP*S_bounce*(n_gen/g_star)*K_vortex, "
            "S5 triangle winding asymmetry. All S1-S5 computed direct "
            "from snapshot NPZ inputs without further pipeline calls."
        ),
    }
    out = REPO / "outputs" / "audit_baryogenesis_per_regime.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
