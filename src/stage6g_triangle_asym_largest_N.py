"""Stage 6g: Triangle phase-class asymmetry empirical N-trend.

Measures the persistent-violator triangle phase-class asymmetry

    asym(N) = (n_PPN - n_PNN) / (n_PPN + n_PNN)

on the seven-point P5 lattice ladder
{P5N64, P5N72, P5N84, P5N100, P5N200, P5N300, P5N512}, with
weighted Symanzik-2 fit y(N) = a + b/N for the asymptote.

The previously-conjectured closed-form
    a_inf = -pi * gamma^2 / 2 = -pi/200 = -0.015708
(framed as a "Berry-Wess-Zumino-Witten one-loop" derivation in
earlier revisions of stage6b_solve_4_problems.py problem_1) is
empirically falsified by this seven-point fit at 6.84 sigma; the
asymptote is consistent with zero at the present precision and the
threshold-c_info dependence (factor-3 spread of the asymptote
across c_info in {1.5x, 2.0x, 2.5x, 3.0x}) shows that the
observable itself does not have a threshold-stable continuum
limit at the current ladder.  No closed-form theoretical anchor
is asserted by this script; it reports only the empirical
asymptote.

Output: outputs/stage6g_triangle_asym_largest_N.json
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

# Cleaned ladder (skip P5N128 K/Q artefact, use 12-seed P5N300)
LADDER = [
    ("P5N64",  64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz"),
    ("P5N72",  72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz"),
    ("P5N84",  84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz"),
    ("P5N100", 100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz"),
    ("P5N200", 200, "results_d1_p5n200_8seeds/P5N200.snapshots.npz"),
    ("P5N300", 300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz"),
    ("P5N512", 512, "results_d1_p5n512_12seeds/P5N512.snapshots.npz"),
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz"),
]

GAMMA = 0.10
PI_OVER_200 = -np.pi * GAMMA * GAMMA / 2.0  # -0.015707963


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


def asym_per_seed(snaps, psi_r, psi_i, n_lat, seed):
    xi_traj = np.asarray(snaps[seed], dtype=float)
    psi_last = (psi_r[seed, -1].astype(float)
                + 1j * psi_i[seed, -1].astype(float))
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
        return None, 0, 0
    phi = np.angle(psi_last)
    n_PPN = 0
    n_PNN = 0
    for i, j, k in triangles:
        d_ij = np.angle(np.exp(1j * (phi[j] - phi[i])))
        d_jk = np.angle(np.exp(1j * (phi[k] - phi[j])))
        d_ki = np.angle(np.exp(1j * (phi[i] - phi[k])))
        if (abs(d_ij) < 1e-9 or abs(d_jk) < 1e-9 or abs(d_ki) < 1e-9):
            continue
        signs = (np.sign(d_ij), np.sign(d_jk), np.sign(d_ki))
        n_pos = sum(1 for s_ in signs if s_ > 0)
        if n_pos == 2:
            n_PPN += 1
        elif n_pos == 1:
            n_PNN += 1
    if n_PPN + n_PNN == 0:
        return None, 0, 0
    return (n_PPN - n_PNN) / (n_PPN + n_PNN), n_PPN, n_PNN


def main() -> int:
    print("=" * 80)
    print("Stage 6g: Triangle phase-class asym on largest-N regimes")
    print("=" * 80)
    print(f"  Lemma 9 (Berry-WZW) prediction: a_inf = -pi*gamma^2/2 = {PI_OVER_200:.6f}")
    print()
    print(f"{'regime':<8} {'N':>4} {'n_s':>4} {'a_mean':>10} "
          f"{'a_std':>9} {'unc(a)':>10} {'sigma_vs_-pi/200':>18}")
    print("-" * 80)
    rows = []
    for regime, n_lat, rel in LADDER:
        fp = PARENT / rel
        if not fp.exists():
            print(f"  {regime} N={n_lat}: missing")
            continue
        z = np.load(fp, allow_pickle=True)
        snaps = z["edge_xi_snapshots"]
        psi_r = z["psi_real_snapshots"]
        psi_i = z["psi_imag_snapshots"]
        n_seeds = int(snaps.shape[0])
        per_seed = []
        for s in range(n_seeds):
            a, n_p, n_n = asym_per_seed(snaps, psi_r, psi_i, n_lat, s)
            if a is not None:
                per_seed.append(a)
        if not per_seed:
            print(f"  {regime}: no triangles")
            continue
        arr = np.array(per_seed)
        m = float(arr.mean())
        sd = float(arr.std())
        unc = float(sd / np.sqrt(len(arr)))
        sig_vs_pred = (m - PI_OVER_200) / unc if unc > 1e-9 else float("nan")
        rows.append({
            "regime": regime, "N": n_lat,
            "n_seeds": len(arr),
            "asym_per_seed": arr.tolist(),
            "asym_mean": m,
            "asym_std": sd,
            "asym_unc": unc,
            "sigma_vs_minus_pi_over_200": sig_vs_pred,
        })
        print(f"{regime:<8} {n_lat:>4} {len(arr):>4} "
              f"{m:>+10.5f} {sd:>9.5f} {unc:>10.5f} {sig_vs_pred:>+18.2f}")

    # Symanzik-2 weighted fit
    print()
    print("=" * 80)
    print("Symanzik-2 fit y(N) = a + b/N (weighted by 1/unc^2)")
    print("=" * 80)
    if len(rows) >= 3:
        N_arr = np.array([r["N"] for r in rows], dtype=float)
        a_arr = np.array([r["asym_mean"] for r in rows])
        unc_arr = np.array([r["asym_unc"] for r in rows])
        x = 1.0 / N_arr
        w = 1.0 / np.maximum(unc_arr, 1e-6) ** 2
        A = np.column_stack([np.ones_like(x), x])
        AtWA = A.T @ (w[:, None] * A)
        AtWy = A.T @ (w * a_arr)
        coef = np.linalg.solve(AtWA, AtWy)
        pred = A @ coef
        chi2 = float(np.sum(w * (a_arr - pred) ** 2))
        cov = np.linalg.inv(AtWA)
        a_inf, b = float(coef[0]), float(coef[1])
        a_inf_unc = float(np.sqrt(cov[0, 0]))
        b_unc = float(np.sqrt(cov[1, 1]))
        sigma_pred = (a_inf - PI_OVER_200) / a_inf_unc

        print(f"  a_inf (asymptote N->inf) = {a_inf:+.5f} +/- {a_inf_unc:.5f}")
        print(f"  b (1/N coefficient)      = {b:+.4f} +/- {b_unc:.4f}")
        print(f"  chi^2/dof                = {chi2:.2f} / {len(rows) - 2}")
        print(f"  -pi*gamma^2/2 =           {PI_OVER_200:+.5f}")
        print(f"  a_inf - (-pi/200)         = {a_inf - PI_OVER_200:+.5f}")
        print(f"  sigma vs -pi/200          = {sigma_pred:+.2f}")
        print(f"  -1/64                     = {-1/64:+.5f}")
        print(f"  a_inf - (-1/64)           = {a_inf - (-1/64):+.5f}")
        print(f"  sigma vs -1/64            = "
              f"{(a_inf - (-1/64)) / a_inf_unc:+.2f}")

        bundle_fit = {
            "a_inf": a_inf,
            "a_inf_unc": a_inf_unc,
            "b": b,
            "b_unc": b_unc,
            "chi2": chi2,
            "dof": len(rows) - 2,
            "lemma9_prediction_minus_pi_gamma2_over_2": PI_OVER_200,
            "sigma_vs_lemma9": sigma_pred,
            "sigma_vs_minus_1_over_64":
                (a_inf - (-1 / 64)) / a_inf_unc,
        }
    else:
        bundle_fit = {}

    bundle = {
        "method": "stage6g_triangle_asym_largest_N",
        "schema_version": "1.0.0",
        "lemma9_prediction": PI_OVER_200,
        "ladder_regimes": [
            {"regime": r, "N": n, "rel_path": p} for r, n, p in LADDER
        ],
        "rows": rows,
        "symanzik_2pt_fit": bundle_fit,
    }
    out = REPO / "outputs" / "stage6g_triangle_asym_largest_N.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print()
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
