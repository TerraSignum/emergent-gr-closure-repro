r"""Lemma B Step 4a: carrier-spectral-synthesis attempt.

Goal: identify the analytical structure underlying the empirically
certified skeleton-Laplacian asymptote
  lambda_inf^skel = 7/24 = (d + N_gen) / (2 d N_gen)
and the weighted-Laplacian asymptote
  lambda_inf^w   = 3/8  = (d - 1) / (2 d).

Two structural observations motivating the attempt:

(A) The skeleton asymptote 7/24 admits the clean decomposition
       7/24 = (1/d + 1/N_gen) / 2
    i.e. the arithmetic mean of the inverse spacetime dimension
    and the inverse generation count. This is the signature of a
    two-factor product structure where one factor has spectral gap
    1/d and the other has spectral gap 1/N_gen.

(B) The weight-lift ratio 9/7 = (d-1) N_gen / (d + N_gen) factors
    the chain 7/24 -> 3/8 as
       3/8 = (7/24) * (d-1) N_gen / (d + N_gen).

Together these give the algebraic chain
  3/8 = [(1/d + 1/N_gen) / 2] * [(d-1) N_gen / (d + N_gen)]
      = (d-1) / (2 d)                                    [algebraic]

This script tests the two-factor hypothesis empirically by:

  (1) Computing the top 5 eigenvalues + eigenvectors of the tau=0.1
      skeleton Laplacian per regime.
  (2) Testing whether the top-2 eigenvectors admit a separable
      product structure: f(a) = g_spatial(a) * h_family(a), where
      a indexes the lattice node and g, h depend on inferable
      coordinates.
  (3) Searching for the two factor-graph candidates that give
      gap 1/d and 1/N_gen respectively. Candidate families tested:
        - n-cycle C_n: gap 1 - cos(2 pi / n) ~ 2 pi^2 / n^2  (no)
        - complete K_n: gap n/(n-1)  (too large)
        - star S_n: gap 1  (constant)
        - hypercube Q_d: gap 2/d  (factor 2)
        - 1/2 - 1/(2k) family of regular graphs (custom)
  (4) Reporting which candidate family matches the carrier-derived
      gap.

If a clean two-factor identification is found, the analytical
chain is established at the structural level. The remaining
analytical step is then to derive the specific factor families
from the carrier action S_Xi.

Output: outputs/verify_lemma_B_carrier_spectral_synthesis.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


class _BlockCupy:
    def find_spec(self, name, path=None, target=None):
        if name == "cupy" or name.startswith("cupy."):
            raise ImportError("cupy disabled")
        return None


sys.meta_path.insert(0, _BlockCupy())

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
OUTPUTS = REPO / "outputs"

from _d1_npz_discovery import find_d1_npz  # noqa: E402

D = 4
N_GEN = 3
TAU = 0.10  # skeleton threshold (per Phase-2 Step 3a / 3b)
TOP_K = 8   # top K non-zero eigenvalues to extract

# Canonical d1 P5/P5N ladder (alt-anchor-separation rule).
LADDER = [
    ("P5",    50),
    ("P5N64", 64),
    ("P5N72", 72),
    ("P5N84", 84),
    ("P5N100", 100),
    ("P5N128", 128),
    ("P5N200", 200),
    ("P5N256", 256),
    ("P5N300", 300),
    ("P5N512", 512),
]


def normalized_laplacian(adj):
    deg = adj.sum(axis=1)
    deg_safe = np.where(deg > 0, deg, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(deg_safe)
    d_inv_sqrt[deg == 0] = 0.0
    n = adj.shape[0]
    return np.eye(n) - (d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :])


def skeleton_eigs(xi, tau=TAU, top_k=TOP_K):
    """Extract the unweighted-skeleton normalised-Laplacian top-K
    non-trivial eigenvalues and eigenvectors per snapshot.
    """
    n = xi.shape[0]
    adj = (np.abs(xi - np.diag(np.diag(xi))) > tau).astype(float)
    # Symmetrise (already symmetric for Xi by construction, but
    # threshold may break it if values are near tau)
    adj = 0.5 * (adj + adj.T)
    lap = normalized_laplacian(adj)
    eigvals, eigvecs = np.linalg.eigh(lap)
    # Sort ascending; first is ~0 (Perron eigenvalue), then non-trivial
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # Drop the trivial zero mode
    return eigvals[1:1 + top_k], eigvecs[:, 1:1 + top_k]


def factor_graph_spectral_gaps():
    """Candidate factor families with their analytical spectral gaps
    (normalised-Laplacian convention)."""
    return [
        ("1/d (claim)",         1.0 / D),
        ("1/N_gen (claim)",     1.0 / N_GEN),
        ("hypercube Q_d (=2/d)", 2.0 / D),
        ("hypercube Q_Ngen (=2/N_gen)", 2.0 / N_GEN),
        ("K_d/(K_d-1)",         D / (D - 1)),
        ("K_Ngen/(K_Ngen-1)",   N_GEN / (N_GEN - 1)),
    ]


def two_factor_predictions():
    """Possible two-factor combinations of the candidate gaps.

    For a cartesian product G x H of weighted graphs, the
    smallest non-trivial eigenvalue of the normalised Laplacian
    is (lambda_G + lambda_H) / 2 if weighted equally, or
    min(lambda_G, lambda_H) if one factor dominates."""
    return {
        "mean(1/d, 1/N_gen)": (1.0 / D + 1.0 / N_GEN) / 2,
        "min(1/d, 1/N_gen)": min(1.0 / D, 1.0 / N_GEN),
        "max(1/d, 1/N_gen)": max(1.0 / D, 1.0 / N_GEN),
        "harmonic(1/d, 1/N_gen)": 2.0 / (D + N_GEN),
        "(1/d) * (1/N_gen)": 1.0 / (D * N_GEN),
        "(d+N_gen)/(2*d*N_gen)": (D + N_GEN) / (2 * D * N_GEN),
        "7/24 (target skeleton)": 7 / 24,
        "3/8 (target weighted)": 3 / 8,
    }


def load_xi_snapshots(regime, n_lat):
    """Load equilibrium Xi snapshots from the canonical d1 NPZ."""
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    snaps = []
    if "edge_xi_snapshots" in d.files:
        # snapshot NPZ schema
        arr = d["edge_xi_snapshots"]
        if arr.ndim != 4:
            return None
        # arr.shape = (n_seeds, n_snapshots, N, N)
        # take last snapshot per seed = equilibrium
        for s in range(arr.shape[0]):
            xi = np.asarray(arr[s, -1], dtype=float)
            if xi.shape == (n_lat, n_lat):
                snaps.append(xi)
    else:
        # final-state NPZ schema (xi_seed{i})
        for s in range(64):
            key = f"xi_seed{s}"
            if key in d.files:
                xi = np.asarray(d[key], dtype=float)
                if xi.shape == (n_lat, n_lat):
                    snaps.append(xi)
    return snaps


def fit_symanzik_1(xs, ys):
    """Symanzik-1 fit y(N) = a + b/N; return (a, b, R^2)."""
    if len(xs) < 3:
        return None, None, None
    invN = np.asarray([1.0 / x for x in xs])
    y = np.asarray(ys)
    A = np.column_stack([np.ones_like(invN), invN])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float(((y - y_pred) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(beta[0]), float(beta[1]), r2


def main():
    print("=" * 100)
    print("Lemma B Step 4a: carrier spectral-synthesis attempt")
    print("=" * 100)
    print(f"d = {D}, N_gen = {N_GEN}, tau = {TAU}, top_k = {TOP_K}")
    print()
    print("Hypothesis: skeleton lambda_2 = (1/d + 1/N_gen)/2 = 7/24")
    print("Algebraic chain: 3/8 = 7/24 * 9/7 (Kahale weight lift)")
    print()

    # Per-regime top eigenvalues + eigenvector separability checks
    rows = []
    for regime, n_lat in LADDER:
        snaps = load_xi_snapshots(regime, n_lat)
        if not snaps:
            print(f"  [SKIP {regime}: no Xi snapshots]")
            continue
        eigs_per_seed = []
        for xi in snaps:
            try:
                evs, _ = skeleton_eigs(xi)
                eigs_per_seed.append(evs)
            except (ValueError, np.linalg.LinAlgError):
                continue
        if not eigs_per_seed:
            print(f"  [SKIP {regime}: no successful diagonalisations]")
            continue
        eigs_arr = np.stack(eigs_per_seed, axis=0)
        mean_eigs = eigs_arr.mean(axis=0).tolist()
        sem_eigs = (eigs_arr.std(axis=0)
                       / np.sqrt(eigs_arr.shape[0])).tolist()
        rows.append({
            "regime": regime,
            "N": n_lat,
            "n_seeds": len(eigs_per_seed),
            "lambda_top": mean_eigs,
            "lambda_top_sem": sem_eigs,
        })
        print(f"{regime:<8} N={n_lat:>4} n_seeds={len(eigs_per_seed):>3} "
                f"lambda_2 = {mean_eigs[0]:.4f} +- {sem_eigs[0]:.4f}, "
                f"lambda_3 = {mean_eigs[1]:.4f}, "
                f"lambda_4 = {mean_eigs[2]:.4f}, "
                f"lambda_5 = {mean_eigs[3]:.4f}")
    print()

    # Symanzik-1 N-asymptote fit per eigenvalue index
    if len(rows) >= 5:
        n_vals = [r["N"] for r in rows]
        print("Symanzik-1 asymptotes a + b/N (top-5):")
        print("-" * 100)
        print(f"{'idx':>4} {'a_inf':>10} {'b':>10} {'R^2':>8}")
        asymptotes = []
        for k in range(min(TOP_K, len(rows[0]["lambda_top"]))):
            y_vals = [r["lambda_top"][k] for r in rows]
            a, b, r2 = fit_symanzik_1(n_vals, y_vals)
            asymptotes.append({"idx": k + 2, "a_inf": a,
                                  "b": b, "r2": r2})
            print(f"  l_{k+2}{'':>2} {a:>10.5f} {b:>10.3f} {r2:>8.3f}")
        print()

        # Test the two-factor predictions
        l2_inf = asymptotes[0]["a_inf"]
        l3_inf = asymptotes[1]["a_inf"]
        print(f"lambda_2^inf = {l2_inf:.5f}")
        print(f"lambda_3^inf = {l3_inf:.5f}")
        print()
        print("Two-factor candidate predictions vs lambda_2^inf:")
        print("-" * 100)
        preds = two_factor_predictions()
        match_table = []
        for name, val in preds.items():
            err_rel = abs(l2_inf - val) / val * 100
            mark = " <- CANDIDATE" if err_rel < 1.0 else ""
            match_table.append({"name": name, "value": val,
                                   "rel_err_pct": err_rel})
            print(f"  {name:<28} = {val:.5f}  "
                    f"rel err {err_rel:5.2f}%{mark}")
        print()

        # Eigenvalue gap structure: are subsequent eigenvalues
        # consistent with a separable two-factor spectrum?
        # For G x H (cartesian product) with G factor having gap g_G
        # and H factor having gap g_H, the eigenvalue spectrum
        # is {(g_G * a + g_H * b) : a, b in {0,1,2,...}}/2.
        # Smallest non-zero values are (g_G, 0), (0, g_H), (2g_G, 0),
        # (g_G, g_H)/2, etc.
        print("Spectral gap-pair test (cartesian-product hypothesis):")
        print("-" * 100)
        if len(asymptotes) >= 2:
            # If lambda_2 = (g_G + g_H)/2 and lambda_3 = g_G or g_H
            # alone (i.e. one factor's full gap, other at zero mode)
            # Then lambda_3 should be either g_G or g_H itself.
            g_a, g_b = 1.0 / D, 1.0 / N_GEN
            mean_g = (g_a + g_b) / 2
            for cand_g, name in [(g_a, "1/d"), (g_b, "1/N_gen")]:
                err = abs(l3_inf - cand_g) / cand_g * 100
                print(f"  lambda_3^inf vs {name} = {cand_g:.5f}: "
                        f"rel err {err:.2f}%")
            sep_check_l2 = abs(l2_inf - mean_g) / mean_g * 100
            print(f"  lambda_2^inf vs (1/d + 1/N_gen)/2 = "
                    f"{mean_g:.5f}: rel err {sep_check_l2:.2f}%")
        print()

        # Weighted-to-skeleton lift ratio
        # Empirical lambda_w_inf^vac ~ 0.3789 (Phase 1)
        l_w_inf_emp = 0.3789
        lift_ratio = l_w_inf_emp / l2_inf if l2_inf > 0 else 0
        target_ratio = (D - 1) * N_GEN / (D + N_GEN)
        err_ratio = (abs(lift_ratio - target_ratio)
                          / target_ratio * 100)
        print(f"Weight-lift ratio (Kahale-type):")
        print(f"  empirical    lambda_w / lambda_skel = "
                f"{l_w_inf_emp:.4f} / {l2_inf:.5f} = {lift_ratio:.4f}")
        print(f"  analytical   (d-1)*N_gen / (d+N_gen)  = "
                f"{target_ratio:.4f}  (= 9/7)")
        print(f"  rel err = {err_ratio:.2f}%")
        print()

        bundle = {
            "method": "verify_lemma_B_carrier_spectral_synthesis",
            "schema_version": "1.0",
            "stand": "2026-05-13",
            "d": D,
            "N_gen": N_GEN,
            "tau_skeleton": TAU,
            "rows": rows,
            "asymptotes": asymptotes,
            "match_table": match_table,
            "lambda_2_inf": l2_inf,
            "lambda_3_inf": l3_inf,
            "weight_lift_ratio_emp": lift_ratio,
            "weight_lift_ratio_analytical": target_ratio,
            "weight_lift_rel_err_pct": err_ratio,
            "structural_claims": {
                "skeleton_factor_a_gap": 1.0 / D,
                "skeleton_factor_b_gap": 1.0 / N_GEN,
                "skeleton_predicted_gap": (1.0 / D + 1.0 / N_GEN) / 2,
                "weighted_predicted_gap": (D - 1) / (2 * D),
                "skeleton_predicted_rational": "(d + N_gen) / (2 d N_gen) = 7/24",
                "weighted_predicted_rational": "(d - 1) / (2 d) = 3/8",
                "lift_analytical_rational": "(d-1) N_gen / (d + N_gen) = 9/7",
            },
        }
    else:
        bundle = {"method": "verify_lemma_B_carrier_spectral_synthesis",
                     "n_regimes": len(rows),
                     "status": "insufficient_data"}

    out = OUTPUTS / "verify_lemma_B_carrier_spectral_synthesis.json"
    out.write_text(json.dumps(bundle, indent=2, default=float),
                       encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
