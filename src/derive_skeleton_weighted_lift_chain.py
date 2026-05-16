r"""Per-regime algebraic-chain audit for the (SG) closure.

Verifies the Q-identity
   3/8 = (7/24) * (9/7)
        = lambda_inf^skel  *  weight-lift factor
        = lambda_inf^vac
empirically on the closure-domain ladder, *per regime*. Companion
script to Remark rmk:phase3_algebraic_chain of the manuscript.

For each regime, this loads the per-seed Xi snapshots (final
iteration), builds the weighted normalised Laplacian and the
skeleton normalised Laplacian (threshold tau = 0.10), reads the
top non-trivial eigenvalues, and reports:

  * skeleton lambda_2  (target 7/24)
  * weighted lambda_2  (target 3/8)
  * weight-lift ratio  (target 9/7)
  * chained product    (= skeleton * lift)

The chained product is compared against the directly-extracted
weighted lambda_2 *per regime* and after Symanzik-1 extrapolation
to N -> infinity. The chain consistency residual is the headline
number reported in the remark (typically <1% end-to-end).

Algebraic identity is checked exactly in Q at the (4, 3) anchor.

Output: outputs/derive_skeleton_weighted_lift_chain.json plus a
console summary.

Honest reading. The chain is an exact Q-identity at the anchor,
not an independent two-step mechanism: the N_gen factor in 9/7
cancels with that in 7/24 in the product, so the factorisation is
post-hoc rather than mechanistic (see par:cross_closure_identities
of the bridge manuscript for the d-only / System-R independence
statement that constrains this reading).
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

# Path discovery uses the bundled helper.
SRC = Path(__file__).resolve().parent
REPO = SRC.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC))
from _d1_npz_discovery import find_d1_npz  # noqa: E402

# ------------------------------------------------------------------
# (d, N_gen) anchor.
# ------------------------------------------------------------------
D = 4
N_GEN = 3
TAU = 0.10
MAX_SEEDS = 8

LADDER = ["P5N100", "P5N128", "P5N200", "P5N256", "P5N300", "P5N512"]
LADDER_N = {"P5N100": 100, "P5N128": 128, "P5N200": 200,
            "P5N256": 256, "P5N300": 300, "P5N512": 512}

# ------------------------------------------------------------------
# Algebraic chain targets (exact in Q at (4, 3)).
# ------------------------------------------------------------------
TARGET_SKEL = Fraction(D + N_GEN, 2 * D * N_GEN)        # 7/24
TARGET_LIFT = Fraction((D - 1) * N_GEN, D + N_GEN)       # 9/7
TARGET_WEIGHTED = Fraction(D - 1, 2 * D)                 # 3/8
assert TARGET_SKEL * TARGET_LIFT == TARGET_WEIGHTED, (
    f"algebraic chain FAILS in Q: {TARGET_SKEL}*{TARGET_LIFT}="
    f"{TARGET_SKEL*TARGET_LIFT} != {TARGET_WEIGHTED}"
)


def normalised_laplacian(adj: np.ndarray) -> np.ndarray:
    w = adj.copy()
    np.fill_diagonal(w, 0.0)
    deg = np.maximum(w.sum(axis=1), 1e-12)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    return 0.5 * (
        np.eye(w.shape[0])
        - (d_inv_sqrt[:, None] * w * d_inv_sqrt[None, :])
        + (np.eye(w.shape[0])
           - (d_inv_sqrt[:, None] * w * d_inv_sqrt[None, :])).T
    )


def laplacian_top_eigs(xi: np.ndarray, weighted: bool,
                        tau: float = TAU, top_k: int = 5) -> np.ndarray:
    if weighted:
        adj = xi.copy()
    else:
        adj = (np.abs(xi - np.diag(np.diag(xi))) > tau).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    if adj.sum() == 0:
        return np.full(top_k, np.nan)
    L = normalised_laplacian(adj)
    eigs = np.linalg.eigvalsh(L)
    return eigs[1:1 + top_k]


def symanzik1_fit(n_arr, y_arr):
    n_arr = np.asarray(n_arr, dtype=np.float64)
    y_arr = np.asarray(y_arr, dtype=np.float64)
    mask = np.isfinite(y_arr)
    if mask.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    n_arr, y_arr = n_arr[mask], y_arr[mask]
    A = np.column_stack([np.ones_like(n_arr), 1.0 / n_arr])
    coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
    pred = A @ coef
    ss_res = float(((y_arr - pred) ** 2).sum())
    ss_tot = float(((y_arr - y_arr.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), float(coef[1]), r2


def process_regime(regime: str) -> dict | None:
    npz_path = find_d1_npz(regime, REPO)
    if npz_path is None:
        print(f"  [skip] {regime}: NPZ not found")
        return None
    d = np.load(npz_path, allow_pickle=True)
    if "edge_xi_snapshots" not in d.files:
        print(f"  [skip] {regime}: edge_xi_snapshots key missing in {npz_path.name}")
        return None
    xi_arr = d["edge_xi_snapshots"][:, -1]
    n_seeds = min(MAX_SEEDS, xi_arr.shape[0])
    w_l2, s_l2 = [], []
    for s in range(n_seeds):
        xi = xi_arr[s].astype(np.float64)
        try:
            ws = laplacian_top_eigs(xi, weighted=True, top_k=1)
            sks = laplacian_top_eigs(xi, weighted=False, top_k=1)
            w_l2.append(float(ws[0]))
            s_l2.append(float(sks[0]))
        except np.linalg.LinAlgError:
            continue
    if not w_l2 or not s_l2:
        return None
    wm = float(np.mean(w_l2))
    sm = float(np.mean(s_l2))
    return {
        "regime": regime,
        "N": LADDER_N[regime],
        "n_seeds": n_seeds,
        "source": str(npz_path.relative_to(REPO.parent) if REPO.parent in npz_path.parents else npz_path),
        "weighted_l2": wm,
        "skeleton_l2": sm,
        "lift_l2": (wm / sm) if sm > 0 else float("nan"),
    }


def main():
    print("=" * 78)
    print("Per-regime algebraic-chain audit (Phase 3 / Lemma B)")
    print("=" * 78)
    print()
    print("  Algebraic chain (exact in Q at (d, N_gen) = (4, 3)):")
    print(f"    skeleton gap (target) = (d+N_gen)/(2 d N_gen) = "
          f"{TARGET_SKEL} = {float(TARGET_SKEL):.6f}")
    print(f"    weight-lift (target)  = (d-1) N_gen/(d+N_gen) = "
          f"{TARGET_LIFT} = {float(TARGET_LIFT):.6f}")
    print(f"    weighted gap (target) = (d-1)/(2 d) = "
          f"{TARGET_WEIGHTED} = {float(TARGET_WEIGHTED):.6f}")
    print(f"    product check: {TARGET_SKEL}*{TARGET_LIFT} == "
          f"{TARGET_WEIGHTED}?  "
          f"{TARGET_SKEL * TARGET_LIFT == TARGET_WEIGHTED}")
    print()
    per_regime = []
    for regime in LADDER:
        r = process_regime(regime)
        if r is not None:
            per_regime.append(r)
            print(f"  {regime:<8} N={r['N']:>4}:  "
                  f"w_l2={r['weighted_l2']:.4f}   "
                  f"sk_l2={r['skeleton_l2']:.4f}   "
                  f"lift={r['lift_l2']:.4f}")
    print()
    if len(per_regime) < 3:
        print("  [warn] not enough regimes available for Symanzik fit; "
              "running with available data only.")

    n_arr = np.array([r["N"] for r in per_regime])
    w_y = np.array([r["weighted_l2"] for r in per_regime])
    s_y = np.array([r["skeleton_l2"] for r in per_regime])
    lift_y = np.array([r["lift_l2"] for r in per_regime])

    w_inf, w_b, w_r2 = symanzik1_fit(n_arr, w_y)
    s_inf, s_b, s_r2 = symanzik1_fit(n_arr, s_y)
    lift_inf, lift_b, lift_r2 = symanzik1_fit(n_arr, lift_y)

    print("-" * 78)
    print("Symanzik-1 extrapolation N -> infinity")
    print("-" * 78)
    print(f"  weighted   lambda_2 -> {w_inf:.5f}   "
          f"target {float(TARGET_WEIGHTED):.5f}   "
          f"rel = {(w_inf - float(TARGET_WEIGHTED))/float(TARGET_WEIGHTED)*100:+.2f}%")
    print(f"  skeleton   lambda_2 -> {s_inf:.5f}   "
          f"target {float(TARGET_SKEL):.5f}   "
          f"rel = {(s_inf - float(TARGET_SKEL))/float(TARGET_SKEL)*100:+.2f}%")
    print(f"  weight-lift          -> {lift_inf:.5f}   "
          f"target {float(TARGET_LIFT):.5f}   "
          f"rel = {(lift_inf - float(TARGET_LIFT))/float(TARGET_LIFT)*100:+.2f}%")
    print()
    chain_recon = s_inf * lift_inf
    rel_chain = (chain_recon - w_inf) / w_inf * 100 if w_inf else float("nan")
    print(f"  end-to-end:  emp_skel * emp_lift = {chain_recon:.5f}  "
          f"vs  emp_weighted = {w_inf:.5f}   "
          f"chain residual = {rel_chain:+.2f}%")
    print()

    bundle = {
        "title": "Per-regime algebraic-chain audit for (SG) closure",
        "anchor": {"d": D, "N_gen": N_GEN, "tau_skeleton": TAU},
        "exact_targets": {
            "skeleton_gap_7_over_24": str(TARGET_SKEL),
            "weight_lift_9_over_7":  str(TARGET_LIFT),
            "weighted_gap_3_over_8": str(TARGET_WEIGHTED),
            "algebraic_identity_holds": True,
        },
        "per_regime": per_regime,
        "symanzik": {
            "weighted": {"a_inf": w_inf, "b": w_b, "r2": w_r2,
                          "target": float(TARGET_WEIGHTED),
                          "rel_err_pct":
                            (w_inf - float(TARGET_WEIGHTED))/float(TARGET_WEIGHTED)*100},
            "skeleton": {"a_inf": s_inf, "b": s_b, "r2": s_r2,
                          "target": float(TARGET_SKEL),
                          "rel_err_pct":
                            (s_inf - float(TARGET_SKEL))/float(TARGET_SKEL)*100},
            "lift":     {"a_inf": lift_inf, "b": lift_b, "r2": lift_r2,
                          "target": float(TARGET_LIFT),
                          "rel_err_pct":
                            (lift_inf - float(TARGET_LIFT))/float(TARGET_LIFT)*100},
        },
        "chain_reconstruction": {
            "empirical_skel_times_empirical_lift": chain_recon,
            "empirical_weighted": w_inf,
            "chain_residual_pct": rel_chain,
        },
        "honest_caveat": (
            "Algebraic identity is exact in Q at (d, N_gen) = (4, 3): "
            "the N_gen factor in 9/7 cancels with that in 7/24, so the "
            "factorisation is post-hoc rather than a two-step "
            "mechanism. See par:cross_closure_identities of the bridge "
            "manuscript for the d-only / System-R independence "
            "statement that constrains this reading."
        ),
    }
    out = OUTPUTS / "derive_skeleton_weighted_lift_chain.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
