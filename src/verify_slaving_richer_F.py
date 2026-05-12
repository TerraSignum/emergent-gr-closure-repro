r"""Iter-21: Slaving with richer functional form aggregator.

The G4 baseline (verify_slaving_reconstruction.py) found that the
naive 4-parameter sigmoid of bare degree (Sum Xi) and Laplacian-
squared diagonal gives R^2 ~ 0.10 on the per-snapshot ff_K, ff_Q
target fields. Q2 Lipschitz-stability test showed Lip = 0 across
all seeds, indicating ff_K/ff_Q are regime-constants invariant
under seed-dependent Xi-fluctuations -- the structurally
meaningful slaving statement at the seed level.

This script extends G4 with richer functional forms to identify
WHICH aggregator captures the residual ~1% per-node K, Q
variance:

  R1. Polynomial features (degree-3) of (deg, lap_2, fiedler 1-3):
      tests if quadratic / cubic combinations of the 5 base
      features improve R^2.

  R2. Random-forest non-parametric regression with the same
      6-feature set: tests if the relationship is non-linear /
      non-polynomial (capturing tree-like decision boundaries).

  R3. Spectral-projection: regress the target on the top-k
      eigenvalues / eigenvectors of the Xi-Laplacian directly
      (non-local global aggregator).

The interpretation: if R1/R2 push R^2 above 0.5, the per-node
K, Q variance is a non-trivial functional of Xi-graph topology.
If even R2 (universal approximator at depth 4-5) gives R^2 < 0.5,
then ff_K, ff_Q carry information NOT derivable from the
per-snapshot Xi alone -- they would be additional state variables
slaved at a higher RG stage.

Output: outputs/verify_slaving_richer_F.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PARENT = REPO.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def load_first_snapshot(regime: str, n_lat: int):
    candidates = [
        PARENT / f"results_d1_{regime.lower()}_24seeds" / f"{regime}.snapshots.npz",
        PARENT / f"results_d1_{regime.lower()}_8seeds" / f"{regime}.snapshots.npz",
        PARENT / f"results_d1_{regime.lower()}_12seeds" / f"{regime}.snapshots.npz",
    ]
    for p in candidates:
        if p.exists():
            d = np.load(p, allow_pickle=True)
            xi = d["edge_xi_snapshots"][0, -1].astype(float).copy()
            np.fill_diagonal(xi, 1.0)
            ff_k = d.get("ff_K_seed0", np.full((n_lat, n_lat), 0.55))
            ff_q = d.get("ff_Q_seed0", np.full((n_lat, n_lat), 0.45))
            return (xi, np.asarray(ff_k, dtype=float),
                     np.asarray(ff_q, dtype=float))
    return None


def base_features(xi):
    """Six base per-node features."""
    n = xi.shape[0]
    xi_off = xi.copy()
    np.fill_diagonal(xi_off, 0.0)
    deg = xi_off.sum(axis=1)
    L = np.diag(deg) - xi_off
    lap_squared_diag = np.diag(L @ L)
    eigvals, eigvecs = np.linalg.eigh(L)
    f1, f2, f3 = eigvecs[:, 1], eigvecs[:, 2], eigvecs[:, 3]
    mean_edge = xi_off.mean(axis=1)
    return np.column_stack([deg, lap_squared_diag, f1, f2, f3, mean_edge])


def polynomial_expand(features, max_degree: int = 3):
    """Add polynomial cross-terms up to total degree max_degree."""
    n_samples, n_feat = features.shape
    cols = [np.ones(n_samples), *[features[:, k] for k in range(n_feat)]]
    if max_degree >= 2:
        for i in range(n_feat):
            for j in range(i, n_feat):
                cols.append(features[:, i] * features[:, j])
    if max_degree >= 3:
        for i in range(n_feat):
            for j in range(i, n_feat):
                for k in range(j, n_feat):
                    cols.append(features[:, i] * features[:, j] * features[:, k])
    return np.column_stack(cols)


def linear_lstsq_r2(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot if ss_tot > 0 else 0.0), coef.tolist()


def fit_random_forest(features, target):
    """Random-forest regression R^2 using sklearn if available."""
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        return None
    rf = RandomForestRegressor(n_estimators=64, max_depth=8, random_state=0)
    rf.fit(features, target)
    pred = rf.predict(features)
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    return float(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)


def spectral_features(xi, k_modes: int = 12):
    """Top-k eigenvectors of L for global spectral projection."""
    xi_off = xi.copy()
    np.fill_diagonal(xi_off, 0.0)
    deg = xi_off.sum(axis=1)
    L = np.diag(deg) - xi_off
    eigvals, eigvecs = np.linalg.eigh(L)
    k = min(k_modes, eigvecs.shape[1])
    return eigvecs[:, 1:1 + k]


LADDER = [("P5", 50), ("P5N64", 64), ("P5N72", 72),
          ("P5N84", 84), ("P5N100", 100)]


def per_node_targets(ff_k, ff_q):
    if ff_k.ndim == 2:
        return ff_k.mean(axis=1), ff_q.mean(axis=1)
    return (np.full(ff_k.shape, float(np.mean(ff_k))),
            np.full(ff_q.shape, float(np.mean(ff_q))))


def main():
    out_path = OUTPUTS / "verify_slaving_richer_F.json"
    print("=" * 90)
    print("Iter-21: Slaving with richer-F aggregators (poly + RF + spectral)")
    print("=" * 90)
    print()
    print(f"{'regime':<8} {'N':>4} {'R2_linear_6':>12} {'R2_poly3':>10} "
          f"{'R2_RF':>8} {'R2_spec12_A':>13} {'R2_spec12_Q':>13}")
    print("-" * 80)
    rows = []
    for regime, n_lat in LADDER:
        snap = load_first_snapshot(regime, n_lat)
        if snap is None:
            print(f"{regime}: snapshot not found -- skip")
            continue
        xi, ff_k, ff_q = snap
        a_target, q_target = per_node_targets(ff_k, ff_q)
        if a_target.std() < 1e-12:
            # Constant target - all aggregators give R^2 = 0 by definition
            rows.append({"regime": regime, "N": n_lat,
                         "skip_reason": "target_is_constant"})
            print(f"{regime:<8} {n_lat:>4}  target const, skip")
            continue
        feats_base = base_features(xi)
        # Linear with intercept on 6 features
        Xlin = np.column_stack([np.ones(feats_base.shape[0]), feats_base])
        r2_lin_a, _ = linear_lstsq_r2(Xlin, a_target)
        # Polynomial degree 3
        Xpoly = polynomial_expand(feats_base, max_degree=3)
        r2_poly_a, _ = linear_lstsq_r2(Xpoly, a_target)
        # Random forest (sklearn)
        r2_rf_a = fit_random_forest(feats_base, a_target)
        # Spectral projection
        spec = spectral_features(xi, k_modes=12)
        Xspec = np.column_stack([np.ones(spec.shape[0]), spec])
        r2_spec_a, _ = linear_lstsq_r2(Xspec, a_target)
        r2_spec_q, _ = linear_lstsq_r2(Xspec, q_target)

        rows.append({
            "regime": regime, "N": int(n_lat),
            "R2_A_linear_6feat": float(r2_lin_a),
            "R2_A_polynomial_degree_3": float(r2_poly_a),
            "R2_A_random_forest": (
                float(r2_rf_a) if r2_rf_a is not None else None),
            "R2_A_spectral_top12": float(r2_spec_a),
            "R2_Q_spectral_top12": float(r2_spec_q),
            "n_features_polynomial": int(Xpoly.shape[1]),
            "n_modes_spectral": 12,
        })
        rf_str = f"{r2_rf_a:>8.3f}" if r2_rf_a is not None else "  (n/a)"
        print(f"{regime:<8} {n_lat:>4} {r2_lin_a:>12.3f} {r2_poly_a:>10.3f} "
              f"{rf_str} {r2_spec_a:>13.3f} {r2_spec_q:>13.3f}")

    if rows and all("R2_A_linear_6feat" in r for r in rows):
        avg_lin = float(np.mean([r["R2_A_linear_6feat"] for r in rows]))
        avg_poly = float(np.mean([r["R2_A_polynomial_degree_3"] for r in rows]))
        avg_spec = float(np.mean([r["R2_A_spectral_top12"] for r in rows]))
        rf_vals = [r["R2_A_random_forest"] for r in rows
                   if r.get("R2_A_random_forest") is not None]
        avg_rf = float(np.mean(rf_vals)) if rf_vals else None
        # Best aggregator
        candidates = {"linear_6feat": avg_lin,
                       "polynomial_3": avg_poly,
                       "spectral_top12": avg_spec}
        if avg_rf is not None:
            candidates["random_forest"] = avg_rf
        best = max(candidates, key=lambda k: candidates[k])
        verdict = (
            f"Best richer-F aggregator: {best} "
            f"(<R^2>_A = {candidates[best]:.3f}). "
            f"Comparison vs G4 baseline 4-param sigmoid (<R^2> ~ 0.10): "
            f"linear_6feat = {avg_lin:.3f}, poly_3 = {avg_poly:.3f}, "
            f"spectral_top12 = {avg_spec:.3f}"
            + (f", RF = {avg_rf:.3f}" if avg_rf is not None else "")
            + ". "
        )
        if candidates[best] > 0.7:
            verdict += (
                "RICHER-F captures variance: "
                f"the {best} aggregator achieves R^2 > 0.7, indicating "
                "the per-node K, Q variance is a non-trivial computable "
                "function of Xi-graph topology beyond simple degree."
            )
        elif candidates[best] > 0.4:
            verdict += (
                "RICHER-F PARTIALLY captures variance: improvement over "
                "naive sigmoid is non-trivial but does not saturate; "
                "additional state information beyond the local Xi-graph "
                "topology is likely required to fully reproduce ff_K, ff_Q."
            )
        else:
            verdict += (
                "RICHER-F INSUFFICIENT: even universal approximators "
                "(random forest + degree-3 polynomial + global spectral) "
                "fail to capture the per-node K, Q variance from Xi alone. "
                "ff_K, ff_Q must carry independent state information set "
                "by the regime (initialisation/RG stage), confirming the "
                "G4 Q2 finding that they are regime-constants invariant "
                "under seed-Xi fluctuations."
            )
    else:
        verdict = "INSUFFICIENT_DATA: snapshots missing or targets constant."

    bundle = {
        "method": (
            "Iter-21 slaving richer-F aggregator scan: tests four "
            "richer functional forms beyond the G4 4-param sigmoid -- "
            "linear-6-feature with intercept, polynomial degree-3 cross-"
            "terms, random-forest (depth=8, n_estimators=64), and global "
            "spectral projection on the top-12 Xi-Laplacian eigenvectors. "
            "Reports R^2 per regime to identify which aggregator (if any) "
            "captures the per-node K, Q variance from Xi-graph topology."
        ),
        "stand": "2026-05-05",
        "literature": [
            "Haken 1983 (Synergetics; slaving principle)",
            "Mori 1965 (Continued-fraction time-correlation functions)",
            "Zwanzig 2001 (Nonequilibrium statistical mechanics)",
            "Breiman 2001 (Random forests)",
        ],
        "rows": rows,
        "verdict": verdict,
    }
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\n{verdict[:300]}...")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
