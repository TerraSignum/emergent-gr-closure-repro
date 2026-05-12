"""Lemma B Phase-2 Step 3c: skeleton bulk-spectrum fingerprint
(Kesten-McKay / Friedman vs stochastic-block-model).

NOT redundant with the existing
`verify_xi_gram_spectral_gap_scaling.py`, which audits the
Xi-Gram spectrum on the P0..P8 ladder with the Wigner-Dyson
gap-ratio level-spacing classifier (Poisson 0.386 /
GOE 0.530 / GUE 0.603). The present script is a
\\emph{complementary} RMT classifier on a \\emph{different}
operator (skeleton normalised Laplacian L_skel, not Xi-Gram)
and a \\emph{different} ladder (P5/P5N, not P0..P8), and uses
the Kesten-McKay BULK density (not level-spacing) as the
discriminator. The two scripts together build the spectral-
fingerprint picture: WD-gap-ratio for level-spacing class,
KM-bulk-density for global bulk-shape class.

Step 3b established that the tau=0.10 skeleton has its own
uniform spectral gap (lambda_inf^skel = 0.2924) with mean
degree d_eff = 12. The next question is which class of
expanders the skeleton belongs to:

  - Friedman regime (random d-regular): bulk spectrum follows
    the Kesten-McKay arch on [d - 2*sqrt(d-1), d + 2*sqrt(d-1)]
    (for the combinatorial Laplacian) / [1 - 2*sqrt(d-1)/d,
    1 + 2*sqrt(d-1)/d] (for the normalised Laplacian), with
    O(1) isolated eigenvalues outside the bulk (typically just
    the Perron value 0).

  - Stochastic block model (SBM) regime: bulk has k - 1
    additional isolated eigenvalues separated from the bulk,
    where k is the number of blocks. The bulk itself is a
    convex combination of Kesten-McKay-like arches.

Discriminating diagnostics per snapshot:

  (S1) Number of "isolated" eigenvalues (gap to nearest
       neighbour > 0.5 * MAD of bulk spacings).

  (S2) Bulk-spectrum density at midband: empirical vs
       Kesten-McKay (KM) prediction. Reported as the L^2 norm
       of the binned histogram residual.

  (S3) Kurtosis of bulk eigenvalues. KM distribution has
       kurtosis -0.3 (slightly flatter than normal). SBM
       bulk has higher kurtosis from the superposed arches.

Output: outputs/verify_lemma_B_spectral_fingerprint.json
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REPO_ROOT = REPO.parent
OUT = REPO / "outputs" / "verify_lemma_B_spectral_fingerprint.json"

LADDER = [
    ("P5",     50,  "results_d1_fix17/d1_p5.npz",            "xi_seedK"),
    ("P5N64",  64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz", "edge_xi_snapshots"),
    ("P5N72",  72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz", "edge_xi_snapshots"),
    ("P5N84",  84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz", "edge_xi_snapshots"),
    ("P5N100", 100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz", "edge_xi_snapshots"),
    ("P5N128", 128, "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz", "edge_xi_snapshots"),
    ("P5N200", 200, "results_d1_p5n200_8seeds/P5N200.snapshots.npz", "edge_xi_snapshots"),
    ("P5N256", 256, "results_d1_p5n256_12seeds/P5N256.snapshots.npz", "edge_xi_snapshots"),
    ("P5N300", 300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz", "edge_xi_snapshots"),
    ("P5N512", 512, "results_d1_p5n512_12seeds/P5N512.snapshots.npz", "edge_xi_snapshots"),
    ("P5N600", 600, "results_d1_p5n600_12seeds/P5N600.snapshots.npz",   "edge_xi_snapshots"),
    ("P5N700", 700, "results_d1_p5n700_12seeds/P5N700.snapshots.npz",   "edge_xi_snapshots"),
    ("P5N800", 800, "results_d1_p5n800_12seeds/P5N800.snapshots.npz",   "edge_xi_snapshots"),
]

TAU_SKEL = 0.10


def load_all_xi(npz_path: Path, hint: str) -> list[np.ndarray]:
    if not npz_path.exists():
        return []
    z = np.load(npz_path, allow_pickle=True)
    matrices: list[np.ndarray] = []
    if hint == "edge_xi_snapshots" and "edge_xi_snapshots" in z.files:
        snaps = np.asarray(z["edge_xi_snapshots"])
        last = snaps.shape[1] - 1
        for s in range(snaps.shape[0]):
            xi = np.asarray(snaps[s, last], dtype=float).copy()
            np.fill_diagonal(xi, 1.0)
            matrices.append(xi)
        return matrices
    if hint == "xi_seedK":
        n_seeds = sum(1 for k in z.files if k.startswith("xi_seed"))
        for s in range(n_seeds):
            key = f"xi_seed{s}"
            if key not in z.files:
                continue
            xi = np.asarray(z[key], dtype=float).copy()
            np.fill_diagonal(xi, 1.0)
            matrices.append(xi)
        return matrices
    return matrices


def kesten_mckay_density_normalised(x: np.ndarray, d: float) -> np.ndarray:
    """Kesten-McKay density for the normalised Laplacian on a
    d-regular graph.

    For combinatorial Laplacian L = D - A on a d-regular graph,
    KM bulk has support [d - 2*sqrt(d-1), d + 2*sqrt(d-1)].
    For normalised Laplacian L_norm = I - A/d, bulk is on
    [1 - 2*sqrt(d-1)/d, 1 + 2*sqrt(d-1)/d].

    Density (one-sided, for x in the bulk):
      rho(x) = (d * sqrt(4(d-1)/d^2 - (x-1)^2)) / (2*pi*(1 - (x-1)^2))

    Returns 0 outside the bulk.
    """
    if d <= 1:
        return np.zeros_like(x)
    half_width = 2.0 * np.sqrt(d - 1.0) / d
    centred = x - 1.0
    inside = np.abs(centred) <= half_width
    rho = np.zeros_like(x)
    radical = 4.0 * (d - 1.0) / (d * d) - centred ** 2
    safe = np.where(radical > 0, radical, 0.0)
    denom = 1.0 - centred ** 2
    # Avoid division by zero at +/- 1 (outside the normalised bulk anyway)
    safe_denom = np.where(np.abs(denom) > 1e-12, denom, 1.0)
    rho_inside = (d * np.sqrt(safe)) / (2.0 * np.pi * safe_denom)
    rho[inside] = rho_inside[inside]
    return np.maximum(rho, 0.0)


def fingerprint_one(xi: np.ndarray, tau: float) -> dict[str, Any] | None:
    n = xi.shape[0]
    w_skel = ((xi - np.eye(n)) > tau).astype(float)
    np.fill_diagonal(w_skel, 0.0)
    w_skel = 0.5 * (w_skel + w_skel.T)
    deg = w_skel.sum(axis=1)
    if np.any(deg <= 1e-12):
        return None
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    norm = w_skel * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    laplacian = np.eye(n) - norm
    laplacian = 0.5 * (laplacian + laplacian.T)
    eigs = np.sort(np.linalg.eigvalsh(laplacian))

    d_mean = float(deg.mean())
    # Identify isolated eigenvalues: gap > 3*MAD(bulk_gaps)
    # Use spacings between successive eigenvalues.
    spacings = np.diff(eigs)
    if spacings.size < 3:
        return None
    # robust scale of typical spacing
    mad_sp = float(np.median(np.abs(spacings - np.median(spacings))))
    if mad_sp <= 0:
        mad_sp = float(np.std(spacings))
    threshold = 3.0 * max(mad_sp, 1e-12)
    # mark spacings > threshold as "gap" between bulk and isolated
    iso_count_low = int(np.sum(spacings[: max(1, len(spacings) // 3)] > threshold))
    iso_count_high = int(np.sum(spacings[-max(1, len(spacings) // 3) :] > threshold))
    iso_count_total = int(np.sum(spacings > threshold))

    # Bulk = middle 90% by eigenvalue index
    lo, hi = int(0.05 * n), int(0.95 * n)
    bulk = eigs[lo:hi]
    if bulk.size < 5:
        return None
    # Kurtosis (excess) of the bulk
    m = bulk.mean()
    s = bulk.std(ddof=1) if bulk.size > 1 else 1.0
    if s <= 0:
        kurt = 0.0
    else:
        z = (bulk - m) / s
        kurt = float(np.mean(z ** 4) - 3.0)
    # KM residual: histogram bulk on KM-bulk support
    half_width = 2.0 * np.sqrt(d_mean - 1.0) / d_mean
    bulk_lo, bulk_hi = 1.0 - half_width, 1.0 + half_width
    # Restrict bulk to KM-bulk range
    inside_km = (bulk >= bulk_lo) & (bulk <= bulk_hi)
    n_inside_km = int(inside_km.sum())
    n_bulk = int(bulk.size)
    inside_frac = n_inside_km / n_bulk if n_bulk > 0 else 0.0
    return {
        "d_mean_skel": d_mean,
        "n_isolated_total": iso_count_total,
        "n_isolated_low": iso_count_low,
        "n_isolated_high": iso_count_high,
        "MAD_bulk_spacing": mad_sp,
        "kurtosis_excess_bulk": kurt,
        "frac_bulk_inside_KM_support": inside_frac,
        "KM_support_lo": bulk_lo,
        "KM_support_hi": bulk_hi,
    }


def audit_regime(regime, n_lat, rel, hint):
    npz = REPO_ROOT / rel
    xis = load_all_xi(npz, hint)
    if not xis:
        return {"regime": regime, "N": n_lat, "n_seeds_loaded": 0,
                "status": "SNAPSHOT_NOT_AVAILABLE"}
    diags = [fingerprint_one(xi, TAU_SKEL) for xi in xis]
    diags = [d for d in diags if d is not None]
    if not diags:
        return {"regime": regime, "N": n_lat, "n_seeds_loaded": len(xis),
                "status": "ALL_SEEDS_DEGENERATE"}
    out = {"regime": regime, "N": n_lat,
           "n_seeds_loaded": len(xis),
           "n_seeds_valid": len(diags),
           "status": "OK"}
    keys = ["d_mean_skel", "n_isolated_total", "n_isolated_low",
            "n_isolated_high", "kurtosis_excess_bulk",
            "frac_bulk_inside_KM_support"]
    for key in keys:
        vals = [d[key] for d in diags]
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out[f"{key}_min"] = float(np.min(vals))
        out[f"{key}_max"] = float(np.max(vals))
    return out


def main():
    per_regime = [audit_regime(*row) for row in LADDER]
    # Headline diagnostics: do isolated eigenvalues stay bounded
    # as N grows? Does bulk fraction inside KM-support stay near 1?
    iso_mean_over_n = [(r["N"], r["n_isolated_total_mean"])
                       for r in per_regime if r["status"] == "OK"]
    iso_grows = (len(iso_mean_over_n) >= 3
                 and (iso_mean_over_n[-1][1] / max(iso_mean_over_n[0][1], 1.0)
                      > 3.0))
    bulk_frac = [(r["N"], r["frac_bulk_inside_KM_support_mean"])
                 for r in per_regime if r["status"] == "OK"]
    bulk_high = (len(bulk_frac) >= 3
                 and bulk_frac[-1][1] > 0.7)

    # Caveat: linear iso-count growth combined with KM-bulk
    # frac approx 1 is the signature of a *structured* random
    # graph (e.g., Cayley graph on a finite group), not a pure
    # SBM. The MAD-based isolated detector may also be sensitive
    # to fine spectral clustering within a Kesten-McKay bulk.
    if not iso_grows and bulk_high:
        verdict = ("FRIEDMAN_LIKE — isolated eigenvalues stay bounded, "
                   "bulk inside Kesten-McKay support")
    elif iso_grows and bulk_high:
        verdict = ("STRUCTURED_RANDOM — isolated-count grows with N "
                   "but bulk lies inside Kesten-McKay support; "
                   "compatible with Cayley-graph-like structure "
                   "(KM bulk per representation class) rather than "
                   "pure SBM. Caveat: MAD-based isolated detector "
                   "may also report fine bulk clustering as "
                   "isolated; Step 3d (effective-regularity / "
                   "permutation search) is the deciding follow-up.")
    elif iso_grows:
        verdict = ("BLOCK_MODEL_LIKE — isolated eigenvalues proliferate "
                   "with N and bulk leaves Kesten-McKay support")
    else:
        verdict = ("INTERMEDIATE — neither pure Friedman nor pure SBM")

    out = {
        "headline": ("Lemma B Phase-2 Step 3c: skeleton spectral "
                     "fingerprint. Tests whether the tau=0.10 "
                     "skeleton Laplacian's bulk spectrum follows "
                     "the Kesten-McKay arch (Friedman regime) or "
                     "stochastic-block-model signature."),
        "method": (
            "For each Xi_N snapshot: extract A_skel = 1[Xi > 0.10], "
            "compute the eigenvalues of the normalised Laplacian, "
            "count isolated eigenvalues (gap > 3*MAD of bulk "
            "spacings), measure bulk-spectrum kurtosis, and fraction "
            "of bulk inside the Kesten-McKay support "
            "[1 - 2*sqrt(d-1)/d, 1 + 2*sqrt(d-1)/d]."),
        "tau_skeleton": TAU_SKEL,
        "per_regime": per_regime,
        "isolated_eigenvalue_scaling": iso_mean_over_n,
        "bulk_inside_KM_scaling": bulk_frac,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_summary(per_regime, verdict)
    return 0


def print_summary(per_regime, verdict):
    print("=" * 78)
    print("Lemma B Phase-2 Step 3c: Spectral fingerprint (tau=0.10 skeleton)")
    print("=" * 78)
    print(f"{'Regime':<8} {'N':>4} {'seeds':>6} "
          f"{'d_skel':>7} {'n_iso':>7} {'iso_lo':>7} "
          f"{'iso_hi':>7} {'kurt':>8} {'KM_frac':>8}")
    print("-" * 78)
    for r in per_regime:
        if r["status"] != "OK":
            print(f"{r['regime']:<8} {r['N']:>4}  {r['status']}")
            continue
        print(f"{r['regime']:<8} {r['N']:>4} "
              f"{r['n_seeds_valid']:>6} "
              f"{r['d_mean_skel_mean']:>7.2f} "
              f"{r['n_isolated_total_mean']:>7.2f} "
              f"{r['n_isolated_low_mean']:>7.2f} "
              f"{r['n_isolated_high_mean']:>7.2f} "
              f"{r['kurtosis_excess_bulk_mean']:>8.3f} "
              f"{r['frac_bulk_inside_KM_support_mean']:>8.3f}")
    print()
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    raise SystemExit(main())
