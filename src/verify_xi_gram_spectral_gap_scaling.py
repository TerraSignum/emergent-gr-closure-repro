"""Direct spectral-gap N-scaling from d1 Xi-Gram eigenvalue spectra
across the cross-regime ladder P0..P8.

Provides a within-framework continuum-limit test of the
defect-excitation gap *without* refactoring the QFE/MS/DQC pipeline
modules (which are hardcoded to P1+P2'). For each regime:

  1. Load Xi matrices per-seed from the d1_p*.npz bundle
  2. Symmetrise + trace-normalise the Gram matrix
     G = (Xi @ Xi.T) / tr(Xi @ Xi.T)
  3. Diagonalise G via numpy.linalg.eigh
  4. Per-seed: extract descending-sorted eigenvalues
     - lambda_max (ground state)
     - lambda_1 (first excitation)
     - relative spectral gap g = (lambda_max - lambda_1) / lambda_max
     - log-spacing uniformity on top-6 eigenvalues
     - gap-ratio statistics on the full spectrum
  5. Across regimes: N-scaling of mean g, gap ratio classification
     Poisson(0.386) / GOE(0.530) / GUE(0.603), random control via
     phase-randomised Xi

Writes:
  data/xi_gram_spectral_gap_scaling.json

Honest framing
--------------
This is the spectrum of the Xi-Gram operator, which is a natural
self-adjoint construction on the lattice state. The "spectral gap"
here is the algebraic distance between the dominant mode and the
first excitation, *not* the Yang-Mills mass gap and *not* the
energy gap of a Hamiltonian in a QFT sense. The Wigner-Dyson
gap-ratio classification follows Atas-Bogomolny-Roux (2013).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from _d1_npz_discovery import find_d1_npz
from verify_galerkin_runner_A_hessian_ricci import edge_to_matrix

DATA = REPO / "data"
PARENT = REPO.parent

LADDER = [
    ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
    ("P4", 42), ("P5", 50), ("P6", 60), ("P7", 72), ("P8", 84),
]


def gram_eigenvalues(xi: np.ndarray) -> np.ndarray:
    """Trace-normalised Gram eigenvalues, descending.

    Uses SVD (numerically more robust than eigvalsh on
    ill-conditioned Gram matrices); returns the squared singular
    values divided by their sum (== Gram eigenvalues / trace(Gram)).
    """
    # Sanitize: replace NaN/Inf with zero
    xi_clean = np.where(np.isfinite(xi), xi, 0.0)
    if not np.any(np.abs(xi_clean) > 1e-15):
        return np.zeros(xi.shape[0])
    try:
        sv = np.linalg.svd(xi_clean, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.zeros(xi.shape[0])
    eigvals = sv ** 2
    s = float(np.sum(eigvals))
    if s < 1e-12:
        return np.zeros(xi.shape[0])
    return np.sort(eigvals / s)[::-1]


def per_seed_gap(eigvals: np.ndarray) -> dict:
    """Compute per-seed gap diagnostics from descending eigenvalues."""
    lam_max = float(eigvals[0])
    lam_1 = float(eigvals[1]) if eigvals.size > 1 else 0.0
    rel_gap = (lam_max - lam_1) / lam_max if lam_max > 0 else 0.0
    top6 = eigvals[: min(6, eigvals.size)]
    top6_pos = np.maximum(top6, 1e-12)
    log_spacing_std = float(np.std(np.diff(np.log10(top6_pos))))
    return {
        "lambda_max": lam_max,
        "lambda_1": lam_1,
        "relative_spectral_gap": float(rel_gap),
        "absolute_spectral_gap": float(lam_max - lam_1),
        "log_spacing_top6_std": log_spacing_std,
    }


def gap_ratio_classification(eigvals: np.ndarray) -> dict:
    """Atas-Bogomolny-Roux 2013 gap-ratio statistic.

    For an unfolded spectrum the mean of r_i = min(s_i, s_{i+1}) /
    max(s_i, s_{i+1}) is approximately:
        Poisson:  0.386 (regular/integrable)
        GOE:      0.530 (chaotic, time-reversal symmetric)
        GUE:      0.603 (time-reversal breaking)
    """
    eigs_asc = np.sort(eigvals)
    spacings = np.diff(eigs_asc)
    spacings = spacings[spacings > 1e-12]
    if spacings.size < 3:
        return {"gap_ratio_mean": None,
                "spectral_class": "INSUFFICIENT",
                "n_spacings": int(spacings.size)}
    ratios = []
    for k in range(spacings.size - 1):
        s1, s2 = spacings[k], spacings[k + 1]
        if max(s1, s2) > 0:
            ratios.append(min(s1, s2) / max(s1, s2))
    if not ratios:
        return {"gap_ratio_mean": None,
                "spectral_class": "EMPTY",
                "n_spacings": int(spacings.size)}
    r_mean = float(np.mean(ratios))
    if r_mean < 0.45:
        cls = "Poisson"
    elif r_mean < 0.57:
        cls = "GOE"
    else:
        cls = "GUE"
    return {"gap_ratio_mean": r_mean, "spectral_class": cls,
            "n_spacings": int(spacings.size),
            "n_ratios": len(ratios)}


def shuffled_control(xi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomise the Xi matrix while preserving symmetry and
    diagonal. Acts as null-hypothesis control for the gap-ratio test."""
    n = xi.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = xi[iu].copy()
    rng.shuffle(vals)
    M = np.zeros_like(xi)
    M[iu] = vals
    M = M + M.T
    np.fill_diagonal(M, 1.0)
    return M


def per_regime_audit(reg: str, n_lat: int, rng: np.random.Generator) -> dict:
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return {"regime": reg, "N": n_lat, "status": "NO_FILE"}
    z = np.load(p, allow_pickle=True)
    if "dense_cell_edge_xi_values" not in z.files:
        return {"regime": reg, "N": n_lat, "status": "NO_XI"}
    edge = z["dense_cell_edge_xi_values"]
    n_seeds = int(edge.shape[0])
    seed_results = []
    seed_gap_class = []
    seed_shuffle_class = []
    for s in range(n_seeds):
        xi = edge_to_matrix(edge[s], n_lat)
        np.fill_diagonal(xi, 1.0)
        eigvals = gram_eigenvalues(xi)
        seed_results.append(per_seed_gap(eigvals))
        seed_gap_class.append(gap_ratio_classification(eigvals))
        # Random control
        xi_shuf = shuffled_control(xi, rng)
        eigvals_shuf = gram_eigenvalues(xi_shuf)
        seed_shuffle_class.append(gap_ratio_classification(eigvals_shuf))
    rel_gaps = np.asarray([r["relative_spectral_gap"] for r in seed_results])
    abs_gaps = np.asarray([r["absolute_spectral_gap"] for r in seed_results])
    log_unif = np.asarray([r["log_spacing_top6_std"] for r in seed_results])
    gap_ratios = np.asarray([
        c["gap_ratio_mean"] for c in seed_gap_class
        if c["gap_ratio_mean"] is not None
    ])
    shuf_ratios = np.asarray([
        c["gap_ratio_mean"] for c in seed_shuffle_class
        if c["gap_ratio_mean"] is not None
    ])
    classes = [c["spectral_class"] for c in seed_gap_class]
    shuf_classes = [c["spectral_class"] for c in seed_shuffle_class]
    return {
        "regime": reg, "N": n_lat, "n_seeds": n_seeds,
        "status": "OK",
        "relative_spectral_gap": {
            "mean": float(np.mean(rel_gaps)),
            "std": float(np.std(rel_gaps)),
            "min": float(np.min(rel_gaps)),
            "max": float(np.max(rel_gaps)),
        },
        "absolute_spectral_gap": {
            "mean": float(np.mean(abs_gaps)),
            "std": float(np.std(abs_gaps)),
        },
        "log_spacing_top6_std": {
            "mean": float(np.mean(log_unif)),
            "std": float(np.std(log_unif)),
        },
        "gap_ratio_mean": {
            "mean": float(np.mean(gap_ratios)) if gap_ratios.size else None,
            "std": float(np.std(gap_ratios)) if gap_ratios.size else None,
            "n_seeds_valid": int(gap_ratios.size),
        },
        "spectral_classes": classes,
        "spectral_class_majority": (
            max(set(classes), key=classes.count) if classes else None
        ),
        "shuffled_gap_ratio_mean": {
            "mean": float(np.mean(shuf_ratios)) if shuf_ratios.size else None,
            "std": float(np.std(shuf_ratios)) if shuf_ratios.size else None,
        },
        "shuffled_spectral_class_majority": (
            max(set(shuf_classes), key=shuf_classes.count)
            if shuf_classes else None
        ),
    }


def loglog_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (y > 0) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    lx, ly = np.log(x[mask]), np.log(y[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = intercept + slope * lx
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return {"slope": float(slope), "intercept": float(intercept),
            "C_prefactor": float(np.exp(intercept)),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
            "n_points": int(mask.sum())}


def main() -> int:
    rng = np.random.default_rng(20260501)
    print("=" * 86)
    print("Xi-Gram spectral-gap N-scaling across the cross-regime ladder")
    print("=" * 86)

    rows = []
    for reg, n_lat in LADDER:
        r = per_regime_audit(reg, n_lat, rng)
        rows.append(r)
        if r["status"] != "OK":
            print(f"  {reg:>10} N={n_lat:>3}: {r['status']}")
            continue
        rg = r["relative_spectral_gap"]
        gr = r["gap_ratio_mean"]
        cls = r["spectral_class_majority"]
        scls = r["shuffled_spectral_class_majority"]
        gr_mean = (gr["mean"] if gr["mean"] is not None else float("nan"))
        print(f"  {reg:>10} N={n_lat:>3} (n_seeds={r['n_seeds']:>2}): "
              f"rel_gap={rg['mean']:.4f}+-{rg['std']:.4f}  "
              f"r_bar={gr_mean:.3f}  class={cls:>7}  shuf={scls:>7}")

    ok = [r for r in rows if r["status"] == "OK"]
    Ns = [r["N"] for r in ok]
    rel_gaps_mean = [r["relative_spectral_gap"]["mean"] for r in ok]
    abs_gaps_mean = [r["absolute_spectral_gap"]["mean"] for r in ok]
    grm = [
        r["gap_ratio_mean"]["mean"] for r in ok
        if r["gap_ratio_mean"]["mean"] is not None
    ]
    Ns_grm = [
        r["N"] for r in ok
        if r["gap_ratio_mean"]["mean"] is not None
    ]

    print()
    print("N-scaling of relative spectral gap:")
    fit_rel = loglog_fit(Ns, rel_gaps_mean)
    if fit_rel:
        print(f"  log-log slope = {fit_rel['slope']:+.4f}  "
              f"R^2 = {fit_rel['r2']:.3f}  ({fit_rel['n_points']} pts)")

    print()
    print("N-scaling of absolute spectral gap:")
    fit_abs = loglog_fit(Ns, abs_gaps_mean)
    if fit_abs:
        print(f"  log-log slope = {fit_abs['slope']:+.4f}  "
              f"R^2 = {fit_abs['r2']:.3f}  ({fit_abs['n_points']} pts)")

    print()
    print("N-scaling of Wigner-Dyson gap_ratio_mean r_bar:")
    if grm:
        print(f"  values:    {[round(v, 3) for v in grm]}")
        print(f"  cross-regime mean = {float(np.mean(grm)):.4f}, "
              f"std = {float(np.std(grm)):.4f}")
        print(f"  Poisson-cutoff (0.45): "
              f"{sum(1 for v in grm if v < 0.45)}/{len(grm)} below")
        print(f"  GOE-band (0.45-0.57): "
              f"{sum(1 for v in grm if 0.45 <= v < 0.57)}/{len(grm)}")
        print(f"  GUE-cutoff (>=0.57): "
              f"{sum(1 for v in grm if v >= 0.57)}/{len(grm)}")

    bundle = {
        "method": "xi_gram_spectral_gap_n_scaling",
        "title": ("Cross-regime Xi-Gram eigenvalue gap with "
                  "Wigner-Dyson gap-ratio classification on P0..P8."),
        "ladder_N": Ns,
        "per_regime": rows,
        "loglog_fit_relative_gap": fit_rel,
        "loglog_fit_absolute_gap": fit_abs,
        "gap_ratio_cross_regime_summary": {
            "regimes_with_valid_gr": Ns_grm,
            "values": grm,
            "mean": float(np.mean(grm)) if grm else None,
            "std": float(np.std(grm)) if grm else None,
        },
        "honest_caveats": [
            "Xi-Gram eigenvalues are not a Hamiltonian spectrum;"
            " the 'spectral gap' here is the algebraic mode-amplitude"
            " distance, not a dynamical mass gap.",
            "Atas-Bogomolny-Roux gap-ratio (2013) is computed on the"
            " full Xi-Gram spectrum without unfolding; classification"
            " thresholds (Poisson 0.386, GOE 0.530, GUE 0.603) are"
            " benchmarks, not statistical tests.",
            "Shuffled control phase-randomises off-diagonal Xi"
            " entries; non-trivial correlation should produce"
            " distinguishable spectral statistics.",
            "P5N64/P5N72/P5N84/P5N100 etc. are deliberately not"
            " included here because they share P5 physics and would"
            " bias the cross-regime ladder.",
        ],
    }
    out = DATA / "xi_gram_spectral_gap_scaling.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
