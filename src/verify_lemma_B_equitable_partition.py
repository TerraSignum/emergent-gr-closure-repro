r"""Lemma B Route 6 + M_F-construction diagnosis and corrected derivation.

This script does three things.

(1) DIAGNOSIS of the legacy family-coupling matrix. The legacy
    construction (verify_lemma_B_family_factor_p5n_canonical.py and
    siblings) builds generation basis vectors psi_g from DISJOINT
    sets of orthogonal Xi-eigenvectors -- psi_1 from modes {0,3,6},
    psi_2 from {1,4,7}, psi_3 from {2,5,8} -- and sets
    M_F[g,h] = psi_g . Xi . psi_h. Because Xi is diagonal in its own
    eigenbasis and the mode sets are disjoint, M_F is EXACTLY
    DIAGONAL in exact arithmetic: the off-diagonal "coupling" is
    pure orthogonality round-off (~1e-16), which the subsequent
    1/sqrt(deg) normalisation (deg ~ 1e-15) amplifies back to O(1).
    The legacy "lambda_2(M_F) = 7/6" is therefore the normalised
    Laplacian of round-off noise. This script reports
    max|M_F_offdiag| and the perturbation sensitivity to make the
    degeneracy explicit.

(2) CORRECTED family-coupling derivation. The fix: project the
    tau=0.10 SKELETON A_skel = 1[Xi > tau] -- which is nonlinear in
    Xi and hence NOT diagonal in Xi's eigenbasis -- instead of Xi
    itself. M_F_skel[g,h] = psi_g . A_skel . psi_h has genuine
    off-diagonal structure. Reported: spectrum, lambda_2,
    (1/d)*lambda_2 vs the directly measured skeleton gap.

(3) EQUITABLE-PARTITION route (Route 6 proper). A genuine 3-class
    VERTEX partition of the skeleton (spectral clustering on the
    three lowest non-trivial L_skel eigenvectors) gives a genuine
    graph quotient B (B_ij = mean skeleton weight class i -> class
    j). Reported: equitability defect (per-vertex within-class
    degree-to-class CV) vs a 20-permutation random-partition null,
    quotient spectrum, lambda_2(quotient), and the (1/d)-dilution
    relation against the directly measured skeleton gap.

Verdict refers to the CORRECTED constructions only. Per the
governing rule, a result enters the manuscript only on success;
the legacy M_F block is reported as numerically void regardless.

Output: outputs/verify_lemma_B_equitable_partition.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REPO_ROOT = REPO.parent
sys.path.insert(0, str(REPO / "src"))
OUTPUTS = REPO / "outputs"

from _d1_npz_discovery import find_d1_npz  # noqa: E402

D = 4
N_GEN = 3
TAU = 0.10
N_MODES_USED = 9
N_RANDOM_PERM = 20
MAX_SEEDS_PER_REGIME = 8
RNG_SEED = 42

LADDER = [
    ("P5", 50), ("P5N64", 64), ("P5N72", 72), ("P5N84", 84),
    ("P5N100", 100), ("P5N128", 128), ("P5N200", 200),
    ("P5N256", 256), ("P5N300", 300), ("P5N512", 512),
]

TARGET_FAMILY_GAP = 7.0 / 6.0    # legacy claimed lambda_2(M_F)
TARGET_SKEL_GAP = 7.0 / 24.0     # (1/d) * 7/6 ; also direct skeleton claim


def load_xi_snapshots(regime: str, n_lat: int):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return []
    d = np.load(p, allow_pickle=True)
    snaps = []
    if "edge_xi_snapshots" in d.files:
        arr = d["edge_xi_snapshots"]
        if arr.ndim == 4:
            for s in range(min(MAX_SEEDS_PER_REGIME, arr.shape[0])):
                xi = np.asarray(arr[s, -1], dtype=float)
                if xi.shape == (n_lat, n_lat):
                    snaps.append(xi)
        return snaps
    for s in range(MAX_SEEDS_PER_REGIME):
        key = f"xi_seed{s}"
        if key not in d.files:
            break
        xi = np.asarray(d[key], dtype=float)
        if xi.shape == (n_lat, n_lat):
            snaps.append(xi)
    return snaps


def _offdiag_nonneg(m: np.ndarray) -> np.ndarray:
    w = np.asarray(m, dtype=float).copy()
    w = 0.5 * (w + w.T)
    np.fill_diagonal(w, 0.0)
    return np.maximum(w, 0.0)


def norm_laplacian(w: np.ndarray):
    """Normalised Laplacian of a non-negative symmetric weight matrix
    (diagonal already zeroed). Returns the full sorted spectrum, or
    None if any vertex is isolated."""
    n = w.shape[0]
    if n < 3:
        return None
    deg = w.sum(axis=1)
    if np.any(deg <= 1e-12):
        return None
    dis = 1.0 / np.sqrt(deg)
    lap = np.eye(n) - (w * dis[:, None] * dis[None, :])
    lap = 0.5 * (lap + lap.T)
    return np.sort(np.linalg.eigvalsh(lap))


def generation_basis(xi: np.ndarray):
    """Legacy generation basis vectors psi_g: 9 highest-|eigenvalue|
    Xi modes grouped 3 sectors x 3 generations, each generation the
    sqrt(|eig|)-weighted sum of its 3 sector modes. Returns the
    (n x 3) psi matrix (columns normalised)."""
    xi_sym = 0.5 * (xi + xi.T)
    eigvals, eigvecs = np.linalg.eigh(xi_sym)
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    if len(eigvals) < N_MODES_USED:
        raise ValueError("need >= 9 modes")
    psi = np.zeros((eigvecs.shape[0], N_GEN))
    for g in range(N_GEN):
        v = np.zeros(eigvecs.shape[0])
        for sector_off in (0, N_GEN, 2 * N_GEN):
            m = sector_off + g
            v += eigvecs[:, m] * np.sqrt(max(abs(eigvals[m]), 1e-12))
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v /= nrm
        psi[:, g] = v
    return psi


def spectral_3_partition(skel: np.ndarray):
    """Genuine 3-class vertex partition of the skeleton: k-means
    (Lloyd, fixed init) on the three lowest non-trivial normalised
    -Laplacian eigenvectors. Returns an integer class vector."""
    n = skel.shape[0]
    deg = skel.sum(axis=1)
    if np.any(deg <= 1e-12):
        return None
    dis = 1.0 / np.sqrt(deg)
    lap = np.eye(n) - (skel * dis[:, None] * dis[None, :])
    lap = 0.5 * (lap + lap.T)
    _, vecs = np.linalg.eigh(lap)
    feat = vecs[:, 1:1 + N_GEN]                       # spectral embedding
    # row-normalise (standard Ng-Jordan-Weiss)
    rn = np.linalg.norm(feat, axis=1, keepdims=True)
    feat = feat / np.where(rn > 1e-12, rn, 1.0)
    # deterministic k-means: init at the N_GEN most mutually distant rows
    centres = [int(np.argmax(np.linalg.norm(feat - feat.mean(0), axis=1)))]
    for _ in range(N_GEN - 1):
        d2 = np.min([np.linalg.norm(feat - feat[c], axis=1)
                     for c in centres], axis=0)
        centres.append(int(np.argmax(d2)))
    C = feat[centres].copy()
    classes = np.zeros(n, dtype=int)
    for _ in range(50):
        dists = np.stack([np.linalg.norm(feat - C[k], axis=1)
                          for k in range(N_GEN)], axis=1)
        new = np.argmin(dists, axis=1)
        if np.array_equal(new, classes):
            break
        classes = new
        for k in range(N_GEN):
            if np.any(classes == k):
                C[k] = feat[classes == k].mean(0)
    return classes


def equitability_cv(skel: np.ndarray, classes: np.ndarray):
    """Mean coefficient of variation of per-vertex within-class
    skeleton degree-to-class-j. Equitable partition <=> -> 0."""
    cv = []
    for i in range(N_GEN):
        vi = np.where(classes == i)[0]
        if vi.size < 2:
            continue
        for j in range(N_GEN):
            vj = np.where(classes == j)[0]
            if vj.size == 0:
                continue
            dj = skel[np.ix_(vi, vj)].sum(axis=1)
            m = dj.mean()
            cv.append(dj.std() / m if m > 1e-12 else 0.0)
    return float(np.mean(cv)) if cv else np.nan


def quotient_matrix(skel: np.ndarray, classes: np.ndarray):
    """B_ij = mean over v in class i of skeleton degree v -> class j."""
    B = np.zeros((N_GEN, N_GEN))
    for i in range(N_GEN):
        vi = np.where(classes == i)[0]
        if vi.size == 0:
            return None
        for j in range(N_GEN):
            vj = np.where(classes == j)[0]
            B[i, j] = skel[np.ix_(vi, vj)].sum(axis=1).mean() if vj.size else 0.0
    return B


def audit_snapshot(xi: np.ndarray, rng: np.random.Generator):
    w = _offdiag_nonneg(xi)
    n = w.shape[0]
    skel = (w > TAU).astype(float)
    spec_skel = norm_laplacian(skel)
    if spec_skel is None:
        return None
    lam_skel = float(spec_skel[1])

    try:
        psi = generation_basis(xi)
    except (ValueError, np.linalg.LinAlgError):
        return None

    # (1) legacy M_F = psi^T Xi psi  -- diagonal by construction
    xi_sym = 0.5 * (xi + xi.T)
    M_legacy = psi.T @ xi_sym @ psi
    legacy_offdiag_max = float(np.abs(M_legacy - np.diag(np.diag(M_legacy))).max())

    # (2) corrected M_F = psi^T A_skel psi  -- genuine off-diagonal
    M_skel = psi.T @ skel @ psi
    skel_offdiag_max = float(np.abs(M_skel - np.diag(np.diag(M_skel))).max())
    spec_MFskel = norm_laplacian(_offdiag_nonneg(M_skel))
    lam_MFskel = float(spec_MFskel[1]) if spec_MFskel is not None else None

    # (3) equitable-partition vertex quotient of the skeleton
    classes = spectral_3_partition(skel)
    quo = {}
    if classes is not None:
        sizes = [int((classes == g).sum()) for g in range(N_GEN)]
        if min(sizes) >= 2:
            B = quotient_matrix(skel, classes)
            spec_B = norm_laplacian(_offdiag_nonneg(B)) if B is not None else None
            cv = equitability_cv(skel, classes)
            null_cv = []
            for _ in range(N_RANDOM_PERM):
                perm = rng.permutation(n)
                rc = np.empty(n, dtype=int)
                a = 0
                for g, sz in enumerate(sizes):
                    rc[perm[a:a + sz]] = g
                    a += sz
                null_cv.append(equitability_cv(skel, rc))
            null_cv = np.asarray(null_cv)
            quo = {
                "class_sizes": sizes,
                "lambda_quotient": (float(spec_B[1])
                                    if spec_B is not None else None),
                "spec_quotient": (None if spec_B is None
                                  else [float(x) for x in spec_B]),
                "equitability_cv": cv,
                "random_cv_mean": float(null_cv.mean()),
                "cv_z_vs_random": (
                    float((cv - null_cv.mean()) / null_cv.std())
                    if null_cv.std() > 1e-9 else None),
            }

    return {
        "n": n,
        "lambda_skel": lam_skel,
        "legacy_MF_offdiag_max": legacy_offdiag_max,
        "corrected_MFskel_offdiag_max": skel_offdiag_max,
        "lambda_MFskel": lam_MFskel,
        "spec_MFskel": (None if spec_MFskel is None
                        else [float(x) for x in spec_MFskel]),
        "dilution_pred_MFskel": (None if lam_MFskel is None
                                 else lam_MFskel / D),
        **quo,
    }


def fit_symanzik_1(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    msk = np.isfinite(ys)
    if msk.sum() < 3:
        return None, None, None
    invN = 1.0 / xs[msk]
    y = ys[msk]
    A = np.column_stack([np.ones_like(invN), invN])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    ssr = float(((y - pred) ** 2).sum())
    r2 = 1.0 - ssr / sst if sst > 0 else 0.0
    return float(beta[0]), float(beta[1]), r2


def main() -> int:
    print("=" * 100)
    print("Lemma B: M_F-construction diagnosis + corrected derivation "
          "+ equitable-partition route")
    print("=" * 100)
    print(f"d = {D}, N_gen = {N_GEN}, tau = {TAU}")
    print()

    rng = np.random.default_rng(RNG_SEED)
    per_regime = []
    for regime, n_lat in LADDER:
        snaps = load_xi_snapshots(regime, n_lat)
        if not snaps:
            print(f"  [SKIP {regime}: no Xi snapshots]")
            continue
        rows = [audit_snapshot(xi, rng) for xi in snaps]
        rows = [r for r in rows if r is not None]
        if not rows:
            print(f"  [SKIP {regime}: no usable snapshots]")
            continue

        def amean(key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        agg = {
            "regime": regime, "N": n_lat, "n_seeds": len(rows),
            "lambda_skel": amean("lambda_skel"),
            "legacy_MF_offdiag_max": amean("legacy_MF_offdiag_max"),
            "corrected_MFskel_offdiag_max": amean("corrected_MFskel_offdiag_max"),
            "lambda_MFskel": amean("lambda_MFskel"),
            "dilution_pred_MFskel": amean("dilution_pred_MFskel"),
            "lambda_quotient": amean("lambda_quotient"),
            "equitability_cv": amean("equitability_cv"),
            "random_cv_mean": amean("random_cv_mean"),
            "cv_z_vs_random": amean("cv_z_vs_random"),
        }
        per_regime.append(agg)
        print(f"{regime:<8} N={n_lat:>4d} n={len(rows):>2d} | "
              f"lam_skel={agg['lambda_skel']:.4f} | "
              f"legacy|offdiag|={agg['legacy_MF_offdiag_max']:.1e} | "
              f"MF_skel: lam2={agg['lambda_MFskel']!s:>7.7} "
              f"(1/d)lam2={agg['dilution_pred_MFskel']!s:>7.7} | "
              f"quot: lam2={agg['lambda_quotient']!s:>7.7} "
              f"CV={agg['equitability_cv']:.3f} "
              f"CVrand={agg['random_cv_mean']:.3f} "
              f"z={agg['cv_z_vs_random']!s:>6.6}")

    if len(per_regime) < 5:
        print("\nInsufficient regimes for Symanzik extrapolation.")
        return 1

    Ns = [r["N"] for r in per_regime]
    fits = {}
    for key in ("lambda_skel", "lambda_MFskel", "dilution_pred_MFskel",
                "lambda_quotient", "equitability_cv", "random_cv_mean"):
        a, b, r2 = fit_symanzik_1(Ns, [r[key] for r in per_regime])
        fits[key] = {"a_inf": a, "b": b, "r2": r2}

    print()
    print("Symanzik-1 (a + b/N) continuum extrapolations:")
    for key, f in fits.items():
        if f["a_inf"] is None:
            continue
        print(f"  {key:<26} a_inf = {f['a_inf']:+.5f}  "
              f"b = {f['b']:+.3f}  R^2 = {f['r2']:.3f}")

    legacy_max = max(r["legacy_MF_offdiag_max"] for r in per_regime)
    print()
    print(f"Legacy M_F max |off-diagonal| across all regimes/seeds: "
          f"{legacy_max:.2e}  (machine eps ~2.2e-16 -> diagonal by "
          f"construction; legacy lambda_2(M_F) is round-off noise)")

    # ---- candidate-rational comparison for the corrected objects ----
    def nearest(val):
        if val is None:
            return None
        cands = {
            "7/6": 7 / 6, "7/24": 7 / 24, "3/8": 3 / 8, "1/3": 1 / 3,
            "1/4": 1 / 4, "1/2": 1 / 2, "2/3": 2 / 3, "3/4": 3 / 4,
            "9/10": 9 / 10, "1": 1.0, "6/5": 6 / 5, "5/4": 5 / 4,
        }
        best = min(cands.items(), key=lambda kv: abs(val - kv[1]))
        return {"rational": best[0], "value": best[1],
                "rel_err_pct": 100.0 * (val - best[1]) / best[1]}

    print()
    print("Corrected constructions vs nearest 5-smooth rational:")
    for key in ("lambda_MFskel", "dilution_pred_MFskel", "lambda_quotient"):
        a = fits[key]["a_inf"]
        nr = nearest(a)
        print(f"  {key:<26} a_inf = {a!s:>9.9}  "
              f"nearest {nr['rational']} ({nr['value']:.5f}), "
              f"rel err {nr['rel_err_pct']:+.2f}%" if nr else
              f"  {key}: n/a")

    bundle = {
        "method": "verify_lemma_B_equitable_partition",
        "stand": "2026-05-14",
        "d": D, "N_gen": N_GEN, "tau": TAU,
        "legacy_MF_diagnosis": {
            "construction": "psi_g . Xi . psi_h with psi_g from "
                            "disjoint orthogonal Xi-eigenvector sets "
                            "{0,3,6}/{1,4,7}/{2,5,8}",
            "max_offdiag_all": float(legacy_max),
            "status": "DIAGONAL_BY_CONSTRUCTION_legacy_lambda2_is_roundoff",
        },
        "corrected_derivations": {
            "MF_skel": "psi_g . A_skel . psi_h  (project the tau=0.10 "
                       "skeleton, nonlinear in Xi -> genuine off-diagonal)",
            "vertex_quotient": "spectral-clustering 3-partition of the "
                               "skeleton; B_ij = mean class-i->class-j "
                               "weight (genuine graph quotient)",
        },
        "per_regime": per_regime,
        "symanzik_fits": fits,
    }
    out = OUTPUTS / "verify_lemma_B_equitable_partition.json"
    out.write_text(json.dumps(bundle, indent=2, default=float),
                   encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
