r"""SUPERSEDED (2026-05-14) -- NUMERICALLY VOID. See banner below.

Lemma B Step 4a (b)-side extension: family-factor spectral gap on
canonical d1 P5/P5N ladder.

================================ SUPERSEDED ================================
The family-coupling matrix M_F built as psi_g . Xi . psi_h, with generation
basis vectors psi_g formed from DISJOINT sets of orthogonal Xi-eigenvectors
({0,3,6}/{1,4,7}/{2,5,8}), is EXACTLY DIAGONAL in exact arithmetic: psi_i and
psi_j (i != j) span orthogonal eigenspaces, so psi_i . Xi . psi_j = 0. The
off-diagonal "coupling" entries are pure orthogonality round-off (~1e-16),
which the subsequent 1/sqrt(deg) normalisation (deg ~ 1e-15) amplifies back
to O(1). The reported lambda_2(M_F) = 7/6 is therefore the normalised
Laplacian of round-off noise -- under 1e-12 perturbations of Xi it scatters
by ~0.35.

Diagnosis + two corrected derivations (project the tau=0.10 skeleton instead
of Xi; genuine vertex-class graph quotient) are in
verify_lemma_B_equitable_partition.py -- NEITHER reproduces 7/6. The M_F /
family-coupling block has been removed from the P4 manuscript; only the real
asymptotes lambda_skel = 7/24 and lambda_w = 3/8 and the pure-algebra
identity 3/8 = (7/24)*(9/7) are retained. Kept for provenance only -- do not
cite its output.
============================================================================

Companion to `verify_lemma_B_family_factor_p1p2prime.py` which
falsified the conjectured family-factor gap = 1/N_gen on the small
P1-P5 family_phase_microscopic dataset (commit f5d5f35). Extends
the same family-coupling-matrix construction to the canonical d1
P5/P5N N-ordered ladder (N = 50, 64, 72, 84, 100, 128, 200, 256,
300, 512). Two possible outcomes:

(i) λ_2 ~ 1.21 (regime-stable continuation of P1-P5): further
    consolidates the falsification of the per-factor "family
    gap = 1/N_gen" interpretation.

(ii) λ_2 -> 1/N_gen = 1/3 as N → ∞ (Symanzik-1 finite-size
     convergence): would VINDICATE the per-factor interpretation
     at the continuum limit, attributing the P1-P5 deviation to
     finite-N effects.

The mode-level assignment heuristic on these snapshots (which do
not carry the GFS-01 pre-computed assignments) follows the same
9-mode pattern documented in the P1-P5 dataset:
- 9 lowest non-trivial Xi-Laplacian modes
- Group as (3 sectors) x (3 generations) = quark_g1, quark_g2,
  quark_g3, lepton_g1, ..., higher_g3
- Project Xi onto the per-generation 3-dim subspace
- Build the 3x3 family-coupling matrix and compute its normalised-
  Laplacian spectral gap.

Output: outputs/verify_lemma_B_family_factor_p5n_canonical.json
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
TARGET_GAP = 1 / N_GEN
N_MODES_USED = 9   # 3 sectors x 3 generations

LADDER = [
    ("P5",     50),
    ("P5N64",  64),
    ("P5N72",  72),
    ("P5N84",  84),
    ("P5N100", 100),
    ("P5N128", 128),
    ("P5N200", 200),
    ("P5N256", 256),
    ("P5N300", 300),
    ("P5N512", 512),
]


def load_xi_snapshots(regime, n_lat):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return []
    d = np.load(p, allow_pickle=True)
    snaps = []
    if "edge_xi_snapshots" in d.files:
        arr = d["edge_xi_snapshots"]
        if arr.ndim == 4:
            for s in range(arr.shape[0]):
                xi = np.asarray(arr[s, -1], dtype=float)
                if xi.shape == (n_lat, n_lat):
                    snaps.append(xi)
    return snaps


def family_coupling_3x3(xi):
    """Project Xi onto 3 generation-level basis vectors and build
    a 3x3 family-coupling matrix. Same construction as the P1-P5
    test: 9 lowest non-trivial modes grouped as 3 sectors x 3
    generations, with each generation receiving the AVERAGE of its
    3 sector-modes."""
    # Symmetrise (Xi may have small numerical asymmetry)
    xi_sym = 0.5 * (xi + xi.T)
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(xi_sym)
    # Sort by absolute eigenvalue descending (largest = most
    # carrier-action-active modes)
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Use the top N_MODES_USED = 9 (skip the Perron / trivial)
    # Note: Xi as bounded operator has no automatic "trivial" mode,
    # so we take all 9 highest-magnitude
    if len(eigvals) < N_MODES_USED:
        raise ValueError(f"need at least {N_MODES_USED} modes")
    # Group into 3 sectors x 3 generations
    # Use the same labelling as the family_phase dataset:
    # modes 0,1,2 -> quark gen 1,2,3
    # modes 3,4,5 -> lepton gen 1,2,3
    # modes 6,7,8 -> higher_mode gen 1,2,3
    gen_vectors = {}
    for g in range(1, N_GEN + 1):
        psi_g = np.zeros(eigvecs.shape[0])
        for sector_offset in (0, N_GEN, 2 * N_GEN):
            mode_idx = sector_offset + (g - 1)
            psi_g += eigvecs[:, mode_idx] * np.sqrt(
                max(abs(eigvals[mode_idx]), 1e-12))
        nrm = np.linalg.norm(psi_g)
        if nrm > 0:
            psi_g /= nrm
        gen_vectors[g] = psi_g
    M = np.zeros((N_GEN, N_GEN))
    for g in range(1, N_GEN + 1):
        psi_g = gen_vectors[g]
        for h in range(1, N_GEN + 1):
            psi_h = gen_vectors[h]
            M[g - 1, h - 1] = psi_g @ xi_sym @ psi_h
    return M


def normalised_laplacian_3x3(W):
    W = 0.5 * (W + W.T)
    W = W - np.diag(np.diag(W))
    W = np.abs(W)
    deg = W.sum(axis=1)
    deg_safe = np.where(deg > 0, deg, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(deg_safe)
    d_inv_sqrt[deg == 0] = 0.0
    n = W.shape[0]
    return np.eye(n) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])


def spectral_gap_norm_lap(M):
    L = normalised_laplacian_3x3(M)
    eigs = np.sort(np.linalg.eigvalsh(L))
    return float(eigs[1])


def fit_symanzik_1(xs, ys):
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
    print("Lemma B Step 4a (b) extension: family-factor spectral "
            "gap on canonical d1 P5/P5N ladder")
    print("=" * 100)
    print(f"d = {D}, N_gen = {N_GEN}, target 1/N_gen = {TARGET_GAP:.5f}")
    print(f"family-coupling construction: 9 highest-|eigenvalue| Xi modes "
            f"grouped 3 sectors x 3 generations")
    print()
    rows = []
    for regime, n_lat in LADDER:
        snaps = load_xi_snapshots(regime, n_lat)
        if not snaps:
            print(f"  [SKIP {regime}: no Xi snapshots]")
            continue
        gaps = []
        for xi in snaps:
            try:
                M = family_coupling_3x3(xi)
                gap = spectral_gap_norm_lap(M)
                gaps.append(gap)
            except (ValueError, np.linalg.LinAlgError):
                continue
        if not gaps:
            print(f"  [SKIP {regime}: no successful diagonalisations]")
            continue
        gaps_arr = np.asarray(gaps)
        mean_gap = gaps_arr.mean()
        sem_gap = gaps_arr.std() / np.sqrt(len(gaps))
        rows.append({
            "regime": regime,
            "N": n_lat,
            "n_seeds": len(gaps),
            "lambda_2_mean": float(mean_gap),
            "lambda_2_sem": float(sem_gap),
        })
        rel_err = (mean_gap - TARGET_GAP) / TARGET_GAP * 100
        rel_err_to_K3 = (mean_gap - 1.5) / 1.5 * 100
        print(f"{regime:<8} N={n_lat:>4d} n_seeds={len(gaps):>3d} "
                f"lambda_2 = {mean_gap:.5f} +- {sem_gap:.5f} "
                f"(vs 1/N_gen {rel_err:+5.1f}%, vs K_3=3/2 "
                f"{rel_err_to_K3:+5.1f}%)")
    print()
    if len(rows) >= 5:
        n_vals = [r["N"] for r in rows]
        y_vals = [r["lambda_2_mean"] for r in rows]
        a, b, r2 = fit_symanzik_1(n_vals, y_vals)
        print(f"Symanzik-1 a + b/N fit on 10-regime canonical ladder:")
        print(f"  a_inf  = {a:.5f}")
        print(f"  b      = {b:+.3f}")
        print(f"  R^2    = {r2:.4f}")
        print()
        print(f"a_inf vs candidate rationals:")
        candidates = [
            ("1/N_gen = 1/3", 1.0 / N_GEN),
            ("(1+gamma)^2 = 1.21", (1.1) ** 2),
            ("K_3 max = 3/2", 1.5),
            ("(d-1)/d = 3/4", 0.75),
            ("alpha_xi = 9/10", 0.9),
            ("alpha_xi + 1/N_gen = 1.233", 0.9 + 1.0 / N_GEN),
            ("d/N_gen = 4/3", 4.0 / 3),
            ("11/9", 11.0 / 9),
        ]
        for name, val in candidates:
            err = (a - val) / val * 100 if val != 0 else 0
            mark = " <- close" if abs(err) < 2 else ""
            print(f"  {name:<28} = {val:.5f}  rel err {err:+6.2f}%{mark}")
    print()

    bundle = {
        "method": "verify_lemma_B_family_factor_p5n_canonical",
        "stand": "2026-05-13",
        "d": D,
        "N_gen": N_GEN,
        "target_lambda_2": TARGET_GAP,
        "rows": rows,
        "symanzik_fit_a_inf": a if len(rows) >= 5 else None,
        "symanzik_fit_b": b if len(rows) >= 5 else None,
        "symanzik_fit_r2": r2 if len(rows) >= 5 else None,
    }
    out = OUTPUTS / "verify_lemma_B_family_factor_p5n_canonical.json"
    out.write_text(json.dumps(bundle, indent=2, default=float),
                       encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
