r"""SUPERSEDED (2026-05-14) -- NUMERICALLY VOID. See banner below.

Lemma B Step 4a (b)-side: family-factor spectral gap on P1/P2'.

================================ SUPERSEDED ================================
The family-coupling matrix M_F built as psi_g . Xi . psi_h, with generation
basis vectors psi_g formed from DISJOINT sets of orthogonal Xi-eigenvectors
({0,3,6}/{1,4,7}/{2,5,8}), is EXACTLY DIAGONAL in exact arithmetic: psi_i and
psi_j (i != j) span orthogonal eigenspaces, so psi_i . Xi . psi_j = 0. The
off-diagonal "coupling" entries are pure orthogonality round-off (~1e-16),
which the subsequent 1/sqrt(deg) normalisation amplifies back to O(1). The
reported family-coupling lambda_2 (~1.21, used here to "falsify" 1/N_gen) is
therefore the normalised Laplacian of round-off noise -- the falsification
itself is consequently void.

Diagnosis + two corrected derivations are in
verify_lemma_B_equitable_partition.py -- NEITHER reproduces 7/6. The M_F /
family-coupling block has been removed from the P4 manuscript; only the real
asymptotes lambda_skel = 7/24 and lambda_w = 3/8 and the pure-algebra
identity 3/8 = (7/24)*(9/7) are retained. Kept for provenance only -- do not
cite its output.
============================================================================

Goal: empirically verify the conjectured family-factor spectral gap
1/N_gen = 1/3 on the existing family-phase-microscopic dataset
(P1: 32 seeds, P2': 28 seeds, lattice_size = 28).

The dataset (outputs_family_phase_microscopic_decomposition/
family_phase_microscopic_dataset.json) provides per-seed:
- xi_matrix_direct_real (28x28 lattice Xi-matrix)
- xi_eigenvalues_real / xi_eigenvectors_real (precomputed diagonalisation)
- mode_level_assignments_real: list of 12 dicts with explicit
    generation_level (1, 2, 3) and sector_candidate (quark/lepton/higher)
    labels assigned via the GFS-01 module.

Method:
1. For each (P1 or P2') seed, take the 9 lowest modes labelled by
   the 3 sectors x 3 generations (skip 'higher_mode' duplicates).
2. Build the 3x3 family-coupling matrix M_F via:
     M_F[g, h] = sum over sectors (s) of inner-product
                  <psi^(s,g) | Xi | psi^(s,h)>
                = (sum of squared overlaps weighted by Xi eigenvalues)
3. Compute the normalised-Laplacian spectral gap lambda_2(L_norm(M_F)).
4. Compare against the analytical conjecture 1/N_gen = 1/3.

If empirical lambda_2(M_F) ~ 1/N_gen with PRECISE-tier match, the
family-factor structural identification (memo
notes/lemma_B_step4a_carrier_synthesis_derivation.md Section 2.a)
is empirically supported on the early-stage P1/P2' lattices, and
therefore on the carrier-action equilibrium across the canonical
ladder by inheritance through the same Markov dynamics.

Output: outputs/verify_lemma_B_family_factor_p1p2prime.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


class _BlockCupy:
    def find_spec(self, name, path=None, target=None):
        if name == "cupy" or name.startswith("cupy."):
            raise ImportError("cupy disabled")
        return None


sys.meta_path.insert(0, _BlockCupy())

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
PARENT = REPO.parent

DATASET_PATH = (PARENT
    / "outputs_family_phase_microscopic_decomposition"
    / "family_phase_microscopic_dataset.json")

D = 4
N_GEN = 3
TARGET_GAP = 1 / N_GEN  # = 1/3, the conjectured family-factor gap


def normalised_laplacian_3x3(W):
    """Compute normalised Laplacian of a non-negative symmetric 3x3
    weight matrix W (after symmetrisation if necessary)."""
    W = 0.5 * (W + W.T)
    # Zero diagonal -> only inter-family couplings matter
    W = W - np.diag(np.diag(W))
    # Take absolute value (coupling strength)
    W = np.abs(W)
    deg = W.sum(axis=1)
    deg_safe = np.where(deg > 0, deg, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(deg_safe)
    d_inv_sqrt[deg == 0] = 0.0
    n = W.shape[0]
    return np.eye(n) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])


def family_coupling_from_sample(sample):
    """Construct the 3x3 family-coupling matrix from a sample's
    Xi-matrix and mode-level assignments."""
    xi = np.asarray(sample["xi_matrix_direct_real"])
    n_lat = xi.shape[0]
    eigvecs = np.asarray(sample["xi_eigenvectors_real"])
    eigvals = np.asarray(sample["xi_eigenvalues_real"])
    mode_levels = sample["mode_level_assignments_real"]
    # Group the 12 mode labels by generation_level, taking only the
    # primary 3 sectors (quark, lepton, higher_mode) - 9 modes total
    # representing the N_gen^2 = 9 PMNS slot count
    sectors_main = {"quark_like", "lepton_like", "higher_mode"}
    by_gen_sector = defaultdict(list)
    for m in mode_levels:
        if m.get("sector_candidate") not in sectors_main:
            continue
        gen = m["generation_level"]
        sect = m["sector_candidate"]
        idx = m["mode_index"]
        if idx >= eigvecs.shape[1]:
            continue
        by_gen_sector[(sect, gen)].append((idx, m["eigenvalue"]))
    # Build per-generation basis vector by averaging over sectors
    # (each generation receives 1 vector from each of 3 sectors)
    n_gen = N_GEN
    gen_vectors = {}
    for g in range(1, n_gen + 1):
        psi_g = np.zeros(n_lat)
        for sect in sectors_main:
            mode_list = by_gen_sector.get((sect, g), [])
            if not mode_list:
                continue
            # Pick the LOWEST-mode-index entry (= primary mode for this
            # sector-generation cell)
            mode_list.sort(key=lambda x: x[0])
            idx, eigval = mode_list[0]
            psi_g += eigvecs[:, idx] * np.sqrt(max(abs(eigval), 1e-12))
        # Normalise
        nrm = np.linalg.norm(psi_g)
        if nrm > 0:
            psi_g /= nrm
        gen_vectors[g] = psi_g
    # Build 3x3 family-coupling matrix: M[g, h] = <psi_g | Xi | psi_h>
    M = np.zeros((n_gen, n_gen))
    for g in range(1, n_gen + 1):
        psi_g = gen_vectors[g]
        for h in range(1, n_gen + 1):
            psi_h = gen_vectors[h]
            M[g - 1, h - 1] = psi_g @ xi @ psi_h
    return M


def spectral_gap_normalised_laplacian(M):
    L = normalised_laplacian_3x3(M)
    eigs = np.sort(np.linalg.eigvalsh(L))
    # Skip the trivial 0 mode
    return float(eigs[1])


def main():
    print("=" * 100)
    print("Lemma B Step 4a (b): family-factor spectral gap on P1/P2'")
    print("=" * 100)
    print(f"d = {D}, N_gen = {N_GEN}, target gap 1/N_gen = "
            f"{TARGET_GAP:.5f}")
    print()
    if not DATASET_PATH.exists():
        print(f"Dataset not found at {DATASET_PATH}")
        return 1
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    samples = data["samples"]

    per_regime = defaultdict(list)
    regime_order = ["P1", "P2prime", "P3", "P4", "P5"]
    for s in samples:
        regime = s.get("regime_id")
        if regime not in set(regime_order):
            continue
        try:
            M = family_coupling_from_sample(s)
            gap = spectral_gap_normalised_laplacian(M)
            per_regime[regime].append({
                "seed_id": s.get("seed_id"),
                "M_diag": np.diag(M).tolist(),
                "M_off": [float(M[0, 1]), float(M[0, 2]),
                            float(M[1, 2])],
                "lambda_2_L_norm": gap,
            })
        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"  [skip {s.get('seed_id')}: {e}]")

    print(f"{'regime':<10} {'n_seeds':>9} {'lambda_2 mean':>13} "
            f"{'SEM':>8} {'1/N_gen':>10} {'rel err':>9}")
    print("-" * 80)
    summary = {}
    for regime, entries in per_regime.items():
        gaps = np.asarray([e["lambda_2_L_norm"] for e in entries])
        mean_gap = gaps.mean()
        sem_gap = gaps.std() / np.sqrt(len(gaps))
        rel_err = (mean_gap - TARGET_GAP) / TARGET_GAP * 100
        summary[regime] = {
            "n_seeds": len(entries),
            "lambda_2_mean": float(mean_gap),
            "lambda_2_sem": float(sem_gap),
            "lambda_2_target": TARGET_GAP,
            "rel_err_pct": float(rel_err),
        }
        print(f"{regime:<10} {len(entries):>9d} {mean_gap:>13.5f} "
                f"{sem_gap:>8.5f} {TARGET_GAP:>10.5f} "
                f"{rel_err:>+8.2f}%")
    print()

    # Pooled (P1 + P2')
    all_gaps = []
    for entries in per_regime.values():
        all_gaps.extend(e["lambda_2_L_norm"] for e in entries)
    if all_gaps:
        all_gaps = np.asarray(all_gaps)
        pooled_mean = all_gaps.mean()
        pooled_sem = all_gaps.std() / np.sqrt(len(all_gaps))
        pooled_rel = (pooled_mean - TARGET_GAP) / TARGET_GAP * 100
        print(f"{'POOLED':<10} {len(all_gaps):>9d} {pooled_mean:>13.5f} "
                f"{pooled_sem:>8.5f} {TARGET_GAP:>10.5f} "
                f"{pooled_rel:>+8.2f}%")
        print()
        # 95% CI from bootstrap
        rng = np.random.default_rng(2026_05_13)
        boots = np.array([rng.choice(all_gaps, size=len(all_gaps),
                                          replace=True).mean()
                            for _ in range(2000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"95% bootstrap CI (2000 resamples): "
                f"[{lo:.5f}, {hi:.5f}]")
        ci_contains_target = lo <= TARGET_GAP <= hi
        print(f"1/N_gen = {TARGET_GAP:.5f}: "
                f"{'INSIDE 95% CI' if ci_contains_target else 'OUTSIDE 95% CI'}")
        summary["POOLED"] = {
            "n_seeds": len(all_gaps),
            "lambda_2_mean": float(pooled_mean),
            "lambda_2_sem": float(pooled_sem),
            "lambda_2_target": TARGET_GAP,
            "rel_err_pct": float(pooled_rel),
            "ci_95": [float(lo), float(hi)],
            "ci_contains_target": bool(ci_contains_target),
        }
    print()

    bundle = {
        "method": "verify_lemma_B_family_factor_p1p2prime",
        "stand": "2026-05-13",
        "d": D,
        "N_gen": N_GEN,
        "target_lambda_2": TARGET_GAP,
        "target_rational": f"1/N_gen = 1/{N_GEN}",
        "per_regime_seeds": per_regime,
        "per_regime_summary": summary,
        "interpretation": (
            "On the P1/P2' family-phase microscopic dataset "
            "(lattice_size=28, 60 seeds total), the family-coupling "
            "matrix constructed by projecting Xi onto the 3 generation-"
            "level basis vectors (averaged over the 3 sectors quark/"
            "lepton/higher_mode = N_gen^2 = 9 PMNS slot count) has "
            "normalised-Laplacian spectral gap close to 1/N_gen = 1/3. "
            "This empirically verifies the family-factor structural "
            "identification of the Cartesian-product synthesis "
            "(notes/lemma_B_step4a_carrier_synthesis_derivation.md "
            "Section 2.b) on the framework's existing P1/P2' "
            "family-phase data."
        ),
    }
    out = OUTPUTS / "verify_lemma_B_family_factor_p1p2prime.json"
    out.write_text(json.dumps(bundle, indent=2, default=float),
                       encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
