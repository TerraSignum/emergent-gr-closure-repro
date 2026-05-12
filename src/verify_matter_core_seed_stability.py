"""Matter-core / heavy-tail seed-stability analysis on the
extended-seed within-P5 ladder.

The heavy-tail of T_00 (top-decile nodes) is hypothesised to be
the matter-core of the lattice. Two diagnostics are computed
per regime:

  (i)  Top-decile Jaccard index across seed pairs:
       Jaccard(s, s') = |M_s ∩ M_s'| / |M_s ∪ M_s'|
       where M_s = top-10% T_00 nodes for seed s. A small Jaccard
       at fixed regime physics is the nucleation-picture
       prediction (matter cores form at stochastically distinct
       lattice positions per initial condition).

  (ii) Co-localisation of heavy-T_00 with heavy-Frobenius:
       Spearman rho between per-node T_00 and per-node Frobenius
       residual, computed seed-by-seed. A large positive rho
       means matter cores carry the closure-residual heavy tail.

Per-regime stability statistics:
   - mean Jaccard, std Jaccard (across seed pairs)
   - mean Spearman, std Spearman (across seeds)
   - coefficient of variation on top-decile T_00 magnitude

Reads the seed-richest file per regime via
verify_within_p5_extended_seeds_runner_A's selection logic.
Computes heavy-tail Jaccard + T_00/Frobenius rho per seed.

Writes:
  data/matter_core_seed_stability_within_p5.json
"""
from __future__ import annotations
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


class _BlockCupy:
    def find_module(self, name, path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self
    def load_module(self, name):
        raise ImportError("cupy disabled")
sys.meta_path.insert(0, _BlockCupy())

from verify_galerkin_runner_A_hessian_ricci import (
    edge_to_matrix, per_seed_galerkin)
from verify_within_p5_extended_seeds_runner_A import (
    find_best_npz, has_kq_persisted, load_seeds, LADDER)

DATA = REPO / "data"


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def per_seed_t00_and_frob(xi, psi, k, q, n_lat) -> tuple[np.ndarray, np.ndarray]:
    prep = per_seed_galerkin(xi, psi, k, q, n_lat, np)
    t00 = np.asarray(prep["t00"], dtype=float).flatten()
    g00 = np.asarray(prep["g_00_h"], dtype=float).flatten()
    # Direct Frobenius: |G_00 + Lambda_t - 8 pi G T_00| using fixed
    # System-R Lambda_t = 0.81. (Per-node 1x1 reduction; full 4x4
    # is the runner-A blind_frob_median; here we use the time-time
    # component only as the matter-core proxy.)
    frob_t = np.abs(g00 + 0.81 - t00)
    return t00, frob_t


def jaccard_pair(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def main() -> int:
    print("=" * 86)
    print("Matter-core / heavy-tail seed-stability across the within-P5 ladder")
    print("=" * 86)

    rows = []
    for reg, n_lat in LADDER:
        p, label = find_best_npz(reg, n_lat)
        if p is None or not has_kq_persisted(p):
            print(f"  {reg} N={n_lat}: SKIP (no usable file)")
            continue
        seeds = load_seeds(p, n_lat)
        if len(seeds) < 4:
            print(f"  {reg} N={n_lat} ({label}): only {len(seeds)} seeds")
        per_seed_data = []
        for xi, psi, k, q in seeds:
            t00, frob = per_seed_t00_and_frob(xi, psi, k, q, n_lat)
            n_top = max(1, int(0.10 * t00.size))
            top_idx = np.argsort(t00)[::-1][:n_top]
            per_seed_data.append({
                "t00": t00, "frob": frob,
                "top_decile_set": set(int(i) for i in top_idx),
                "top_decile_t00_max": float(t00[top_idx].max()),
                "top_decile_t00_mean": float(t00[top_idx].mean()),
                "spearman_t00_frob": spearman(t00, frob),
            })

        if len(per_seed_data) < 2:
            continue
        jaccards = [
            jaccard_pair(per_seed_data[i]["top_decile_set"],
                         per_seed_data[j]["top_decile_set"])
            for i, j in combinations(range(len(per_seed_data)), 2)
        ]
        rhos = [r["spearman_t00_frob"] for r in per_seed_data]
        top_means = [r["top_decile_t00_mean"] for r in per_seed_data]
        top_maxs = [r["top_decile_t00_max"] for r in per_seed_data]

        print(f"  {reg:<8} N={n_lat:>4} ({label:>9}, n_seeds={len(seeds):>2}):")
        print(f"    top-decile Jaccard:          mean={float(np.mean(jaccards)):.4f}  std={float(np.std(jaccards)):.4f}")
        print(f"    Spearman(T_00, Frob_time):   mean={float(np.mean(rhos)):.4f}  std={float(np.std(rhos)):.4f}")
        print(f"    top-decile T_00 magnitude:   mean={float(np.mean(top_means)):.3f}  CV={float(np.std(top_means)/np.mean(top_means)) if np.mean(top_means) > 0 else 0:.3f}")

        rows.append({
            "regime": reg, "N": n_lat,
            "source_label": label,
            "n_seeds": len(seeds),
            "n_pairs": len(jaccards),
            "jaccard_mean": float(np.mean(jaccards)),
            "jaccard_std": float(np.std(jaccards)),
            "jaccard_min": float(np.min(jaccards)),
            "jaccard_max": float(np.max(jaccards)),
            "spearman_t00_frob_mean": float(np.mean(rhos)),
            "spearman_t00_frob_std": float(np.std(rhos)),
            "top_decile_t00_mean_per_seed_mean": float(np.mean(top_means)),
            "top_decile_t00_mean_per_seed_cv":
                float(np.std(top_means) / np.mean(top_means))
                if np.mean(top_means) > 0 else 0.0,
            "top_decile_t00_max_per_seed_mean": float(np.mean(top_maxs)),
            "all_jaccards": jaccards,
            "all_spearmans_t00_frob": rhos,
        })

    bundle = {
        "method": "matter_core_seed_stability_within_p5",
        "title": ("Heavy-tail T_00 top-decile Jaccard and "
                  "T_00/Frobenius Spearman correlation per regime "
                  "on the extended-seed within-P5 ladder."),
        "interpretation": {
            "jaccard": ("low Jaccard = nucleation cores at "
                        "stochastically distinct positions per seed; "
                        "high Jaccard = lattice-pinned positions"),
            "spearman_t00_frob": ("positive Spearman = matter cores "
                                  "carry the closure residual heavy "
                                  "tail (matter-localised "
                                  "anisotropy)"),
        },
        "per_regime": rows,
    }
    out = DATA / "matter_core_seed_stability_within_p5.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
