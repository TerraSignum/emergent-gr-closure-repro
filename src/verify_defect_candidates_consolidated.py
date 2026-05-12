"""Three additional defect-candidate hypotheses for the heavy-tail
per-node residual location, after the vortex-winding hypothesis
was falsified (lift~0.83, z~-0.3).

Hypotheses tested here:

  H_triangle: heavy-tail nodes sit at triangle-defect concentrations
              (metric strain: places where d_ij + d_jk < d_ik
              violates triangle inequality before the fast-slow
              flow of Section sec:a1 stabilises the metric tube).
              This is the per-node frustration field 𝔣 from the
              P4 manuscript residual-action L_res.

  H_walls: heavy-tail nodes sit at domain-wall-connected regions
              (top decile of wall_map row-sum).

  H_T00: heavy-tail nodes ARE the locations of high source-tensor
              energy density T_00. This is the user's matter-
              localisation intuition tested directly with the
              energy density, not with topological winding.

Each hypothesis is tested with the same overlap-statistic
methodology as the falsified vortex-winding test, against the
top-decile per-node Frobenius residual mask.

Output: outputs/defect_candidates_consolidated_audit.json
"""
from __future__ import annotations
import json
import sys
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

from _d1_npz_discovery import find_d1_npz
from verify_galerkin_runner_A_hessian_ricci import (
    edge_to_matrix, per_seed_galerkin)


LADDER_REGIMES = [
    ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
    ("P4", 42), ("P5", 50), ("P6", 60), ("P5N64", 64),
    ("P7", 72), ("P8", 84), ("P5N100", 100),
]


def per_node_residual_struct(prep, xp):
    g_00 = prep["g_00_h"]
    g_ij = prep["g_ij_h"]
    t00 = prep["t00"]
    t_ij = prep["t_ij"]
    eye3 = prep["eye3"]
    LAMBDA_T = 0.81
    LAMBDA_S = -0.005
    res00 = g_00 + LAMBDA_T - t00
    spatial_res = (g_ij + LAMBDA_S * eye3[None, :, :]) - t_ij
    sq = res00 ** 2 + (spatial_res ** 2).sum(axis=(1, 2))
    return xp.sqrt(sq)


def overlap_stats(top_mask, defect_mask):
    n = len(top_mask)
    n_top = int(top_mask.sum())
    n_def = int(defect_mask.sum())
    n_both = int((top_mask & defect_mask).sum())
    if n == 0 or n_top == 0 or n_def == 0:
        return {
            "n_total": n, "n_top": n_top, "n_defect": n_def,
            "n_overlap": n_both, "expected_random": float("nan"),
            "lift": float("nan"), "z_score": float("nan"),
        }
    expected = n_top * n_def / n
    var = expected * (1 - n_top / n) * (1 - n_def / n)
    z = (n_both - expected) / max(np.sqrt(var), 1e-12)
    return {
        "n_total": n, "n_top": n_top, "n_defect": n_def,
        "n_overlap": n_both,
        "expected_random": float(expected),
        "lift": float(n_both / max(expected, 1e-12)),
        "z_score": float(z),
    }


def per_node_triangle_defect_count(triangle_defects_flat_seed, n_lat):
    """Reshape (n_lat^3,) flat tensor into (n_lat, n_lat, n_lat)
    triangle indicator and count, for each node a, how many
    triangles (a, j, k) have nonzero defect entry."""
    arr = np.asarray(triangle_defects_flat_seed)
    if arr.size != n_lat ** 3:
        return None
    cube = arr.reshape(n_lat, n_lat, n_lat)
    # Count nonzero defect-flagged triangles incident to each node a
    # by summing over (j, k) for each a.
    return (cube != 0).sum(axis=(1, 2)).astype(np.float64)


def analyse_regime(regime, n_lat):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    n_seeds = min(edge_arr.shape[0], 32)

    has_triangle = "triangle_defects_flat" in d.files
    has_walls = "wall_map" in d.files
    triangle_flat = d["triangle_defects_flat"] if has_triangle else None
    wall = d["wall_map"] if has_walls else None
    # triangle_defects_flat is shape (n_seeds * n_lat^3,)? or
    # (n_seeds, n_lat^3)? Determine from total size.
    if has_triangle:
        per_seed_size = triangle_flat.size // n_seeds
        # If per_seed_size == n_lat^3, fine. Else skip.
        if per_seed_size != n_lat ** 3:
            has_triangle = False

    pooled = {
        "residual": [],
        "triangle_count": [],
        "wall_degree": [],
        "t00": [],
    }
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        residual = np.asarray(per_node_residual_struct(prep, np))
        t00 = np.asarray(prep["t00"])

        pooled["residual"].append(residual)
        pooled["t00"].append(t00)

        if has_triangle:
            seed_chunk = triangle_flat[
                s * n_lat ** 3 : (s + 1) * n_lat ** 3]
            tc = per_node_triangle_defect_count(seed_chunk, n_lat)
            if tc is None:
                tc = np.zeros(n_lat)
            pooled["triangle_count"].append(tc)
        else:
            pooled["triangle_count"].append(np.zeros(n_lat))

        if has_walls and wall.ndim == 3:
            wd = np.asarray(wall[s]).sum(axis=1)
        else:
            wd = np.zeros(n_lat)
        pooled["wall_degree"].append(wd)

    res = np.concatenate(pooled["residual"])
    tri = np.concatenate(pooled["triangle_count"])
    wdeg = np.concatenate(pooled["wall_degree"])
    t00 = np.concatenate(pooled["t00"])

    p90_res = np.percentile(res, 90)
    top_mask = res >= p90_res

    p90_tri = np.percentile(tri, 90) if tri.max() > 0 else np.inf
    top_tri_mask = (tri >= p90_tri) if np.isfinite(p90_tri) \
        else np.zeros_like(tri, dtype=bool)
    p90_wdeg = np.percentile(wdeg, 90) if wdeg.max() > 0 else np.inf
    top_wall_mask = (wdeg >= p90_wdeg) if np.isfinite(p90_wdeg) \
        else np.zeros_like(wdeg, dtype=bool)
    p90_t00 = np.percentile(t00, 90)
    top_t00_mask = t00 >= p90_t00

    rec = {
        "regime": regime,
        "N": n_lat,
        "n_nodes_total": int(len(res)),
        "has_triangle_data": has_triangle,
        "has_wall_data": has_walls,
        "overlap_top10_residual_vs_top10_triangle_defect":
            overlap_stats(top_mask, top_tri_mask),
        "overlap_top10_residual_vs_top10_wall_degree":
            overlap_stats(top_mask, top_wall_mask),
        "overlap_top10_residual_vs_top10_t00":
            overlap_stats(top_mask, top_t00_mask),
        "summary_means_at_top10_residual": {
            "triangle_defect_top10": float(tri[top_mask].mean()),
            "triangle_defect_bot90": float(tri[~top_mask].mean()),
            "wall_degree_top10": float(wdeg[top_mask].mean()),
            "wall_degree_bot90": float(wdeg[~top_mask].mean()),
            "t00_top10": float(t00[top_mask].mean()),
            "t00_bot90": float(t00[~top_mask].mean()),
        },
    }
    return rec


def main():
    results = []
    print("=" * 130)
    print("Three additional defect-candidate hypotheses for the heavy-tail per-node residual:")
    print("  H_triangle: top10 residual vs top10 per-node triangle-defect count")
    print("  H_walls:    top10 residual vs top10 wall_map degree")
    print("  H_T00:      top10 residual vs top10 source energy density T_00")
    print("=" * 130)
    print()
    print(f"{'reg':<8} {'N':>3} | "
          f"{'triangle: lift':>14} {'z':>5} | "
          f"{'walls:  lift':>13} {'z':>5} | "
          f"{'T00:   lift':>12} {'z':>5}")
    print("-" * 100)
    for reg, n_lat in LADDER_REGIMES:
        rec = analyse_regime(reg, n_lat)
        if rec is None:
            continue
        results.append(rec)
        ot = rec["overlap_top10_residual_vs_top10_triangle_defect"]
        ow = rec["overlap_top10_residual_vs_top10_wall_degree"]
        oe = rec["overlap_top10_residual_vs_top10_t00"]
        tri_disp = f"{ot['lift']:>14.2f} {ot['z_score']:>+5.1f}" \
            if not np.isnan(ot.get('lift', np.nan)) else f"{'(no data)':>20}"
        wall_disp = f"{ow['lift']:>13.2f} {ow['z_score']:>+5.1f}" \
            if not np.isnan(ow.get('lift', np.nan)) else f"{'(no data)':>19}"
        t00_disp = f"{oe['lift']:>12.2f} {oe['z_score']:>+5.1f}"
        print(f"{reg:<8} {n_lat:>3} | {tri_disp} | {wall_disp} | {t00_disp}")

    print()
    print("=" * 130)
    print("VERDICT")
    print("=" * 130)

    def avg_lift(key):
        vals = [r[key]["lift"] for r in results
                if not np.isnan(r[key].get("lift", np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    def avg_z(key):
        vals = [r[key]["z_score"] for r in results
                if not np.isnan(r[key].get("z_score", np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"  Average lift / z across regimes:")
    print(f"    triangle-defect: lift={avg_lift('overlap_top10_residual_vs_top10_triangle_defect'):.2f}, "
          f"z={avg_z('overlap_top10_residual_vs_top10_triangle_defect'):+.1f}")
    print(f"    wall-degree:    lift={avg_lift('overlap_top10_residual_vs_top10_wall_degree'):.2f}, "
          f"z={avg_z('overlap_top10_residual_vs_top10_wall_degree'):+.1f}")
    print(f"    T_00 energy:    lift={avg_lift('overlap_top10_residual_vs_top10_t00'):.2f}, "
          f"z={avg_z('overlap_top10_residual_vs_top10_t00'):+.1f}")
    print()
    print(f"  T_00 mean ratios at top10-residual vs bot90-residual:")
    for r in results:
        s = r["summary_means_at_top10_residual"]
        ratio = s["t00_top10"] / max(s["t00_bot90"], 1e-12)
        print(f"    {r['regime']:<8} N={r['N']:>3}: T_00 top10={s['t00_top10']:.4f}, "
              f"bot90={s['t00_bot90']:.4f}, ratio={ratio:.3f}")

    out_path = REPO / "outputs" / "defect_candidates_consolidated_audit.json"
    out = {
        "method": "consolidated_three_defect_candidate_overlap_audit",
        "schema_version": "1.0.0",
        "candidates_tested": [
            "per-node triangle-defect count (metric strain / frustration)",
            "per-node wall_map row-sum (domain-wall connectivity)",
            "per-node T_00 (source energy density, matter-localisation test)",
        ],
        "structural_lambda": {"t": 0.81, "s": -0.005},
        "per_regime": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
