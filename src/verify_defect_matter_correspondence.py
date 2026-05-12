"""Direct test of the user-proposed hypothesis:

  the 10% heavy-tail per-node residual nodes correspond to
  topological vortex defect cores, i.e. the discrete locations
  where matter (in the framework's DM-vortex source decomposition)
  emerges.

For each regime + seed we have:
  - per-node 4x4 Galerkin Frobenius residual (struct Lambda)
  - per-node winding_map (vortex winding number per node)
  - per-node wall_map (domain-wall connectivity)
  - per-node carrier amplitude (|psi|)

The test: form the top-decile-residual mask (10% highest residual
per regime+seed) and the topological-defect mask (|winding| > 0
per node), then compute the overlap statistic.

If overlap >> 0.10 (random expectation), the heavy-tail tracks
the topological-defect locations and the framework's DM-vortex
source IS what we have been calling 'heavy-tail'.

We also test against:
  - top-decile of |psi| amplitude departure from regime-mean
    (alternative 'matter-density' proxy)
  - top-decile of wall_map row sum (domain-wall presence)

Output: outputs/defect_matter_correspondence_audit.json
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
    """For two binary masks, compute overlap counts and the
    chi-square / standard-error overlap z-score under random
    independence."""
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


def analyse_regime(regime, n_lat):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)

    if "winding_map" not in d.files:
        return {"regime": regime, "N": n_lat, "skipped": "no winding_map"}

    edge_arr = d["dense_cell_edge_xi_values"]
    amp_arr = d["dense_cell_node_amplitude_values"]
    phase_arr = d["dense_cell_node_phase_values"]
    winding = d["winding_map"]  # shape (n_seeds, n_lat)
    wall = d["wall_map"] if "wall_map" in d.files else None
    n_seeds = min(edge_arr.shape[0], winding.shape[0], 32)

    pooled = {"residual": [], "winding": [], "wall_degree": [],
              "amp_dev": []}
    for s in range(n_seeds):
        xi_mat = edge_to_matrix(edge_arr[s], n_lat)
        np.fill_diagonal(xi_mat, 1.0)
        psi = amp_arr[s] * np.exp(1j * phase_arr[s])
        k_field = d.get(f"ff_K_seed{s}", np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}", np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
        residual = np.asarray(per_node_residual_struct(prep, np))

        wnum = np.asarray(winding[s])
        if wall is not None and wall.ndim == 3:
            w_deg = np.asarray(wall[s]).sum(axis=1)
        else:
            w_deg = np.zeros(n_lat)
        amp = np.abs(psi)
        amp_dev = np.abs(amp - amp.mean())

        pooled["residual"].append(residual)
        pooled["winding"].append(wnum)
        pooled["wall_degree"].append(w_deg)
        pooled["amp_dev"].append(amp_dev)

    res = np.concatenate(pooled["residual"])
    wnd = np.concatenate(pooled["winding"])
    wdeg = np.concatenate(pooled["wall_degree"])
    amp_dev = np.concatenate(pooled["amp_dev"])

    p90_res = np.percentile(res, 90)
    top_mask = res >= p90_res

    nonzero_winding_mask = np.abs(wnd) > 1e-9
    p90_wdeg = np.percentile(wdeg, 90) if wdeg.max() > 0 else np.inf
    top_wall_mask = wdeg >= p90_wdeg if np.isfinite(p90_wdeg) else np.zeros_like(wdeg, dtype=bool)
    p90_amp = np.percentile(amp_dev, 90)
    top_amp_dev_mask = amp_dev >= p90_amp

    rec = {
        "regime": regime,
        "N": n_lat,
        "n_nodes_total": int(len(res)),
        "winding_summary": {
            "n_nonzero_total": int(nonzero_winding_mask.sum()),
            "fraction_nonzero": float(nonzero_winding_mask.mean()),
            "max_abs_winding": float(np.abs(wnd).max()),
            "winding_distinct_values": int(len(np.unique(wnd))),
        },
        "overlap_top10_residual_vs_nonzero_winding":
            overlap_stats(top_mask, nonzero_winding_mask),
        "overlap_top10_residual_vs_top10_walldegree":
            overlap_stats(top_mask, top_wall_mask),
        "overlap_top10_residual_vs_top10_ampdev":
            overlap_stats(top_mask, top_amp_dev_mask),
    }
    # Direct mean residual at vortex-defect nodes vs non-defect nodes
    if nonzero_winding_mask.any() and (~nonzero_winding_mask).any():
        rec["mean_residual_at_vortex_nodes"] = float(
            res[nonzero_winding_mask].mean())
        rec["mean_residual_at_non_vortex_nodes"] = float(
            res[~nonzero_winding_mask].mean())
        rec["vortex_vs_non_vortex_residual_ratio"] = (
            rec["mean_residual_at_vortex_nodes"]
            / max(rec["mean_residual_at_non_vortex_nodes"], 1e-12))
    return rec


def main():
    results = []
    print("=" * 130)
    print("Defect <-> matter-emergence correspondence test:")
    print("Do the 10% heavy-tail per-node residual nodes overlap with")
    print("topological vortex / wall / amplitude-anomaly nodes?")
    print("=" * 130)
    print()
    print(f"{'reg':<8} {'N':>3} | {'wind frac':>9} {'max|w|':>7} | "
          f"{'res top10 vs nonzero-winding':>29} {'vs top10 wall-deg':>20} {'vs top10 amp-dev':>17} |"
          f" {'vortex/non ratio':>17}")
    print("-" * 130)
    for reg, n_lat in LADDER_REGIMES:
        rec = analyse_regime(reg, n_lat)
        if rec is None or rec.get("skipped"):
            continue
        results.append(rec)
        ws = rec["winding_summary"]
        ov_w = rec["overlap_top10_residual_vs_nonzero_winding"]
        ov_l = rec["overlap_top10_residual_vs_top10_walldegree"]
        ov_a = rec["overlap_top10_residual_vs_top10_ampdev"]
        vr = rec.get("vortex_vs_non_vortex_residual_ratio", float("nan"))
        print(f"{reg:<8} {n_lat:>3} | "
              f"{ws['fraction_nonzero']:>9.3f} {ws['max_abs_winding']:>7.0f} | "
              f"lift={ov_w['lift']:>5.2f} z={ov_w['z_score']:>+5.1f}{' ':6} "
              f"lift={ov_l['lift']:>5.2f} z={ov_l['z_score']:>+5.1f} "
              f"lift={ov_a['lift']:>5.2f} z={ov_a['z_score']:>+5.1f} | "
              f"{vr:>17.3f}")

    print()
    print("=" * 130)
    print("VERDICT")
    print("=" * 130)
    valid = [r for r in results if "vortex_vs_non_vortex_residual_ratio" in r]
    if valid:
        avg_lift_winding = float(np.mean([
            r["overlap_top10_residual_vs_nonzero_winding"]["lift"]
            for r in valid]))
        avg_z_winding = float(np.mean([
            r["overlap_top10_residual_vs_nonzero_winding"]["z_score"]
            for r in valid]))
        avg_lift_wall = float(np.mean([
            r["overlap_top10_residual_vs_top10_walldegree"]["lift"]
            for r in valid if not np.isnan(r["overlap_top10_residual_vs_top10_walldegree"]["lift"])]))
        avg_z_wall = float(np.mean([
            r["overlap_top10_residual_vs_top10_walldegree"]["z_score"]
            for r in valid if not np.isnan(r["overlap_top10_residual_vs_top10_walldegree"]["z_score"])]))
        avg_lift_amp = float(np.mean([
            r["overlap_top10_residual_vs_top10_ampdev"]["lift"]
            for r in valid]))
        avg_ratio = float(np.mean([
            r["vortex_vs_non_vortex_residual_ratio"] for r in valid]))
        print(f"  Average overlap-lift across regimes:")
        print(f"    top10-residual vs nonzero-winding: lift={avg_lift_winding:.2f}, z={avg_z_winding:+.1f}")
        print(f"    top10-residual vs top10-walldegree: lift={avg_lift_wall:.2f}, z={avg_z_wall:+.1f}")
        print(f"    top10-residual vs top10-amp-dev:    lift={avg_lift_amp:.2f}")
        print(f"  Mean vortex/non-vortex residual ratio: {avg_ratio:.3f}")
        print()
        print("  Interpretation:")
        print("    If lift > 1.5 and |z| > 3 systematically for nonzero-winding,")
        print("    the heavy-tail per-node residual is concentrated at topological")
        print("    vortex defect cores -> the 10% IS the discrete location of")
        print("    matter (DM-vortex source) in the relational lattice.")
        print("    If lift ~ 1 (random), the heavy-tail is NOT topological matter.")

    out_path = REPO / "outputs" / "defect_matter_correspondence_audit.json"
    out = {
        "method": "heavy_tail_residual_vs_topological_defect_overlap",
        "schema_version": "1.0.0",
        "structural_lambda": {"t": 0.81, "s": -0.005},
        "per_regime": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
