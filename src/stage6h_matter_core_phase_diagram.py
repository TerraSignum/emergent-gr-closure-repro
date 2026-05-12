"""Stage 6h: Matter-core percentile phase diagram.

Layer-nested matter-core characterisation on the cleaned eleven-regime
canonical-physics ladder N in [50,512] (Pipeline-A snapshots).  For
each regime we compute per-node Delta_a (per-node relative-Frobenius
residual under structural Lambda_t = alpha_xi^2, isotropic Lambda_s),
then define five nested core layers by percentile threshold:

  C^(p95)  = {a : Delta_a > a_95}    bulk-edge / source-active support
  C^(p99)  = {a : Delta_a > a_99}    transition shell
  C^(p99.5)= {a : Delta_a > a_99.5}  matter-core mantle
  C^(p99.9)= {a : Delta_a > a_99.9}  core spine outer
  C^(sup)  = {a : Delta_a >= max-eps} core spine maximum

For each layer we report:
  - support fraction mu_N(C^X) = |C^X|/N
  - core mass M(C^X) = sum_a |psi_a|^2 over a in C^X (matter readout)
  - Jaccard overlap with the |T_00| top decile, top-5%, top-1%
  - Jaccard overlap with K-field top-5%, Q-field top-5%
  - sign distribution of R_time (R_00) on the layer
  - layer nesting: C^(99.9) subset C^(99.5) subset C^(99) subset C^(95)

Output: outputs/stage6h_matter_core_phase_diagram.json
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

    def load_module(self, name):
        raise ImportError("cupy disabled")


sys.meta_path.insert(0, _BlockCupy())

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from _d1_npz_discovery import find_d1_npz  # noqa: E402
from stage6f_full_tensor_norm_audit import (  # noqa: E402
    LADDER, LAMBDA_T, LAMBDA_S, load_canonical, load_snapshots,
    per_node_relative_delta,
)
from verify_galerkin_runner_A_hessian_ricci import per_seed_galerkin  # noqa: E402
from verify_per_eigendirection_residual import (  # noqa: E402
    per_node_eigendirection_residuals,
)


PERCENTILE_LAYERS = [
    ("C95", 95.0, "bulk-edge / source-active heavy-tail support"),
    ("C99", 99.0, "transition shell"),
    ("C99_5", 99.5, "matter-core mantle"),
    ("C99_9", 99.9, "core spine outer"),
]


def _gather_regime(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    if "snapshots" in p.name.lower():
        seeds = load_snapshots(p, n_lat)
    else:
        seeds = load_canonical(p, n_lat)
    return seeds


def _node_quantities(xi_mat, psi, k_field, q_field, n_lat):
    """Per-node observables on a single seed."""
    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
    res = per_node_eigendirection_residuals(prep, LAMBDA_T, LAMBDA_S)
    R_time = res["R_time"]
    R_diag = res["R_diag"]
    R_off = res["R_off"]
    t_eigs = res["T_eigvals"]
    t00 = np.asarray(prep["t00"])
    R_norm = np.sqrt(R_time ** 2
                      + (R_diag ** 2).sum(axis=1)
                      + R_off ** 2)
    T_norm = np.sqrt(t00 ** 2 + (t_eigs ** 2).sum(axis=1))
    delta = R_norm / np.maximum(T_norm, 1e-12)

    # Per-node K(x), Q(x) row-mean (matches P4-B sec:matter_loading
    # convention: K_field is a 2D matrix, K(x) is the row-mean over j)
    if k_field.ndim == 2:
        # Exclude self-coupling on the diagonal for the row-mean
        eye = np.eye(n_lat, dtype=bool)
        k_row_mean = np.where(eye, 0.0, k_field).sum(axis=1) / max(n_lat - 1, 1)
        q_row_mean = np.where(eye, 0.0, q_field).sum(axis=1) / max(n_lat - 1, 1)
    else:
        k_row_mean = np.asarray(k_field).flatten()
        q_row_mean = np.asarray(q_field).flatten()

    # Xi-row-sum degree on the multiplicative Xi-graph (P4-B
    # sec:matter_loading convention: top-5% selected by xi_row_sum,
    # i.e. matter-cluster cores with strong synchronisation neighbours)
    xi_no_diag = xi_mat.copy()
    np.fill_diagonal(xi_no_diag, 0.0)
    xi_degree = xi_no_diag.sum(axis=1)
    # Defect-degree: opposite selection on (1-Xi) (low Xi = strong defect)
    defect_degree = (1.0 - xi_no_diag - np.eye(n_lat)).sum(axis=1)

    # Matter mass density |psi|^2
    psi_sq = (psi.real ** 2 + psi.imag ** 2)

    return {
        "delta": delta,
        "t00": np.abs(t00),
        "psi_sq": psi_sq,
        "k_row_mean": k_row_mean,
        "q_row_mean": q_row_mean,
        "xi_degree": xi_degree,
        "defect_degree": defect_degree,
        "R_time": R_time,
    }


def _layer_indices(arr, percentile):
    """Indices of nodes whose value exceeds the per-seed percentile threshold."""
    thr = np.percentile(arr, percentile)
    return set(int(i) for i in np.where(arr > thr)[0])


def _topX_indices(arr, fraction):
    """Indices of nodes in the top-X% by value (descending)."""
    if arr.size == 0:
        return set()
    thr = np.quantile(arr, 1.0 - fraction)
    return set(int(i) for i in np.where(arr > thr)[0])


def _jaccard(a, b):
    if not a and not b:
        return float("nan")
    return len(a & b) / max(len(a | b), 1)


def _seed_layer_audit(q):
    """Per-seed layer-nested audit on the per-node observables."""
    layers = {}
    for name, pct, _desc in PERCENTILE_LAYERS:
        layers[name] = _layer_indices(q["delta"], pct)
    # sup as the (top-1 node) by Delta
    arg_sup = int(np.argmax(q["delta"]))
    layers["Csup"] = {arg_sup}

    t00_top10 = _topX_indices(q["t00"], 0.10)
    t00_top5 = _topX_indices(q["t00"], 0.05)
    t00_top1 = _topX_indices(q["t00"], 0.01)
    psi_top5 = _topX_indices(q["psi_sq"], 0.05)
    psi_top1 = _topX_indices(q["psi_sq"], 0.01)
    xi_deg_top5 = _topX_indices(q["xi_degree"], 0.05)
    defect_deg_top5 = _topX_indices(q["defect_degree"], 0.05)
    k_top5 = _topX_indices(q["k_row_mean"], 0.05)
    q_top5 = _topX_indices(q["q_row_mean"], 0.05)

    out = {}
    for name in ["C95", "C99", "C99_5", "C99_9", "Csup"]:
        S = layers[name]
        # sign distribution of R_time on layer
        if S:
            r_arr = q["R_time"][np.array(sorted(S))]
            frac_neg = float((r_arr < 0).mean())
        else:
            frac_neg = float("nan")
        out[name] = {
            "support_fraction": len(S) / max(len(q["delta"]), 1),
            "core_mass_psi_sq_sum": (float(q["psi_sq"][np.array(sorted(S))].sum())
                                       if S else 0.0),
            "core_mass_psi_sq_frac": ((float(q["psi_sq"][np.array(sorted(S))].sum())
                                          / max(float(q["psi_sq"].sum()), 1e-12))
                                         if S else 0.0),
            "jaccard_t00_top10": _jaccard(S, t00_top10),
            "jaccard_t00_top5": _jaccard(S, t00_top5),
            "jaccard_t00_top1": _jaccard(S, t00_top1),
            "jaccard_psi_top5": _jaccard(S, psi_top5),
            "jaccard_psi_top1": _jaccard(S, psi_top1),
            "jaccard_xi_degree_top5": _jaccard(S, xi_deg_top5),
            "jaccard_defect_degree_top5": _jaccard(S, defect_deg_top5),
            "jaccard_K_top5": _jaccard(S, k_top5),
            "jaccard_Q_top5": _jaccard(S, q_top5),
            "frac_R_time_negative": frac_neg,
        }
    # Layer-nesting verification
    nesting = {
        "C99_9_in_C99_5": (layers["C99_9"] <= layers["C99_5"]) if layers["C99_9"] else True,
        "C99_5_in_C99":   (layers["C99_5"] <= layers["C99"]) if layers["C99_5"] else True,
        "C99_in_C95":     (layers["C99"]   <= layers["C95"]) if layers["C99"] else True,
    }
    out["nesting"] = nesting
    return out


def _per_regime_aggregate(reg, n_lat, seeds):
    seed_audits = []
    for xi_mat, psi, k_field, q_field in seeds:
        q = _node_quantities(xi_mat, psi, k_field, q_field, n_lat)
        seed_audits.append(_seed_layer_audit(q))
    # Aggregate per layer (mean over seeds)
    agg = {"regime": reg, "N": n_lat, "n_seeds": len(seed_audits)}
    layer_keys = ["C95", "C99", "C99_5", "C99_9", "Csup"]
    metric_keys = [
        "support_fraction", "core_mass_psi_sq_sum", "core_mass_psi_sq_frac",
        "jaccard_t00_top10", "jaccard_t00_top5", "jaccard_t00_top1",
        "jaccard_psi_top5", "jaccard_psi_top1",
        "jaccard_xi_degree_top5", "jaccard_defect_degree_top5",
        "jaccard_K_top5", "jaccard_Q_top5",
        "frac_R_time_negative",
    ]
    for lk in layer_keys:
        agg[lk] = {}
        for mk in metric_keys:
            vals = [a[lk][mk] for a in seed_audits if not np.isnan(a[lk][mk])]
            agg[lk][mk] = float(np.mean(vals)) if vals else float("nan")
            agg[lk][mk + "_std"] = float(np.std(vals)) if vals else float("nan")
    # Nesting fraction (fraction of seeds where nesting holds for each pair)
    agg["nesting"] = {
        k: float(np.mean([a["nesting"][k] for a in seed_audits]))
        for k in ["C99_9_in_C99_5", "C99_5_in_C99", "C99_in_C95"]
    }
    return agg


def _symanzik2(n_arr, y_arr):
    n_arr = np.asarray(n_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)
    if len(n_arr) < 3:
        return float("nan"), float("nan")
    x_mat = np.column_stack([np.ones_like(n_arr), 1.0 / n_arr])
    coef, *_ = np.linalg.lstsq(x_mat, y_arr, rcond=None)
    return float(coef[0]), float(coef[1])


def main() -> int:
    print("=" * 110)
    print("Stage 6h: matter-core percentile phase diagram (layer-nested)")
    print("=" * 110)
    print(f"  Lambda_t = {LAMBDA_T}, Lambda_s = {LAMBDA_S}")
    print()

    rows = []
    for reg, n_lat in LADDER:
        seeds = _gather_regime(reg, n_lat)
        if seeds is None:
            print(f"  {reg} N={n_lat}: NPZ not found, skipping")
            continue
        agg = _per_regime_aggregate(reg, n_lat, seeds)
        rows.append(agg)
        c95 = agg["C95"]["support_fraction"]
        m99_5 = agg["C99_5"]["core_mass_psi_sq_frac"]
        m99_9 = agg["C99_9"]["core_mass_psi_sq_frac"]
        j_t99_5 = agg["C99_5"]["jaccard_t00_top5"]
        j_q99_5 = agg["C99_5"]["jaccard_Q_top5"]
        print(f"  {reg:<8} N={n_lat:>4} ns={agg['n_seeds']:>2}  "
              f"|C95|={c95:.4f}  M(C99.5)/M={m99_5:.3f}  "
              f"M(C99.9)/M={m99_9:.3f}  "
              f"Jacc(C99.5,T00top5)={j_t99_5:.2f}  "
              f"Jacc(C99.5,Qtop5)={j_q99_5:.2f}")

    if len(rows) < 3:
        print("  Not enough regimes")
        return 0

    print()
    print("=" * 110)
    print("Per-layer Symanzik-2 fits on core-mass-fraction vs N")
    print("=" * 110)
    layer_fits = {}
    for layer in ["C95", "C99", "C99_5", "C99_9", "Csup"]:
        n_arr = [r["N"] for r in rows]
        m_arr = [r[layer]["core_mass_psi_sq_frac"] for r in rows]
        y_inf, b = _symanzik2(n_arr, m_arr)
        layer_fits[layer] = {"y_inf": y_inf, "b_over_N": b}
        print(f"  {layer:>6s}: M/M_total -> {y_inf:+.4f} + ({b:+.3f})/N")

    print()
    print("=" * 110)
    print("Per-regime layer-nesting holding-fraction (1.0 = always nested)")
    print("=" * 110)
    for r in rows:
        n = r["nesting"]
        print(f"  {r['regime']:<8} N={r['N']:>4}  "
              f"C99.9 in C99.5={n['C99_9_in_C99_5']:.2f}  "
              f"C99.5 in C99={n['C99_5_in_C99']:.2f}  "
              f"C99 in C95={n['C99_in_C95']:.2f}")

    print()
    print("=" * 110)
    print("Layer-by-layer overlap with K-field-top5 and T_00-top5 (cross-regime mean)")
    print("=" * 110)
    summary = {}
    for layer in ["C95", "C99", "C99_5", "C99_9", "Csup"]:
        def _m(key):
            return float(np.mean(
                [r[layer][key] for r in rows if not np.isnan(r[layer][key])]
            ))
        j_t = _m("jaccard_t00_top5")
        j_k = _m("jaccard_K_top5")
        j_q = _m("jaccard_Q_top5")
        j_psi = _m("jaccard_psi_top5")
        j_xi = _m("jaccard_xi_degree_top5")
        j_def = _m("jaccard_defect_degree_top5")
        f_neg = _m("frac_R_time_negative")
        summary[layer] = {
            "jaccard_t00_top5_mean": j_t,
            "jaccard_K_top5_mean": j_k,
            "jaccard_Q_top5_mean": j_q,
            "jaccard_psi_top5_mean": j_psi,
            "jaccard_xi_degree_top5_mean": j_xi,
            "jaccard_defect_degree_top5_mean": j_def,
            "frac_R_time_negative_mean": f_neg,
        }
        print(f"  {layer:>6s}: Jacc(T00)={j_t:.2f} Jacc(K)={j_k:.2f} "
              f"Jacc(Q)={j_q:.2f} Jacc(psi)={j_psi:.2f} "
              f"Jacc(Xi-deg)={j_xi:.2f} Jacc(defect-deg)={j_def:.2f} "
              f"R_time<0: {f_neg:.2f}")

    bundle = {
        "method": "stage6h_matter_core_phase_diagram",
        "schema_version": "1.0.0",
        "lambda_t": LAMBDA_T,
        "lambda_s": LAMBDA_S,
        "ladder_skipped": ["P5N128 (K/Q persistence bug)"],
        "layers": [
            {"name": n, "percentile": p, "interpretation": d}
            for n, p, d in PERCENTILE_LAYERS
        ] + [{"name": "Csup", "percentile": "max", "interpretation": "core spine"}],
        "regimes": rows,
        "symanzik_fits_core_mass_fraction": layer_fits,
        "cross_regime_summary": summary,
    }
    out = REPO / "outputs" / "stage6h_matter_core_phase_diagram.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print()
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
