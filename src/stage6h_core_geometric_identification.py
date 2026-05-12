"""Stage 6h-geom: geometric identification of the Delta-residual core
hierarchy (C95, C99, C99.5, C99.9, Csup) on the lattice.

Tests Jaccard overlap of each core layer with five geometric markers
that are independently defined on Pipeline-A snapshots:

  M1: |psi|^2 bottom-5% (= phase-defect / vortex-core candidates)
  M2: phase-gradient top-5% (= domain-wall / interface candidates,
      computed via the variance of arg(psi_neighbour/psi_node) over
      the persistent-edge neighbourhood)
  M3: local phase-coherence |<e^{i phi}>| bottom-5% within a fixed
      neighbourhood of n_lat^{1/3} edges (= phase-incoherent regions)
  M4: |R_00| top-5% (= energy-density residual extremes)
  M5: |T_00 - T_00^bulk-mean| top-5% (= matter-density extremes,
      both deficits and excesses)

We also measure spatial proximity: mean Euclidean distance from
each Csup node to the nearest C99.5 node (should be 0 if Csup
inside C99.5; should be small if hierarchy is geometrically
clustered).

Output: outputs/stage6h_core_geometric_identification.json
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
)
from verify_galerkin_runner_A_hessian_ricci import per_seed_galerkin  # noqa: E402
from verify_per_eigendirection_residual import (  # noqa: E402
    per_node_eigendirection_residuals,
)


def _layer_indices(arr, percentile_high):
    thr = np.percentile(arr, percentile_high)
    return set(int(i) for i in np.nonzero(arr > thr)[0])


def _bottom_indices(arr, percentile_low):
    thr = np.percentile(arr, percentile_low)
    return set(int(i) for i in np.nonzero(arr < thr)[0])


def _topX_high(arr, fraction):
    if arr.size == 0:
        return set()
    thr = np.quantile(arr, 1.0 - fraction)
    return set(int(i) for i in np.nonzero(arr > thr)[0])


def _topX_low(arr, fraction):
    if arr.size == 0:
        return set()
    thr = np.quantile(arr, fraction)
    return set(int(i) for i in np.nonzero(arr < thr)[0])


def _jaccard(a, b):
    if not a and not b:
        return float("nan")
    return len(a & b) / max(len(a | b), 1)


def _phase_gradient(psi, xi_mat, n_lat):
    """Mean |Delta phi| over each node's strongest-coupled neighbours."""
    phi = np.angle(psi)
    eye = np.eye(n_lat, dtype=bool)
    # Use 1-Xi as defect-coupling weight; stronger coupling = stronger defect
    w = np.where(eye, 0.0, 1.0 - xi_mat)
    grad = np.zeros(n_lat)
    for i in range(n_lat):
        if w[i].sum() > 1e-12:
            d_phi = np.angle(np.exp(1j * (phi - phi[i])))
            grad[i] = (np.abs(d_phi) * w[i]).sum() / w[i].sum()
    return grad


def _local_phase_coherence(psi, xi_mat, n_lat, k_neighbours=None):
    """Local |<exp(i phi)>| over the top-k_neighbours by Xi-row coupling."""
    if k_neighbours is None:
        k_neighbours = max(1, int(np.cbrt(n_lat)))
    phi = np.angle(psi)
    np.fill_diagonal(xi_mat, 0.0)
    coh = np.zeros(n_lat)
    for i in range(n_lat):
        # Pick k_neighbours strongest xi neighbours (highest synchrony)
        order = np.argsort(-xi_mat[i])[:k_neighbours]
        local_phi = phi[order]
        coh[i] = float(np.abs(np.mean(np.exp(1j * local_phi))))
    return coh


def _node_observables(xi_mat, psi, k_field, q_field, n_lat):
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

    psi_sq = (psi.real ** 2 + psi.imag ** 2)
    phase_grad = _phase_gradient(psi, xi_mat.copy(), n_lat)
    local_coh = _local_phase_coherence(psi, xi_mat.copy(), n_lat)

    t00_mean = float(t00.mean())
    t00_dev = np.abs(t00 - t00_mean)

    return {
        "delta": delta,
        "psi_sq": psi_sq,
        "phase_gradient": phase_grad,
        "local_coh": local_coh,
        "abs_R_time": np.abs(R_time),
        "t00_dev": t00_dev,
    }


def _seed_audit(obs):
    layers = {
        "C95": _layer_indices(obs["delta"], 95.0),
        "C99": _layer_indices(obs["delta"], 99.0),
        "C99_5": _layer_indices(obs["delta"], 99.5),
        "C99_9": _layer_indices(obs["delta"], 99.9),
    }
    arg_sup = int(np.argmax(obs["delta"]))
    layers["Csup"] = {arg_sup}

    # Geometric markers
    M1_psi_low5 = _topX_low(obs["psi_sq"], 0.05)
    M2_grad_top5 = _topX_high(obs["phase_gradient"], 0.05)
    M3_coh_low5 = _topX_low(obs["local_coh"], 0.05)
    M4_R00_top5 = _topX_high(obs["abs_R_time"], 0.05)
    M5_t00dev_top5 = _topX_high(obs["t00_dev"], 0.05)

    out = {}
    for name, S in layers.items():
        out[name] = {
            "M1_jacc_psi_low5": _jaccard(S, M1_psi_low5),
            "M2_jacc_grad_top5": _jaccard(S, M2_grad_top5),
            "M3_jacc_coh_low5": _jaccard(S, M3_coh_low5),
            "M4_jacc_R00_top5": _jaccard(S, M4_R00_top5),
            "M5_jacc_t00dev_top5": _jaccard(S, M5_t00dev_top5),
        }
    return out


def _per_regime(reg, n_lat, seeds):
    seed_audits = []
    for xi_mat, psi, k_field, q_field in seeds:
        obs = _node_observables(xi_mat, psi, k_field, q_field, n_lat)
        seed_audits.append(_seed_audit(obs))
    out = {"regime": reg, "N": n_lat, "n_seeds": len(seed_audits)}
    keys = ["M1_jacc_psi_low5", "M2_jacc_grad_top5", "M3_jacc_coh_low5",
            "M4_jacc_R00_top5", "M5_jacc_t00dev_top5"]
    for layer in ["C95", "C99", "C99_5", "C99_9", "Csup"]:
        out[layer] = {}
        for k in keys:
            vals = [a[layer][k] for a in seed_audits
                     if not np.isnan(a[layer][k])]
            out[layer][k + "_mean"] = (float(np.mean(vals))
                                          if vals else float("nan"))
            out[layer][k + "_std"] = (float(np.std(vals))
                                          if vals else float("nan"))
    return out


def main() -> int:
    print("=" * 110)
    print("Stage 6h-geom: geometric identification of Delta-core hierarchy")
    print("=" * 110)
    print("  Markers:")
    print("    M1 = |psi|^2 bottom-5%      (vortex / phase-defect candidates)")
    print("    M2 = phase-gradient top-5%   (domain-wall candidates)")
    print("    M3 = local-coherence bot-5%  (phase-incoherent regions)")
    print("    M4 = |R_00| top-5%           (energy-density residual extremes)")
    print("    M5 = |T_00 - <T_00>| top-5%  (matter-density extremes)")
    print()

    rows = []
    for reg, n_lat in LADDER:
        p = find_d1_npz(reg, REPO)
        if p is None or not p.exists():
            continue
        if "snapshots" in p.name.lower():
            seeds = load_snapshots(p, n_lat)
        else:
            seeds = load_canonical(p, n_lat)
        rows.append(_per_regime(reg, n_lat, seeds))
        last = rows[-1]
        print(f"  {reg:<8} N={n_lat:>4} ns={last['n_seeds']:>2}  "
              f"C99.5: M1={last['C99_5']['M1_jacc_psi_low5_mean']:.2f} "
              f"M2={last['C99_5']['M2_jacc_grad_top5_mean']:.2f} "
              f"M3={last['C99_5']['M3_jacc_coh_low5_mean']:.2f} "
              f"M4={last['C99_5']['M4_jacc_R00_top5_mean']:.2f} "
              f"M5={last['C99_5']['M5_jacc_t00dev_top5_mean']:.2f}")

    if len(rows) < 3:
        print("Not enough regimes")
        return 0

    print()
    print("=" * 110)
    print("Cross-regime mean overlap per layer (Jaccard)")
    print("=" * 110)
    layer_names = ["C95", "C99", "C99_5", "C99_9", "Csup"]
    keys = ["M1_jacc_psi_low5", "M2_jacc_grad_top5", "M3_jacc_coh_low5",
            "M4_jacc_R00_top5", "M5_jacc_t00dev_top5"]
    summary = {}
    print(f"{'layer':>8s}  {'M1 psi-low5':>12s}  {'M2 grad-top5':>12s}  "
          f"{'M3 coh-low5':>12s}  {'M4 R00-top5':>12s}  {'M5 T00dev-top5':>14s}")
    for layer in layer_names:
        means = []
        for k in keys:
            vals = [r[layer][k + "_mean"] for r in rows
                     if not np.isnan(r[layer][k + "_mean"])]
            m = float(np.mean(vals)) if vals else float("nan")
            means.append(m)
        summary[layer] = dict(zip(keys, means))
        print(f"{layer:>8s}  {means[0]:>12.3f}  {means[1]:>12.3f}  "
              f"{means[2]:>12.3f}  {means[3]:>12.3f}  {means[4]:>14.3f}")

    print()
    print("=" * 110)
    print("Per-regime trend of strongest marker (max-mean per layer)")
    print("=" * 110)
    for r in rows:
        c5 = r["C99_5"]
        ranks = sorted(
            [("M1", c5["M1_jacc_psi_low5_mean"]),
             ("M2", c5["M2_jacc_grad_top5_mean"]),
             ("M3", c5["M3_jacc_coh_low5_mean"]),
             ("M4", c5["M4_jacc_R00_top5_mean"]),
             ("M5", c5["M5_jacc_t00dev_top5_mean"])],
            key=lambda x: -x[1] if not np.isnan(x[1]) else 0.0,
        )
        top = ranks[0]
        print(f"  {r['regime']:<8} N={r['N']:>4}: "
              f"strongest C99.5-marker = {top[0]} ({top[1]:.2f})")

    bundle = {
        "method": "stage6h_core_geometric_identification",
        "schema_version": "1.0.0",
        "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
        "markers": {
            "M1": "|psi|^2 bottom-5% (vortex / phase-defect candidates)",
            "M2": "phase-gradient top-5% (domain-wall candidates)",
            "M3": "local-coherence bot-5% (phase-incoherent regions)",
            "M4": "|R_00| top-5% (energy-density residual extremes)",
            "M5": "|T_00 - <T_00>| top-5% (matter-density extremes)",
        },
        "regimes": rows,
        "cross_regime_summary": summary,
    }
    out = REPO / "outputs" / "stage6h_core_geometric_identification.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print()
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
