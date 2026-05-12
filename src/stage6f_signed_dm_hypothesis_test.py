"""Rigorous test of the bulk-negative-as-DM hypothesis.

The signed bulk-core balance audit (stage6f_signed_bulk_core_balance.py)
established that on every regime in the cleaned ten-regime ladder,
the per-node trace residual tr(R) carries opposite-sign amplitude on
bulk versus matter-core, with S_core > 0 (decreasing with N) and
S_bulk approximately = -0.02 per node (regime-stable). The hypothesis
under test here is whether the bulk-negative content can be
identified as a smooth additional DM-like source.

Three falsifiable predictions are tested:

  (T1) DISTINCTNESS from existing T^DM-vortex.
       The framework's existing dark-matter sector is the integer-
       quantized vortex contribution T^DM-vortex with winding
       |q| in {1, 2, ...} localized at vortex centers. If the
       bulk-negative residual were already captured by T^DM-vortex,
       its amplitude would correlate with vortex-density proxies.
       We test Spearman rho(|tr(R)|_bulk, vortex-proxy). Low
       correlation supports T1; high correlation falsifies the
       distinct-DM hypothesis.

  (T2) SPATIAL DISTRIBUTION matches a smooth (non-vortex)
       background.
       A genuine smooth DM background would correlate POSITIVELY
       with distance-to-nearest-vortex (i.e., bulk-DM is most
       prominent away from defect centers). We test
       Spearman rho(|tr(R)|_bulk, d_to_nearest_vortex). Positive
       correlation supports T2.

  (T3) NUMERICAL DENSITY consistency with the cosmological DM
       gap.
       The framework's GCC pipeline reports
       Omega_DM h^2 = 0.068 against observed Omega_DM h^2 = 0.12,
       i.e., a remaining gap of about Omega_DM,extra h^2 = 0.052
       (factor 0.43 of observed). We test whether the lattice
       ratio S_bulk / (S_bulk + S_core_at_lowN) is consistent
       with that fraction. The lattice ratio is dimensionless;
       the prediction is at the order-of-magnitude level only.

For each prediction we report the per-regime audit and a verdict:
  - PASS if all 10 regimes are individually consistent
  - PARTIAL if 7-9 regimes are consistent
  - FAIL if 6 or fewer

A unanimous PASS on T1+T2+T3 would warrant integration into the
papers; anything less should stay in this audit document.

Output: outputs/stage6f_signed_dm_hypothesis_test.json
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

    def load_module(self, _name):
        raise ImportError("cupy disabled")


sys.meta_path.insert(0, _BlockCupy())

from stage6f_full_tensor_norm_audit import (  # noqa: E402
    LADDER, gather_regime, LAMBDA_T, LAMBDA_S, per_node_relative_delta)
from verify_galerkin_runner_A_hessian_ricci import (  # noqa: E402
    edge_to_matrix, ELL_0, D_MIN, EPS_D, XI_THRESH, per_seed_galerkin)
from _d1_npz_discovery import find_d1_npz  # noqa: E402

OUT = REPO / "outputs" / "stage6f_signed_dm_hypothesis_test.json"

CORE_THRESHOLD_TAU = 0.05
COSMO_OMEGA_DM_OBSERVED = 0.12
COSMO_OMEGA_DM_FRAMEWORK = 0.068  # GCC-04 thermal freeze-out, P1
COSMO_OMEGA_DM_EXTRA_TARGET = (
    COSMO_OMEGA_DM_OBSERVED - COSMO_OMEGA_DM_FRAMEWORK)
COSMO_FRACTION_TARGET = COSMO_OMEGA_DM_EXTRA_TARGET / COSMO_OMEGA_DM_OBSERVED


def _spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    rx = np.argsort(np.argsort(x[mask]))
    ry = np.argsort(np.argsort(y[mask]))
    n = len(rx)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _vortex_proxy(xi_mat, n_lat):
    """Per-node vortex-density proxy: |sum_neighbors winding-phase|.
    High where vortex defect cores live."""
    np.fill_diagonal(xi_mat, 1.0)
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(np.float64)
    deg = adj.sum(axis=1) + 1e-12
    xi_log = -np.log(np.maximum(xi_off, 1e-12))
    proxy = (xi_log * adj).sum(axis=1) / deg
    return proxy


def _distance_to_nearest_high_amp(amp, xi_mat, n_lat):
    """Crude per-node 'distance' to the highest-amplitude vortex-like
    node, measured in lattice graph hops weighted by edge xi."""
    target_idx = int(np.argmax(amp))
    np.fill_diagonal(xi_mat, 1.0)
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(np.float64)
    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat = np.where(adj > 0, d_mat, np.inf)
    n = n_lat
    dist = np.full(n, np.inf)
    dist[target_idx] = 0.0
    visited = np.zeros(n, dtype=bool)
    for _ in range(n):
        u = -1
        best = np.inf
        for i in range(n):
            if not visited[i] and dist[i] < best:
                best = dist[i]
                u = i
        if u < 0 or best == np.inf:
            break
        visited[u] = True
        for v in range(n):
            if not visited[v] and adj[u, v] > 0:
                alt = dist[u] + d_mat[u, v]
                if alt < dist[v]:
                    dist[v] = alt
    return dist


def _signed_trace_per_node(prep, lambda_t=LAMBDA_T, lambda_s=LAMBDA_S):
    g_00 = prep["g_00_h"]
    g_ij = prep["g_ij_h"]
    t00 = prep["t00"]
    t_ij = prep["t_ij"]
    res_00 = g_00 + lambda_t - t00
    eye3 = np.eye(3)
    res_d = (g_ij + lambda_s * eye3[None, :, :]) - t_ij
    return res_00 + res_d[:, 0, 0] + res_d[:, 1, 1] + res_d[:, 2, 2]


def _process_regime(regime, n_lat):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    try:
        from stage6f_full_tensor_norm_audit import (
            load_canonical, load_snapshots)
    except ImportError:
        return None
    seeds = (load_snapshots(p, n_lat)
             if "snapshots" in p.name.lower()
             else load_canonical(p, n_lat))
    out = []
    for xi_mat, psi, k_field, q_field in seeds:
        try:
            prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
            tr = _signed_trace_per_node(prep)
            df = per_node_relative_delta(prep, LAMBDA_T, LAMBDA_S)["delta_full"]
            t00 = prep["t00"]
            amp = np.abs(psi).flatten() if hasattr(psi, "flatten") else np.abs(psi)
            v_proxy = _vortex_proxy(np.asarray(xi_mat), n_lat)
            mask_core = df > CORE_THRESHOLD_TAU
            mask_bulk = ~mask_core
            try:
                d_vortex = _distance_to_nearest_high_amp(
                    np.asarray(amp), np.asarray(xi_mat), n_lat)
            except Exception:  # noqa: BLE001
                d_vortex = np.full(n_lat, np.nan)
            out.append({
                "tr_signed": tr, "delta_full": df, "t00": t00,
                "vortex_proxy": v_proxy, "d_vortex": d_vortex,
                "mask_core": mask_core, "mask_bulk": mask_bulk,
            })
        except Exception:  # noqa: BLE001
            continue
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime, n_lat in LADDER:
        seeds = _process_regime(regime, n_lat)
        if not seeds:
            continue
        # Pool over seeds for this regime
        tr = np.concatenate([s["tr_signed"] for s in seeds])
        df = np.concatenate([s["delta_full"] for s in seeds])
        t00 = np.concatenate([s["t00"] for s in seeds])
        vp = np.concatenate([s["vortex_proxy"] for s in seeds])
        dv = np.concatenate([s["d_vortex"] for s in seeds])
        mask_core = df > CORE_THRESHOLD_TAU
        mask_bulk = ~mask_core

        # T1: |tr_R| in bulk vs vortex_proxy → expect LOW positive
        rho_T1 = _spearman(np.abs(tr[mask_bulk]), vp[mask_bulk])
        # T2: |tr_R| in bulk vs distance-to-nearest-vortex → expect POSITIVE
        finite_dv = np.isfinite(dv[mask_bulk])
        if finite_dv.sum() < 10:
            rho_T2 = float("nan")
        else:
            tr_bulk_abs = np.abs(tr[mask_bulk][finite_dv])
            dv_bulk = dv[mask_bulk][finite_dv]
            rho_T2 = _spearman(tr_bulk_abs, dv_bulk)
        # T2 positive control: tr_core vs t00 → expect HIGH POSITIVE (already established)
        rho_T2_control = _spearman(np.abs(tr[mask_core]), t00[mask_core])
        # T3: lattice ratio |S_bulk| / (|S_core| + |S_bulk|) per regime
        s_core = float(tr[mask_core].sum()) / len(tr)
        s_bulk = float(tr[mask_bulk].sum()) / len(tr)
        lattice_fraction_bulk = (
            abs(s_bulk) / (abs(s_core) + abs(s_bulk))
            if (abs(s_core) + abs(s_bulk)) > 0 else float("nan"))

        rows.append({
            "regime": regime,
            "N": int(n_lat),
            "n_node": int(len(tr)),
            "core_fraction": float(mask_core.mean()),
            "S_core_per_node": s_core,
            "S_bulk_per_node": s_bulk,
            # Predictions
            "T1_spearman_bulk_tr_vs_vortex_proxy": rho_T1,
            "T1_low_correlation_pass": (
                abs(rho_T1) < 0.30 if rho_T1 == rho_T1 else None),
            "T2_spearman_bulk_tr_vs_dist_to_vortex": rho_T2,
            "T2_positive_correlation_pass": (
                rho_T2 > 0.10 if rho_T2 == rho_T2 else None),
            "T2_control_core_tr_vs_t00": rho_T2_control,
            "T3_lattice_bulk_fraction": lattice_fraction_bulk,
        })
        print(f"  {regime:>10s}  N={n_lat:>4d}  "
              f"T1 rho={rho_T1:+.3f}  "
              f"T2 rho(d_v)={rho_T2:+.3f}  "
              f"control rho(t00)={rho_T2_control:+.3f}  "
              f"T3 |S_b|/(|S_b|+|S_c|)={lattice_fraction_bulk:.3f}")

    # Aggregate verdicts
    n_total = len(rows)
    n_T1_pass = sum(1 for r in rows
                    if r.get("T1_low_correlation_pass") is True)
    n_T2_pass = sum(1 for r in rows
                    if r.get("T2_positive_correlation_pass") is True)
    median_T3 = float(np.median(
        [r["T3_lattice_bulk_fraction"] for r in rows
         if r["T3_lattice_bulk_fraction"] == r["T3_lattice_bulk_fraction"]]))

    out = {
        "method": "Three-prediction test of bulk-negative-as-DM hypothesis",
        "core_threshold_tau": CORE_THRESHOLD_TAU,
        "cosmo_target_dm_extra_fraction": COSMO_FRACTION_TARGET,
        "per_regime": rows,
        "summary": {
            "n_regimes": n_total,
            "T1_n_pass": n_T1_pass,
            "T1_verdict": _verdict(n_T1_pass, n_total),
            "T2_n_pass": n_T2_pass,
            "T2_verdict": _verdict(n_T2_pass, n_total),
            "T3_median_lattice_fraction": median_T3,
            "T3_within_factor_2_of_cosmo_target": (
                0.5 * COSMO_FRACTION_TARGET <= median_T3
                <= 2.0 * COSMO_FRACTION_TARGET),
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print(f"  T1 (distinctness from vortex-DM): "
          f"{n_T1_pass}/{n_total} pass — "
          f"{out['summary']['T1_verdict']}")
    print(f"  T2 (smooth-distributed signature): "
          f"{n_T2_pass}/{n_total} pass — "
          f"{out['summary']['T2_verdict']}")
    print(f"  T3 (cosmological-DM-fraction match): "
          f"median lattice fraction = {median_T3:.3f}, "
          f"target = {COSMO_FRACTION_TARGET:.3f} "
          f"(within factor 2: "
          f"{out['summary']['T3_within_factor_2_of_cosmo_target']})")
    print()
    print(f"Wrote {OUT}")


def _verdict(n_pass, n_total):
    if n_total == 0:
        return "NO_DATA"
    if n_pass == n_total:
        return "PASS"
    if n_pass >= n_total - 3:
        return "PARTIAL"
    return "FAIL"


if __name__ == "__main__":
    main()
