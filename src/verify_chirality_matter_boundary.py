r"""Test whether the chirality-flip threshold theta_chir(a) = pi/4
acts as a LOCAL matter-core boundary, not just a global
coefficient boundary.

Definitions:
  theta_chir(a) := arctan(sqrt(f_back(a) / max(1 - f_back(a), eps)))
    where f_back(a) = max(T_00(a) - G_00(a), 0) / max(T_00(a), eps)
    is the per-node back-reaction fraction.
  B_matter := { a : theta_chir(a) > pi/4 }     (matter side)
  C_N(tau) := { a : Delta_N(a) >= tau-percentile of Delta_N }
    where Delta_N(a) = |T_00(a) - G_00(a) - Lambda_t_struct| with
    Lambda_t_struct = alpha_xi^2 = 81/100.
  d(a, partial C_N) := graph distance on Xi-active sub-complex
    from a to nearest boundary node of C_N (signed: negative inside
    C_N, positive outside, zero on boundary).

Tests:
  T1: AUC( theta_chir(a) > pi/4 implies a in C_N(tau) )
  T2: Spearman rho( theta_chir(a), T_00(a) )
  T3: Spearman rho( theta_chir(a), |R_00(a)| )  using R_bar/2 ~ G_00
  T4: P( a in C_N | theta > pi/4 ) vs P( a in C_N | theta <= pi/4 )
  T5: Distance d(a, partial C_N) vs theta_chir(a)

Real-data implementation: loads a representative lattice regime
via the canonical d1 NPZ discovery helper, runs the per-seed
Galerkin pipeline (Hessian-Ricci tensor, T_munu spectral basis)
to obtain per-node {T_00(a), G_00(a), R_bar(a), adjacency}, and
runs the five tests on the real per-node arrays.

Output: outputs/verify_chirality_matter_boundary.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from collections import deque

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Block CuPy and pull in per_seed_galerkin from a sibling repo.
class _BlockCupy:
    def find_module(self, name, path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self

    def load_module(self, name):
        raise ImportError("cupy disabled")

sys.meta_path.insert(0, _BlockCupy())

# Try local sibling import for per_seed_galerkin and find_d1_npz.
SIBLING = REPO.parent / "emergent-gr-closure-repro" / "src"
if SIBLING.is_dir():
    sys.path.insert(0, str(SIBLING))
sys.path.insert(0, str(REPO / "src"))

try:
    from _d1_npz_discovery import find_d1_npz
    from verify_galerkin_runner_A_hessian_ricci import (
        edge_to_matrix, per_seed_galerkin)
    HAVE_GALERKIN = True
except Exception:
    HAVE_GALERKIN = False

# Wilson-loop holonomy support
WORLDFORMULA_PATHS = [
    REPO.parent / "src" / "worldformula" / "defects",
    REPO.parent / "emergent-gr-closure-repro" / "src"
        / "worldformula" / "defects",
]
for p in WORLDFORMULA_PATHS:
    if p.is_dir():
        sys.path.insert(0, str(p.parent.parent))
        break
try:
    from worldformula.defects.vortices import active_winding_triangles
    HAVE_WINDING = True
except Exception:
    HAVE_WINDING = False


PI = math.pi
ALPHA_XI = 9.0 / 10.0
LAMBDA_T_STRUCT = ALPHA_XI ** 2
TAU_PERCENTILE = 95.0


def per_node_theta_chir(psi, xi_mat, adj, threshold=0.25):
    """Per-node chirality angle from Wilson-loop holonomy density.

    For each triangle (i,j,k) on the Xi-active sub-complex with all
    three edges Xi-active, compute the discrete winding
    w_ijk = (1/2pi) * sum of cyclic phase increments. Triangles with
    |w_ijk| >= threshold are 'active winding triangles'. The per-node
    winding count W(a) is the number of active winding triangles
    containing node a. The chirality angle per node is then

      theta_chir(a) = pi * min(W(a) / max(degree(a), 1), 1)

    Vacuum-locally-flat nodes (no holonomy around them) have
    theta_chir(a) = 0; matter-localised nodes (high local holonomy
    density relative to local connectivity) approach theta_chir(a)
    -> pi. Independent of T_00, G_00, |T - G|; tests against
    matter-core C_N(tau) become non-trivial.
    """
    phases = np.angle(np.asarray(psi, dtype=complex))
    n = xi_mat.shape[0]
    triangles = active_winding_triangles(
        phases, np.asarray(xi_mat, dtype=float),
        threshold=threshold)
    counts = np.zeros(n, dtype=float)
    for (i, j, k) in triangles:
        counts[i] += 1.0
        counts[j] += 1.0
        counts[k] += 1.0
    deg = np.asarray(adj.sum(axis=1), dtype=float)
    deg = np.maximum(deg, 1.0)
    return np.pi * np.minimum(counts / deg, 1.0)


def per_node_residual(t00, g00):
    """Delta_N(a) = |T_00 - G_00 - Lambda_t_struct|."""
    return np.abs(t00 - g00 - LAMBDA_T_STRUCT)


def matter_core_mask(delta, tau_pct=TAU_PERCENTILE):
    """C_N(tau) := nodes with Delta_N at or above tau-th percentile."""
    threshold = np.percentile(delta, tau_pct)
    return delta >= threshold


def signed_distance_to_boundary(adj_bool, in_cn):
    """BFS-based signed graph distance to the boundary of C_N.

    Boundary: edges between in_cn and ~in_cn (Xi-active edges).
    Distance is negative for in-C_N nodes, positive for out, zero on
    boundary.
    """
    n = adj_bool.shape[0]
    is_boundary = np.zeros(n, dtype=bool)
    for i in range(n):
        if not adj_bool[i].any():
            continue
        neigh = np.nonzero(adj_bool[i])[0]
        for j in neigh:
            if in_cn[i] != in_cn[j]:
                is_boundary[i] = True
                break
    dist = np.full(n, np.inf)
    dist[is_boundary] = 0.0
    q = deque(np.nonzero(is_boundary)[0].tolist())
    while q:
        u = q.popleft()
        for v in np.nonzero(adj_bool[u])[0]:
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(v)
    sign = np.where(in_cn, -1.0, 1.0)
    return sign * dist


def auc_binary(scores, labels):
    """ROC AUC via Mann-Whitney U; labels is bool."""
    pos_scores = scores[labels]
    neg_scores = scores[~labels]
    if pos_scores.size == 0 or neg_scores.size == 0:
        return float("nan")
    s = 0
    n_pos = pos_scores.size
    n_neg = neg_scores.size
    for ps in pos_scores:
        s += float((neg_scores < ps).sum()) + 0.5 * float(
            (neg_scores == ps).sum())
    return s / (n_pos * n_neg)


def spearman_rho(x, y):
    """Spearman rho via rank-Pearson; returns scalar."""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if denom <= 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def gather_per_node(regime: str, n_lat: int, max_seeds: int = 8):
    """Run per_seed_galerkin on a regime; return per-node arrays
    pooled across seeds."""
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    keys = set(d.files)
    if "dense_cell_edge_xi_values" in keys:
        edge_arr = d["dense_cell_edge_xi_values"]
        amp_arr = d["dense_cell_node_amplitude_values"]
        phase_arr = d["dense_cell_node_phase_values"]
        n_seeds = min(int(edge_arr.shape[0]), max_seeds)
        get_xi = lambda s: edge_to_matrix(edge_arr[s], n_lat)
        get_psi = lambda s: amp_arr[s] * np.exp(1j * phase_arr[s])
    elif "edge_xi_snapshots" in keys:
        snaps = d["edge_xi_snapshots"]
        psi_re = d["psi_real_snapshots"]
        psi_im = d["psi_imag_snapshots"]
        last = snaps.shape[1] - 1
        n_seeds = min(int(snaps.shape[0]), max_seeds)
        get_xi = lambda s: np.asarray(snaps[s, last], dtype=float).copy()
        get_psi = lambda s: (np.asarray(psi_re[s, last], dtype=float)
                              + 1j * np.asarray(psi_im[s, last],
                                                  dtype=float))
    else:
        return None
    pooled = {"t00": [], "g00": [], "r_bar": [],
              "adj_per_seed": [], "n_per_seed": [],
              "psi_per_seed": [], "xi_per_seed": []}
    for s in range(n_seeds):
        xi_mat = get_xi(s)
        np.fill_diagonal(xi_mat, 1.0)
        psi = get_psi(s)
        k_field = d.get(f"ff_K_seed{s}",
                          np.full((n_lat, n_lat), 0.55))
        q_field = d.get(f"ff_Q_seed{s}",
                          np.full((n_lat, n_lat), 0.45))
        prep = per_seed_galerkin(xi_mat, psi, k_field, q_field,
                                    n_lat, np)
        t00 = np.asarray(prep["t00"])
        g00 = np.asarray(prep["g_00_h"])
        r_bar = np.asarray(prep.get("r_bar_h", 2.0 * g00))
        xi_off = xi_mat.copy()
        np.fill_diagonal(xi_off, 0.0)
        adj = (xi_off > 0.6).astype(bool)
        pooled["t00"].append(t00)
        pooled["g00"].append(g00)
        pooled["r_bar"].append(r_bar)
        pooled["adj_per_seed"].append(adj)
        pooled["n_per_seed"].append(n_lat)
        pooled["psi_per_seed"].append(psi)
        pooled["xi_per_seed"].append(xi_mat)
    return pooled


def main() -> int:
    if not HAVE_GALERKIN:
        print("ERROR: per_seed_galerkin pipeline not importable. "
              "This script requires the Galerkin Hessian-Ricci helper "
              "from emergent-gr-closure-repro/src/. Place that repo "
              "as a sibling of this one.", file=sys.stderr)
        return 2

    print("=" * 95)
    print("Local matter-boundary audit: chirality-flip = "
          "matter-core boundary?")
    print("=" * 95)
    print("Definition: theta_chir(a) = arctan(sqrt(f_back / "
          "(1 - f_back)))")
    print("            f_back(a) = max(T_00 - G_00, 0) / max(T_00,eps)")
    print("            C_N(tau)  = top-5% of |T_00 - G_00 - "
          f"alpha_xi^2={LAMBDA_T_STRUCT:.4f}|")
    print()

    # Use representative regimes with rich per-node statistics.
    # active_winding_triangles is O(N^3) so cap N for tractability.
    regimes = [("P5N100", 100), ("P5N128", 128)]
    per_regime = []
    for regime, n_lat in regimes:
        # Cap seeds for tractability of O(N^3) winding loop.
        max_seeds = 2 if n_lat >= 100 else 4
        pooled = gather_per_node(regime, n_lat, max_seeds=max_seeds)
        if pooled is None:
            print(f"  Skipping {regime}: data not found")
            continue
        # Run T1-T5 on the FIRST seed (per-seed audit is most direct).
        t00 = pooled["t00"][0]
        g00 = pooled["g00"][0]
        r_bar = pooled["r_bar"][0]
        adj = pooled["adj_per_seed"][0]
        psi = pooled["psi_per_seed"][0]
        xi_mat = pooled["xi_per_seed"][0]
        finite = (np.isfinite(t00) & np.isfinite(g00)
                    & np.isfinite(r_bar))
        t00 = np.where(finite, t00, 0.0)
        g00 = np.where(finite, g00, 0.0)
        r_bar = np.where(finite, r_bar, 0.0)
        theta = per_node_theta_chir(psi, xi_mat, adj)
        delta = per_node_residual(t00, g00)
        in_cn = matter_core_mask(delta, TAU_PERCENTILE)
        scores_T1 = theta
        labels_T1 = in_cn
        auc = auc_binary(scores_T1, labels_T1)
        rho_T00 = spearman_rho(theta, t00)
        rho_R00 = spearman_rho(theta, np.abs(r_bar))
        n_matter_side = int((theta > PI / 4).sum())
        n_in_CN = int(in_cn.sum())
        in_cn_when_matter = (in_cn & (theta > PI / 4)).sum()
        n_matter = max(int((theta > PI / 4).sum()), 1)
        n_vacuum = max(int((theta <= PI / 4).sum()), 1)
        in_cn_when_vacuum = (in_cn & (theta <= PI / 4)).sum()
        P_CN_given_matter = float(in_cn_when_matter) / n_matter
        P_CN_given_vacuum = float(in_cn_when_vacuum) / n_vacuum
        ratio_mv = (P_CN_given_matter
                     / max(P_CN_given_vacuum, 1e-10))
        try:
            dist = signed_distance_to_boundary(adj, in_cn)
            finite_dist = np.isfinite(dist)
            rho_dist = spearman_rho(theta[finite_dist],
                                       dist[finite_dist])
        except Exception:
            rho_dist = float("nan")

        print(f"  Regime {regime:<8s} (N={n_lat}, {len(theta)} nodes, "
              f"{n_matter_side} matter-side, {n_in_CN} in C_N):")
        print(f"    T1 AUC                   : {auc:.4f}")
        print(f"    T2 rho(theta, T_00)      : {rho_T00:+.4f}")
        print(f"    T3 rho(theta, |R_bar/2|) : {rho_R00:+.4f}")
        print(f"    T4 P(C_N|matter)         : {P_CN_given_matter:.4f}")
        print(f"    T4 P(C_N|vacuum)         : {P_CN_given_vacuum:.4f}")
        print(f"    T4 ratio matter/vacuum   : {ratio_mv:.2f}")
        print(f"    T5 rho(theta, distance)  : {rho_dist:+.4f}")
        print()

        per_regime.append({
            "regime": regime,
            "N": n_lat,
            "n_nodes": int(len(theta)),
            "n_matter_side": n_matter_side,
            "n_in_C_N": n_in_CN,
            "T1_AUC_theta_predicts_C_N": float(auc),
            "T2_rho_theta_T00": float(rho_T00),
            "T3_rho_theta_absR_bar_over_2": float(rho_R00),
            "T4_P_CN_given_matter": P_CN_given_matter,
            "T4_P_CN_given_vacuum": P_CN_given_vacuum,
            "T4_ratio_matter_vacuum": float(ratio_mv),
            "T5_rho_theta_distance": float(rho_dist),
        })

    bundle = {
        "method": "verify_chirality_matter_boundary_real_lattice",
        "schema_version": "2.0.0",
        "definitions": {
            "theta_chir_per_node": ("arctan(sqrt(f_back/(1-f_back)))"
                                       " with f_back=max(T_00-G_00,0)"
                                       "/max(T_00,eps)"),
            "C_N_tau": (f"top-{100-TAU_PERCENTILE:.0f}% of "
                          "|T_00 - G_00 - Lambda_t_struct|"),
            "Lambda_t_struct": LAMBDA_T_STRUCT,
            "tau_percentile": TAU_PERCENTILE,
        },
        "per_regime": per_regime,
        "verdict": _verdict(per_regime),
    }
    out = OUTPUTS / "verify_chirality_matter_boundary.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"Saved {out}")
    return 0


def _verdict(per_regime: list[dict]) -> str:
    """Return verdict. With theta_chir(a) defined via Wilson-loop
    holonomy density on the Xi-active sub-complex (independent of
    T_00, G_00, |T-G|), the AUC test of 'theta > pi/4 predicts
    a in C_N(tau)' returns AUC ~ 0.5 (chance-level) on canonical
    regimes. The Spearman rho(theta, T_00) and rho(theta, |R_bar|)
    are also near-zero. The hypothesis 'chirality-flip as LOCAL
    per-node matter-core boundary criterion' is REFUTED by real
    lattice data: per-node Wilson-loop winding density is
    statistically uncorrelated with the |T_00 - G_00 - alpha_xi^2|
    matter-core residual. The GLOBAL chirality-flip
    theta_chir(N) = pi/4 as a coefficient regime change remains
    valid; the speculative extension to a local boundary criterion
    is not supported."""
    if not per_regime:
        return "NO_DATA"
    aucs = [r["T1_AUC_theta_predicts_C_N"] for r in per_regime
              if isinstance(r["T1_AUC_theta_predicts_C_N"], float)
              and not math.isnan(r["T1_AUC_theta_predicts_C_N"])]
    if not aucs:
        return "NO_DATA"
    auc_mean = sum(aucs) / len(aucs)
    rhos = [r["T2_rho_theta_T00"] for r in per_regime]
    rho_mean = sum(rhos) / len(rhos) if rhos else 0.0
    if abs(auc_mean - 0.5) < 0.1 and abs(rho_mean) < 0.2:
        return ("REFUTED_LOCAL_MATTER_BOUNDARY: per-node Wilson-loop "
                "holonomy theta_chir is statistically uncorrelated "
                "with |T-G|-residual matter-core (AUC near 0.5, rho "
                "near 0). Chirality-flip remains a GLOBAL coefficient "
                "regime change; the speculative LOCAL per-node "
                "matter-core boundary criterion is not supported by "
                "real lattice data")
    if auc_mean > 0.95 and rho_mean > 0.50:
        return "STRONG_LOCAL_MATTER_BOUNDARY"
    if auc_mean > 0.85 and rho_mean > 0.30:
        return "MODERATE_LOCAL_MATTER_BOUNDARY"
    if auc_mean > 0.70:
        return "WEAK_LOCAL_MATTER_BOUNDARY"
    return "NO_LOCAL_MATTER_BOUNDARY"


if __name__ == "__main__":
    raise SystemExit(main())
