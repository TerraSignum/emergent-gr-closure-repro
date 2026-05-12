r"""Multi-$N$ spectral-embedding off-diagonal $T_{ij}$ convergence test.

Direct full-tensor companion to the diagonal-block analyses above.
Computes the spatial $3\times 3$ pressure tensor $T_{ij}$ on the full
canonical lattice ladder $N\in\{18,28,30,36,42,50,60,72,84\}$ via
spectral-graph-Laplacian embedding (top-3 non-zero eigenvectors of the
normalized graph Laplacian as the natural 3D spatial coordinate frame),
on every available seed per regime. Reads the already-bundled D1 NPZ
files; no new lattice run required.

For each (regime, seed) pair, evaluates:
  (1) spatial $T_{ij}$ eigenvalues $\lambda_{1,2,3}$ (coordinate-invariant);
  (2) eigenvalue spread $\mathrm{std}(\lambda)/|\mathrm{mean}(\lambda)|$
      (relative anisotropy);
  (3) off-diagonal Frobenius norm
      $\|T_{ij} - \tfrac{1}{3}(\mathrm{tr}\,T)\,\delta_{ij}\|_F$
      (deviation from spatial isotropy).

Aggregates per regime and reports the multi-$N$ trend of (2) and (3).
A decreasing trend with $N$ supports asymptotic spatial isotropy in
the continuum limit, completing the full-tensor convergence reading.
A constant or growing trend genuine anisotropy and is reported as
a substantive physical finding rather than a numerical artefact.

Usage:
    python ./src/verify_lambda_offdiagonal_Tij_spectral_multiN.py
"""
from __future__ import annotations
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Coefficients (framework-fixed; same as single-regime script).
A_K, A_Q = 1.0, 0.5
Z_XI = KAPPA_XI = ZETA_1 = OMEGA = 1.0
ZETA_3 = 0.5
ELL_0 = 1.0
D_MIN = 0.1
XI_THRESH = 0.1
EPS_D = D_MIN ** 2

# Canonical lattice ladder.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _d1_npz_discovery import find_d1_npz

LADDER_REGIMES = [
    ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
    ("P4", 42), ("P5", 50),
    ("P6", 60), ("P7", 72), ("P8", 84),
]
LADDER = [(r, n, find_d1_npz(r, REPO)) for r, n in LADDER_REGIMES]

# Seeds per regime to evaluate (all 4 production seeds per regime;
# canonical D1 packs additional seeds for window statistics, but the
# first 4 entries are the principal seed sequence).
N_SEEDS_PER_REGIME = 4


def edge_to_matrix(edges, n):
    """Convert flat upper-triangular edge list to symmetric N x N matrix."""
    import numpy as np
    if hasattr(edges, "shape") and edges.shape == (n, n):
        return edges.copy()
    edges = list(edges)
    mat = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if idx < len(edges):
                mat[i, j] = edges[idx]
                mat[j, i] = edges[idx]
                idx += 1
    return mat


def evaluate_t_ij(xi_mat, psi, n_lat):
    """Compute the spatial 3x3 T_ij tensor via spectral graph-Laplacian
    embedding. Returns (T_ij, eigenvalues_of_T)."""
    import numpy as np
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(float)
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(deg + 1e-12)
    l_norm = (np.eye(n_lat)
              - (deg_inv_sqrt[:, None] * weight_adj * deg_inv_sqrt[None, :]))
    eigvals_l, eigvecs_l = np.linalg.eigh(l_norm)
    spatial = eigvecs_l[:, 1:4]  # top 3 non-zero modes

    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat = np.maximum(d_mat, D_MIN)
    d_sq = d_mat ** 2
    d_sq[adj == 0] = np.inf
    weight_grad = (xi_off * adj) / (d_sq + EPS_D)
    weight_grad[adj == 0] = 0.0

    grad_psi = np.zeros((n_lat, 3), dtype=complex)
    for a in range(n_lat):
        for b in range(n_lat):
            if adj[a, b] == 0:
                continue
            d_alpha = spatial[b] - spatial[a]
            grad_psi[a] += (weight_grad[a, b] * (psi[b] - psi[a])
                            * d_alpha / d_mat[a, b])
    omega_a = weight_grad.sum(axis=1)
    grad_psi /= omega_a[:, None] + 1e-12

    coeff = 2 * (0.5 * Z_XI + KAPPA_XI + ZETA_1 * OMEGA)
    iso_subtract = (0.5 * Z_XI + KAPPA_XI + ZETA_1 * OMEGA)
    t_ij = np.zeros((3, 3), dtype=float)
    for a in range(n_lat):
        g = grad_psi[a]
        outer = np.real(np.conj(g[:, None]) * g[None, :])
        norm_sq = float(np.sum(np.abs(g) ** 2))
        t_ij += coeff * outer - iso_subtract * norm_sq * np.eye(3)
    t_ij /= n_lat

    eigvals_t = np.linalg.eigvalsh(t_ij)
    return t_ij, eigvals_t


def offdiag_frobenius(t_ij):
    """Compute the deviation of T_ij from its isotropic part:
    || T_ij - (1/3) tr(T) delta_ij ||_F.
    This is the non-isotropic contribution to the spatial pressure tensor.
    """
    import numpy as np
    iso = (1.0 / 3.0) * float(np.trace(t_ij))
    aniso = t_ij - iso * np.eye(3)
    return float(np.linalg.norm(aniso, ord="fro"))


def main():
    try:
        import numpy as np
    except ImportError:
        print("numpy unavailable.")
        return

    print("=" * 78)
    print("Multi-N spectral-embedding off-diagonal T_ij convergence")
    print("on the canonical lattice ladder")
    print(f"  N in {[n for _, n, _ in LADDER]}")
    print(f"  seeds per regime: up to {N_SEEDS_PER_REGIME}")
    print("=" * 78)
    print()

    per_regime_results = {}
    aggregate_means = []

    for regime, n_lat, npz_path in LADDER:
        if npz_path is None or not npz_path.exists():
            print(f"[{regime}, N={n_lat}] NPZ not found: {npz_path}")
            continue

        print(f"[{regime}, N={n_lat}] reading {npz_path.name} ...")
        d = np.load(npz_path, allow_pickle=True)
        edge_arr = d["dense_cell_edge_xi_values"]
        amp_arr = d["dense_cell_node_amplitude_values"]
        phase_arr = d["dense_cell_node_phase_values"]
        n_seeds_avail = min(edge_arr.shape[0], N_SEEDS_PER_REGIME)

        per_seed = []
        for seed_idx in range(n_seeds_avail):
            xi_mat = edge_to_matrix(edge_arr[seed_idx], n_lat)
            np.fill_diagonal(xi_mat, 1.0)
            psi = amp_arr[seed_idx] * np.exp(1j * phase_arr[seed_idx])
            t_ij, eig_t = evaluate_t_ij(xi_mat, psi, n_lat)
            trace_t = float(np.trace(t_ij))
            mean_eig = float(np.mean(eig_t))
            std_eig = float(np.std(eig_t))
            spread = (std_eig / abs(mean_eig)
                      if abs(mean_eig) > 1e-12 else float("inf"))
            offdiag = offdiag_frobenius(t_ij)
            per_seed.append({
                "seed": seed_idx,
                "eigvals": [float(e) for e in eig_t],
                "trace": trace_t,
                "spread": spread,
                "offdiag_frobenius": offdiag,
            })

        spreads = [s["spread"] for s in per_seed
                   if math.isfinite(s["spread"])]
        offdiags = [s["offdiag_frobenius"] for s in per_seed]
        agg = {
            "regime": regime,
            "N": n_lat,
            "n_seeds": n_seeds_avail,
            "per_seed": per_seed,
            "mean_spread": (sum(spreads) / len(spreads) if spreads
                            else float("nan")),
            "mean_offdiag_frobenius": (sum(offdiags) / len(offdiags)
                                       if offdiags else float("nan")),
        }
        per_regime_results[regime] = agg
        aggregate_means.append((n_lat, agg["mean_spread"],
                                agg["mean_offdiag_frobenius"]))

        print(f"  mean spread = {agg['mean_spread']:.3f} "
              f"({agg['mean_spread']*100:.0f}%)")
        print(f"  mean off-diagonal Frobenius = "
              f"{agg['mean_offdiag_frobenius']:.4f}")
        print()

    print("=" * 78)
    print("Multi-N aggregate trend:")
    print(f"  {'N':>5} {'mean spread':>14} {'mean offdiag-Frob':>22}")
    for n, sp, of in aggregate_means:
        print(f"  {n:5d} {sp*100:13.1f}% {of:22.5f}")

    out = {
        "schema_version": "1.0.0",
        "title": "Multi-N spectral-embedding T_ij convergence",
        "ladder_N": [n for _, n, _ in LADDER],
        "n_seeds_per_regime": N_SEEDS_PER_REGIME,
        "per_regime": per_regime_results,
        "trend": [{"N": n, "mean_spread": sp,
                   "mean_offdiag_frobenius": of}
                  for n, sp, of in aggregate_means],
    }
    out_path = OUTPUTS / "lambda_offdiagonal_Tij_spectral_multiN.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
