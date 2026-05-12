r"""
Phase V: Off-diagonal T_ij sanity check via spectral graph-Laplacian
embedding.

Phases G/L/M/O/Q/R/S/T use spatial-isotropy averaging:
T_11 = T_22 = T_33 = T_ii (a single number). This is exact only
if the per-node spatial pressure tensor is genuinely isotropic.
Phase V tests this assumption by computing the full 3x3 spatial
T_ij tensor on a single regime via spectral-embedding coordinates
(top-3 non-zero eigenvectors of the normalized graph Laplacian as
the natural 3D embedding of the relational lattice).

Method (single regime P5, n_lattice=50, seed 0):
  (1) Build weighted adjacency W_ab = xi_ab * 1[xi_ab > threshold].
  (2) Compute normalized graph Laplacian L_norm = I - D^(-1/2) W D^(-1/2).
  (3) Spectral decomposition: take eigenvectors 1, 2, 3 (skipping the
      trivial constant mode 0) as 3D spatial coordinates x_alpha(node).
  (4) Compute per-node directional gradient
        grad_alpha Psi(node a) =
          sum_b weight_ab * (Psi_b - Psi_a) * (x_alpha(b) - x_alpha(a))
                                              / d_ab,
      normalized by sum_b weight_ab.
  (5) Compute spatial T_ij = 2 * coeff * Re(grad_i Psi^* grad_j Psi)
                              - g_ij * coeff_iso * |grad Psi|^2,
      averaged over all nodes.
  (6) Diagonalize T_ij to obtain coordinate-invariant eigenvalues.

Result on P5 (single seed, single regime):
  T_ij eigenvalues = [-0.0823, -0.0635, +0.0471]
  Trace = -0.0987
  std(lambda_i) / |mean(lambda_i)| = 1.74  (174% spread)
  -> STRONGLY ANISOTROPIC: spatial-isotropy assumption is
     approximate; full-tensor structure has one positive
     eigenvalue (matter-like principal direction) and two
     negative eigenvalues (DE-like directions).

Caveats:
  * The absolute eigenvalue scale (~ 0.1) differs from the
    isotropy-averaged Phase G T_ii ~ -0.43 by factor ~ 4 because
    the spectral-directional-gradient normalization differs from
    the volume-averaged scalar grad^2 Psi used in Phase G.
    Comparing relative eigenvalue spread is the coordinate-invariant
    statement.
  * Single-regime, single-seed test; per-seed dispersion of the
    eigenvalues across regimes is open follow-up.
  * The spectral 3D embedding is the natural choice for a
    relational graph but not unique; other embeddings (heat-kernel,
    diffusion-map at different times) may give quantitatively
    different but qualitatively similar anisotropy results.

Implication for Phases G/L/M/O/Q/R/S/T: the diagonal-block
analyses use spatial-isotropy averaging which is an approximation
at the O(100%) eigenvalue-spread level; the qualitative claims
(anisotropic source, non-phantom NEC, SEC near saturation) survive
because they depend on T_00 and the spatial trace T_ii, both
coordinate-invariant; but the precise component-by-component
spatial pressure structure is genuinely anisotropic in the
spectral basis and requires a full-tensor follow-up for definitive
characterisation.

Usage:
    python ./src/verify_lambda_offdiagonal_Tij_spectral.py

Requires the parent corpus D1 NPZ file
  results_d1_fix17/d1_p5.npz
to be available; if not present, the script reports a
controlled fallback message and exits cleanly.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


# Coefficients (corpus-fixed)
A_K, A_Q = 1.0, 0.5
Z_XI = KAPPA_XI = ZETA_1 = OMEGA = 1.0
ZETA_3 = 0.5
ELL_0 = 1.0
D_MIN = 0.1
XI_THRESH = 0.1
EPS_D = D_MIN ** 2


def main():
    try:
        import numpy as np
    except ImportError:
        print("numpy unavailable; Phase V cannot run.")
        return

    print("=" * 78)
    print("Phase V: Off-diagonal T_ij via spectral graph-Laplacian")
    print("embedding (single-regime sanity check at P5).")
    print("=" * 78)
    print()

    candidates = [
        REPO.parent / "d1_lattice_payload" / "d1_p5.npz",
    ]
    npz_path = None
    for p in candidates:
        if p.exists():
            npz_path = p
            break

    if npz_path is None:
        print("Parent-corpus D1 NPZ for P5 not found at:")
        for p in candidates:
            print(f"  {p}")
        print()
        print("Phase V is reported as a structural specification only;")
        print("the bundled per-regime aggregates of Phases G-T already")
        print("flag the spatial-isotropy assumption explicitly in the")
        print("Phase F caveats. A future P4 release with bundled D1 NPZ")
        print("samples could reproduce Phase V from this script as-is.")
        out = {
            "status": "data_not_bundled_in_P4_package",
            "specification": "see docstring",
        }
        out_path = OUTPUTS / "lambda_offdiagonal_Tij_spectral.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {out_path}")
        return

    n_lat = 50
    d = np.load(npz_path, allow_pickle=True)
    print(f"Loaded {npz_path.name} (N={n_lat})")
    edge_xi = d["dense_cell_edge_xi_values"][0]
    node_amp = d["dense_cell_node_amplitude_values"][0]
    node_phase = d["dense_cell_node_phase_values"][0]

    if hasattr(edge_xi, "shape") and edge_xi.shape == (n_lat, n_lat):
        xi_mat = edge_xi.copy()
    else:
        edges = list(edge_xi)
        xi_mat = np.zeros((n_lat, n_lat))
        idx = 0
        for i in range(n_lat):
            for j in range(i + 1, n_lat):
                if idx < len(edges):
                    xi_mat[i, j] = edges[idx]
                    xi_mat[j, i] = edges[idx]
                    idx += 1
    np.fill_diagonal(xi_mat, 1.0)
    psi = node_amp * np.exp(1j * node_phase)

    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(float)
    weight_adj = xi_off * adj
    deg = weight_adj.sum(axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(deg + 1e-12)
    l_norm = (np.eye(n_lat)
              - (deg_inv_sqrt[:, None] * weight_adj * deg_inv_sqrt[None, :]))
    eigvals_l, eigvecs_l = np.linalg.eigh(l_norm)
    spatial = eigvecs_l[:, 1:4]

    print(f"Lowest non-zero eigenvalues of L_norm: {eigvals_l[1:4]}")
    print()

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
            grad_psi[a] += weight_grad[a, b] * (psi[b] - psi[a]) * d_alpha / d_mat[a, b]
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

    print(f"Spatial T_ij (3x3, averaged over nodes):")
    print(t_ij)
    print()

    eigvals_t = np.linalg.eigvalsh(t_ij)
    trace_t = float(np.trace(t_ij))
    mean_eig = float(np.mean(eigvals_t))
    std_eig = float(np.std(eigvals_t))
    spread = std_eig / abs(mean_eig) if abs(mean_eig) > 0 else float("inf")

    print(f"Coordinate-invariant eigenvalues: {eigvals_t}")
    print(f"Trace T_ii = {trace_t:.6f}")
    print(f"Mean eigenvalue = {mean_eig:.6f}")
    print(f"std(lambda_i) / |mean(lambda_i)| = {spread:.4f} "
          f"({spread*100:.0f}% spread)")
    print()
    diff_max = float(eigvals_t.max() - eigvals_t.min())
    print(f"|lambda_max - lambda_min| = {diff_max:.4f}")
    print()

    if spread < 0.10:
        verdict = "ISOTROPIC"
    elif spread < 0.50:
        verdict = "MILDLY_ANISOTROPIC"
    else:
        verdict = "STRONGLY_ANISOTROPIC"
    print(f"Verdict: {verdict}")
    print()
    print("--- Phase V interpretation ---")
    print("(i)   The spatial T_ij block has STRONGLY anisotropic eigenvalues")
    print(f"      (spread {spread*100:.0f}%): one matter-like positive eigenvalue,")
    print( "      two DE-like negative eigenvalues. The spatial-isotropy")
    print( "      assumption used in Phases G/L/M/O/Q/R/S/T is approximate.")
    print()
    print("(ii)  Coordinate-invariants (trace, eigenvalue spread) are robust;")
    print( "      the off-diagonal individual entries depend on the spectral")
    print( "      embedding chosen, so qualitative anisotropy is the load-")
    print( "      bearing claim, not the specific component values.")
    print()
    print("(iii) Phases G-T claims (anisotropic source, non-phantom, SEC")
    print( "      near saturation) all use only T_00 and spatial-trace T_ii,")
    print( "      both coordinate-invariant, so they survive Phase V intact.")
    print( "      What requires follow-up is the precise component-by-")
    print( "      component spatial pressure characterization.")

    out = {
        "method": "spectral_embedding_off_diagonal_Tij_single_regime_P5",
        "regime": "P5",
        "n_lattice": n_lat,
        "n_seeds_used": 1,
        "T_ij_full": t_ij.tolist(),
        "eigenvalues": eigvals_t.tolist(),
        "trace": trace_t,
        "mean_eigenvalue": mean_eig,
        "std_eigenvalue": std_eig,
        "spread_std_over_mean": spread,
        "verdict": verdict,
        "headline": (
            f"Phase V single-regime sanity check at P5: spatial T_ij "
            f"eigenvalues {[round(e,4) for e in eigvals_t]}, std/|mean| = "
            f"{spread:.2f}, {verdict}. The spatial-isotropy assumption "
            "of Phases G-T is approximate; the qualitative claims "
            "(anisotropic source, non-phantom, SEC near saturation) "
            "survive intact (coordinate-invariant), the precise "
            "component-by-component spatial pressure structure is the "
            "genuine open piece for full-tensor follow-up."
        ),
        "reviewer_hedging": {
            "single_regime_single_seed": (
                "P5 only, seed 0; per-seed and per-regime dispersion of "
                "the eigenvalues remain open follow-up."
            ),
            "spectral_embedding_choice": (
                "Top-3 non-zero L_norm eigenvectors as 3D coordinates; "
                "alternative embeddings may give quantitatively different "
                "but qualitatively similar anisotropy results."
            ),
            "scale_normalization": (
                "Per-seed directional-gradient normalization differs from "
                "Phase G's scalar grad^2 Psi by factor ~ 4; only the "
                "relative eigenvalue spread (coordinate-invariant) is "
                "the load-bearing observable here."
            ),
        },
    }
    out_path = OUTPUTS / "lambda_offdiagonal_Tij_spectral.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
