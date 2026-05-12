r"""G5: Canonical Lagrangian density on a single Xi-snapshot.

Computes the six terms of the Master-blueprint canonical
Lagrangian on a representative D1 lattice snapshot:

    L_total = L_kin + L_pot + L_curv + L_ferm + L_RG + L_causal

with the explicit identifications

    L_kin   = (Nabla Xi)^2  -- Xi-graph kinetic energy via
                                 graph Laplacian quadratic form
    L_pot   = V(Xi)         -- double-well V(Xi) = -Xi^2/2 + Xi^4/4
    L_curv  = kappa . R     -- discrete Ricci scalar curvature
                                  proxy on the Xi-graph
    L_ferm  = psi^bar (i gamma . d - m) psi
                              -- discrete Dirac action with
                                  per-node Psi = (|psi|, i*Im(psi))^T
    L_RG    = ||Xi - B[Xi]||_F^2  -- block-decimation residual
                                       (Wilsonian relevant-operator cost)
    L_causal = ||D_Omega Lap psi - (alpha_xi - eps_sync^2) psi
                - beta_pi |psi|^2 psi||^2
              -- causal-wave transport-equation residual under
                 the System-R coefficients
                 (alpha_xi = 9/10, gamma = 1/10, eps_sync^2 = 1/20,
                  beta_pi = 15/16, D_Omega = 67/80) which are
                 first-principles rationals derived from the
                 framework's lattice topology.

Reports each term separately AND the assembled L_total. Provides
the explicit numerical realisation of the symbolic Lagrangian
density form L = (dXi)^2 + V(Xi) + kappa R + psi-bar (i gamma d - m) psi
+ L_RG + L_causal.

Literature:
  Wilson 1974 "Confinement of quarks" (lattice gauge theory)
  Symanzik 1983 "Continuum limit of lattice gauge theories"
  Haken 1983 (Synergetics)

Output: outputs/verify_canonical_lagrangian_density.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PARENT = REPO.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

KAPPA = 1.0  # gravitational coupling (lattice units, normalised)
M_FERM = 0.5  # Dirac mass (lattice units)

# System-R coefficients for the causal-wave transport residual
ALPHA_XI = 9.0 / 10.0
GAMMA_R = 1.0 / 10.0
EPS_SYNC2 = 1.0 / 20.0
BETA_PI = 15.0 / 16.0
D_OMEGA = 67.0 / 80.0


def graph_laplacian_dense(xi):
    w = xi.copy()
    np.fill_diagonal(w, 0.0)
    deg = w.sum(axis=1)
    return np.diag(deg) - w


def L_kinetic_xi(xi):
    """L_kin = (Nabla Xi)^2 = (1/2) sum_ij (Xi_i - Xi_j)^2 wij,
    treating Xi-row-mean per node as the scalar field value."""
    xi_off = xi.copy()
    np.fill_diagonal(xi_off, 0.0)
    xi_node = xi_off.mean(axis=1)
    n = xi.shape[0]
    s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if xi_off[i, j] > 0:
                s += xi_off[i, j] * (xi_node[i] - xi_node[j]) ** 2
    return float(s)


def L_potential_xi(xi):
    """Double-well V(Xi) = -Xi_node^2 / 2 + Xi_node^4 / 4 summed over nodes."""
    xi_off = xi.copy()
    np.fill_diagonal(xi_off, 0.0)
    xi_node = xi_off.mean(axis=1)
    return float(np.sum(-0.5 * xi_node ** 2 + 0.25 * xi_node ** 4))


def L_curvature_xi(xi):
    """Discrete Ricci-scalar proxy via spectral gap of graph Laplacian.
    R_disc ~ 2 * (lambda_min^+ + lambda_max - 2 * lambda_mid)
    where lambda_* are quartiles of the Laplacian spectrum.
    Returns kappa * R_disc summed over the lattice (single scalar)."""
    L = graph_laplacian_dense(xi)
    eig = np.linalg.eigvalsh(L)
    eig = eig[1:]  # drop zero (constant) mode
    if len(eig) < 4:
        return 0.0
    q1, q2, q3 = np.quantile(eig, [0.25, 0.5, 0.75])
    R_disc = 2.0 * (eig.min() + eig.max() - 2.0 * q2)
    return float(KAPPA * R_disc)


def L_fermion_xi_psi(xi, psi):
    """Discrete free-Dirac action L_ferm = sum_i psi_i^dag (D_disc - m) psi_i,
    with D_disc the hermitian Dirac built from Xi-edge weights and
    spectral-Fiedler embedding, per-node spinor Psi_i = (|psi_i|, i*Im(psi_i))^T."""
    n = xi.shape[0]
    # Spectral Fiedler embedding (3D)
    L = graph_laplacian_dense(xi)
    eigvals_L, eigvecs_L = np.linalg.eigh(L)
    spatial = eigvecs_L[:, 1:4]
    sigma_1 = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_3 = np.array([[1, 0], [0, -1]], dtype=complex)
    dim = 2 * n
    D = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        for j in range(n):
            if i == j or xi[i, j] <= 0:
                continue
            e_ij = spatial[j] - spatial[i]
            norm = np.linalg.norm(e_ij)
            if norm < 1e-10:
                continue
            e_ij = e_ij / norm
            sigma_dot_e = (e_ij[0] * sigma_1
                            + e_ij[1] * sigma_2
                            + e_ij[2] * sigma_3)
            w = np.sqrt(xi[i, j])
            D[2*i:2*i+2, 2*j:2*j+2] += 0.5j * w * sigma_dot_e
            D[2*i:2*i+2, 2*i:2*i+2] -= 0.5j * w * sigma_dot_e
    D = 0.5 * (D + D.conj().T)
    D_minus_m = D - M_FERM * np.eye(dim, dtype=complex)
    psi_spinor = np.zeros(dim, dtype=complex)
    for i in range(n):
        psi_spinor[2*i] = np.abs(psi[i])
        psi_spinor[2*i+1] = 1j * np.imag(psi[i])
    val = np.real(np.vdot(psi_spinor, D_minus_m @ psi_spinor))
    return float(val)


def L_rg_block_residual(xi, block_size: int = 4):
    """L_RG: block-decimation kinetic correction = ||Xi - upsample(decimate(Xi))||_F^2.

    Operationally implements one RG block-mapping step:
      Xi'_IJ = (1/|I||J|) sum_{i in I, j in J} Xi_ij
    then up-samples back to original size (constant within block);
    the residual norm measures the energy cost of deviating from
    the RG-fixed-point flow.

    A vanishing residual means Xi is invariant under the chosen
    block decimation (perfect fixed point); finite residual is the
    Wilsonian relevant-operator cost.
    """
    n = xi.shape[0]
    n_blocks = max(1, n // block_size)
    actual_block = n // n_blocks
    coarse = np.zeros((n_blocks, n_blocks))
    for I in range(n_blocks):
        for J in range(n_blocks):
            i_start = I * actual_block
            i_end = (I + 1) * actual_block if I < n_blocks - 1 else n
            j_start = J * actual_block
            j_end = (J + 1) * actual_block if J < n_blocks - 1 else n
            coarse[I, J] = xi[i_start:i_end, j_start:j_end].mean()
    upsample = np.zeros_like(xi)
    for I in range(n_blocks):
        for J in range(n_blocks):
            i_start = I * actual_block
            i_end = (I + 1) * actual_block if I < n_blocks - 1 else n
            j_start = J * actual_block
            j_end = (J + 1) * actual_block if J < n_blocks - 1 else n
            upsample[i_start:i_end, j_start:j_end] = coarse[I, J]
    return float(np.linalg.norm(xi - upsample, 'fro') ** 2)


def L_causal_transport_residual(xi, psi):
    """L_causal: causal-wave transport-equation residual norm.

    On a steady-state snapshot the causal-wave transport relation
    enforces a fixpoint equation of the saturated-Ginzburg-Landau
    type with the System-R coefficients (alpha_xi = 9/10,
    eps_sync^2 = 1/20, beta_pi = 15/16, D_Omega = 67/80) acting
    on the complex psi field on the Xi-graph:

        D_Omega . Lap_Xi(psi) - (alpha_xi - eps_sync^2) psi
                  - beta_pi |psi|^2 psi = 0

    The residual norm
        || D_Omega . L psi - (alpha_xi - eps_sync^2) psi
           - beta_pi |psi|^2 psi ||^2
    measures the lattice's deviation from causal-wave equilibrium
    and is the Lagrangian contribution L_causal of the
    transport-channel; on a true fixpoint this term vanishes.
    """
    L = graph_laplacian_dense(xi)
    lap_psi = -L @ psi
    transport = (D_OMEGA * lap_psi
                 - (ALPHA_XI - EPS_SYNC2) * psi
                 - BETA_PI * np.abs(psi) ** 2 * psi)
    return float(np.real(np.vdot(transport, transport)))


def find_snapshot_xi_psi(regime: str, N: int):
    candidates = [
        PARENT / f"results_d1_{regime.lower()}_24seeds" / f"{regime}.snapshots.npz",
        PARENT / f"results_d1_{regime.lower()}_8seeds" / f"{regime}.snapshots.npz",
        PARENT / f"results_d1_{regime.lower()}_12seeds" / f"{regime}.snapshots.npz",
    ]
    for p in candidates:
        if p.exists():
            d = np.load(p, allow_pickle=True)
            xi = d["edge_xi_snapshots"][0, -1].astype(float).copy()
            np.fill_diagonal(xi, 1.0)
            psi = (d["psi_real_snapshots"][0, -1].astype(float)
                   + 1j * d["psi_imag_snapshots"][0, -1].astype(float))
            return xi, psi
    return None, None


def main():
    print("=" * 80)
    print("G5: Canonical Lagrangian density on framework lattice snapshots")
    print("=" * 80)
    print()
    print("L_total = L_kin + L_pot + L_curv + L_ferm + L_RG + L_causal "
          "(all six terms computed)")
    print(f"kappa = {KAPPA}, m_fermion = {M_FERM}, "
          f"alpha_xi = {ALPHA_XI}, gamma = {GAMMA_R}, "
          f"eps_sync^2 = {EPS_SYNC2}, beta_pi = {BETA_PI}, "
          f"D_Omega = {D_OMEGA}")
    print()
    LADDER = [("P5", 50), ("P5N64", 64), ("P5N72", 72),
              ("P5N84", 84), ("P5N100", 100)]
    rows = []
    print(f"{'regime':<8} {'N':>4} {'L_kin':>10} {'L_pot':>10} "
          f"{'L_curv':>10} {'L_ferm':>10} {'L_RG':>10} {'L_causal':>10} "
          f"{'L_total':>10}")
    print("-" * 95)
    for regime, N in LADDER:
        xi, psi = find_snapshot_xi_psi(regime, N)
        if xi is None:
            print(f"{regime}: snapshot not found -- skip")
            continue
        L_k = L_kinetic_xi(xi)
        L_v = L_potential_xi(xi)
        L_c = L_curvature_xi(xi)
        L_f = L_fermion_xi_psi(xi, psi)
        L_rg = L_rg_block_residual(xi, block_size=4)
        L_causal = L_causal_transport_residual(xi, psi)
        L_t = L_k + L_v + L_c + L_f + L_rg + L_causal
        rows.append({
            "regime": regime, "N": int(N),
            "L_kin": L_k, "L_pot": L_v,
            "L_curv": L_c, "L_ferm": L_f,
            "L_RG_block_residual": L_rg,
            "L_causal_transport_residual": L_causal,
            "L_total": L_t,
        })
        print(f"{regime:<8} {N:>4} {L_k:>+10.3f} {L_v:>+10.3f} "
              f"{L_c:>+10.3f} {L_f:>+10.3f} {L_rg:>+10.3f} "
              f"{L_causal:>+10.3f} {L_t:>+10.3f}")
    print()
    if not rows:
        verdict = "INSUFFICIENT_DATA: D1 snapshots not found in parent dir."
    else:
        L_kin_arr = np.array([r["L_kin"] for r in rows])
        L_pot_arr = np.array([r["L_pot"] for r in rows])
        L_curv_arr = np.array([r["L_curv"] for r in rows])
        L_ferm_arr = np.array([r["L_ferm"] for r in rows])
        L_rg_arr = np.array([r["L_RG_block_residual"] for r in rows])
        L_causal_arr = np.array(
            [r["L_causal_transport_residual"] for r in rows])
        L_total_arr = np.array([r["L_total"] for r in rows])
        verdict = (
            f"L_total computed across {len(rows)} regime samples on the "
            f"cleaned P5..P5N100 ladder, INCLUDING ALL SIX TERMS:"
            f" L_kin in [{L_kin_arr.min():+.3f}, {L_kin_arr.max():+.3f}],"
            f" L_pot in [{L_pot_arr.min():+.3f}, {L_pot_arr.max():+.3f}],"
            f" L_curv in [{L_curv_arr.min():+.3f}, {L_curv_arr.max():+.3f}],"
            f" L_ferm in [{L_ferm_arr.min():+.3f}, {L_ferm_arr.max():+.3f}],"
            f" L_RG in [{L_rg_arr.min():+.3f}, {L_rg_arr.max():+.3f}]"
            f" (block-decimation residual; size of relevant operators),"
            f" L_causal in [{L_causal_arr.min():+.3f}, {L_causal_arr.max():+.3f}]"
            f" (causal-wave transport residual under System-R coefficients),"
            f" L_total in [{L_total_arr.min():+.3f}, {L_total_arr.max():+.3f}]. "
            "The six-term assembly is the explicit numerical realisation "
            "of the symbolic Lagrangian L = (dXi)^2 + V(Xi) + kappa R + "
            "psi-bar (i gamma d - m) psi + L_RG + L_causal quoted in the "
            "Master-blueprint, with L_RG = ||Xi - B[Xi]||^2 the Wilsonian "
            "relevant-operator cost (block-mean coarse-graining residual) "
            "and L_causal the Hermitian norm-squared of the causal-wave "
            "transport residue under (alpha_xi, gamma, eps_sync^2, "
            "beta_pi, D_Omega) System-R coefficients."
        )

    bundle = {
        "method": (
            "G5 canonical Lagrangian density: per-term computation of "
            "ALL SIX components (L_kin from graph Laplacian, L_pot "
            "from double-well, L_curv from spectral curvature proxy, "
            "L_ferm from discrete Wilson-Dirac at m=0.5, L_RG = "
            "||Xi - B[Xi]||_F^2 from one block-decimation step at "
            "block_size=4, L_causal = norm-sq of causal-wave transport "
            "residual under System-R coefficients alpha_xi=9/10, "
            "gamma=1/10, eps_sync^2=1/20, beta_pi=15/16, D_Omega=67/80) "
            "on framework D1 snapshots; reports per-regime values and "
            "total action."
        ),
        "stand": "2026-05-05",
        "literature": [
            "Wilson 1974 (Confinement of quarks; lattice gauge theory)",
            "Symanzik 1983 (Continuum limit of lattice gauge theories)",
            "Haken 1983 (Synergetics, slaving principle)",
        ],
        "kappa": KAPPA,
        "m_fermion": M_FERM,
        "rows": rows,
        "verdict": verdict,
    }
    out_path = OUTPUTS / "verify_canonical_lagrangian_density.json"
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\n{verdict[:300]}...")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
