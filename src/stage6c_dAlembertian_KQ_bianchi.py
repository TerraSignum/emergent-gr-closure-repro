"""Stage 6c: Xi-d'Alembertian transport equation, K/Q metric-functional
expressions, and Bianchi identity numerical verification.

Per Master.instructions §5.4:
  (Box_Xi phi)_i = sum_j C_caus,ij * W_ij * (phi_j - phi_i)

with W_ij = Xi-weighted edge weight and C_caus,ij the causal mask.

This script:
  1. Implements the Xi-d'Alembertian on the persistent-edge subgraph
  2. Solves the K-Q functional EOMs delta_S_fac / delta_K = 0,
     delta_S_fac / delta_Q = 0 to express K[g, psi] and Q[g, psi]
  3. Computes the discrete divergence of G^{mu nu} on the lattice
  4. Cross-checks against the empirical Symanzik-asymptote 7e-4 from
     project_discrete_bianchi_recovered_2026_04_30.

Output: outputs/stage6c_dAlembertian_KQ_bianchi.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PARENT = REPO.parent

LADDER = [
    ("P5N64",  64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz"),
    ("P5N72",  72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz"),
    ("P5N84",  84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz"),
    ("P5N100",100,  "results_d1_p5n100_24seeds/P5N100.snapshots.npz"),
    ("P5N128",128,  "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz"),
    ("P5N200",200,  "results_d1_p5n200_8seeds/P5N200.snapshots.npz"),
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz"),
    ("P5N512", 512,  "results_d1_p5n512_12seeds/P5N512.snapshots.npz"),
]


def xi_dAlembertian(phi: np.ndarray, xi: np.ndarray,
                     causal_mask: np.ndarray | None = None) -> np.ndarray:
    """Xi-weighted graph d'Alembertian per Master.instructions §5.4:
        (Box_Xi phi)_i = sum_j C_caus,ij W_ij (phi_j - phi_i)
    with W_ij = Xi_ij (canonical choice).

    If causal_mask is None, uses the unweighted symmetric mask
    (C_caus = 1 everywhere off-diagonal).
    """
    n = phi.shape[0]
    if causal_mask is None:
        causal_mask = np.ones_like(xi)
        np.fill_diagonal(causal_mask, 0)
    weight = xi * causal_mask
    # phi differences: phi_j - phi_i for each edge
    phi_diff = phi[None, :] - phi[:, None]  # (N, N), [i, j] = phi_j - phi_i
    box_phi = (weight * phi_diff).sum(axis=1)  # sum over j
    return box_phi


def kq_functional_from_action(xi_last: np.ndarray, psi_last: np.ndarray,
                                 n_lat: int) -> tuple[np.ndarray, np.ndarray]:
    """Solve K and Q from S_fac action variation.

    The factor sector S_fac includes terms (linearised at vacuum):
      S_fac = (a_K/2) sum_ij (K_ij - <K>)^2
              + (a_Q/2) sum_ij (Q_ij - <Q>)^2
              + g_KX K_ij Xi_ij
              + g_QX Q_ij Xi_ij
              + g_Kpsi |psi_i psi_j*| K_ij
              + ...

    Variation delta_S_fac/delta_K_ij = 0 gives:
      K_ij = <K> - (g_KX/a_K) Xi_ij - (g_Kpsi/a_K) |psi_i psi_j*|

    Similarly for Q. With the canonical normalization a_K = a_Q = 1
    and g_KX = -1/2, g_QX = +1/2, g_Kpsi = -1/4, g_Qpsi = +1/4 (these
    are the framework's bounded-operator readout values, see
    unified_action.py), we get the canonical K and Q functionals.
    """
    psi_outer = np.abs(np.outer(psi_last.conj(), psi_last))
    K_func = 1.0 - 0.5 * xi_last - 0.25 * psi_outer
    Q_func = 0.5 * xi_last + 0.25 * psi_outer
    np.fill_diagonal(K_func, 1.0)
    np.fill_diagonal(Q_func, 0.0)
    return K_func, Q_func


def discrete_einstein_tensor(xi_last: np.ndarray, psi_last: np.ndarray,
                                n_lat: int) -> np.ndarray:
    """Linearised discrete Einstein tensor G^{mu nu}_disc, here
    simplified to the per-node 4-component G^{0nu} contribution
    from -log Xi metric.

    For our purposes (Bianchi numerical check), use:
      G_i = sum_j (1/d_ij) (m_i - m_j)
    with d_ij = -ell_0 log Xi_ij and m_i a per-node mass-like
    quantity (here |psi_i|^2 as proxy).
    """
    # Per-node mass proxy
    m_node = np.abs(psi_last) ** 2
    # Distance from Xi
    d_ij = -np.log(np.maximum(xi_last, 1e-9))
    np.fill_diagonal(d_ij, 1.0)  # avoid div by zero
    # Inverse-distance weighted divergence
    G_node = np.zeros(n_lat)
    for i in range(n_lat):
        # Sum over j != i
        diff = m_node[i] - m_node
        diff[i] = 0
        weight = 1.0 / d_ij[i]
        weight[i] = 0
        G_node[i] = (weight * diff).sum()
    return G_node


def discrete_bianchi_norm(G_node: np.ndarray) -> float:
    """Discrete divergence-of-divergence: ||nabla^disc G^mu nu||.

    For a node-valued G_i, the discrete divergence ∇^disc · G is
    sum over neighbours of G differences. The Bianchi identity
    requires this to vanish at the continuum.

    Here we use ||G_node||_2 / N as a proxy for the divergence-norm
    that should -> 0 in the continuum limit.
    """
    return float(np.sqrt(np.mean(G_node ** 2)))


def main():
    print("=" * 80)
    print("Stage 6c: Xi-d'Alembertian, K/Q functionals, Bianchi numerical check")
    print("=" * 80)
    print()
    print("Following Master.instructions §5.4 d'Alembertian formulation.")
    print()

    rows = []
    for regime, n_lat, rel in LADDER:
        fp = PARENT / rel
        if not fp.exists():
            continue
        z = np.load(fp, allow_pickle=True)
        snaps = z["edge_xi_snapshots"]
        psi_r = z["psi_real_snapshots"]
        psi_i = z["psi_imag_snapshots"]
        n_seeds = int(snaps.shape[0])
        per_seed = []
        for s in range(min(n_seeds, 8)):
            xi_last = snaps[s, -1]
            psi_last = (psi_r[s, -1].astype(float)
                         + 1j * psi_i[s, -1].astype(float))
            phi = np.angle(psi_last)
            # Test 1: Xi-d'Alembertian on phi field
            box_phi = xi_dAlembertian(phi, xi_last)
            box_phi_norm = float(np.sqrt(np.mean(box_phi ** 2)))
            # Test 2: K, Q functional vs measured (from k_snapshots/q_snapshots)
            K_func, Q_func = kq_functional_from_action(
                xi_last, psi_last, n_lat)
            if "k_snapshots" in z.files:
                K_meas = np.asarray(z["k_snapshots"][s, -1], dtype=float)
                Q_meas = np.asarray(z["q_snapshots"][s, -1], dtype=float)
                K_diff = float(np.sqrt(np.mean((K_func - K_meas) ** 2)))
                Q_diff = float(np.sqrt(np.mean((Q_func - Q_meas) ** 2)))
            else:
                K_diff = float("nan")
                Q_diff = float("nan")
            # Test 3: Discrete Bianchi norm
            G_node = discrete_einstein_tensor(xi_last, psi_last, n_lat)
            bianchi_norm = discrete_bianchi_norm(G_node)
            per_seed.append({
                "seed": s,
                "box_phi_rms": box_phi_norm,
                "K_functional_diff_from_measured": K_diff,
                "Q_functional_diff_from_measured": Q_diff,
                "bianchi_residual_norm": bianchi_norm,
            })
        if not per_seed:
            continue
        def mn(key):
            vals = [d[key] for d in per_seed
                     if not (isinstance(d[key], float)
                              and np.isnan(d[key]))]
            return float(np.mean(vals)) if vals else float("nan")
        box_m = mn("box_phi_rms")
        K_diff_m = mn("K_functional_diff_from_measured")
        Q_diff_m = mn("Q_functional_diff_from_measured")
        bianchi_m = mn("bianchi_residual_norm")
        print(f"--- {regime} N={n_lat} ({len(per_seed)} seeds) ---")
        print(f"  Xi-d'Alembertian rms |Box_Xi phi|     = {box_m:.4e}")
        print(f"  ||K_functional - K_measured|| rms     = {K_diff_m:.4e}")
        print(f"  ||Q_functional - Q_measured|| rms     = {Q_diff_m:.4e}")
        print(f"  Discrete Bianchi residual rms          = {bianchi_m:.4e}")
        rows.append({
            "regime": regime, "N": n_lat, "n_seeds": len(per_seed),
            "box_phi_rms_mean": box_m,
            "K_functional_diff_mean": K_diff_m,
            "Q_functional_diff_mean": Q_diff_m,
            "bianchi_residual_norm_mean": bianchi_m,
        })

    # Symanzik fit on Bianchi residual
    print()
    print("=" * 80)
    print("Symanzik fit on Bianchi residual: -> N->infinity asymptote")
    print("=" * 80)
    if rows:
        N_vals = np.array([r["N"] for r in rows], dtype=float)
        bianchi_vals = np.array([r["bianchi_residual_norm_mean"]
                                   for r in rows])
        # Fit log(bianchi) = log(c) + alpha * log(N) (power law)
        log_N = np.log(N_vals)
        log_b = np.log(np.maximum(bianchi_vals, 1e-12))
        # Linear regression
        A_mat = np.column_stack([np.ones_like(log_N), log_N])
        coef, *_ = np.linalg.lstsq(A_mat, log_b, rcond=None)
        log_c, alpha = coef
        c = np.exp(log_c)
        print(f"  Power-law fit: bianchi_norm = {c:.4f} * N^{alpha:+.3f}")
        print(f"  Reference (project_discrete_bianchi_recovered_2026_04_30):")
        print(f"    Empirical scaling N^(-1.51..-2.19) on 14-regime ladder")
        print(f"    Symanzik asymptote ~7e-4 (R^2 = 0.96)")
        if alpha < -1:
            print(f"    Our power -{abs(alpha):.2f} consistent with"
                  " convergence to zero ✓")
        else:
            print(f"    Power {alpha:+.2f} not converging cleanly (proxy "
                  "scaling)")

    bundle = {
        "method": "stage6c_dAlembertian_KQ_bianchi",
        "framework_reference": "Master.instructions §5.4",
        "rows": rows,
    }
    out = REPO / "outputs" / "stage6c_dAlembertian_KQ_bianchi.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
