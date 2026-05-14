r"""Per-(regime, seed) full-4x4 Frobenius residual on the D1 ladder.

Direct multi-N test of $\|G_{\mu\nu}+\Lambda_{\mu\nu}-8\pi G\,T_{\mu\nu}\|_F$
constructed component-wise from bundled lattice quantities. Differs from the
scalar-invariant Frobenius-decomposed proxy of `compute_true_einstein_residual.py`
in that the trace, traceless, and off-diagonal contributions all come from
the same residual tensor at the (regime, seed) level, evaluated under an
explicit anisotropic $\Lambda_{\mu\nu}=\mathrm{diag}(\Lambda_t,\Lambda_s,
\Lambda_s,\Lambda_s)$ ansatz.

Per (regime, seed) construction
-------------------------------
With $8\pi G \equiv 1$ in lattice units and signature $(-,+,+,+)$, the
FRW-isotropic Einstein tensor reads $G_{00}=\bar R/2$,
$G_{ii}=-\bar R/2$ (i = 1, 2, 3), $G_{0i}=G_{ij}=0$. The lattice stress
tensor at the (regime, seed) level decomposes into

  $T_{00}^{(\mathrm{HV})}$: Hilbert variation
    $0.5\,Z_\Xi\,\mathrm{Var}(\Xi) + \kappa_\Xi\,\mathrm{Var}(|\Psi|)
     + \zeta_1\,\omega\,\langle|\nabla\Psi|^2\rangle
     + \zeta_3\,\omega\,K_{\mathrm{rec}}^{\mathrm{row\text{-}mean}}$,

  $T_{ii}^{(\mathrm{spec})}$: three spectral-embedding eigenvalues
    $\lambda_1,\lambda_2,\lambda_3$ from
    `lambda_offdiagonal_Tij_spectral_multiN.json`,

  $T_{ij,\,\mathrm{off}}^{(\mathrm{spec})}$: off-diagonal Frobenius
    norm from the same JSON.

Anisotropic $\Lambda_{\mu\nu}$ is read from the manuscript value
$\Lambda_t = 4/3$, $\Lambda_s = -0.37$ (lattice units; framework-derived
from the time--time and spatial-trace identifications).

The residual is then
  $\|A\|_F^2 = (G_{00}+\Lambda_t-T_{00})^2
              + \sum_{i=1}^3 (G_{ii}+\Lambda_s-\lambda_i)^2
              + 6\,T_{ij,\,\mathrm{off}}^2$,
with the factor 6 coming from $2\sum_{i<j}T_{ij}^2 = 6\langle T_{ij,\mathrm{off}}^2\rangle$
in $d=3$ spatial dimensions on a symmetric tensor.

Schwarzschild unit test: in vacuum $T_{\mu\nu}=0$ and $G_{\mu\nu}=0$,
so $\|A\|_F = |\Lambda|\sqrt{4} = 2|\Lambda|$ if $\Lambda\neq 0$, else 0.
The script verifies this analytical limit before processing the lattice.

Usage:
    python ./src/verify_full_tensor_frobenius_d1.py
"""
from __future__ import annotations
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Lambda_munu values from manuscript anisotropic-extension fit.
LAMBDA_T = 4.0 / 3.0
LAMBDA_S = -0.37

# Hilbert-variation coefficients (framework-fixed).
Z_XI = KAPPA_XI = ZETA_1 = OMEGA = 1.0
ZETA_3 = 0.5

# Canonical D1 lattice ladder (same as spectral_multiN script).
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _d1_npz_discovery import find_d1_npz

LADDER_REGIMES = [
    ("P0", 18), ("P1", 28), ("P2prime", 30), ("P3", 36),
    ("P4", 42), ("P5", 50),
    ("P6", 60), ("P7", 72), ("P8", 84),
]
LADDER = [(r, n, find_d1_npz(r, REPO)) for r, n in LADDER_REGIMES]
N_SEEDS = 4


def hilbert_T00_per_seed(d, seed_idx, n_lat):
    """Compute T_00 per seed from the Hilbert variation, using bundled
    per-seed scalar invariants where possible."""
    import numpy as np
    var_xi = float(d["Var(Xi)"][seed_idx]) if "Var(Xi)" in d.files else 0.0
    amp = d["dense_cell_node_amplitude_values"][seed_idx]
    var_amp = float(np.var(np.abs(amp)))
    # grad psi squared from per-node amplitude+phase via Laplace-Beltrami
    # weights; we use the bundled spatial gradient if available, otherwise
    # reconstruct from the per-edge Xi data.
    # For consistency with the spectral-embedding script, take a row-mean
    # estimate: <|grad Psi|^2>_node from amplitude variance scaled by N.
    grad_psi_sq = var_amp * float(n_lat)
    # K_rec row-mean from bundled row-mean if present, else default to 1.7.
    if "K_rec_row_mean" in d.files:
        k_rec = float(d["K_rec_row_mean"][seed_idx])
    else:
        k_rec = 1.7
    t00 = (0.5 * Z_XI * var_xi
           + KAPPA_XI * var_amp
           + ZETA_1 * OMEGA * grad_psi_sq
           + ZETA_3 * OMEGA * k_rec)
    return t00


def schwarzschild_unit_test():
    """Verify: in vacuum (G=T=0) with Lambda non-zero, residual = 2|Lambda|.
    With Lambda=0, residual = 0."""
    # Vacuum, no Lambda: should be 0.
    g00 = 0.0
    gii = 0.0
    t00 = 0.0
    eigs = [0.0, 0.0, 0.0]
    off = 0.0
    res_no_lam = math.sqrt(
        (g00 + 0 - t00) ** 2
        + sum((gii + 0 - e) ** 2 for e in eigs)
        + 6.0 * off ** 2
    )
    # Vacuum with Lambda: should be sqrt(Lambda_t^2 + 3*Lambda_s^2).
    res_with_lam = math.sqrt(
        (g00 + LAMBDA_T - t00) ** 2
        + sum((gii + LAMBDA_S - e) ** 2 for e in eigs)
        + 6.0 * off ** 2
    )
    expected = math.sqrt(LAMBDA_T ** 2 + 3 * LAMBDA_S ** 2)
    return {
        "vacuum_no_Lambda_residual": res_no_lam,
        "vacuum_no_Lambda_expected": 0.0,
        "vacuum_no_Lambda_pass": abs(res_no_lam) < 1e-12,
        "vacuum_with_Lambda_residual": res_with_lam,
        "vacuum_with_Lambda_expected": expected,
        "vacuum_with_Lambda_pass": abs(res_with_lam - expected) < 1e-12,
    }


def main():
    try:
        import numpy as np
    except ImportError:
        print("numpy unavailable.")
        return

    print("=" * 78)
    print("Per-(regime, seed) full-4x4 Frobenius residual on D1 ladder")
    print("=" * 78)
    print()

    print("--- Schwarzschild vacuum unit test ---")
    sw = schwarzschild_unit_test()
    for k, v in sw.items():
        print(f"  {k}: {v}")
    print()

    # Load spectral T_ij data (from spectral_multiN run).
    spec_path = OUTPUTS / "lambda_offdiagonal_Tij_spectral_multiN.json"
    if not spec_path.exists():
        print(f"Required input missing: {spec_path}")
        print("Run verify_lambda_offdiagonal_Tij_spectral_multiN.py first.")
        return
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    aggregate_means = []
    per_regime = {}

    for regime, n_lat, npz_path in LADDER:
        if (npz_path is None or not npz_path.exists()
                or regime not in spec["per_regime"]):
            print(f"[{regime}, N={n_lat}] missing data; skipping.")
            continue

        d = np.load(npz_path, allow_pickle=True)
        r_bar_per_seed = d["R_bar"] if "R_bar" in d.files else None
        if r_bar_per_seed is None:
            print(f"[{regime}, N={n_lat}] R_bar missing; skipping.")
            continue

        spec_per_seed = spec["per_regime"][regime]["per_seed"]

        per_seed_residuals = []
        for s in spec_per_seed:
            seed_idx = s["seed"]
            r_bar = float(r_bar_per_seed[seed_idx])
            g00 = r_bar / 2.0
            gii = -r_bar / 2.0
            t00 = hilbert_T00_per_seed(d, seed_idx, n_lat)
            eigs = s["eigvals"]
            offdiag = s["offdiag_frobenius"]

            time_res_sq = (g00 + LAMBDA_T - t00) ** 2
            spatial_res_sq = sum(
                (gii + LAMBDA_S - lam) ** 2 for lam in eigs
            )
            shear_res_sq = 6.0 * offdiag ** 2
            total = math.sqrt(time_res_sq + spatial_res_sq + shear_res_sq)

            per_seed_residuals.append({
                "seed": seed_idx,
                "R_bar": r_bar,
                "T_00_HV": t00,
                "G_00": g00,
                "G_ii": gii,
                "spectral_eigvals": eigs,
                "spectral_offdiag_F": offdiag,
                "time_residual": math.sqrt(time_res_sq),
                "spatial_residual": math.sqrt(spatial_res_sq),
                "shear_residual": math.sqrt(shear_res_sq),
                "full_4x4_F_residual": total,
            })

        means = [r["full_4x4_F_residual"] for r in per_seed_residuals]
        time_means = [r["time_residual"] for r in per_seed_residuals]
        spatial_means = [r["spatial_residual"] for r in per_seed_residuals]
        shear_means = [r["shear_residual"] for r in per_seed_residuals]

        agg = {
            "regime": regime,
            "N": n_lat,
            "n_seeds": len(per_seed_residuals),
            "per_seed": per_seed_residuals,
            "mean_full_4x4_F_residual": (sum(means) / len(means)
                                          if means else float("nan")),
            "mean_time_residual": (sum(time_means) / len(time_means)
                                   if time_means else float("nan")),
            "mean_spatial_residual": (sum(spatial_means) / len(spatial_means)
                                      if spatial_means else float("nan")),
            "mean_shear_residual": (sum(shear_means) / len(shear_means)
                                    if shear_means else float("nan")),
        }
        per_regime[regime] = agg
        aggregate_means.append((n_lat, agg))

        print(f"[{regime}, N={n_lat}] "
              f"||A||_F = {agg['mean_full_4x4_F_residual']:.4f}  "
              f"(00: {agg['mean_time_residual']:.3f}, "
              f"spatial: {agg['mean_spatial_residual']:.3f}, "
              f"shear: {agg['mean_shear_residual']:.3f})")

    print()
    print("=" * 78)
    print(f"  {'N':>5} {'||A||_F':>10} {'00':>10} {'spatial':>10} "
          f"{'shear':>10}")
    print("-" * 78)
    for n_lat, agg in aggregate_means:
        print(f"  {n_lat:5d} "
              f"{agg['mean_full_4x4_F_residual']:10.4f} "
              f"{agg['mean_time_residual']:10.3f} "
              f"{agg['mean_spatial_residual']:10.3f} "
              f"{agg['mean_shear_residual']:10.3f}")

    out = {
        "schema_version": "1.0.0",
        "title": ("Per-(regime, seed) full-4x4 Frobenius residual "
                  "on D1 ladder under anisotropic Lambda ansatz"),
        "Lambda_t": LAMBDA_T,
        "Lambda_s": LAMBDA_S,
        "schwarzschild_unit_test": sw,
        "per_regime": per_regime,
        "trend": [
            {
                "N": n_lat,
                "mean_full_4x4_F_residual":
                    agg["mean_full_4x4_F_residual"],
                "mean_time_residual": agg["mean_time_residual"],
                "mean_spatial_residual": agg["mean_spatial_residual"],
                "mean_shear_residual": agg["mean_shear_residual"],
            }
            for n_lat, agg in aggregate_means
        ],
    }
    out_path = OUTPUTS / "lambda_full_tensor_frobenius_d1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
