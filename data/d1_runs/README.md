# `data/d1_runs/` — D1 Lattice NPZ Files (Tier-2 reproduction)

This directory is the **bundled location** for the per-regime D1
lattice state files used by the per-node 4×4 Galerkin Frobenius
residual scripts (Runner A Hessian-Ricci, Runner B 3-Λ variants,
Runner D basis-invariance, and several supporting analyses). Without
the NPZ files in this directory, those scripts gracefully degrade and
the test suite verifies the recomputed numbers against the bundled
frozen JSON certificates in `data/galerkin_*.json` rather than
re-running the per-node Galerkin pipeline.

## Expected files

| Filename | Regime | $N$ | Approx. size |
|---|---|---|---|
| `d1_p0.npz` | P0 | 18 | 4 MB |
| `d1_p1.npz` | P1 | 28 | 25 MB |
| `d1_p2prime.npz` | P2′ | 30 | 28 MB |
| `d1_p3.npz` | P3 | 36 | 49 MB |
| `d1_p4.npz` | P4 | 42 | 79 MB |
| `d1_p5.npz` | P5 | 50 | 134 MB |
| `d1_p6.npz` | P6 | 60 | 232 MB |
| `d1_p7.npz` | P7 | 72 | 288 MB |
| `d1_p8.npz` | P8 | 84 | 365 MB |

Total: approximately 1.15 GB.

## Optional within-regime extension files

| Filename | Regime | $N$ | Notes |
|---|---|---|---|
| `d1_p5n64.npz` | P5N64 | 64 | Multi-$N$ within-regime point at P5-physics |
| `d1_p5n100.npz` | P5N100 | 100 | Same; second within-regime point |

## Required NPZ field schema

Each NPZ file must contain at minimum the following fields (the
discovery helper `src/_d1_npz_discovery.py` does not validate
schemas — it only resolves paths):

- `dense_cell_edge_xi_values` — shape `(n_seeds, n_lat*(n_lat-1)/2)`
  flat upper-triangular Ξ-matrix per seed
- `dense_cell_node_amplitude_values` — shape `(n_seeds, n_lat)`
- `dense_cell_node_phase_values` — shape `(n_seeds, n_lat)`
- `R_bar` — shape `(n_seeds,)` per-seed lattice Ricci scalar
- `R_bar_by_level` — shape `(n_seeds, 3)` per-seed Ricci scalar at three
  coarse-graining levels (used for the cg-2 reading)
- `Delta_curv` — shape `(n_seeds,)` per-seed traceless-curvature
  diagnostic
- `Delta_curv_by_level` — shape `(n_seeds, 3)`
- `ff_K_seed<S>` — shape `(n_lat, n_lat)` per-edge $K(x)$ field
  (one entry per seed; `ff_K_seed0`, `ff_K_seed1`, ...)
- `ff_Q_seed<S>` — shape `(n_lat, n_lat)` per-edge $Q(x)$ field
- (additional fields are written by the lattice campaign but not used
  by the Galerkin scripts)

## How to populate

The NPZ files are produced by the underlying lattice campaign of the
parent Emergence project (not bundled in this repo because they
exceed the size threshold of typical reproducibility packages). When
available, place them into this directory and the Tier-2 Galerkin
scripts will pick them up automatically via the discovery helper.
For the bundled developer setup, the helper also falls back to
`<repo-parent>/d1_lattice_payload/`, `<repo-parent>/d1_lattice_payload/`,
`<repo-parent>/results_d1_p5n64/`, and `<repo-parent>/results_d1_p5n100/`.

## Without the NPZ files

The `tests/test_galerkin_*.py` suite verifies the bundled
`data/galerkin_runner_A_hessian_ricci.json`,
`data/galerkin_runner_B_lambda_variants.json`,
`data/galerkin_runner_D_basis_invariance.json` and friends against
their content schemas and tier-classification verdicts. The
`src/verify_galerkin_*.py` scripts can be invoked but will print a
helpful message and exit cleanly when their NPZ input is not found.
