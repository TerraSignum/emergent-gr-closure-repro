# Archive 2026-04-30 — orphan figures from P4 trim

These figure files were referenced by sections that were removed
or substantially rewritten during the 2026-04-29/30 P4 reduction
work (sec:appB removal, sec:ppn delegation, sec:bh delegation,
fig11 regeneration with P5N100 datapoint integrated).

They are kept here in case any future revision wants to recover
them; the parent paper now references only figures that survive
the live `paper/figures/` directory.

| File | Original use |
|---|---|
| `fig10_galerkin_per_node_full.{pdf,png}` | older variant of the per-node Galerkin convergence plot, superseded by `fig11_runner_A_hessian_ricci` |
| `fig4b_chirality_multi_n.{pdf,png}` | chirality-deviation multi-N plot, content now in `emergent-gr-h3c-witnesses-repro` companion |
| `fig4b_four_verification_paths.{pdf,png}` | four-path verification chain, content split between parent fig4 and the H3c companion |
| `fig9_delta_E_9point_frobenius.{pdf,png}` | nine-point Frobenius ladder, content in fig11 + H3c companion |
| `fig_regime_artifact_audit.png` | regime-artifact audit, content text-only in current revision |

To restore any of these, move the file back to
`paper/figures/` and add the `\includegraphics{...}` reference
in `paper/manuscript.tex`.
