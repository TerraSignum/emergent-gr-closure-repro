r"""Tau-threshold-sweep robustness audit for the (SG) skeleton closure.

The (SG) skeleton-Laplacian closure lambda_inf^skel = 7/24 was identified
at threshold tau = 0.10. Question: is the result robust to choice of tau?

Sweep tau in {0.05, 0.07, 0.10, 0.15, 0.20, 0.30} on existing snapshots
(no new compute). For each tau, compute skeleton-Laplacian lambda_2 per
seed, average across seeds, fit Symanzik-1 to extract asymptote.

A robust closure shows:
  - asymptote flat across tau in [0.07, 0.20]
  - asymptote drifts only at boundary (tau too small = full graph,
    tau too large = disconnected sub-graph)

Output: outputs/verify_tau_sweep_robustness.json + console.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
REPO = SRC.parent
OUTPUTS = REPO / "outputs"
GENERATED = REPO / "paper" / "generated"

sys.path.insert(0, str(SRC))
from _d1_ladder_discovery import discover_d1_ladder  # noqa: E402
from adaptive_pipeline import hierarchical_bayes  # noqa: E402

TAU_VALUES = [0.05, 0.07, 0.10, 0.15, 0.20, 0.30]
TARGET = 7.0 / 24.0  # registered skeleton closure


def skeleton_lambda2(xi: np.ndarray, tau: float) -> float | None:
    n = xi.shape[0]
    adj = (np.abs(xi - np.diag(np.diag(xi))) > tau).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    if adj.sum() == 0:
        return None
    deg = np.maximum(adj.sum(axis=1), 1e-12)
    d_inv = 1.0 / np.sqrt(deg)
    L = np.eye(n) - (d_inv[:, None] * adj * d_inv[None, :])
    L = 0.5 * (L + L.T)
    try:
        eigs = np.linalg.eigvalsh(L)
        return float(eigs[1])
    except np.linalg.LinAlgError:
        return None


def main():
    print("=" * 78)
    print(f"Tau-sweep robustness audit (target lambda_inf^skel = 7/24 = {TARGET:.5f})")
    print("=" * 78)
    ladder = discover_d1_ladder(REPO)
    # Use only regimes with at least 8 seeds (excluding tiny early ones)
    ladder = [(r, n, p) for (r, n, p) in ladder if n >= 100]

    results = {}
    for tau in TAU_VALUES:
        ns = []
        lams = []
        for regime, n_lat, npz_path in ladder:
            try:
                d = np.load(npz_path, allow_pickle=True)
                if "edge_xi_snapshots" not in d.files:
                    continue
                xi_arr = d["edge_xi_snapshots"][:, -1]
                per_seed = []
                for s in range(min(8, xi_arr.shape[0])):
                    xi = xi_arr[s].astype(np.float64)
                    val = skeleton_lambda2(xi, tau)
                    if val is not None:
                        per_seed.append(val)
                for v in per_seed:
                    ns.append(n_lat)
                    lams.append(v)
            except OSError:
                continue
        ns_arr = np.asarray(ns)
        lams_arr = np.asarray(lams)
        if len(lams) < 4:
            print(f"  tau={tau:.2f}: insufficient data ({len(lams)} obs)")
            continue
        cmp = hierarchical_bayes.model_comparison(ns_arr, lams_arr)
        best_key = cmp["selected"]
        best = cmp[best_key]
        mu, sigma = best["asymptote_mean"], best["asymptote_std"]
        z = (mu - TARGET) / max(sigma, 1e-12)
        results[f"tau_{tau:.2f}"] = {
            "tau": tau,
            "n_obs": int(len(lams)),
            "n_regimes": len(set(ns)),
            "selected_model": best_key,
            "asymptote_mean": mu,
            "asymptote_std": sigma,
            "asymptote_ci95": [mu - 1.96 * sigma, mu + 1.96 * sigma],
            "z_vs_target": z,
            "rel_err_pct": (mu - TARGET) / TARGET * 100,
        }
        print(f"  tau={tau:.2f}: lambda_inf^skel = {mu:.5f} +/- {sigma:.5f}  "
              f"(z={z:+.2f}, rel {(mu-TARGET)/TARGET*100:+.2f}%, "
              f"{cmp['selected']}, {len(lams)} obs)")

    # Robustness verdict
    asymptotes = [v["asymptote_mean"] for v in results.values()]
    if asymptotes:
        rng = max(asymptotes) - min(asymptotes)
        mid_taus = ["tau_0.07", "tau_0.10", "tau_0.15", "tau_0.20"]
        mid_asy = [results[k]["asymptote_mean"] for k in mid_taus if k in results]
        mid_rng = max(mid_asy) - min(mid_asy) if mid_asy else float("nan")
        verdict = ("ROBUST" if mid_rng < 0.01 else
                    "SENSITIVE" if mid_rng < 0.03 else "BREAKS_DOWN")
        print()
        print(f"  Full-range asymptote spread: {rng:.5f}")
        print(f"  Mid-range (tau in [0.07, 0.20]) spread: {mid_rng:.5f}")
        print(f"  VERDICT: {verdict}")
        results["robustness_verdict"] = verdict
        results["full_range_spread"] = rng
        results["mid_range_spread"] = mid_rng

    report = {
        "title": "Tau-threshold robustness audit (skeleton closure 7/24)",
        "target": TARGET,
        "by_tau": results,
    }
    out_path = OUTPUTS / "verify_tau_sweep_robustness.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # Emit tex macros
    GENERATED.mkdir(parents=True, exist_ok=True)
    macros = GENERATED / "tau_sweep_macros.tex"
    lines = ["% AUTO-GENERATED by src/verify_tau_sweep_robustness.py"]
    for tau in TAU_VALUES:
        k = f"tau_{tau:.2f}"
        if k in results:
            v = results[k]
            label = f"{int(tau*100):02d}"
            lines.append(f"\\newcommand{{\\tauSweepAsy{label}}}{{{v['asymptote_mean']:.5f}}}")
            lines.append(f"\\newcommand{{\\tauSweepRel{label}}}{{{v['rel_err_pct']:+.2f}\\%}}")
    if "robustness_verdict" in results:
        lines.append(f"\\newcommand{{\\tauSweepVerdict}}"
                       f"{{{results['robustness_verdict']}}}")
        lines.append(f"\\newcommand{{\\tauSweepMidRange}}"
                       f"{{{results['mid_range_spread']:.5f}}}")
    macros.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print(f"Saved: {macros}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
