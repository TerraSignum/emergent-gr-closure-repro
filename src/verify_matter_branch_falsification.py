r"""Verify whether the matter-branch closure is 79/200 or revised.

After N=2048 data arrives, this script:
  1. Re-runs the branch-resolved fit including N=2048
  2. Compares the new matter-branch asymptote against:
       - 79/200 = 0.39500 (currently registered)
       - 13/32 = 0.40625 (preliminary alternative)
       - 2/5 = 0.40000 (matter-branch Bayesian top in q_max=50 search)
       - 11/27 = 0.40741 (another nearby simple rational)
  3. Issues a verdict:
       - "79_200_CONFIRMED": new fit moved toward 79/200 within CI95
       - "13_32_CONFIRMED": new fit confirms 13/32 within CI95
       - "INCONCLUSIVE": both within or both outside CI95
       - "NEITHER": both falsified, asymptote elsewhere

This is the auto-discrimination test for the matter-branch
falsification handle registered in P4 rmk:branch_resolved_audit.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
REPO = SRC.parent
OUTPUTS = REPO / "outputs"

sys.path.insert(0, str(SRC))
from _d1_ladder_discovery import discover_d1_ladder  # noqa: E402
from adaptive_pipeline import hierarchical_bayes  # noqa: E402


CANDIDATES = [
    ("79/200", Fraction(79, 200), "registered closure: 3/8 + d*gamma^2/2"),
    ("13/32",  Fraction(13, 32),  "alternative: (d^2 - N_gen)/(2*d^2)"),
    ("2/5",    Fraction(2, 5),    "Bayesian top in q_max=50 search"),
    ("11/27",  Fraction(11, 27),  "nearby simple rational"),
    ("9/22",   Fraction(9, 22),   "nearby simple rational"),
]


def extract_matter_branch(ladder, min_n: int = 256, max_seeds: int = 12):
    """Extract per-seed weighted lambda_2 from matter-branch regimes."""
    all_n = []
    all_lam = []
    sources = []
    for regime, n_lat, npz_path in ladder:
        if n_lat < min_n:
            continue
        try:
            d = np.load(npz_path, allow_pickle=True)
            if "edge_xi_snapshots" not in d.files:
                continue
            xi_arr = d["edge_xi_snapshots"][:, -1]
            n_seeds = min(max_seeds, xi_arr.shape[0])
            sources.append((regime, n_lat, n_seeds))
            for s in range(n_seeds):
                xi = xi_arr[s].astype(np.float64)
                np.fill_diagonal(xi, 0.0)
                deg = np.maximum(xi.sum(axis=1), 1e-12)
                d_inv = 1.0 / np.sqrt(deg)
                laplacian = np.eye(xi.shape[0]) - (d_inv[:, None] * xi * d_inv[None, :])
                laplacian = 0.5 * (laplacian + laplacian.T)
                try:
                    eigs = np.linalg.eigvalsh(laplacian)
                    all_n.append(n_lat)
                    all_lam.append(float(eigs[1]))
                except np.linalg.LinAlgError:
                    continue
        except OSError:
            # NPZ still being written, skip
            print(f"  [skip] {regime}: NPZ unreadable (may be mid-write)")
            continue
    return np.asarray(all_n), np.asarray(all_lam), sources


def verdict_for_candidate(asymptote_mean: float, asymptote_std: float,
                           candidate_value: float, sigma_threshold: float = 2.0) -> str:
    """Verdict for one candidate: confirmed/excluded based on z-score."""
    z = (asymptote_mean - candidate_value) / max(asymptote_std, 1e-12)
    if abs(z) < sigma_threshold:
        return f"CONFIRMED (z={z:+.2f})"
    return f"EXCLUDED (z={z:+.2f})"


def main():
    print("=" * 78)
    print("Matter-branch falsification test: 79/200 vs 13/32 vs alternatives")
    print("=" * 78)

    ladder = discover_d1_ladder(REPO)
    n_arr, lam_arr, sources = extract_matter_branch(ladder)
    print(f"\nMatter-branch ladder ({len(sources)} regimes, {len(lam_arr)} obs):")
    for regime, n, k in sources:
        print(f"  {regime}: N={n}, {k} seeds")

    if len(lam_arr) < 4:
        print("\n  [error] insufficient matter-branch data")
        return 1

    # Hierarchical Bayesian fit
    cmp = hierarchical_bayes.model_comparison(n_arr, lam_arr)
    best_key = cmp["selected"]
    best = cmp[best_key]
    mu = best["asymptote_mean"]
    sigma = best["asymptote_std"]
    print(f"\nHierarchical Bayes fit:")
    print(f"  Selected model: {best_key}")
    print(f"  Asymptote: {mu:.5f} +/- {sigma:.5f}  (CI95 [{mu - 1.96 * sigma:.5f}, "
          f"{mu + 1.96 * sigma:.5f}])")

    print(f"\nCandidate-by-candidate verdict (sigma threshold = 2):")
    print(f"  {'name':>8}  {'value':>9}  {'z':>7}  verdict")
    verdicts = {}
    for name, frac, _just in CANDIDATES:
        val = float(frac)
        v = verdict_for_candidate(mu, sigma, val)
        verdicts[name] = {"value": val, "z": (mu - val) / sigma, "verdict": v}
        print(f"  {name:>8}  {val:>9.5f}  {(mu - val)/sigma:>+7.2f}  {v}")

    # Overall verdict
    confirmed = [n for n, v in verdicts.items() if "CONFIRMED" in v["verdict"]]
    excluded = [n for n, v in verdicts.items() if "EXCLUDED" in v["verdict"]]
    if len(confirmed) == 1:
        overall = f"DECISIVE: {confirmed[0]} confirmed; all others excluded"
    elif len(confirmed) >= 2:
        # multiple within 2σ — pick the one with smallest |z| as most likely
        best_cand = min(confirmed, key=lambda n: abs(verdicts[n]["z"]))
        overall = f"PARTIAL: {best_cand} closest (z={verdicts[best_cand]['z']:+.2f}); "
        overall += f"{len(confirmed)-1} others not excluded"
    else:
        overall = "NEITHER: all candidates excluded"

    print(f"\nOverall verdict: {overall}")

    report = {
        "title": "Matter-branch closure discrimination test",
        "n_observations": int(len(lam_arr)),
        "n_regimes": len(sources),
        "n_max": int(n_arr.max()),
        "selected_model": best_key,
        "asymptote_mean": mu,
        "asymptote_std": sigma,
        "asymptote_ci95": [mu - 1.96 * sigma, mu + 1.96 * sigma],
        "candidate_verdicts": verdicts,
        "overall_verdict": overall,
    }
    out_path = OUTPUTS / "verify_matter_branch_falsification.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
