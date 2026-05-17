r"""Per-seed branch classifier for matter/vacuum identification.

Critical finding (user reminder, 2026-05-17): the "matter-branch" 79/200
closure was originally identified from the TOP-PERCENTILE / matter-core
behavior, not from a pooled mean over all seeds. At intermediate N
(N=256, 300, 512), individual seeds split into:
  - VAC-mode: lambda_2 < 0.385 (close to 3/8 = 0.375)
  - MATTER-mode: lambda_2 > 0.395 (close to 79/200 = 0.395)
  - TRANSITION: in between
Pooling ALL seeds at high-N gives a mixed mean that has no clean
rational interpretation.

This script:
  1. Per regime, per seed -> compute lambda_2(iter=120)
  2. Classify each seed as VAC / MATTER / TRANSITION
  3. Re-compute branch-specific asymptotes using ONLY the seeds
     in the relevant branch
  4. Test if matter-mode (top-percentile) seeds match 79/200 at
     high N
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

# Branch-classifier thresholds (centred between 3/8 = 0.375 and
# 79/200 = 0.395; transition zone is +/- 0.01 around the midpoint
# 0.385).
VAC_MAX = 0.385      # lambda_2 < 0.385  -> VAC-mode seed
MATTER_MIN = 0.395    # lambda_2 > 0.395  -> MATTER-mode seed


def classify_seed(lam: float) -> str:
    if lam < VAC_MAX:
        return "VAC"
    if lam > MATTER_MIN:
        return "MATTER"
    return "TRANSITION"


def per_seed_lambdas(ladder, min_n: int = 200):
    """Returns dict regime -> list of per-seed lambda_2 values."""
    out = {}
    for regime, n_lat, npz_path in ladder:
        if n_lat < min_n:
            continue
        try:
            d = np.load(npz_path, allow_pickle=True)
            if "edge_xi_snapshots" not in d.files:
                continue
            xi_arr = d["edge_xi_snapshots"][:, -1]
            lams = []
            for s in range(xi_arr.shape[0]):
                xi = xi_arr[s].astype(np.float64)
                np.fill_diagonal(xi, 0.0)
                deg = np.maximum(xi.sum(axis=1), 1e-12)
                d_inv = 1.0 / np.sqrt(deg)
                L = np.eye(xi.shape[0]) - (d_inv[:, None] * xi * d_inv[None, :])
                L = 0.5 * (L + L.T)
                eigs = np.linalg.eigvalsh(L)
                lams.append(float(eigs[1]))
            out[regime] = {"N": n_lat, "lambdas": lams}
        except OSError:
            print(f"  [skip] {regime}: NPZ unreadable")
    return out


def main():
    print("=" * 78)
    print("Per-seed branch classifier (VAC / MATTER / TRANSITION)")
    print("=" * 78)
    print(f"Thresholds: VAC < {VAC_MAX:.3f}, MATTER > {MATTER_MIN:.3f}")
    print()

    ladder = discover_d1_ladder(REPO)
    data = per_seed_lambdas(ladder)

    # Per-regime classification summary
    print(f"  {'regime':<10} {'N':>5} {'#seeds':>7} "
          f"{'VAC':>5} {'MAT':>5} {'TRA':>5} "
          f"{'vac_mean':>10} {'mat_mean':>10}")
    branch_data = {"vac_seeds": [], "vac_ns": [],
                   "mat_seeds": [], "mat_ns": [],
                   "per_regime": {}}
    for regime, info in data.items():
        n = info["N"]
        lams = info["lambdas"]
        classes = [classify_seed(l) for l in lams]
        n_vac = classes.count("VAC")
        n_mat = classes.count("MATTER")
        n_tra = classes.count("TRANSITION")
        vac_lams = [l for l, c in zip(lams, classes) if c == "VAC"]
        mat_lams = [l for l, c in zip(lams, classes) if c == "MATTER"]
        vac_mean = float(np.mean(vac_lams)) if vac_lams else float("nan")
        mat_mean = float(np.mean(mat_lams)) if mat_lams else float("nan")
        print(f"  {regime:<10} {n:>5} {len(lams):>7} "
              f"{n_vac:>5} {n_mat:>5} {n_tra:>5} "
              f"{vac_mean:>10.5f} {mat_mean:>10.5f}")
        branch_data["per_regime"][regime] = {
            "N": n, "lams": lams, "classes": classes,
            "vac_lams": vac_lams, "mat_lams": mat_lams,
            "vac_mean": vac_mean, "mat_mean": mat_mean,
        }
        # Accumulate per-seed for the global per-branch Symanzik fit
        for l in vac_lams:
            branch_data["vac_seeds"].append(l)
            branch_data["vac_ns"].append(n)
        for l in mat_lams:
            branch_data["mat_seeds"].append(l)
            branch_data["mat_ns"].append(n)

    # Symanzik fits per branch using ONLY classified-branch seeds
    print()
    print("-" * 78)
    print("Per-branch Symanzik fits (using only classified seeds)")
    print("-" * 78)

    def fit_branch(name, ns, lams, target_value, target_label):
        if len(lams) < 4:
            print(f"  {name}: insufficient data ({len(lams)} obs)")
            return None
        ns_arr = np.asarray(ns)
        lams_arr = np.asarray(lams)
        cmp = hierarchical_bayes.model_comparison(ns_arr, lams_arr)
        best = cmp[cmp["selected"]]
        mu, sigma = best["asymptote_mean"], best["asymptote_std"]
        z = (mu - target_value) / max(sigma, 1e-12)
        print(f"  {name}: {len(lams)} obs across N in [{ns_arr.min()},"
              f" {ns_arr.max()}]")
        print(f"    Symanzik selected: {cmp['selected']}")
        print(f"    Asymptote: {mu:.5f} +/- {sigma:.5f}")
        print(f"    Target {target_label} = {target_value:.5f}: z = {z:+.2f}")
        return {"asymptote_mean": mu, "asymptote_std": sigma,
                "selected": cmp["selected"], "z_target": z, "n_obs": len(lams)}

    vac_fit = fit_branch("VAC-mode seeds", branch_data["vac_ns"],
                          branch_data["vac_seeds"], 3 / 8, "3/8")
    mat_fit = fit_branch("MATTER-mode seeds", branch_data["mat_ns"],
                          branch_data["mat_seeds"], 79 / 200, "79/200")

    # Save report
    report = {
        "title": "Per-seed branch classifier + per-branch Symanzik fit",
        "thresholds": {"vac_max": VAC_MAX, "matter_min": MATTER_MIN},
        "per_regime": {k: {kk: vv for kk, vv in v.items()
                              if kk not in ("lams", "classes",
                                              "vac_lams", "mat_lams")}
                       for k, v in branch_data["per_regime"].items()},
        "per_seed_data": branch_data["per_regime"],
        "vac_fit": vac_fit,
        "matter_fit": mat_fit,
    }
    out_path = OUTPUTS / "verify_per_seed_branch_classifier.json"
    out_path.write_text(json.dumps(report, indent=2, default=str),
                          encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # Emit macros for manuscript
    if mat_fit is not None and vac_fit is not None:
        GENERATED.mkdir(parents=True, exist_ok=True)
        lines = [
            "% AUTO-GENERATED by src/verify_per_seed_branch_classifier.py",
            f"\\newcommand{{\\perSeedVacAsymptote}}{{{vac_fit['asymptote_mean']:.5f}}}",
            f"\\newcommand{{\\perSeedVacStd}}{{{vac_fit['asymptote_std']:.5f}}}",
            f"\\newcommand{{\\perSeedVacZ}}{{{vac_fit['z_target']:+.2f}}}",
            f"\\newcommand{{\\perSeedMatAsymptote}}{{{mat_fit['asymptote_mean']:.5f}}}",
            f"\\newcommand{{\\perSeedMatStd}}{{{mat_fit['asymptote_std']:.5f}}}",
            f"\\newcommand{{\\perSeedMatZ}}{{{mat_fit['z_target']:+.2f}}}",
            f"\\newcommand{{\\perSeedVacObs}}{{{vac_fit['n_obs']}}}",
            f"\\newcommand{{\\perSeedMatObs}}{{{mat_fit['n_obs']}}}",
        ]
        macros = GENERATED / "per_seed_branch_macros.tex"
        macros.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved: {macros}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
