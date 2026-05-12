"""Per-percentile structural-rational alpha audit for the
direct Einstein-gap full-tensor Frobenius residual.

Reads the Stage 6f bulk-percentile audit
(outputs/stage6f_full_tensor_norm_audit.json), extracts the
per-N median, mean, p90, p95, p99, sup of the per-node
relative full-tensor norm
||G_munu + Lambda^back - 8 pi G T_munu^Xi||_F /
||T_munu^Xi||_F, and fits each per-N percentile sequence
under a family of single-parameter power laws

  Delta_perc(N) = C * N^{-alpha}

with alpha drawn from candidate framework rationals
{1/3, 1/2, 2/3, 4/5, alpha_xi^2 = 81/100, 13/16, D_Omega = 67/80,
17/20 = alpha_xi - gamma/2, alpha_xi = 9/10, 1, 4/3, 11/8}.

Outputs:
  outputs/verify_full_tensor_alpha_family.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "outputs" / "stage6f_full_tensor_norm_audit.json"
OUT = REPO / "outputs" / "verify_full_tensor_alpha_family.json"

CANDIDATES = [
    ("1/3",                      1.0 / 3.0),
    ("1/2 (sqrt sub-Gaussian)",  1.0 / 2.0),
    ("2/3 (Theorem~15.18 bound)", 2.0 / 3.0),
    ("4/5 = alpha_xi - gamma",   4.0 / 5.0),
    ("alpha_xi^2 = 81/100 = Lambda_t",  81.0 / 100.0),
    ("13/16",                    13.0 / 16.0),
    ("D_Omega = beta_pi - gamma = 67/80",  67.0 / 80.0),
    ("17/20 = alpha_xi - gamma/2",        17.0 / 20.0),
    ("alpha_xi = 9/10",          9.0 / 10.0),
    ("1 (linear)",               1.0),
    ("4/3",                      4.0 / 3.0),
    ("11/8",                     11.0 / 8.0),
]


def fit_fixed(log_n, log_y, alpha):
    log_c = float(np.mean(log_y + alpha * log_n))
    pred = log_c - alpha * log_n
    ss_res = float(np.sum((log_y - pred) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n_pts = len(log_n)
    nll = 0.5 * n_pts * math.log(ss_res / n_pts) if ss_res > 0 else -1e9
    aicc = 2.0 + 2 * nll
    if n_pts - 2 > 0:
        aicc += 4.0 / (n_pts - 2)
    return {"alpha": alpha, "log_c": log_c, "r2": r2,
            "nll": nll, "AICc": aicc}


def fit_free(log_n, log_y):
    coef = np.polyfit(log_n, log_y, 1)
    slope, log_c = float(coef[0]), float(coef[1])
    alpha = -slope
    pred = log_c - alpha * log_n
    ss_res = float(np.sum((log_y - pred) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n_pts = len(log_n)
    nll = 0.5 * n_pts * math.log(ss_res / n_pts) if ss_res > 0 else -1e9
    aicc = 4.0 + 2 * nll
    if n_pts - 3 > 0:
        aicc += 12.0 / (n_pts - 3)
    return {"alpha": alpha, "log_c": log_c, "r2": r2,
            "nll": nll, "AICc": aicc}


def main():
    src = json.loads(SRC.read_text(encoding="utf-8"))
    n_arr, percs = [], {"median": [], "mean": [], "p90": [],
                         "p95": [], "p99": [], "sup": []}
    for r in src["per_regime"]:
        n_arr.append(int(r["N"]))
        df = r["delta_full"]
        for k in percs:
            percs[k].append(float(df[k]))
    n_arr = np.array(n_arr, dtype=float)
    log_n = np.log(n_arr)

    out = {
        "method": ("Per-percentile alpha-family power-law audit on "
                   "the direct Einstein-gap full-tensor Frobenius "
                   "residual ||G + Lambda - T||_F / ||T||_F."),
        "ladder_N": n_arr.tolist(),
        "per_percentile": {},
    }

    print(f'{"Percentile":<10} {"free_a":>8} {"R^2_free":>10}'
          f' {"best_cand":<35} {"a*":>8} {"diff%":>8}'
          f' {"R^2_cand":>10} {"dAICc":>10}')
    print("-" * 110)
    for stat in ["median", "mean", "p90", "p95", "p99", "sup"]:
        vals = np.array(percs[stat], dtype=float)
        if (vals <= 0).any() or not np.isfinite(vals).all():
            continue
        log_y = np.log(vals)
        free = fit_free(log_n, log_y)
        cand_fits = []
        for label, alpha in CANDIDATES:
            f = fit_fixed(log_n, log_y, alpha)
            f["label"] = label
            cand_fits.append(f)
        cand_fits.sort(key=lambda f: f["AICc"])
        a_min = min(f["AICc"] for f in cand_fits + [free])
        for f in cand_fits:
            f["delta_AICc"] = f["AICc"] - a_min
        free["delta_AICc"] = free["AICc"] - a_min

        # Closest by alpha distance
        best_dist = min(
            cand_fits,
            key=lambda f: abs(f["alpha"] - free["alpha"])
                              / max(f["alpha"], 1e-9))
        rel_diff_pct = abs(best_dist["alpha"] - free["alpha"]) / \
            best_dist["alpha"] * 100
        print(f'{stat:<10} {free["alpha"]:>8.4f} {free["r2"]:>10.3f}'
              f' {best_dist["label"][:34]:<35} '
              f'{best_dist["alpha"]:>8.4f} {rel_diff_pct:>7.2f}%'
              f' {best_dist["r2"]:>10.3f}'
              f' {best_dist["delta_AICc"]:>+10.3f}')

        out["per_percentile"][stat] = {
            "values": vals.tolist(),
            "free_fit": free,
            "candidate_fits": cand_fits,
            "best_distance_match": best_dist,
        }

    # Bootstrap CI on free-alpha for median, mean, p99, sup
    rng = np.random.default_rng(2026)
    n_boot = 2000
    bootstrap = {}
    for stat in ["median", "mean", "p99", "sup"]:
        if stat not in out["per_percentile"]:
            continue
        vals = np.array(percs[stat], dtype=float)
        log_y = np.log(vals)
        alphas = []
        for _ in range(n_boot):
            idx = rng.choice(len(n_arr), len(n_arr), replace=True)
            try:
                slope, _ = np.polyfit(log_n[idx], log_y[idx], 1)
                alphas.append(-slope)
            except Exception:  # noqa: BLE001
                pass
        alphas = np.array(alphas)
        bootstrap[stat] = {
            "n_boot": int(n_boot),
            "alpha_mean": float(alphas.mean()),
            "alpha_std":  float(alphas.std()),
            "alpha_CI95_low":  float(np.percentile(alphas, 2.5)),
            "alpha_CI95_high": float(np.percentile(alphas, 97.5)),
        }
    out["bootstrap_CI95"] = bootstrap

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print("Bootstrap CI95 (n=2000, resampling over N-points):")
    for stat, b in bootstrap.items():
        print(f"  {stat}: alpha = {b['alpha_mean']:.4f} +/- "
              f"{b['alpha_std']:.4f}, "
              f"CI95 = [{b['alpha_CI95_low']:.4f}, "
              f"{b['alpha_CI95_high']:.4f}]")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
