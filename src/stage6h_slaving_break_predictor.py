"""Stage 6h: per-node Lipschitz-slaving residual epsilon_K(a) as
candidate independent identifier of the closure-defect support.

Tests three claims:
  (i) epsilon_K(a) := |K(a) - K_slaved(Xi)(a)| with K_slaved =
      ridge regression of K(a) on the six Xi-features used in
      the G4 Lipschitz-slaving theorem (degree, Laplacian^2 diag,
      Fiedler_1..3, mean_edge_weight) is well-defined per-node.
  (ii) per-node correlations rho(epsilon_K, Delta), rho(epsilon_K,
      rho_phase), rho(epsilon_K, |R_00|) measure whether
      epsilon_K identifies the same nodes as Delta does, or an
      independent sub-structure.
  (iii) Layer C^(p)_epsilon := top-(1-p/100)% by epsilon_K vs
      C^(p)_Delta := top-(1-p/100)% by Delta. Jaccard overlap
      tells whether epsilon_K is Delta-redundant (Jaccard high)
      or independent (Jaccard low).

Plus the A -> P scaling check: is layer-mass-fraction P_n
consistent with (rho-shift)^2 ~ A_n^2?

Within-P5 ladder only (8 regimes; PRE 5, POST 3).

Output: outputs/stage6h_slaving_break_predictor.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


class _BlockCupy:
    def find_spec(self, name, path=None, target=None):
        if name == "cupy" or name.startswith("cupy."):
            raise ImportError("cupy disabled")
        return None

    def load_module(self, name):
        raise ImportError("cupy disabled")


sys.meta_path.insert(0, _BlockCupy())

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from _d1_npz_discovery import find_d1_npz  # noqa: E402
from stage6f_full_tensor_norm_audit import (  # noqa: E402
    LAMBDA_T, LAMBDA_S, load_canonical, load_snapshots,
)
from verify_galerkin_runner_A_hessian_ricci import per_seed_galerkin  # noqa: E402
from verify_per_eigendirection_residual import (  # noqa: E402
    per_node_eigendirection_residuals,
)

N_STAR = 50
N_GEN = 3
D = 4
N_FLIP = N_STAR * math.sqrt(D * N_GEN)
RIDGE_LAMBDA = 1e-3

LADDER = [
    ("P5",     50),
    ("P5N64",  64),
    ("P5N72",  72),
    ("P5N84",  84),
    ("P5N100", 100),
    ("P5N200", 200),
    ("P5N300", 300),
    ("P5N512", 512),
    ("P5N256", 256),
]

LAYER_DEFS = [
    ("Csup",  "max"),
    ("C99_9", 99.9),
    ("C99_5", 99.5),
    ("C99",   99.0),
    ("C95",   95.0),
]


def chirality_phase(n_lat):
    x = math.log(n_lat / N_STAR) / math.log(D * N_GEN)
    th = math.atan(N_GEN ** (2 * x - 1))
    s2 = math.sin(th) ** 2
    return s2, ("PRE" if th < math.pi / 4 else "POST")


def _xi_features(xi_mat, n_lat):
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    deg = xi_off.sum(axis=1)
    laplacian = np.diag(deg) - xi_off
    lap2 = np.diag(laplacian @ laplacian)
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    f1 = eigvecs[:, 1] if n_lat > 1 else np.zeros(n_lat)
    f2 = eigvecs[:, 2] if n_lat > 2 else np.zeros(n_lat)
    f3 = eigvecs[:, 3] if n_lat > 3 else np.zeros(n_lat)
    mean_edge = xi_off.mean(axis=1)
    feat = np.column_stack([deg, lap2, f1, f2, f3, mean_edge])
    mu = feat.mean(axis=0)
    sd = feat.std(axis=0) + 1e-12
    return (feat - mu) / sd


def _ridge_predict(features, target):
    a_mat = np.column_stack([np.ones(features.shape[0]), features])
    aTa = a_mat.T @ a_mat + RIDGE_LAMBDA * np.eye(a_mat.shape[1])
    aTy = a_mat.T @ target
    beta = np.linalg.solve(aTa, aTy)
    return a_mat @ beta


def _per_seed(xi_mat, psi, k_field, q_field, n_lat):
    """Per-seed observables: delta, R_time, rho, epsilon_K, t00."""
    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
    res = per_node_eigendirection_residuals(prep, LAMBDA_T, LAMBDA_S)
    R_time = res["R_time"]
    R_diag = res["R_diag"]
    R_off = res["R_off"]
    t_eigs = res["T_eigvals"]
    t00 = np.asarray(prep["t00"])
    R_norm = np.sqrt(R_time ** 2
                      + (R_diag ** 2).sum(axis=1)
                      + R_off ** 2)
    T_norm = np.sqrt(t00 ** 2 + (t_eigs ** 2).sum(axis=1))
    delta = R_norm / np.maximum(T_norm, 1e-12)

    # Per-node K, Q row-mean targets
    if k_field.ndim == 2:
        eye = np.eye(n_lat, dtype=bool)
        k_target = np.where(eye, 0.0, k_field).sum(axis=1) / max(n_lat - 1, 1)
        q_target = np.where(eye, 0.0, q_field).sum(axis=1) / max(n_lat - 1, 1)
    else:
        k_target = np.full(n_lat, float(np.mean(k_field)))
        q_target = np.full(n_lat, float(np.mean(q_field)))

    feats = _xi_features(xi_mat, n_lat)
    k_pred = _ridge_predict(feats, k_target)
    q_pred = _ridge_predict(feats, q_target)
    eps_K = np.abs(k_target - k_pred)
    eps_Q = np.abs(q_target - q_pred)

    # phase coherence rho
    xi_no = xi_mat.copy()
    np.fill_diagonal(xi_no, 0.0)
    safe = np.maximum(xi_no.sum(axis=1), 1e-12)
    eiphi = np.exp(1j * np.angle(psi))
    nbhd = (xi_no @ eiphi) / safe
    rho = np.real(np.conj(eiphi) * nbhd)

    psi_sq = psi.real ** 2 + psi.imag ** 2

    return {
        "delta": delta,
        "abs_R_time": np.abs(R_time),
        "rho": rho,
        "eps_K": eps_K,
        "eps_Q": eps_Q,
        "t00": np.abs(t00),
        "psi_sq": psi_sq,
    }


def _gather(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    if "snapshots" in p.name.lower():
        seeds = load_snapshots(p, n_lat)
    else:
        seeds = load_canonical(p, n_lat)
    return [_per_seed(xi, psi, k, q, n_lat) for (xi, psi, k, q) in seeds]


def _layer_indices(arr, percentile_or_max):
    if percentile_or_max == "max":
        return {int(np.argmax(arr))}
    thr = float(np.percentile(arr, percentile_or_max))
    return set(int(i) for i in np.nonzero(arr > thr)[0])


def _jaccard(a, b):
    if not a and not b:
        return float("nan")
    return len(a & b) / max(len(a | b), 1)


def _pearson(a, b):
    if a.size < 3:
        return float("nan")
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    if a.size < 3:
        return float("nan")
    return _pearson(np.argsort(np.argsort(a)).astype(float),
                     np.argsort(np.argsort(b)).astype(float))


def _per_regime_audit(seeds_obs):
    """Per-regime per-seed correlations + cross-Jaccard, then aggregate."""
    pearson = {"eps_K_vs_delta": [], "eps_K_vs_rho": [],
                "eps_K_vs_R": [], "eps_K_vs_t00": [],
                "eps_K_vs_eps_Q": [], "delta_vs_rho": [],
                "delta_vs_R": []}
    spearman = {"eps_K_vs_delta": [], "eps_K_vs_rho": [],
                  "eps_K_vs_R": []}
    jaccard = {ln: {"top5": [], "top1": [], "top0_5": [], "top0_1": []}
               for ln, _ in LAYER_DEFS}
    layer_means_eps_K = {ln: [] for ln, _ in LAYER_DEFS}
    layer_means_delta = {ln: [] for ln, _ in LAYER_DEFS}
    rho_shifts = {ln: [] for ln, _ in LAYER_DEFS}
    psi_sq_fractions = {ln: [] for ln, _ in LAYER_DEFS}

    for obs in seeds_obs:
        # per-seed Pearson + Spearman
        for src_a, src_b, key in [
            (obs["eps_K"], obs["delta"], "eps_K_vs_delta"),
            (obs["eps_K"], obs["rho"], "eps_K_vs_rho"),
            (obs["eps_K"], obs["abs_R_time"], "eps_K_vs_R"),
            (obs["eps_K"], obs["t00"], "eps_K_vs_t00"),
            (obs["eps_K"], obs["eps_Q"], "eps_K_vs_eps_Q"),
            (obs["delta"], obs["rho"], "delta_vs_rho"),
            (obs["delta"], obs["abs_R_time"], "delta_vs_R"),
        ]:
            pearson[key].append(_pearson(src_a, src_b))

        for src_a, src_b, key in [
            (obs["eps_K"], obs["delta"], "eps_K_vs_delta"),
            (obs["eps_K"], obs["rho"], "eps_K_vs_rho"),
            (obs["eps_K"], obs["abs_R_time"], "eps_K_vs_R"),
        ]:
            spearman[key].append(_spearman(src_a, src_b))

        # Cross-Jaccard at four percentile levels:
        # eps_K-top vs delta-top
        for thr_label, thr_pct in [("top5", 95.0), ("top1", 99.0),
                                      ("top0_5", 99.5), ("top0_1", 99.9)]:
            S_e = _layer_indices(obs["eps_K"], thr_pct)
            S_d = _layer_indices(obs["delta"], thr_pct)
            for ln, _ in LAYER_DEFS:
                pass  # handled below
            # Aggregate for THIS threshold (independent of LAYER_DEFS)
        # Per-layer Jaccard and stats
        for ln, pc in LAYER_DEFS:
            S_d = _layer_indices(obs["delta"], pc)
            S_e = _layer_indices(obs["eps_K"], pc)
            j_thr = ("top5" if pc == 95.0
                     else "top1" if pc == 99.0
                     else "top0_5" if pc == 99.5
                     else "top0_1" if pc == 99.9
                     else "top0_1")
            jaccard[ln][j_thr].append(_jaccard(S_d, S_e))
            if S_d:
                idx = np.array(sorted(S_d), dtype=int)
                layer_means_delta[ln].append(float(obs["delta"][idx].mean()))
                rho_shifts[ln].append(
                    float(obs["rho"][idx].mean() - obs["rho"].mean()))
                psi_sq_fractions[ln].append(
                    float(obs["psi_sq"][idx].sum()
                          / max(obs["psi_sq"].sum(), 1e-12)))
            if S_e:
                idx_e = np.array(sorted(S_e), dtype=int)
                layer_means_eps_K[ln].append(
                    float(obs["eps_K"][idx_e].mean()))

    # Aggregate
    out = {
        "n_seeds": len(seeds_obs),
        "pearson_mean": {k: float(np.nanmean(v)) for k, v in pearson.items()},
        "pearson_std": {k: float(np.nanstd(v)) for k, v in pearson.items()},
        "spearman_mean": {k: float(np.nanmean(v)) for k, v in spearman.items()},
        "layer_jaccard_eps_vs_delta": {
            ln: {k: float(np.nanmean(v)) if v else float("nan")
                 for k, v in pairs.items() if v}
            for ln, pairs in jaccard.items()
        },
        "layer_rho_shift": {
            ln: float(np.nanmean(rho_shifts[ln])) if rho_shifts[ln]
            else float("nan") for ln, _ in LAYER_DEFS
        },
        "layer_psi_sq_fraction": {
            ln: float(np.nanmean(psi_sq_fractions[ln])) if psi_sq_fractions[ln]
            else float("nan") for ln, _ in LAYER_DEFS
        },
    }
    return out


def main() -> int:
    print("=" * 110)
    print("Stage 6h: Slaving-residual epsilon_K(a) as candidate")
    print("identifier of the closure-defect support (within-P5 only)")
    print("=" * 110)
    print()

    rows = []
    for reg, n_lat in LADDER:
        s2, phase = chirality_phase(n_lat)
        seeds_obs = _gather(reg, n_lat)
        if seeds_obs is None:
            continue
        agg = _per_regime_audit(seeds_obs)
        agg.update({
            "regime": reg, "N": n_lat, "phase": phase, "sin2t": s2,
        })
        rows.append(agg)
        # Quick line
        p = agg["pearson_mean"]
        j_top1 = agg["layer_jaccard_eps_vs_delta"].get("C99", {}).get("top1", float("nan"))
        j_top5 = agg["layer_jaccard_eps_vs_delta"].get("C95", {}).get("top5", float("nan"))
        print(f"  {reg:<8s} N={n_lat:>4d} {phase}  "
              f"Pearson(eK,D)={p['eps_K_vs_delta']:+.3f}  "
              f"(eK,rho)={p['eps_K_vs_rho']:+.3f}  "
              f"(eK,R)={p['eps_K_vs_R']:+.3f}  "
              f"Jacc(eK_top5, D_top5)={j_top5:.2f}  "
              f"Jacc(top1)={j_top1:.2f}")

    if len(rows) < 3:
        print("Not enough regimes")
        return 0

    # Aggregate per phase
    print()
    print("=" * 110)
    print("Cross-regime aggregate per phase")
    print("=" * 110)
    summary = {}
    for phase in ("PRE", "POST"):
        grp = [r for r in rows if r["phase"] == phase]
        n_grp = len(grp)
        keys = ["eps_K_vs_delta", "eps_K_vs_rho", "eps_K_vs_R",
                "eps_K_vs_t00", "eps_K_vs_eps_Q", "delta_vs_rho",
                "delta_vs_R"]
        means_p = {k: float(np.mean([r["pearson_mean"][k] for r in grp]))
                    for k in keys}
        means_s = {k: float(np.mean([r["spearman_mean"][k] for r in grp]))
                    for k in ["eps_K_vs_delta", "eps_K_vs_rho", "eps_K_vs_R"]}
        # Aggregate Jaccards by threshold across layers
        jacc_thresholds = {"top5": [], "top1": [], "top0_5": [], "top0_1": []}
        for ln, pc in LAYER_DEFS:
            j_label = ("top5" if pc == 95.0
                       else "top1" if pc == 99.0
                       else "top0_5" if pc == 99.5
                       else "top0_1" if pc == 99.9
                       else None)
            if j_label is None:
                continue
            for r in grp:
                if (ln in r["layer_jaccard_eps_vs_delta"]
                        and j_label in r["layer_jaccard_eps_vs_delta"][ln]):
                    jacc_thresholds[j_label].append(
                        r["layer_jaccard_eps_vs_delta"][ln][j_label])
        jacc_means = {k: float(np.nanmean(v)) if v else float("nan")
                       for k, v in jacc_thresholds.items()}
        summary[phase] = {
            "n_regimes": n_grp,
            "pearson_mean": means_p,
            "spearman_mean": means_s,
            "layer_jaccard_eps_vs_delta_means": jacc_means,
        }
        print(f"\n{phase} (n={n_grp}):")
        print("  Pearson means:")
        for k in keys:
            print(f"    {k:>20s}: {means_p[k]:+.3f}")
        print("  Spearman means:")
        for k in ["eps_K_vs_delta", "eps_K_vs_rho", "eps_K_vs_R"]:
            print(f"    {k:>20s}: {means_s[k]:+.3f}")
        print("  Cross-Jaccard eps-top vs delta-top:")
        for k, v in jacc_means.items():
            print(f"    {k:>10s}: {v:.3f}")

    # A -> P scaling test: P_n ~ A_n^2
    print()
    print("=" * 110)
    print("A -> P scaling test:  P_n (psi^2 fraction) vs A_n^2 (rho-shift)^2")
    print("=" * 110)
    for phase in ("PRE", "POST"):
        grp = [r for r in rows if r["phase"] == phase]
        if not grp:
            continue
        print(f"\n{phase}:")
        print(f"  {'layer':>8s} {'P_n':>10s} {'A_n=|rho_sh|':>14s} "
              f"{'A_n^2':>10s} {'P_n/A_n^2':>12s}")
        for ln, _ in LAYER_DEFS:
            P = float(np.mean([r["layer_psi_sq_fraction"][ln] for r in grp
                                if not np.isnan(r["layer_psi_sq_fraction"][ln])]))
            A = abs(float(np.mean([r["layer_rho_shift"][ln] for r in grp
                                    if not np.isnan(r["layer_rho_shift"][ln])])))
            A2 = A * A
            ratio = P / A2 if A2 > 1e-12 else float("nan")
            print(f"  {ln:>8s} {P:>10.4f} {A:>14.4f} {A2:>10.4f} {ratio:>12.3f}")

    # Verdict
    print()
    print("=" * 110)
    print("Verdict")
    print("=" * 110)
    final = {}
    for phase in ("PRE", "POST"):
        s = summary[phase]
        p_corr = s["pearson_mean"]["eps_K_vs_delta"]
        j_top1 = s["layer_jaccard_eps_vs_delta_means"].get("top1", float("nan"))
        j_top5 = s["layer_jaccard_eps_vs_delta_means"].get("top5", float("nan"))
        if abs(p_corr) > 0.7 and j_top5 > 0.5:
            verdict = "EPS_K_REDUNDANT_WITH_DELTA"
        elif abs(p_corr) < 0.3 and j_top5 < 0.2:
            verdict = "EPS_K_INDEPENDENT_IDENTIFIER"
        else:
            verdict = "PARTIAL_OVERLAP"
        final[phase] = {
            "Pearson_eps_K_vs_delta": p_corr,
            "Jaccard_top5": j_top5,
            "Jaccard_top1": j_top1,
            "verdict": verdict,
        }
        print(f"  {phase}: Pearson(eps_K, delta)={p_corr:+.3f}, "
              f"Jaccard(top5)={j_top5:.2f}, Jaccard(top1)={j_top1:.2f}  "
              f"=> {verdict}")

    bundle = {
        "method": "stage6h_slaving_break_predictor",
        "schema_version": "1.0.0",
        "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
        "n_flip": N_FLIP,
        "ladder_within_p5_only": [r for r, _ in LADDER],
        "regimes": rows,
        "phase_summary": summary,
        "final_verdict": final,
    }
    out = REPO / "outputs" / "stage6h_slaving_break_predictor.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print()
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
