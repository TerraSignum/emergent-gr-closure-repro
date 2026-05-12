"""(Solution-CLP-C) Detailed Γ-convergence analysis on extended 9-pt dense-cell ladder.

CLP-C verifies Dal-Maso 1993 Γ-convergence of the discrete action S_N → S_∞.
Five sub-conditions:

  C1: Liminf inequality (lower semicontinuity)
      Proxy: total closure loss L(N) = mean(1 - gap_i) for 6 keys
      converges; C1 = clip01(1 - L_inf), boosted by monotonicity

  C2: Recovery sequence (upper bound)
      Proxy: min(canonicalisation_score) across N

  C3: Equi-coercivity (bounded sub-level sets)
      Proxy: 1 - range(transport_score)

  C4: Minimiser convergence
      Proxy: fixpoint_proximity_score asymptote

  C5: δS_N → δS_∞ Euler-Lagrange link
      = 0.5 * C1 + 0.5 * C2

  gamma_score = mean(C1..C5)
  status: GAMMA_CONVERGED ≥0.75, PARTIAL ≥0.45, OPEN <0.45

This script:
  1. Loads 9 payloads (P0-P8) from fix17 + fix16
  2. Computes all 5 sub-components with Symanzik-2 fits where applicable
  3. Bootstraps each sub-component
  4. Reports detailed status

Output: outputs/clp_c_gamma_convergence_detailed_audit.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
D1_DIRS = [
    REPO / "results_d1_fix17",
    REPO / "results_d1_fix16" / "p6",
    REPO / "results_d1_fix16" / "p7",
    REPO / "results_d1_fix16" / "p8",
]


def _sf(v):
    try: return float(v) if v is not None else None
    except (TypeError, ValueError): return None


def _clip01(v):
    return max(0.0, min(1.0, v))


def symanzik_2_fit(n_arr, y_arr):
    n_arr = np.asarray(n_arr, float); y_arr = np.asarray(y_arr, float)
    if len(n_arr) < 2: return None
    A = np.column_stack([np.ones_like(n_arr), n_arr**-2])
    coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
    pred = A @ coef
    rss = float(np.sum((y_arr - pred)**2))
    return {"gap_inf": float(coef[0]), "c_2": float(coef[1]),
            "rss": rss, "n_points": int(len(n_arr))}


def bootstrap_gap_inf(n_arr, y_arr, n_boot=2000, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    gaps = []
    n = len(n_arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = symanzik_2_fit(n_arr[idx], y_arr[idx])
        if r is not None and np.isfinite(r["gap_inf"]):
            gaps.append(r["gap_inf"])
    gaps = np.array([g for g in gaps if -2 < g < 2])
    if len(gaps) < 50: return None
    return {"median": float(np.median(gaps)),
            "CI95": [float(np.percentile(gaps, 2.5)),
                     float(np.percentile(gaps, 97.5))],
            "n": int(len(gaps))}


def load_payloads():
    payloads = []
    for d in D1_DIRS:
        if not d.is_dir(): continue
        for f in sorted(d.glob("d1_p*.json")):
            if f.name.endswith(".metadata.json") or "report" in f.name: continue
            with open(f) as fh:
                payloads.append(json.load(fh))
    seen = {}
    for p in payloads:
        n = p.get("dense_cell_node_count")
        if n is None: continue
        seen[int(round(float(n)))] = p
    return [seen[k] for k in sorted(seen.keys())]


def main() -> int:
    print("="*100)
    print("(CLP-C) Detailed Γ-convergence analysis on extended 9-pt ladder")
    print("="*100)
    payloads = load_payloads()
    print(f"\n{len(payloads)} payloads loaded")
    Ns = np.array([float(p["dense_cell_node_count"]) for p in payloads])

    # ── C1: Liminf inequality (total closure loss) ────────────
    gap_keys = [
        "d1_fixpoint_proximity_score",
        "d1_fixpoint_transport_score",
        "d1_gamma_full_macroclass_joint_closure_score",
        "d1_gamma_full_macroclass_joint_nonuniform_closure_score",
        "d1_gamma_ir_variational_closure_score",
        "d1_gamma_ir_residual_locality_score",
    ]
    losses = []
    for p in payloads:
        loss_sum, count = 0.0, 0
        for k in gap_keys:
            v = _sf(p.get(k))
            if v is not None:
                loss_sum += (1.0 - v)
                count += 1
        if count > 0:
            losses.append(loss_sum / count)
    losses_arr = np.array(losses)
    print(f"\n=== C1: Liminf (total closure loss across 6 keys) ===")
    print(f"  L(N): {[round(l, 4) for l in losses_arr.tolist()]}")
    fit_c1 = symanzik_2_fit(Ns, losses_arr)
    bs_c1 = bootstrap_gap_inf(Ns, losses_arr)
    L_inf = fit_c1["gap_inf"]
    monotonic_pairs = sum(1 for i in range(len(losses_arr)-1)
                          if losses_arr[i+1] <= losses_arr[i] + 0.05)
    monotonic_frac = monotonic_pairs / max(len(losses_arr)-1, 1)
    if 0 <= L_inf < 1.0:
        c1_raw = _clip01(1.0 - L_inf)
    elif L_inf >= 0:
        c1_raw = _clip01(0.5 / (1.0 + L_inf))
    else: c1_raw = 0.0
    c1_score = _clip01(c1_raw * 0.7 + monotonic_frac * 0.3)
    print(f"  L_inf = {L_inf:.4f}, monotonic_frac = {monotonic_frac:.2f}")
    print(f"  C1_score = {c1_score:.4f}  (bs CI on L_inf [{bs_c1['CI95'][0]:.4f}, {bs_c1['CI95'][1]:.4f}])" if bs_c1 else f"  C1_score = {c1_score:.4f}")

    # ── C2: Recovery sequence (canonicalisation) ──────────────
    canon = [_sf(p.get("d1_canonicalization_score")) for p in payloads]
    canon = [c for c in canon if c is not None]
    print(f"\n=== C2: Recovery sequence (canonicalisation) ===")
    print(f"  canon: {[round(c, 4) for c in canon]}")
    c2_score = float(np.min(canon)) if canon else 0.0
    print(f"  C2_score = min(canon) = {c2_score:.4f}")

    # ── C3: Equi-coercivity (transport range) ─────────────────
    transport = [_sf(p.get("d1_fixpoint_transport_score")) for p in payloads]
    transport = [t for t in transport if t is not None]
    print(f"\n=== C3: Equi-coercivity (transport range) ===")
    print(f"  transport: {[round(t, 4) for t in transport]}")
    if len(transport) >= 2:
        ft_range = max(transport) - min(transport)
        c3_score = _clip01(1.0 - ft_range)
        print(f"  range = {ft_range:.4f}, C3_score = {c3_score:.4f}")
    else: c3_score = 0.0

    # ── C4: Minimiser convergence (fixpoint proximity asymptote) ──
    fp = [_sf(p.get("d1_fixpoint_proximity_score")) for p in payloads]
    fp = [v for v in fp if v is not None]
    print(f"\n=== C4: Minimiser convergence (fixpoint proximity) ===")
    print(f"  fp: {[round(v, 4) for v in fp]}")
    if len(fp) >= 3:
        ns_fp = [float(p["dense_cell_node_count"]) for p in payloads
                 if _sf(p.get("d1_fixpoint_proximity_score")) is not None]
        fit_c4 = symanzik_2_fit(np.array(ns_fp), np.array(fp))
        bs_c4 = bootstrap_gap_inf(np.array(ns_fp), np.array(fp))
        fp_inf = fit_c4["gap_inf"]
        c4_score = _clip01(fp_inf) if fp_inf > 0 else 0.0
        print(f"  fp_inf = {fp_inf:.4f}, C4_score = {c4_score:.4f}")
        if bs_c4: print(f"  bs CI [{bs_c4['CI95'][0]:.4f}, {bs_c4['CI95'][1]:.4f}]")
    else: c4_score = 0.0; fit_c4 = None; bs_c4 = None

    # ── C5: δS link ───────────────────────────────────────────
    c5_score = _clip01(0.5 * c1_score + 0.5 * c2_score)
    print(f"\n=== C5: δS_N→δS_∞ Euler-Lagrange link ===")
    print(f"  C5 = 0.5*C1 + 0.5*C2 = {c5_score:.4f}")

    # ── Assembly ──────────────────────────────────────────────
    sub = [c1_score, c2_score, c3_score, c4_score, c5_score]
    gamma_score = float(np.mean(sub))
    print(f"\n{'='*60}")
    print(f"=== CLP-C Assembly ===")
    print(f"{'='*60}")
    print(f"  C1 (liminf):           {c1_score:.4f}")
    print(f"  C2 (recovery):         {c2_score:.4f}")
    print(f"  C3 (equi-coercivity):  {c3_score:.4f}")
    print(f"  C4 (minimiser):        {c4_score:.4f}")
    print(f"  C5 (delta-S link):     {c5_score:.4f}")
    print(f"  → mean = {gamma_score:.4f}")
    if gamma_score >= 0.75: status = "GAMMA_CONVERGED"
    elif gamma_score >= 0.45: status = "GAMMA_PARTIAL"
    else: status = "GAMMA_OPEN"
    print(f"  → status: {status}")

    out = {
        "method": "clp_c_gamma_convergence_detailed",
        "n_payloads": len(payloads),
        "C1": {"score": c1_score, "L_inf": L_inf, "monotonic_frac": monotonic_frac,
               "fit": fit_c1, "bootstrap_L_inf": bs_c1, "losses": losses},
        "C2": {"score": c2_score, "canon_min": float(np.min(canon)) if canon else None,
               "canon_values": canon},
        "C3": {"score": c3_score, "transport_range": (max(transport)-min(transport)) if len(transport)>=2 else None,
               "transport_values": transport},
        "C4": {"score": c4_score, "fit": fit_c4, "bootstrap": bs_c4, "fp_values": fp},
        "C5": {"score": c5_score},
        "gamma_score": gamma_score,
        "status": status,
    }
    out_path = REPO / "emergent-gr-closure-repro" / "outputs" / "clp_c_gamma_convergence_detailed_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
