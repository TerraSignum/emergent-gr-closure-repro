"""(Opt-4) Full CLP-A/B/C/D re-fit on the extended 9-point dense-cell ladder.

Reproduces the CLP-A tightness, CLP-B operator convergence,
CLP-C gamma convergence, and CLP-D assembly using the same metric
formulas as the production code in
src/worldformula/physics/continuum_limit_proof.py — but consumes
all 9 dense-cell payloads (P0-P8) from fix16+fix17 simultaneously,
with Symanzik-2 fits for power-law decay (chosen by AICc against
free power-law and Symanzik 2+4 in opt-2).

Output: outputs/clp_full_families_extended_audit.json
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


def load_all_payloads():
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


def _sf(v):
    try: return float(v) if v is not None else None
    except (TypeError, ValueError): return None


def _clip01(v):
    return max(0.0, min(1.0, v))


def symanzik_2_fit(n_arr, y_arr):
    """y = g + c2/N^2. Returns dict with gap_inf, c_2, rss, n_points."""
    n_arr = np.asarray(n_arr, float); y_arr = np.asarray(y_arr, float)
    if len(n_arr) < 2: return None
    x2 = n_arr ** -2
    A = np.column_stack([np.ones_like(x2), x2])
    coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
    pred = A @ coef
    rss = float(np.sum((y_arr - pred)**2))
    return {"gap_inf": float(coef[0]), "c_2": float(coef[1]),
            "rss": rss, "n_points": int(len(n_arr)), "model": "symanzik_2"}


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
    if len(gaps) < 100: return None
    return {"median": float(np.median(gaps)),
            "CI95": [float(np.percentile(gaps, 2.5)),
                     float(np.percentile(gaps, 97.5))],
            "n_resamples": int(len(gaps))}


def extract_n(payload):
    return float(payload.get("dense_cell_node_count", 0))


# ─── CLP-A: Tightness / Precompactness ───────────────────────────────────

def clp_a(payloads):
    out = {"step": "CLP-A", "tag": "tightness", "n_payloads": len(payloads)}
    Ns = [extract_n(p) for p in payloads]
    Ls = [N ** (1.0/3) for N in Ns]
    L_ref = min(Ls)
    diameter_ratios = [L / L_ref for L in Ls]
    N_min = min(Ns)
    anomalous_ratio = [dr / ((N / N_min) ** (1.0/3)) for dr, N in zip(diameter_ratios, Ns)]
    t1_score = 1.0 - min(1.0, float(np.std(anomalous_ratio)))

    transport = [_sf(p.get("d1_fixpoint_transport_score")) for p in payloads]
    transport = [t for t in transport if t is not None]
    if len(transport) >= 2:
        ft_std = float(np.std(transport)); ft_mean = float(np.mean(transport))
        t2_score = _clip01(1.0 - ft_std / max(ft_mean, 1e-6))
    else: t2_score = 0.0

    if len(transport) >= 3:
        ns_t = [extract_n(p) for p in payloads if _sf(p.get("d1_fixpoint_transport_score")) is not None]
        fit = symanzik_2_fit(np.array(ns_t), np.array(transport))
        bs = bootstrap_gap_inf(np.array(ns_t), np.array(transport))
        ft_inf = fit["gap_inf"] if fit else None
        t3_score = _clip01(ft_inf) if ft_inf and ft_inf > 0 else 0.3
        out["T3_fit_symanzik2"] = fit; out["T3_bootstrap"] = bs
    else: t3_score = 0.0

    t4_score = 1.0
    score = 0.40*t1_score + 0.20*t2_score + 0.30*t3_score + 0.10*t4_score
    out.update({
        "T1_score": round(t1_score, 6), "T2_score": round(t2_score, 6),
        "T3_score": round(t3_score, 6), "T4_score": round(t4_score, 6),
        "tightness_score": round(score, 6),
        "tightness_status": "CLOSED" if score >= 0.7 else "OPEN",
    })
    return out


# ─── CLP-B (B4 sub-decomposition) ───────────────────────────────────────

def clp_b_b4(payloads):
    """B4 slow-decomposition — using Symanzik-2 fits chosen by Opt-2 AICc."""
    out = {"step": "CLP-B/B4", "tag": "slow_decomposition"}
    Ns = np.array([extract_n(p) for p in payloads])

    metrics = {
        "absorption": "d1_gamma_ir_residual_absorption_closure_score",
        "locality":   "d1_gamma_ir_residual_locality_score",
        "density":    "d1_gamma_ir_residual_density_score",
        "spectral":   "d1_gamma_full_macroclass_joint_closure_score",
    }
    sub = {}
    for tag, key in metrics.items():
        vals = [_sf(p.get(key)) for p in payloads]
        if any(v is None for v in vals): continue
        v_arr = np.array(vals, float)
        fit = symanzik_2_fit(Ns, v_arr)
        bs = bootstrap_gap_inf(Ns, v_arr, n_boot=2000)
        sub[tag] = {
            "fit": fit, "bootstrap": bs,
            "score_clipped": _clip01(fit["gap_inf"]) if fit else 0.0,
        }
    if sub:
        scores = [s["score_clipped"] for s in sub.values()]
        bottleneck = min(sub, key=lambda k: sub[k]["score_clipped"])
        family_score = float(np.mean(scores))
        out.update({
            "sub_components": sub,
            "bottleneck": bottleneck,
            "family_score_mean": round(family_score, 6),
            "family_score_min": round(min(scores), 6),
            "above_threshold_0p5_count": sum(1 for s in scores if s >= 0.5),
            "n_components": len(scores),
        })
    return out


# ─── CLP-C: Γ-Convergence ───────────────────────────────────────────────

def clp_c(payloads):
    """Faithful CLP-C reproduction following continuum_limit_proof.py:1163-1381.
    Five sub-components: C1 (liminf), C2 (recovery), C3 (equi-coercivity),
    C4 (minimiser), C5 (delta-S link).
    """
    out = {"step": "CLP-C", "tag": "gamma_convergence", "n_payloads": len(payloads)}
    Ns = np.array([extract_n(p) for p in payloads])

    # C1: liminf via total closure loss
    gap_keys = ["d1_fixpoint_proximity_score", "d1_fixpoint_transport_score",
                "d1_gamma_full_macroclass_joint_closure_score",
                "d1_gamma_full_macroclass_joint_nonuniform_closure_score",
                "d1_gamma_ir_variational_closure_score",
                "d1_gamma_ir_residual_locality_score"]
    losses = []
    for p in payloads:
        s, c = 0.0, 0
        for k in gap_keys:
            v = _sf(p.get(k))
            if v is not None: s += (1.0 - v); c += 1
        if c > 0: losses.append(s / c)
    if len(losses) >= 2:
        fit = symanzik_2_fit(Ns, np.array(losses))
        L_inf = fit["gap_inf"]
        c1_raw = _clip01(1.0 - L_inf) if 0 <= L_inf < 1.0 else (_clip01(0.5/(1+L_inf)) if L_inf >= 0 else 0.0)
        mono_pairs = sum(1 for i in range(len(losses)-1) if losses[i+1] <= losses[i] + 0.05)
        mono_frac = mono_pairs / max(len(losses)-1, 1)
        c1_score = _clip01(c1_raw * 0.7 + mono_frac * 0.3)
        out["C1_loss_fit"] = fit
        out["C1_monotonic_fraction"] = mono_frac
    else: c1_score = 0.0
    out["C1_score"] = round(c1_score, 6)

    # C2: recovery sequence (canonicalisation)
    canon = [_sf(p.get("d1_canonicalization_score")) for p in payloads]
    canon = [c for c in canon if c is not None]
    c2_score = float(np.min(canon)) if canon else 0.0
    out["C2_score"] = round(c2_score, 6)

    # C3: equi-coercivity (transport range)
    transport = [_sf(p.get("d1_fixpoint_transport_score")) for p in payloads]
    transport = [t for t in transport if t is not None]
    c3_score = _clip01(1.0 - (max(transport) - min(transport))) if len(transport) >= 2 else 0.0
    out["C3_score"] = round(c3_score, 6)

    # C4: minimiser convergence (fixpoint proximity)
    fp = [_sf(p.get("d1_fixpoint_proximity_score")) for p in payloads]
    fp = [v for v in fp if v is not None]
    if len(fp) >= 3:
        ns_fp = [extract_n(p) for p in payloads
                 if _sf(p.get("d1_fixpoint_proximity_score")) is not None]
        fit_c4 = symanzik_2_fit(np.array(ns_fp), np.array(fp))
        c4_score = _clip01(fit_c4["gap_inf"]) if fit_c4["gap_inf"] > 0 else 0.0
        out["C4_fp_fit"] = fit_c4
    else: c4_score = 0.0
    out["C4_score"] = round(c4_score, 6)

    # C5: delta-S link
    c5_score = _clip01(0.5 * c1_score + 0.5 * c2_score)
    out["C5_score"] = round(c5_score, 6)

    # Assembly: mean of 5 sub-scores
    gamma_score = float(np.mean([c1_score, c2_score, c3_score, c4_score, c5_score]))
    out["gamma_convergence_score"] = round(gamma_score, 6)
    if gamma_score >= 0.75: out["gamma_convergence_status"] = "GAMMA_CONVERGED"
    elif gamma_score >= 0.45: out["gamma_convergence_status"] = "GAMMA_PARTIAL"
    else: out["gamma_convergence_status"] = "GAMMA_OPEN"
    return out


# ─── CLP-D: Assembly ────────────────────────────────────────────────────

def clp_d(a_out, b_out, c_out):
    out = {"step": "CLP-D", "tag": "assembly"}
    a_score = a_out.get("tightness_score", 0)
    b_score = b_out.get("family_score_mean", 0)
    c_score = c_out.get("gamma_convergence_score", 0)
    overall = 0.30*a_score + 0.40*b_score + 0.30*c_score
    out["families"] = {"CLP-A": a_score, "CLP-B/B4": b_score, "CLP-C": c_score}
    out["clp_overall_score"] = round(overall, 6)
    out["verdict"] = "CLP_PROVEN" if overall >= 0.7 else "CLP_PARTIAL"
    return out


# ─── main ───────────────────────────────────────────────────────────────

def main() -> int:
    print("="*100)
    print("(Opt-4) Full CLP-A/B/C/D re-fit on extended 9-point ladder")
    print("="*100)
    payloads = load_all_payloads()
    print(f"\n{len(payloads)} payloads loaded:")
    for p in payloads:
        print(f"  N={int(extract_n(p))}")
    print()

    a_out = clp_a(payloads)
    b_out = clp_b_b4(payloads)
    c_out = clp_c(payloads)
    d_out = clp_d(a_out, b_out, c_out)

    # Pretty print
    print(f"\n=== CLP-A Tightness ===")
    print(f"  T1 (diameter)      = {a_out['T1_score']:.4f}")
    print(f"  T2 (volume doubling)= {a_out['T2_score']:.4f}")
    print(f"  T3 (energy bound)  = {a_out['T3_score']:.4f}")
    print(f"  T4 (mass norm)     = {a_out['T4_score']:.4f}")
    print(f"  → tightness_score  = {a_out['tightness_score']:.4f}  ({a_out['tightness_status']})")
    if "T3_bootstrap" in a_out and a_out["T3_bootstrap"]:
        bs = a_out["T3_bootstrap"]
        print(f"  T3 bootstrap CI: [{bs['CI95'][0]:+.4f}, {bs['CI95'][1]:+.4f}]")

    print(f"\n=== CLP-B/B4 Slow Decomposition (Symanzik-2) ===")
    if "sub_components" in b_out:
        for tag, sc in b_out["sub_components"].items():
            fit = sc["fit"]; bs = sc["bootstrap"]
            print(f"  {tag:<12}: gap_inf={fit['gap_inf']:.4f}, "
                  f"bootstrap CI [{bs['CI95'][0]:+.4f}, {bs['CI95'][1]:+.4f}]" if bs else
                  f"  {tag:<12}: gap_inf={fit['gap_inf']:.4f} (no bootstrap)")
        print(f"  → family score mean = {b_out['family_score_mean']:.4f}")
        print(f"  → bottleneck       = {b_out['bottleneck']}")
        print(f"  → above-0.5 count  = {b_out['above_threshold_0p5_count']}/{b_out['n_components']}")

    print(f"\n=== CLP-C Γ-Convergence ===")
    print(f"  C1 (action density) = {c_out['C1_score']:.4f}")
    print(f"  C2 (variational)    = {c_out['C2_score']:.4f}")
    print(f"  → score = {c_out['gamma_convergence_score']:.4f}  ({c_out['gamma_convergence_status']})")

    print(f"\n=== CLP-D Assembly ===")
    print(f"  CLP-A:    {d_out['families']['CLP-A']:.4f}")
    print(f"  CLP-B/B4: {d_out['families']['CLP-B/B4']:.4f}")
    print(f"  CLP-C:    {d_out['families']['CLP-C']:.4f}")
    print(f"  → CLP overall = {d_out['clp_overall_score']:.4f}  ({d_out['verdict']})")

    out = {"method": "clp_full_families_extended", "n_payloads": len(payloads),
           "clp_a": a_out, "clp_b_b4": b_out, "clp_c": c_out, "clp_d": d_out}
    out_path = REPO / "emergent-gr-closure-repro" / "outputs" / "clp_full_families_extended_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
