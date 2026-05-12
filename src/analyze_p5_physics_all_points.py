"""Compute the full within-P5 physics for ALL ladder points uniformly,
to enable a clean within-regime sound-wave test (T11 redux), Var(Xi)
N-scaling, eigvalue-band stability, and FFT-power-per-k-bin without
cross-regime contamination.

Within-P5 ladder (single regime, varying lattice N):
  N = 50, 64, 72, 84, 100, 128, 200, 300

Per-point quantities computed:
  1. Var(Xi_full)           (per-seed, then mean +/- std)
  2. mean(|Xi|)             (per-seed, then mean +/- std)
  3. Eigvalue spectrum      (Gram trace-normalised, descending)
  4. IR / MID / UV bands    (largest-3 / middle-3 / smallest-3)
  5. FFT power per k-bin    (radial, 6 bins from 0 to pi)
  6. Lambda_t per N         (from within_P5_bootstrap_audit)
  7. T_00 per N             (from per_regime_audit if available)

Output: outputs/p5_physics_all_points.json
        AND a clean within-P5 T11-style sound-wave verdict.

DO NOT modify the manuscript; report only.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


# ============================================================
# Within-P5 ladder + npz path table
# ============================================================
P5_LADDER = [
    ("P5",      50,  "results_d1_fix17/d1_p5.npz",                "d1"),
    ("P5N64",   64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz",             "snap"),
    ("P5N72",   72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz",   "snap"),
    ("P5N84",   84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz",   "snap"),
    ("P5N100", 100,  "results_d1_p5n100_24seeds/P5N100.snapshots.npz",           "snap"),
    ("P5N128", 128,  "results_d1_p5n128_kq_fixed/P5N128.snapshots.npz",           "snap"),
    ("P5N200", 200,  "results_d1_p5n200_8seeds/P5N200.snapshots.npz",    "snap"),
    ("P5N300", 300,  "results_d1_p5n300_12seeds/P5N300.snapshots.npz",    "snap"),
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz",  "snap"),
    ("P5N512", 512,  "results_d1_p5n512_12seeds/P5N512.snapshots.npz",  "snap"),
]


def load_xi_matrices(rel_path: str, n_lat: int, kind: str,
                     max_seeds: int = 32):
    """Return list of N x N Xi matrices for the given ladder point."""
    fp = REPO.parent / rel_path
    if not fp.exists():
        return []
    z = np.load(fp, allow_pickle=True)

    if kind == "snap":
        if "edge_xi_snapshots" not in z.files:
            return []
        snaps = z["edge_xi_snapshots"]  # shape (n_seeds, n_snap, N, N)
        n_seeds_actual = int(min(snaps.shape[0], max_seeds))
        # Use FINAL snapshot only (last along axis 1)
        last_idx = snaps.shape[1] - 1
        out = []
        for s in range(n_seeds_actual):
            m = np.asarray(snaps[s, last_idx], dtype=float)
            if m.shape == (n_lat, n_lat):
                out.append(m)
        return out

    # kind == "d1": prefer xi_seed* keys, fall back to dense_cell_edge_*
    xi_keys = sorted(
        [k for k in z.files if k.startswith("xi_seed") and k[7:].isdigit()],
        key=lambda k: int(k[7:]))
    if xi_keys:
        ns = min(len(xi_keys), max_seeds)
        out = []
        for k in xi_keys[:ns]:
            m = np.asarray(z[k], dtype=float)
            if m.shape == (n_lat, n_lat):
                out.append(m)
        if out:
            return out
    if "dense_cell_edge_xi_values" in z.files:
        edge = z["dense_cell_edge_xi_values"]
        if (edge.ndim == 3 and edge.shape[1] == n_lat
                and edge.shape[2] == n_lat):
            ns = min(int(edge.shape[0]), max_seeds)
            return [np.asarray(edge[s], dtype=float) for s in range(ns)]
    return []


def gram_eigenvalues(xi):
    x = np.where(np.isfinite(xi), xi, 0.0)
    if not np.any(np.abs(x) > 1e-15):
        return np.zeros(xi.shape[0])
    try:
        sv = np.linalg.svd(x, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.zeros(xi.shape[0])
    eig = sv ** 2
    s = float(np.sum(eig))
    if s < 1e-12:
        return np.zeros(xi.shape[0])
    return np.sort(eig / s)[::-1]


def fft_power_per_kbin(xi, k_bins):
    n = xi.shape[0]
    x = np.where(np.isfinite(xi), xi, 0.0)
    x = x - np.mean(x)
    F = np.fft.fft2(x)
    P = np.abs(F) ** 2
    k1 = 2 * math.pi * np.fft.fftfreq(n)
    K1, K2 = np.meshgrid(k1, k1, indexing="ij")
    K_RAD = np.sqrt(K1 ** 2 + K2 ** 2)
    bins = []
    for i in range(len(k_bins) - 1):
        m = (K_RAD >= k_bins[i]) & (K_RAD < k_bins[i + 1])
        bins.append(float(np.mean(P[m])) if np.any(m) else 0.0)
    tot = sum(bins)
    if tot > 0:
        bins_norm = [v / tot for v in bins]
    else:
        bins_norm = bins
    return bins, bins_norm


def per_regime_xi_summary(xi_list, k_bins):
    var_full, mabs = [], []
    eigs_top3, eigs_bot3, eigs_mid3 = [], [], []
    bin_powers_all = []
    for xi in xi_list:
        x = np.where(np.isfinite(xi), xi, 0.0)
        var_full.append(float(np.var(x)))
        mabs.append(float(np.mean(np.abs(x))))
        eig = gram_eigenvalues(x)
        if eig.size >= 6:
            eigs_top3.append(eig[:3])
            eigs_bot3.append(eig[-3:])
            mid_start = max(0, x.shape[0] // 2 - 1)
            eigs_mid3.append(eig[mid_start:mid_start + 3])
        _, bins_norm = fft_power_per_kbin(x, k_bins)
        bin_powers_all.append(bins_norm)
    bp = np.array(bin_powers_all) if bin_powers_all else np.zeros((0, len(k_bins) - 1))
    out = {
        "n_seeds": len(xi_list),
        "var_xi_mean": float(np.mean(var_full)) if var_full else float("nan"),
        "var_xi_std": float(np.std(var_full)) if var_full else float("nan"),
        "mabs_xi_mean": float(np.mean(mabs)) if mabs else float("nan"),
        "mabs_xi_std": float(np.std(mabs)) if mabs else float("nan"),
    }
    if eigs_top3:
        out["uv_top3"] = np.mean(eigs_top3, axis=0).tolist()
        out["ir_bot3"] = np.mean(eigs_bot3, axis=0).tolist()
        out["mid3"] = np.mean(eigs_mid3, axis=0).tolist()
        out["uv_mean"] = float(np.mean(out["uv_top3"]))
        out["ir_mean"] = float(np.mean(out["ir_bot3"]))
        out["mid_mean"] = float(np.mean(out["mid3"]))
    if bp.size > 0:
        out["fft_bin_power_mean"] = bp.mean(axis=0).tolist()
        out["fft_bin_power_std"] = bp.std(axis=0).tolist()
    return out


def main() -> int:
    print("=" * 78)
    print("Within-P5 ladder: full physics for all points")
    print("=" * 78)

    bs = json.loads(
        (REPO / "outputs" / "within_P5_bootstrap_audit.json").read_text())
    seeds = bs["per_regime_seeds"]
    lambda_t_per_N = {
        int(tag.split("=")[1]): {
            "mean": float(np.mean(seeds[tag])),
            "std": float(np.std(seeds[tag])),
            "n": len(seeds[tag]),
        }
        for tag in seeds.keys()
    }

    pr = json.loads(
        (REPO / "outputs" / "per_regime_lambda_t_universal_audit.json"
         ).read_text())
    t00_by_regime = {r["regime"]: r for r in pr["per_regime"]}

    K_BINS = np.linspace(0, math.pi, 6 + 1)
    bundle = {"k_bins": K_BINS.tolist(), "ladder": []}

    print(f"\n{'tag':>8} {'N':>4} {'kind':>5} {'#seeds':>7} "
          f"{'Var(Xi)':>10} {'mabs':>8} {'Lt':>8} {'T_00':>8}")
    for reg, n_lat, rel, kind in P5_LADDER:
        xis = load_xi_matrices(rel, n_lat, kind, max_seeds=32)
        if not xis:
            print(f"{reg:>8} {n_lat:>4} {kind:>5} {0:>7} [no data]")
            continue
        summary = per_regime_xi_summary(xis, K_BINS)
        Lt = lambda_t_per_N.get(n_lat, {})
        T00 = (t00_by_regime.get(reg) or {}).get("T_00_med")
        Lt_val = Lt.get("mean")
        Lt_std = Lt.get("std")
        kappa = (Lt_val / T00) if (T00 is not None
                                    and Lt_val is not None
                                    and T00 > 0) else None
        row = {
            "regime": reg, "N": n_lat, "kind": kind,
            **summary,
            "Lambda_t_mean": Lt_val,
            "Lambda_t_std": Lt_std,
            "T_00_med": T00,
            "kappa_t": kappa,
        }
        bundle["ladder"].append(row)
        print(f"{reg:>8} {n_lat:>4} {kind:>5} {summary['n_seeds']:>7} "
              f"{summary['var_xi_mean']:>10.5f} "
              f"{summary['mabs_xi_mean']:>8.4f} "
              f"{(Lt_val if Lt_val is not None else float('nan')):>8.4f} "
              f"{(T00 if T00 is not None else float('nan')):>8.4f}")

    rows = bundle["ladder"]
    if not rows:
        print("No rows -- abort")
        return 1

    # ----- Var(Xi) N-scaling -----
    print()
    print("=" * 78)
    print("Var(Xi) N-scaling within-P5  (free-power fit y = a + b * N^-p)")
    print("=" * 78)
    Ns = np.array([r["N"] for r in rows], dtype=float)
    var_arr = np.array([r["var_xi_mean"] for r in rows], dtype=float)
    # log-space slope
    log_N = np.log(Ns)
    log_var = np.log(var_arr)
    slope, intercept = np.polyfit(log_N, log_var, 1)
    print(f"  log Var(Xi) = {intercept:.3f} + ({slope:+.3f}) * log N")
    print(f"  -> Var(Xi) ~ N^{slope:.3f}")
    print(f"  Asymptotic Var(Xi) at N->inf is {0.0 if slope < 0 else 'divergent'}")
    bundle["var_xi_scaling"] = {
        "slope": float(slope),
        "intercept": float(intercept),
        "interpretation": "Var(Xi) ~ N^slope; slope < 0 means continuum-zero",
    }

    # ----- Eigvalue band stability -----
    print()
    print("=" * 78)
    print("Eigvalue bands within-P5  (Gram normalised, descending)")
    print("=" * 78)
    print(f"{'N':>4} {'UV mean (top3)':>17} {'MID mean':>10} "
          f"{'IR mean (bot3)':>17}")
    for r in rows:
        if "uv_mean" not in r:
            continue
        print(f"{r['N']:>4} {r['uv_mean']:>17.5f} {r['mid_mean']:>10.5f} "
              f"{r['ir_mean']:>17.7f}")
    uv_arr = np.array([r["uv_mean"] for r in rows if "uv_mean" in r])
    mid_arr = np.array([r["mid_mean"] for r in rows if "mid_mean" in r])
    ir_arr = np.array([r["ir_mean"] for r in rows if "ir_mean" in r])
    cv_uv = float(uv_arr.std() / uv_arr.mean()) if uv_arr.mean() > 0 else float("nan")
    cv_mid = float(mid_arr.std() / mid_arr.mean()) if mid_arr.mean() > 0 else float("nan")
    cv_ir = float(ir_arr.std() / ir_arr.mean()) if ir_arr.mean() > 0 else float("nan")
    print(f"\n  CV across N within-P5:")
    print(f"    UV-band (top3):    {cv_uv:.3f}")
    print(f"    MID-band:          {cv_mid:.3f}")
    print(f"    IR-band (bot3):    {cv_ir:.3f}")
    bundle["eigvalue_band_cv"] = {"uv": cv_uv, "mid": cv_mid, "ir": cv_ir}

    # ----- FFT k-bin sound-wave test (within-P5) -----
    print()
    print("=" * 78)
    print("FFT power per k-bin within-P5  (sound-wave proper test)")
    print("=" * 78)
    BIN_LABELS = [f"[{K_BINS[i]:.2f},{K_BINS[i+1]:.2f}]"
                  for i in range(len(K_BINS) - 1)]
    print(f"  Bins: {BIN_LABELS}")
    print()
    print(f"{'N':>4} | "
          + "  ".join(f"bin{i}".rjust(7) for i in range(len(K_BINS) - 1)))
    bin_matrix = []
    for r in rows:
        if "fft_bin_power_mean" not in r:
            continue
        cells = "  ".join(f"{v:>7.4f}" for v in r["fft_bin_power_mean"])
        print(f"{r['N']:>4} | {cells}")
        bin_matrix.append(r["fft_bin_power_mean"])
    bin_matrix = np.array(bin_matrix) if bin_matrix else np.zeros((0, 0))
    if bin_matrix.size > 0:
        cv_per_bin = []
        print()
        print("  CV per k-bin across within-P5 ladder:")
        print(f"  {'bin':>4} {'k-range':>16} {'mean':>9} {'std':>9} {'CV':>7}")
        for i in range(bin_matrix.shape[1]):
            col = bin_matrix[:, i]
            mu = float(col.mean())
            sd = float(col.std())
            cv = sd / mu if mu > 0 else float("nan")
            cv_per_bin.append(cv)
            print(f"  {i:>4} {BIN_LABELS[i]:>16} {mu:>9.5f} "
                  f"{sd:>9.5f} {cv:>7.3f}")
        bundle["fft_cv_per_bin"] = cv_per_bin
        valid = [(i, c) for i, c in enumerate(cv_per_bin)
                 if c is not None and math.isfinite(c)]
        if len(valid) >= 4:
            low_k_cv = float(np.mean([c for i, c in valid[:2]]))
            high_k_cv = float(np.mean([c for i, c in valid[-2:]]))
            ratio = low_k_cv / high_k_cv if high_k_cv > 0 else float("nan")
            print()
            print(f"  Low-k CV (bins 0-1):  {low_k_cv:.3f}")
            print(f"  High-k CV (last 2):   {high_k_cv:.3f}")
            print(f"  Ratio low_k/high_k:   {ratio:.3f}")
            if ratio < 0.5:
                v = ("SUPPORTED on within-P5: low-k modes far more "
                     f"N-stable than high-k (ratio {ratio:.2f}). Sound-wave "
                     "analogy holds within a single regime.")
            elif ratio < 1.0:
                v = (f"PARTIAL within-P5: low-k more stable than high-k "
                     f"(ratio {ratio:.2f}).")
            elif ratio < 2.0:
                v = (f"NEUTRAL within-P5: similar stability "
                     f"(ratio {ratio:.2f}).")
            else:
                v = (f"REJECTED within-P5: low-k LESS stable "
                     f"(ratio {ratio:.2f}).")
            print(f"  VERDICT: {v}")
            bundle["fft_sound_wave_verdict"] = v
            bundle["fft_low_high_ratio"] = ratio

    # ----- Lambda_t / Var(Xi) within-P5 -----
    print()
    print("=" * 78)
    print("Lambda_t / Var(Xi) within-P5 (K/Q-coupling diagnosis)")
    print("=" * 78)
    print(f"{'N':>4} {'Var(Xi)':>10} {'Lambda_t':>10} {'L/Var':>10} "
          f"{'L/Var/N':>10}")
    for r in rows:
        if r["Lambda_t_mean"] is None:
            continue
        L_V = r["Lambda_t_mean"] / r["var_xi_mean"]
        L_V_N = L_V / r["N"]
        r["Lambda_over_Var_Xi"] = L_V
        r["Lambda_over_Var_Xi_per_N"] = L_V_N
        print(f"{r['N']:>4} {r['var_xi_mean']:>10.5f} "
              f"{r['Lambda_t_mean']:>10.4f} {L_V:>10.2f} {L_V_N:>10.4f}")

    # Save bundle
    out = REPO / "outputs" / "p5_physics_all_points.json"
    out.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
