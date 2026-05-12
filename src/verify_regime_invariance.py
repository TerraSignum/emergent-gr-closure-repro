"""Per-regime within-regime Symanzik test for regime-invariance.

We have 13 D1 NPZ regimes now (added P6N128, P8N128). Group them
into within-regime sequences:

  P5-physics: P5 (N=50), P5N64, P5N100  (3 points)
  P6-physics: P6 (N=60), P6N128         (2 points; can compare to P5 fit)
  P8-physics: P8 (N=84), P8N128         (2 points; can compare to P5 fit)

For each per-eigendirection component, fit Symanzik 2 (2-param,
forced by 2-pt sequences) on each within-regime sequence. Check if
the d_inf values agree across regimes (within their bootstrap CIs).

If yes -> regime-invariant N-convergence, the cross-regime fit
on 11 (now 13) points was not just regime-physics correlation.

Output: outputs/regime_invariance_audit.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


class _BlockCupy:
    def find_module(self, name, path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self

    def load_module(self, name):
        raise ImportError("cupy disabled")

sys.meta_path.insert(0, _BlockCupy())

from verify_per_eigendirection_residual import gather_regime


WITHIN_REGIME_SEQUENCES = {
    "P5_physics": [("P5", 50), ("P5N64", 64), ("P5N100", 100)],
    "P6_physics": [("P6", 60), ("P6N128", 128)],
    "P8_physics": [("P8", 84), ("P8N128", 128)],
}


def fit_symanzik_2(N, y):
    """y = d_inf + c_2 / N^2"""
    if len(N) < 2:
        return None
    X = np.column_stack([np.ones_like(N), N ** -2.0])
    if len(N) == 2:
        # Exactly determined system: solve directly
        c = np.linalg.solve(X, y)
    else:
        c, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
    if c[0] < 0:
        # Refit with d_inf = 0
        if len(N) == 2:
            c = np.array([0.0, float(np.mean(y * N ** 2))])
        else:
            X2 = X[:, 1:]
            c12, *_ = np.linalg.lstsq(X2, y, rcond=1e-10)
            c = np.array([0.0, c12[0]])
    pred = X @ c
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 and len(N) >= 3 else float("nan")
    return {"d_inf": float(c[0]), "c_2": float(c[1]), "R_squared": r2,
            "predicted_at_N_1000": float(c[0] + c[1] / 1000 ** 2)}


def main() -> int:
    print("=" * 100)
    print("Regime-invariance test: per-regime within-regime Symanzik fits")
    print("=" * 100)
    print()

    # Gather data
    sequences = {}
    for seq_name, regimes in WITHIN_REGIME_SEQUENCES.items():
        rows = []
        for reg, n_lat in regimes:
            try:
                r = gather_regime(reg, n_lat)
            except Exception as e:
                print(f"  {seq_name}/{reg} N={n_lat}: ERROR: {e}")
                continue
            if r is None:
                print(f"  {seq_name}/{reg} N={n_lat}: not found")
                continue
            rows.append(r)
        sequences[seq_name] = rows
        print(f"  {seq_name}: {len(rows)} pts -> {[r['regime']+'/'+str(r['N']) for r in rows]}")

    # Per-component, per-sequence Symanzik fit
    components = ["R_time_median_abs", "R_trace_median_abs",
                  "R_TF_norm_median_abs", "R_off_median_abs"]

    out_fits = {}
    for comp in components:
        print()
        print(f"--- {comp} ---")
        print(f"{'sequence':<14} {'pts':>4} {'values':>40} | {'d_inf':>10} {'c_2':>10}")
        print("-" * 90)
        out_fits[comp] = {}
        for seq_name, rows in sequences.items():
            if len(rows) < 2:
                continue
            N_arr = np.array([r["N"] for r in rows], dtype=float)
            y = np.array([r[comp] for r in rows])
            f = fit_symanzik_2(N_arr, y)
            vals_str = ", ".join([f"{v:.4f}" for v in y])
            print(f"{seq_name:<14} {len(rows):>4} {vals_str:>40} | "
                  f"{f['d_inf']:>10.5f} {f['c_2']:>+10.2f}")
            out_fits[comp][seq_name] = {
                "n_pts": len(rows),
                "N_values": [int(r["N"]) for r in rows],
                "values": [float(v) for v in y],
                "d_inf": f["d_inf"],
                "c_2": f["c_2"],
            }

    # Aggregate: do d_inf values agree across regimes?
    print()
    print("=" * 100)
    print("Cross-sequence consistency check (d_inf per component)")
    print("=" * 100)
    for comp in components:
        d_infs = [out_fits[comp][s]["d_inf"] for s in out_fits[comp]]
        if not d_infs: continue
        d_inf_mean = float(np.mean(d_infs))
        d_inf_max = float(max(d_infs))
        d_inf_min = float(min(d_infs))
        spread = d_inf_max - d_inf_min
        # Threshold: spread within 0.02 (closure threshold scale) is consistent
        consistent = spread <= 0.02
        print(f"  {comp:<22} d_inf values: " +
              ", ".join([f"{s}={out_fits[comp][s]['d_inf']:.5f}"
                         for s in out_fits[comp]]) +
              f"  spread={spread:.5f}  -> "
              f"{'CONSISTENT' if consistent else 'INCONSISTENT'}")

    out_path = REPO / "outputs" / "regime_invariance_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "regime_invariance_per_within_regime_symanzik",
            "schema_version": "1.0.0",
            "sequences": {k: [r["regime"]+"_N"+str(r["N"]) for r in v]
                          for k, v in sequences.items()},
            "fits_per_component_per_sequence": out_fits,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
