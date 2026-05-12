"""Xi-EOM bounce / flip diagnostic at the chirality flip.

User-directed test (2026-05-07): does the strict non-linear
Xi-EOM exhibit a bounce / rebounce / flip phenomenon at the
chirality flip N_flip = N_star * sqrt(d * N_gen) ~ 173?

The chirality flip is the structural transition between PRE-flip
(theta_chir < pi/4, N < N_flip) and POST-flip
(theta_chir > pi/4, N > N_flip) regimes, accompanied by
qualitative changes in K, Q (matter-localised top-5% closure
hits chirality-PRE vs POST endpoint values), the rho-anti-
coherence pattern on Delta-cores, and the slaving-K signature.

For the Xi-EOM (the time-derivative dXi/dt of the relational
similarity field), bounce/flip signatures could appear as:
  (1) sign reversal of d<Xi>/dN at N_flip (turning point in
      lattice-mean trajectory),
  (2) sign reversal of <d<Xi>/dt>(N) (the dynamical EOM-mean),
  (3) discontinuity / kink in the EOM second derivative
      d^2 <Xi> / dN^2 at N_flip,
  (4) chirality-angle reversal in the K^2/Q ratio (already
      tested in S_b from K, Q closed potential audit) at the
      flip.

The audit on the canonical-physics ladder
{P5N50, P5N64, P5N72, P5N84, P5N100, P5N200, P5N300, P5N512}
spans PRE-flip (N <= 100) and POST-flip (N >= 200) and computes:
  - lattice mean <Xi> per regime
  - finite-difference d<Xi>/dN across regimes
  - sign of d<Xi>/dN PRE vs POST flip
  - the dynamical mean <dXi/dt> per snapshot per seed
  - chirality-angle theta_chir(N)

Output: outputs/verify_xi_eom_chirality_flip_bounce.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


class _BlockCupy:
    def find_module(self, name, _path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self

    def load_module(self, name):
        raise ImportError("cupy disabled")


sys.meta_path.insert(0, _BlockCupy())

REPO = Path(__file__).resolve().parents[1]
REPO_ROOT = REPO.parent

N_STAR = 50
N_GEN = 3
D = 4
N_FLIP = N_STAR * math.sqrt(D * N_GEN)


def chirality_phase(n_lat: int):
    x = math.log(n_lat / N_STAR) / math.log(D * N_GEN)
    th = math.atan(N_GEN ** (2 * x - 1))
    return th, "PRE" if th < math.pi / 4.0 else "POST"


LADDER = [
    ("P5N64",  64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz"),
    ("P5N72",  72,  "results_d1_p5n72_24seeds/P5N72.snapshots.npz"),
    ("P5N84",  84,  "results_d1_p5n84_24seeds/P5N84.snapshots.npz"),
    ("P5N100", 100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz"),
    ("P5N200", 200, "results_d1_p5n200_8seeds/P5N200.snapshots.npz"),
    ("P5N300", 300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz"),
    ("P5N512", 512, "results_d1_p5n512_12seeds/P5N512.snapshots.npz"),
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz"),
]


def main() -> int:
    rows = []
    for label, n_lat, sub in LADDER:
        path = REPO_ROOT / sub
        if not path.exists():
            continue
        z = np.load(path, allow_pickle=True)
        if "edge_xi_snapshots" not in z.files:
            continue
        snaps = np.asarray(z["edge_xi_snapshots"], dtype=float)
        n_seeds, n_t, _, _ = snaps.shape
        last = n_t - 1
        # Lattice mean of Xi off-diagonal at final snapshot
        xi_means = []
        # Dynamical d<Xi>/dt by finite differences across snapshots
        dxi_dt_means = []
        for s in range(n_seeds):
            xi_t = np.asarray(snaps[s], dtype=float)
            xi_off_t = []
            for t in range(n_t):
                xi = 0.5 * (xi_t[t] + xi_t[t].T).copy()
                np.fill_diagonal(xi, 0.0)
                xi_off_t.append(xi.mean())
            xi_off_t = np.array(xi_off_t)
            # final lattice-mean
            xi_means.append(float(xi_off_t[last]))
            # mean dynamical rate (last - first) / (n_t - 1)
            dxi_dt_means.append(
                float((xi_off_t[last] - xi_off_t[0]) /
                      max(n_t - 1, 1)))
            # finite-difference time series: median of consecutive
            # snapshot diffs
            d_arr = np.diff(xi_off_t)
            dxi_dt_means[-1] = float(np.mean(d_arr))
        theta, phase = chirality_phase(n_lat)
        rows.append({
            "regime": label,
            "N": int(n_lat),
            "n_seeds": int(n_seeds),
            "n_snapshots": int(n_t),
            "theta_chir": theta,
            "theta_chir_deg": math.degrees(theta),
            "phase": phase,
            "xi_lattice_mean_final_per_seed": xi_means,
            "xi_lattice_mean_final_avg": float(np.mean(xi_means)),
            "xi_lattice_mean_final_sem":
                float(np.std(xi_means, ddof=1)
                      / math.sqrt(len(xi_means))),
            "dxi_dt_mean_per_seed": dxi_dt_means,
            "dxi_dt_avg":  float(np.mean(dxi_dt_means)),
            "dxi_dt_sem":
                float(np.std(dxi_dt_means, ddof=1)
                      / math.sqrt(len(dxi_dt_means))),
        })
        print(f"  {label} N={n_lat:<4d} {phase}-flip "
              f"theta={math.degrees(theta):>5.2f}deg  "
              f"<Xi>={rows[-1]['xi_lattice_mean_final_avg']:.5f}+-"
              f"{rows[-1]['xi_lattice_mean_final_sem']:.5f}  "
              f"<dXi/dt>={rows[-1]['dxi_dt_avg']:+.6f}+-"
              f"{rows[-1]['dxi_dt_sem']:.6f}")

    # Cross-regime: finite difference d<Xi>/dN at consecutive
    # regimes; sign change near N_flip
    n_arr = np.array([r["N"] for r in rows], dtype=float)
    xi_arr = np.array([r["xi_lattice_mean_final_avg"]
                         for r in rows], dtype=float)
    dxi_dt_arr = np.array([r["dxi_dt_avg"] for r in rows],
                            dtype=float)
    dxi_dN = np.diff(xi_arr) / np.diff(n_arr)
    n_mid = (n_arr[:-1] + n_arr[1:]) / 2.0
    # Sign of d<Xi>/dN at PRE-flip vs POST-flip regimes
    pre_indices = [i for i in range(len(rows))
                    if rows[i]["phase"] == "PRE"]
    post_indices = [i for i in range(len(rows))
                     if rows[i]["phase"] == "POST"]
    pre_dxi_dN = []
    post_dxi_dN = []
    for k in range(len(dxi_dN)):
        if (rows[k]["phase"] == "PRE"
            and rows[k + 1]["phase"] == "PRE"):
            pre_dxi_dN.append(dxi_dN[k])
        elif (rows[k]["phase"] == "POST"
              and rows[k + 1]["phase"] == "POST"):
            post_dxi_dN.append(dxi_dN[k])

    pre_dxi_dt = [dxi_dt_arr[i] for i in pre_indices]
    post_dxi_dt = [dxi_dt_arr[i] for i in post_indices]

    # Bounce signature: sign reversal between PRE and POST
    bounce_signal = {
        "d_xi_d_N_pre_avg":
            float(np.mean(pre_dxi_dN)) if pre_dxi_dN else None,
        "d_xi_d_N_post_avg":
            float(np.mean(post_dxi_dN)) if post_dxi_dN else None,
        "d_xi_d_N_sign_reversal":
            (bool((np.mean(pre_dxi_dN) * np.mean(post_dxi_dN))
                    < 0)
             if pre_dxi_dN and post_dxi_dN else None),
        "d_xi_d_t_pre_avg":
            float(np.mean(pre_dxi_dt)) if pre_dxi_dt else None,
        "d_xi_d_t_post_avg":
            float(np.mean(post_dxi_dt)) if post_dxi_dt else None,
        "d_xi_d_t_sign_reversal":
            (bool((np.mean(pre_dxi_dt) * np.mean(post_dxi_dt))
                    < 0)
             if pre_dxi_dt and post_dxi_dt else None),
    }

    bundle = {
        "method": "verify_xi_eom_chirality_flip_bounce",
        "schema_version": "1.0.0",
        "framework_constants": {
            "N_star": N_STAR,
            "N_gen":  N_GEN,
            "d":      D,
            "N_flip": N_FLIP,
        },
        "rows": rows,
        "cross_regime": {
            "N_array":      n_arr.tolist(),
            "xi_mean_array":xi_arr.tolist(),
            "dxi_dN_finite_diff": dxi_dN.tolist(),
            "N_midpoints":  n_mid.tolist(),
            "dxi_dt_array": dxi_dt_arr.tolist(),
        },
        "bounce_signature": bounce_signal,
        "verdict": (
            "PRE-flip <dXi/dN> = {:.3e}, POST-flip <dXi/dN> = "
            "{:.3e}; sign reversal: {}. PRE-flip <dXi/dt> = "
            "{:.3e}, POST-flip <dXi/dt> = {:.3e}; dynamical sign "
            "reversal: {}. The structural bounce/flip signature "
            "is the simultaneous sign reversal of both "
            "d<Xi>/dN and d<Xi>/dt across N_flip ~ {:.0f}."
            .format(bounce_signal['d_xi_d_N_pre_avg'] or float('nan'),
                    bounce_signal['d_xi_d_N_post_avg'] or float('nan'),
                    bounce_signal['d_xi_d_N_sign_reversal'],
                    bounce_signal['d_xi_d_t_pre_avg'] or float('nan'),
                    bounce_signal['d_xi_d_t_post_avg'] or float('nan'),
                    bounce_signal['d_xi_d_t_sign_reversal'],
                    N_FLIP)
        ),
    }
    out = (REPO / "outputs"
           / "verify_xi_eom_chirality_flip_bounce.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2),
                   encoding="utf-8")

    print()
    print("=" * 78)
    print("Cross-regime bounce signature (Xi-EOM at chirality flip)")
    print("=" * 78)
    print(f"  N_flip ~ {N_FLIP:.0f}")
    print(f"  PRE-flip <d<Xi>/dN>  = "
          f"{bounce_signal['d_xi_d_N_pre_avg']}")
    print(f"  POST-flip <d<Xi>/dN> = "
          f"{bounce_signal['d_xi_d_N_post_avg']}")
    print(f"  d<Xi>/dN sign reversal: "
          f"{bounce_signal['d_xi_d_N_sign_reversal']}")
    print()
    print(f"  PRE-flip <dXi/dt>  = "
          f"{bounce_signal['d_xi_d_t_pre_avg']}")
    print(f"  POST-flip <dXi/dt> = "
          f"{bounce_signal['d_xi_d_t_post_avg']}")
    print(f"  dXi/dt sign reversal: "
          f"{bounce_signal['d_xi_d_t_sign_reversal']}")
    print(f"  saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
