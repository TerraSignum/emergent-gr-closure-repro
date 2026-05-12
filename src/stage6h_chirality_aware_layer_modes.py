"""Stage 6h: chirality-aware layer-resolved Fourier mode spectrum on
the Delta-residual core hierarchy.

Tests whether the layer-nested Delta-cores
(S_src^5%, C^(95), C^(99), C^(99.5), C^(99.9), C^(sup))
have distinct Fourier mode signatures
M_n := |<exp(i * 2n * theta_a^local)>_{a in layer}|, n=1,2,3,4
relative to chirality-mixed K, Q layer signatures, and whether
this signature changes pre- vs post- chirality flip
(N_flip = N_* * sqrt(d * N_gen) ~ 173, with sin^2 theta_chir <
0.5 pre and > 0.5 post).

Per-node observable definitions:
  theta_a^local := arg(psi_a / exp(i * <arg(psi_b)>_b))
                  (per-seed lattice-mean phase reference)
  layer        := index set defined by Delta-percentile or by
                  Xi-row-sum top fraction
  M_n^layer    := |<exp(i * 2n * theta_a^local)>_{a in layer}|

Layers per seed:
  S_src5      = top-5% by Xi-row-sum (companion paper P4-B
                 source-active heavy-tail support; selection method
                 of verify_KQ_top5_full_structural_closure.py)
  C95, C99, C99_5, C99_9, Csup
              = Delta-residual percentile core layers from
                stage6h_matter_core_phase_diagram.py

Z-score per (layer, mode):
  z_n := (M_n^layer - mean_random) / sigma_random
where random = 200 bootstrap supports of the same cardinality
drawn uniformly without replacement from the lattice.

Pre/post split:
  PRE  = {P5, P6, P5N64, P7, P5N72, P8, P5N84, P5N100}    (8)
  POST = {P5N200, P5N300, P5N512}                          (3)

Output: outputs/stage6h_chirality_aware_layer_modes.json
"""
from __future__ import annotations

import json
import math
import random
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
    per_node_relative_delta,
)
from verify_galerkin_runner_A_hessian_ricci import per_seed_galerkin  # noqa: E402

N_STAR = 50
N_GEN = 3
D = 4
N_FLIP = N_STAR * math.sqrt(D * N_GEN)
N_BOOT = 200
RNG_SEED = 0xC0FFEE
MODE_ORDERS = (1, 2, 3, 4)  # 2theta, 4theta, 6theta, 8theta

# Within-P5 lattice-extension ladder ONLY.  Excludes P6, P7, P8
# (separate canonical-physics regimes with different coupling
# parameters; the chirality-flow argument theta_chir(N) refers to
# the within-P5 lattice-extension dimension, not to cross-regime
# variation).  Excludes P5N128 (K/Q persistence artefact).
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


def chirality_phase(n_lat: int) -> tuple[float, str]:
    x = math.log(n_lat / N_STAR) / math.log(D * N_GEN)
    th = math.atan(N_GEN ** (2 * x - 1))
    s2 = math.sin(th) ** 2
    return s2, ("PRE" if th < math.pi / 4 else "POST")


def _theta_local(psi: np.ndarray) -> np.ndarray:
    """theta_a^local = arg(psi_a / exp(i * mean_arg_psi))."""
    mean_arg = float(np.angle(np.mean(psi)))
    rotated = psi * np.exp(-1j * mean_arg)
    return np.angle(rotated)


def _xi_row_sum_top_indices(xi_mat: np.ndarray, n_lat: int,
                              fraction: float) -> set[int]:
    xi = xi_mat.copy()
    np.fill_diagonal(xi, 0.0)
    deg = xi.sum(axis=1)
    n_top = max(1, int(np.ceil(fraction * n_lat)))
    top = np.argsort(-deg)[:n_top]
    return set(int(i) for i in top)


def _delta_layer(delta: np.ndarray, percentile: float) -> set[int]:
    thr = float(np.percentile(delta, percentile))
    return set(int(i) for i in np.nonzero(delta > thr)[0])


def _delta_argsup(delta: np.ndarray) -> set[int]:
    return {int(np.argmax(delta))}


def _mode_amplitude(theta_local: np.ndarray, support: set[int],
                      mode_n: int) -> float:
    if not support:
        return float("nan")
    idx = np.array(sorted(support), dtype=int)
    z = np.exp(1j * 2.0 * mode_n * theta_local[idx])
    return float(np.abs(np.mean(z)))


def _z_score_random(theta_local: np.ndarray, layer_size: int,
                      mode_n: int, rng: random.Random,
                      n_boot: int = N_BOOT) -> tuple[float, float]:
    n_lat = theta_local.size
    if layer_size <= 0 or layer_size >= n_lat:
        return float("nan"), float("nan")
    samples = []
    for _ in range(n_boot):
        idx = rng.sample(range(n_lat), layer_size)
        z = np.exp(1j * 2.0 * mode_n * theta_local[np.array(idx, dtype=int)])
        samples.append(float(np.abs(np.mean(z))))
    arr = np.array(samples)
    return float(arr.mean()), float(arr.std() + 1e-12)


def _seed_modes(xi_mat, psi, k_field, q_field, n_lat, rng):
    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
    delta_full = per_node_relative_delta(prep, LAMBDA_T, LAMBDA_S)["delta_full"]
    theta_local = _theta_local(psi)

    layers = {
        "S_src5": _xi_row_sum_top_indices(xi_mat.copy(), n_lat, 0.05),
        "C95":    _delta_layer(delta_full, 95.0),
        "C99":    _delta_layer(delta_full, 99.0),
        "C99_5":  _delta_layer(delta_full, 99.5),
        "C99_9":  _delta_layer(delta_full, 99.9),
        "Csup":   _delta_argsup(delta_full),
    }

    out = {}
    for layer_name, S in layers.items():
        layer_size = len(S)
        layer_data = {
            "size": layer_size,
            "support_fraction": layer_size / max(n_lat, 1),
            "modes": {},
        }
        for n in MODE_ORDERS:
            M = _mode_amplitude(theta_local, S, n)
            mu_r, sd_r = _z_score_random(theta_local, layer_size, n, rng)
            z = (M - mu_r) / sd_r if sd_r > 1e-12 else float("nan")
            layer_data["modes"][f"M{n}"] = {
                "amplitude": M,
                "random_mean": mu_r,
                "random_sigma": sd_r,
                "z_random": z,
            }
        out[layer_name] = layer_data
    return out


def _gather_regime(reg, n_lat, rng):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    if "snapshots" in p.name.lower():
        seeds = load_snapshots(p, n_lat)
    else:
        seeds = load_canonical(p, n_lat)
    seed_audits = []
    for xi_mat, psi, k_field, q_field in seeds:
        seed_audits.append(_seed_modes(xi_mat, psi, k_field, q_field,
                                          n_lat, rng))
    return seed_audits


def _agg(seed_audits, layer_name, mode_key, field):
    vals = [s[layer_name]["modes"][mode_key][field] for s in seed_audits
             if not np.isnan(s[layer_name]["modes"][mode_key][field])]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def main() -> int:
    print("=" * 110)
    print("Stage 6h: chirality-aware layer-resolved Fourier mode spectrum")
    print("=" * 110)
    print(f"  Modes: 2t, 4t, 6t, 8t  (n=1..4)")
    print(f"  Layers: S_src5, C95, C99, C99_5, C99_9, Csup")
    print(f"  N_flip ~ {N_FLIP:.1f}; PRE = sin^2t<0.5, POST = sin^2t>=0.5")
    print()

    rng = random.Random(RNG_SEED)
    rows = []
    for reg, n_lat in LADDER:
        s2, phase = chirality_phase(n_lat)
        seed_audits = _gather_regime(reg, n_lat, rng)
        if seed_audits is None:
            continue
        rows.append({
            "regime": reg, "N": n_lat,
            "sin2_theta_chir": s2, "phase": phase,
            "n_seeds": len(seed_audits),
            "layers": {
                ln: {
                    "size_mean": float(np.mean(
                        [s[ln]["size"] for s in seed_audits])),
                    "modes": {
                        mk: {
                            "amplitude_mean": _agg(seed_audits, ln, mk,
                                                   "amplitude")[0],
                            "amplitude_std": _agg(seed_audits, ln, mk,
                                                  "amplitude")[1],
                            "z_random_mean": _agg(seed_audits, ln, mk,
                                                  "z_random")[0],
                            "z_random_std": _agg(seed_audits, ln, mk,
                                                 "z_random")[1],
                        } for mk in [f"M{n}" for n in MODE_ORDERS]
                    },
                } for ln in ["S_src5", "C95", "C99", "C99_5", "C99_9", "Csup"]
            },
        })
        ln = "C99_5"
        modes = rows[-1]["layers"][ln]["modes"]
        print(f"  {reg:<8s} N={n_lat:>4d} {phase} sin^2t={s2:.3f}  "
              f"C99.5: M1={modes['M1']['amplitude_mean']:.3f}"
              f"(z={modes['M1']['z_random_mean']:+.1f})  "
              f"M2={modes['M2']['amplitude_mean']:.3f}"
              f"(z={modes['M2']['z_random_mean']:+.1f})  "
              f"M3={modes['M3']['amplitude_mean']:.3f}"
              f"(z={modes['M3']['z_random_mean']:+.1f})  "
              f"M4={modes['M4']['amplitude_mean']:.3f}"
              f"(z={modes['M4']['z_random_mean']:+.1f})")

    if len(rows) < 3:
        print("Not enough regimes")
        return 0

    print()
    print("=" * 110)
    print("Pre/post-flip mean amplitudes and z-scores per layer (cross-regime)")
    print("=" * 110)
    pre_post = {"PRE": [r for r in rows if r["phase"] == "PRE"],
                 "POST": [r for r in rows if r["phase"] == "POST"]}
    summary = {}
    for phase, grp in pre_post.items():
        summary[phase] = {}
        for ln in ["S_src5", "C95", "C99", "C99_5", "C99_9", "Csup"]:
            summary[phase][ln] = {}
            for n in MODE_ORDERS:
                mk = f"M{n}"
                amp = float(np.mean(
                    [r["layers"][ln]["modes"][mk]["amplitude_mean"]
                     for r in grp]))
                zsc = float(np.mean(
                    [r["layers"][ln]["modes"][mk]["z_random_mean"]
                     for r in grp]))
                z_above3 = sum(
                    1 for r in grp
                    if r["layers"][ln]["modes"][mk]["z_random_mean"] > 3.0
                )
                summary[phase][ln][mk] = {
                    "amp_mean": amp, "z_mean": zsc,
                    "n_z_above3": z_above3, "n_regimes": len(grp),
                }

    # Print compact table
    for phase in ("PRE", "POST"):
        n_grp = len(pre_post[phase])
        print(f"\n{phase} (n={n_grp}):")
        print(f"{'layer':>8s}  {'M1 amp':>7} {'M1 z':>6} {'M1 >=3':>6}  "
              f"{'M2 amp':>7} {'M2 z':>6} {'M2 >=3':>6}  "
              f"{'M3 amp':>7} {'M3 z':>6} {'M3 >=3':>6}  "
              f"{'M4 amp':>7} {'M4 z':>6} {'M4 >=3':>6}")
        for ln in ["S_src5", "C95", "C99", "C99_5", "C99_9", "Csup"]:
            cells = []
            for n in MODE_ORDERS:
                mk = f"M{n}"
                s = summary[phase][ln][mk]
                cells.append((s["amp_mean"], s["z_mean"], s["n_z_above3"]))
            print(f"{ln:>8s}  "
                  f"{cells[0][0]:>7.3f} {cells[0][1]:>+6.2f} "
                  f"{cells[0][2]:>3d}/{n_grp}  "
                  f"{cells[1][0]:>7.3f} {cells[1][1]:>+6.2f} "
                  f"{cells[1][2]:>3d}/{n_grp}  "
                  f"{cells[2][0]:>7.3f} {cells[2][1]:>+6.2f} "
                  f"{cells[2][2]:>3d}/{n_grp}  "
                  f"{cells[3][0]:>7.3f} {cells[3][1]:>+6.2f} "
                  f"{cells[3][2]:>3d}/{n_grp}")

    # Verdict
    print()
    print("=" * 110)
    print("Decision matrix per (layer, phase, mode): MODE_ANCHORED if z>3 in")
    print("  >= 6/8 PRE regimes  OR  >= 3/3 POST regimes")
    print("=" * 110)
    verdict = {}
    for phase in ("PRE", "POST"):
        verdict[phase] = {}
        thr = 6 if phase == "PRE" else 3
        for ln in ["S_src5", "C95", "C99", "C99_5", "C99_9", "Csup"]:
            verdict[phase][ln] = {}
            for n in MODE_ORDERS:
                mk = f"M{n}"
                s = summary[phase][ln][mk]
                anchored = s["n_z_above3"] >= thr
                verdict[phase][ln][mk] = ("MODE_ANCHORED" if anchored
                                            else "RANDOM")
    for phase in ("PRE", "POST"):
        print(f"\n{phase} verdict matrix:")
        print(f"{'layer':>8s}  {'M1':>14} {'M2':>14} {'M3':>14} {'M4':>14}")
        for ln in ["S_src5", "C95", "C99", "C99_5", "C99_9", "Csup"]:
            v1 = verdict[phase][ln]["M1"]
            v2 = verdict[phase][ln]["M2"]
            v3 = verdict[phase][ln]["M3"]
            v4 = verdict[phase][ln]["M4"]
            print(f"{ln:>8s}  {v1:>14s} {v2:>14s} {v3:>14s} {v4:>14s}")

    # Branch-interface verdict: do POST-Delta-cores show a distinctly
    # different mode pattern than PRE-Delta-cores or S_src5?
    print()
    print("=" * 110)
    print("Branch-interface verdict")
    print("=" * 110)

    def dom(d):
        anchored = [n for n in MODE_ORDERS
                     if d[f"M{n}"]["n_z_above3"] >= (6 if d == d else 3)]
        if not anchored:
            return None
        return max(anchored,
                    key=lambda n: d[f"M{n}"]["amp_mean"])

    def dom_amp(d):
        nbest = max(MODE_ORDERS, key=lambda n: d[f"M{n}"]["amp_mean"])
        return nbest

    print(f"{'layer':>8s}  {'PRE dom':>10s}  {'POST dom':>10s}  "
          f"{'PRE M_max':>10s}  {'POST M_max':>10s}")
    branch_cases = []
    for ln in ["S_src5", "C95", "C99", "C99_5", "C99_9", "Csup"]:
        pre_dom = dom_amp(summary["PRE"][ln])
        post_dom = dom_amp(summary["POST"][ln])
        pre_max = summary["PRE"][ln][f"M{pre_dom}"]["amp_mean"]
        post_max = summary["POST"][ln][f"M{post_dom}"]["amp_mean"]
        diff = (pre_dom != post_dom)
        branch_cases.append((ln, pre_dom, post_dom, diff))
        print(f"{ln:>8s}  M{pre_dom:>4d} (a={pre_max:.3f})   "
              f"M{post_dom:>4d} (a={post_max:.3f})   "
              f"{'DIFFERENT' if diff else 'SAME'}")

    branch_interface_confirmed = any(
        ln.startswith("C99") and diff
        for (ln, _, _, diff) in branch_cases
    )
    final_verdict = ("BRANCH_INTERFACE_CONFIRMED"
                       if branch_interface_confirmed
                       else "NO_BRANCH_INTERFACE_FROM_MODES")
    print(f"\n  FINAL: {final_verdict}")

    bundle = {
        "method": "stage6h_chirality_aware_layer_modes",
        "schema_version": "1.0.0",
        "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
        "n_flip": N_FLIP,
        "modes_definition":
            "M_n^layer := |<exp(i * 2n * theta_a^local)>_{a in layer}|",
        "theta_local_definition":
            "theta_a^local = arg(psi_a / exp(i * <arg(psi)>_lattice))",
        "n_bootstrap_random_supports": N_BOOT,
        "regimes": rows,
        "pre_post_summary": summary,
        "verdict_matrix": verdict,
        "branch_interface_dominant_mode_per_layer": [
            {"layer": ln, "PRE_dom_mode": pd, "POST_dom_mode": po,
             "different": d}
            for (ln, pd, po, d) in branch_cases
        ],
        "final_verdict": final_verdict,
    }
    out = REPO / "outputs" / "stage6h_chirality_aware_layer_modes.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print()
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
