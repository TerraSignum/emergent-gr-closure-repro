"""Stage 6h v2: pooled-over-seeds layer-mode and phase-coherence
analysis on the within-P5 chirality-flow ladder.

Improvements over v1 (stage6h_chirality_aware_layer_modes.py):
  - Pool layer indices over all seeds of a regime (C99.5 stops being
    1-node-trivial; gets 24-150 nodes per regime).
  - Two theta-local definitions:
      theta_A_a = arg(psi_a / exp(i * <arg psi>_lattice))
      theta_B_a = arg(psi_a / nbhd_phase_a)
                 where nbhd_phase_a = arg(<psi_b * xi_ab>_b!=a)
  - Per-node phase coherence with neighbours:
      rho_a = Re(<exp(-i theta_a) * exp(i theta_b) * xi_ab>_b!=a)
            in [-1, 1]; high = locked to neighbourhood, low/neg =
            anti-correlated / phase-incoherent
  - Layer-vs-lattice z-score for each statistic via bootstrap
    over RANDOM SUPPORTS OF EQUAL POOLED SIZE drawn uniformly from
    the seed-pooled lattice.

Within-P5 ladder ONLY (P6, P7, P8 excluded - separate regimes):
  PRE  = {P5(50), P5N64, P5N72, P5N84, P5N100}      (5)
  POST = {P5N200, P5N300, P5N512}                    (3)

Output: outputs/stage6h_layer_modes_v2.json
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
N_BOOT = 500
RNG_SEED = 0xC0FFEE
MODE_ORDERS = (1, 2, 3, 4)

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


def _xi_top_indices(xi_mat, n_lat, fraction):
    xi = xi_mat.copy()
    np.fill_diagonal(xi, 0.0)
    deg = xi.sum(axis=1)
    n_top = max(1, int(np.ceil(fraction * n_lat)))
    return set(int(i) for i in np.argsort(-deg)[:n_top])


def _delta_layer(delta, percentile):
    thr = float(np.percentile(delta, percentile))
    return set(int(i) for i in np.nonzero(delta > thr)[0])


def _per_seed_observables(xi_mat, psi, k_field, q_field, n_lat):
    """Compute per-seed per-node observables.

    Returns dict with arrays of length n_lat:
      delta:   per-node Frobenius residual
      theta_A: psi-argument minus lattice-mean argument
      theta_B: psi-argument minus xi-weighted neighbourhood argument
      rho:    Re(e^(-i theta_a) * <e^(i theta_b) * xi_ab>) in [-1,1]
    """
    prep = per_seed_galerkin(xi_mat, psi, k_field, q_field, n_lat, np)
    delta_full = per_node_relative_delta(prep, LAMBDA_T, LAMBDA_S)["delta_full"]

    # theta_A: lattice-mean reference
    mean_arg = float(np.angle(np.mean(psi)))
    psi_A = psi * np.exp(-1j * mean_arg)
    theta_A = np.angle(psi_A)

    # theta_B: xi-weighted neighbourhood reference
    xi_no = xi_mat.copy()
    np.fill_diagonal(xi_no, 0.0)
    # nbhd_mean_psi[a] = sum_b xi_ab * psi_b / sum_b xi_ab
    w_sum = xi_no.sum(axis=1)
    safe = np.maximum(w_sum, 1e-12)
    nbhd_psi = (xi_no @ psi) / safe
    nbhd_phase = np.angle(nbhd_psi)
    theta_B = np.angle(psi * np.exp(-1j * nbhd_phase))

    # rho_a = Re( e^(-i theta_a) * <e^(i theta_b) xi_ab>_b )
    # i.e. xi-weighted real coherence to neighbours
    eiphi = np.exp(1j * np.angle(psi))
    nbhd_eiphi = (xi_no @ eiphi) / safe
    rho = np.real(np.conj(eiphi) * nbhd_eiphi)

    return {
        "delta": delta_full,
        "theta_A": theta_A,
        "theta_B": theta_B,
        "rho": rho,
    }


def _layer_pooled_arrays(seeds_obs, layer_selector):
    """Pool selected layer-nodes over all seeds.

    Returns concatenated arrays of theta_A, theta_B, rho.
    """
    a_list, b_list, r_list = [], [], []
    for obs, xi in seeds_obs:
        S = layer_selector(obs, xi)
        if not S:
            continue
        idx = np.array(sorted(S), dtype=int)
        a_list.append(obs["theta_A"][idx])
        b_list.append(obs["theta_B"][idx])
        r_list.append(obs["rho"][idx])
    if not a_list:
        return np.array([]), np.array([]), np.array([])
    return (np.concatenate(a_list),
            np.concatenate(b_list),
            np.concatenate(r_list))


def _full_lattice_arrays(seeds_obs):
    a_list, b_list, r_list = [], [], []
    for obs, _ in seeds_obs:
        a_list.append(obs["theta_A"])
        b_list.append(obs["theta_B"])
        r_list.append(obs["rho"])
    return (np.concatenate(a_list),
            np.concatenate(b_list),
            np.concatenate(r_list))


def _mode_amplitude_pooled(theta_array, mode_n):
    if theta_array.size == 0:
        return float("nan")
    z = np.exp(1j * 2.0 * mode_n * theta_array)
    return float(np.abs(np.mean(z)))


def _bootstrap_random_supports(theta_full, layer_size, n_boot, rng,
                                  mode_n):
    n_full = theta_full.size
    samples = []
    for _ in range(n_boot):
        idx = rng.sample(range(n_full), min(layer_size, n_full))
        samples.append(_mode_amplitude_pooled(theta_full[np.array(idx)],
                                                  mode_n))
    return np.array(samples)


def _rho_summary(rho_arr):
    if rho_arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(rho_arr)),
        "median": float(np.median(rho_arr)),
        "std": float(np.std(rho_arr)),
        "n": int(rho_arr.size),
    }


def _gather_regime(reg, n_lat):
    p = find_d1_npz(reg, REPO)
    if p is None or not p.exists():
        return None
    if "snapshots" in p.name.lower():
        seeds = load_snapshots(p, n_lat)
    else:
        seeds = load_canonical(p, n_lat)
    seeds_obs = []
    for xi_mat, psi, k_field, q_field in seeds:
        obs = _per_seed_observables(xi_mat, psi, k_field, q_field, n_lat)
        seeds_obs.append((obs, xi_mat))
    return seeds_obs


def main() -> int:
    print("=" * 110)
    print("Stage 6h v2: pooled-over-seeds layer modes & phase coherence")
    print("Within-P5 ladder, two theta-local definitions + rho local")
    print("=" * 110)

    rng = random.Random(RNG_SEED)
    layer_defs = {
        "S_src5": lambda obs, xi:
            _xi_top_indices(xi, xi.shape[0], 0.05),
        "C95":   lambda obs, _xi: _delta_layer(obs["delta"], 95.0),
        "C99":   lambda obs, _xi: _delta_layer(obs["delta"], 99.0),
        "C99_5": lambda obs, _xi: _delta_layer(obs["delta"], 99.5),
        "C99_9": lambda obs, _xi: _delta_layer(obs["delta"], 99.9),
        "Csup":  lambda obs, _xi: {int(np.argmax(obs["delta"]))},
    }

    rows = []
    for reg, n_lat in LADDER:
        s2, phase = chirality_phase(n_lat)
        seeds_obs = _gather_regime(reg, n_lat)
        if seeds_obs is None or not seeds_obs:
            print(f"  {reg} N={n_lat}: NPZ not found, skipping")
            continue

        # full-lattice pooled arrays (for random-support bootstrap)
        theta_A_full, theta_B_full, rho_full = _full_lattice_arrays(seeds_obs)
        rho_full_summary = _rho_summary(rho_full)

        layer_results = {}
        for ln, sel in layer_defs.items():
            theta_A_lay, theta_B_lay, rho_lay = _layer_pooled_arrays(
                seeds_obs, sel)
            layer_size = theta_A_lay.size
            entry = {
                "pooled_size": int(layer_size),
                "rho_layer": _rho_summary(rho_lay),
                "rho_lattice": rho_full_summary,
                "modes_theta_A": {},
                "modes_theta_B": {},
            }
            # rho z-score: layer-mean vs bootstrap-random-mean of size layer_size
            if layer_size > 0:
                rho_boot_means = []
                for _ in range(N_BOOT):
                    idx = rng.sample(range(rho_full.size),
                                       min(layer_size, rho_full.size))
                    rho_boot_means.append(
                        float(np.mean(rho_full[np.array(idx)])))
                rb = np.array(rho_boot_means)
                z_rho = ((entry["rho_layer"]["mean"] - rb.mean())
                          / max(rb.std(), 1e-12))
                entry["rho_z_random"] = float(z_rho)
            else:
                entry["rho_z_random"] = float("nan")

            for mode_n in MODE_ORDERS:
                M_A = _mode_amplitude_pooled(theta_A_lay, mode_n)
                M_B = _mode_amplitude_pooled(theta_B_lay, mode_n)
                if layer_size > 0:
                    boot_A = _bootstrap_random_supports(
                        theta_A_full, layer_size, N_BOOT, rng, mode_n)
                    boot_B = _bootstrap_random_supports(
                        theta_B_full, layer_size, N_BOOT, rng, mode_n)
                    z_A = (M_A - boot_A.mean()) / max(boot_A.std(), 1e-12)
                    z_B = (M_B - boot_B.mean()) / max(boot_B.std(), 1e-12)
                else:
                    z_A = z_B = float("nan")
                entry["modes_theta_A"][f"M{mode_n}"] = {
                    "amplitude": M_A, "z_random": float(z_A),
                }
                entry["modes_theta_B"][f"M{mode_n}"] = {
                    "amplitude": M_B, "z_random": float(z_B),
                }
            layer_results[ln] = entry

        rows.append({
            "regime": reg, "N": n_lat,
            "sin2_theta_chir": s2, "phase": phase,
            "n_seeds": len(seeds_obs),
            "layers": layer_results,
        })

        # quick line per regime: C99.5 layer with theta_B (most physical)
        e = layer_results["C99_5"]
        m1 = e["modes_theta_B"]["M1"]
        m2 = e["modes_theta_B"]["M2"]
        m3 = e["modes_theta_B"]["M3"]
        m4 = e["modes_theta_B"]["M4"]
        rho = e["rho_layer"]["mean"]
        rho_lat = e["rho_lattice"]["mean"]
        zrho = e["rho_z_random"]
        print(f"  {reg:<8s} N={n_lat:>4d} {phase} sin^2t={s2:.3f} "
              f"|C99.5|={e['pooled_size']:>4d}  "
              f"theta_B: M1={m1['amplitude']:.3f}(z={m1['z_random']:+.1f}) "
              f"M2={m2['amplitude']:.3f}(z={m2['z_random']:+.1f}) "
              f"M3={m3['amplitude']:.3f}(z={m3['z_random']:+.1f}) "
              f"M4={m4['amplitude']:.3f}(z={m4['z_random']:+.1f})  "
              f"rho_lay={rho:+.3f} (lat={rho_lat:+.3f}, z={zrho:+.1f})")

    if len(rows) < 3:
        print("Not enough regimes")
        return 0

    # Aggregate per phase
    print()
    print("=" * 110)
    print("Cross-regime aggregate per phase: rho-shift and theta-mode-anchoring")
    print("=" * 110)
    summary = {}
    for phase in ("PRE", "POST"):
        grp = [r for r in rows if r["phase"] == phase]
        summary[phase] = {}
        for ln in layer_defs:
            entry = {
                "rho_layer_mean": float(np.mean([
                    r["layers"][ln]["rho_layer"]["mean"] for r in grp])),
                "rho_lattice_mean": float(np.mean([
                    r["layers"][ln]["rho_lattice"]["mean"] for r in grp])),
                "rho_z_mean": float(np.mean([
                    r["layers"][ln]["rho_z_random"] for r in grp])),
                "rho_z_strong": int(sum(
                    1 for r in grp
                    if abs(r["layers"][ln]["rho_z_random"]) > 3.0)),
                "n_regimes": len(grp),
            }
            for tag in ("theta_A", "theta_B"):
                for mode_n in MODE_ORDERS:
                    mk = f"M{mode_n}"
                    z_arr = [r["layers"][ln][f"modes_{tag}"][mk]["z_random"]
                              for r in grp]
                    a_arr = [r["layers"][ln][f"modes_{tag}"][mk]["amplitude"]
                              for r in grp]
                    n_strong = sum(1 for z in z_arr if z > 3.0)
                    entry[f"{tag}_{mk}_z_mean"] = float(np.mean(z_arr))
                    entry[f"{tag}_{mk}_amp_mean"] = float(np.mean(a_arr))
                    entry[f"{tag}_{mk}_n_z_above3"] = int(n_strong)
            summary[phase][ln] = entry

    # Print rho table
    print(f"\nRho-coherence (cross-regime mean):")
    print(f"{'phase':>6s} {'layer':>8s} {'rho_layer':>10s} {'rho_lattice':>12s} "
          f"{'shift':>8s} {'z_mean':>8s} {'|z|>3':>8s}")
    for phase in ("PRE", "POST"):
        grp_n = summary[phase][next(iter(layer_defs))]["n_regimes"]
        for ln in layer_defs:
            s = summary[phase][ln]
            shift = s["rho_layer_mean"] - s["rho_lattice_mean"]
            print(f"{phase:>6s} {ln:>8s} {s['rho_layer_mean']:>+10.4f} "
                  f"{s['rho_lattice_mean']:>+12.4f} {shift:>+8.4f} "
                  f"{s['rho_z_mean']:>+8.2f} {s['rho_z_strong']:>3d}/{grp_n}")

    # Print theta-B mode table (theta_B is the more physical reference)
    print(f"\nTheta_B mode-amplitudes (relative to neighbourhood phase):")
    print(f"{'phase':>6s} {'layer':>8s}  {'M1 amp':>7} {'M1 z':>6} {'M1 >3':>5}  "
          f"{'M2 amp':>7} {'M2 z':>6} {'M2 >3':>5}  "
          f"{'M3 amp':>7} {'M3 z':>6} {'M3 >3':>5}  "
          f"{'M4 amp':>7} {'M4 z':>6} {'M4 >3':>5}")
    for phase in ("PRE", "POST"):
        grp_n = summary[phase][next(iter(layer_defs))]["n_regimes"]
        for ln in layer_defs:
            s = summary[phase][ln]
            cells = []
            for n in MODE_ORDERS:
                cells.append((s[f"theta_B_M{n}_amp_mean"],
                              s[f"theta_B_M{n}_z_mean"],
                              s[f"theta_B_M{n}_n_z_above3"]))
            print(f"{phase:>6s} {ln:>8s}  "
                  f"{cells[0][0]:>7.3f} {cells[0][1]:>+6.2f} "
                  f"{cells[0][2]:>2d}/{grp_n}  "
                  f"{cells[1][0]:>7.3f} {cells[1][1]:>+6.2f} "
                  f"{cells[1][2]:>2d}/{grp_n}  "
                  f"{cells[2][0]:>7.3f} {cells[2][1]:>+6.2f} "
                  f"{cells[2][2]:>2d}/{grp_n}  "
                  f"{cells[3][0]:>7.3f} {cells[3][1]:>+6.2f} "
                  f"{cells[3][2]:>2d}/{grp_n}")

    # Print theta-A mode table
    print(f"\nTheta_A mode-amplitudes (relative to lattice-mean phase):")
    print(f"{'phase':>6s} {'layer':>8s}  {'M1 amp':>7} {'M1 z':>6} {'M1 >3':>5}  "
          f"{'M2 amp':>7} {'M2 z':>6} {'M2 >3':>5}  "
          f"{'M3 amp':>7} {'M3 z':>6} {'M3 >3':>5}  "
          f"{'M4 amp':>7} {'M4 z':>6} {'M4 >3':>5}")
    for phase in ("PRE", "POST"):
        grp_n = summary[phase][next(iter(layer_defs))]["n_regimes"]
        for ln in layer_defs:
            s = summary[phase][ln]
            cells = []
            for n in MODE_ORDERS:
                cells.append((s[f"theta_A_M{n}_amp_mean"],
                              s[f"theta_A_M{n}_z_mean"],
                              s[f"theta_A_M{n}_n_z_above3"]))
            print(f"{phase:>6s} {ln:>8s}  "
                  f"{cells[0][0]:>7.3f} {cells[0][1]:>+6.2f} "
                  f"{cells[0][2]:>2d}/{grp_n}  "
                  f"{cells[1][0]:>7.3f} {cells[1][1]:>+6.2f} "
                  f"{cells[1][2]:>2d}/{grp_n}  "
                  f"{cells[2][0]:>7.3f} {cells[2][1]:>+6.2f} "
                  f"{cells[2][2]:>2d}/{grp_n}  "
                  f"{cells[3][0]:>7.3f} {cells[3][1]:>+6.2f} "
                  f"{cells[3][2]:>2d}/{grp_n}")

    bundle = {
        "method": "stage6h_layer_modes_v2",
        "schema_version": "2.0.0",
        "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
        "n_flip": N_FLIP,
        "n_bootstrap_random_supports": N_BOOT,
        "ladder_within_p5_only": [r for r, _ in LADDER],
        "definitions": {
            "theta_A": "arg(psi_a) - <arg psi>_lattice",
            "theta_B": "arg(psi_a / nbhd_phase_a),  "
                       "nbhd_phase = arg(<psi_b * xi_ab>_b)",
            "rho_a":   "Re(<exp(-i theta_a) * exp(i theta_b) xi_ab>_b)",
            "M_n":     "|<exp(i 2n theta)>_{a in pooled-layer}|",
        },
        "regimes": rows,
        "pre_post_summary": summary,
    }
    out = REPO / "outputs" / "stage6h_layer_modes_v2.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print()
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
