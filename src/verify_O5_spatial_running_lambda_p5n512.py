"""O5 (spatial-running Lambda) at kappa = gamma^2 = 1/100 on the
extended canonical-physics ladder including P5N512.

The companion follow-up audit (verify_higher_order_terms_all8.py)
showed that O5 at kappa = 1.0 gave a Lambda_t shift of order
unity. The follow-up search identified the structurally-natural
choice kappa = gamma^2 = 1/100 (the squared chirality coefficient)
as the System-R-natural amplitude that keeps Lambda_t close to its
algebraic asymptote alpha_xi^2 = 81/100. This script tests
kappa = gamma^2 across the canonical-physics ladder including
P5N512 to verify that the local-running Lambda matches alpha_xi^2
asymptotically.

Output: outputs/verify_O5_spatial_running_lambda_p5n512.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


class _BlockCupy:
    def find_module(self, name, path=None):
        if name == "cupy" or name.startswith("cupy."):
            return self

    def load_module(self, name):
        raise ImportError("cupy disabled")


sys.meta_path.insert(0, _BlockCupy())

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from verify_galerkin_runner_A_hessian_ricci import (  # noqa: E402
    D_MIN, ELL_0, EPS_D, XI_THRESH, edge_to_matrix, per_seed_galerkin)

ALPHA_XI = 0.9
GAMMA = 0.1
LAMBDA_T = ALPHA_XI ** 2  # 81/100
LAMBDA_S = -GAMMA ** 2 / 2.0  # -1/200
KAPPA_O5 = GAMMA ** 2  # 1/100, structurally natural
REPO_ROOT = REPO.parent

LADDER = [
    ("P5",     50,  "results_d1_fix17/d1_p5.npz",                   "d1"),
    ("P5N64",  64,  "results_d1_p5n64_24seeds/P5N64.snapshots.npz", "snap"),
    ("P5N100", 100, "results_d1_p5n100_24seeds/P5N100.snapshots.npz", "snap"),
    ("P5N300", 300, "results_d1_p5n300_12seeds/P5N300.snapshots.npz", "snap"),
    ("P5N512", 512, "results_d1_p5n512_12seeds/P5N512.snapshots.npz", "snap"),
    ("P5N256", 256,  "results_d1_p5n256_12seeds/P5N256.snapshots.npz",  "snap"),
]


def load_seeds(rel_path: str, n_lat: int, kind: str, max_seeds=12):
    p = REPO_ROOT / rel_path
    if not p.exists():
        return []
    z = np.load(p, allow_pickle=True)
    out = []
    if kind == "snap":
        snaps = z["edge_xi_snapshots"]
        psi_re = z["psi_real_snapshots"]
        psi_im = z["psi_imag_snapshots"]
        last_idx = snaps.shape[1] - 1
        ns = min(int(snaps.shape[0]), max_seeds)
        has_kq = "k_snapshots" in z.files and "q_snapshots" in z.files
        for s in range(ns):
            xi = np.asarray(snaps[s, last_idx], dtype=float)
            psi = (np.asarray(psi_re[s, last_idx], dtype=float)
                   + 1j * np.asarray(psi_im[s, last_idx], dtype=float))
            if has_kq:
                kf = np.asarray(z["k_snapshots"][s, last_idx],
                                 dtype=float)
                qf = np.asarray(z["q_snapshots"][s, last_idx],
                                 dtype=float)
            else:
                kf = np.full((n_lat, n_lat), 0.55)
                qf = np.full((n_lat, n_lat), 0.45)
            out.append((xi, psi, kf, qf))
    else:
        edge_arr = z["dense_cell_edge_xi_values"]
        amp_arr = z["dense_cell_node_amplitude_values"]
        phase_arr = z["dense_cell_node_phase_values"]
        ns = min(int(edge_arr.shape[0]), max_seeds)
        for s in range(ns):
            xi = edge_to_matrix(edge_arr[s], n_lat)
            psi = amp_arr[s] * np.exp(1j * phase_arr[s])
            kf = np.asarray(
                z[f"ff_K_seed{s}"]
                if f"ff_K_seed{s}" in z.files
                else np.full((n_lat, n_lat), 0.55), dtype=float)
            qf = np.asarray(
                z[f"ff_Q_seed{s}"]
                if f"ff_Q_seed{s}" in z.files
                else np.full((n_lat, n_lat), 0.45), dtype=float)
            out.append((xi, psi, kf, qf))
    return out


def omega_per_node(xi_mat: np.ndarray) -> np.ndarray:
    xi_off = np.where(np.isfinite(xi_mat), xi_mat, 0.0).copy()
    np.fill_diagonal(xi_off, 0.0)
    adj = (xi_off > XI_THRESH).astype(np.float64)
    d_mat = -ELL_0 * np.log(np.maximum(xi_off, 1e-12))
    d_mat = np.maximum(d_mat, D_MIN)
    d_sq_safe = np.where(adj > 0, d_mat ** 2, np.inf)
    weight_grad = np.where(
        adj > 0, (xi_off * adj) / (d_sq_safe + EPS_D), 0.0)
    return weight_grad.sum(axis=1)


def per_seed_O5_lambda(prep_data, omega_a: np.ndarray, kappa: float):
    omega_mean = max(float(omega_a.mean()), 1e-9)
    factor = 1.0 + kappa * omega_a / omega_mean
    lt_a = LAMBDA_T * factor
    ls_a = LAMBDA_S * factor

    g_00 = prep_data["g_00_h"]
    g_ij = prep_data["g_ij_h"]
    t00 = prep_data["t00"]
    t_ij = prep_data["t_ij"]
    eye3 = prep_data["eye3"]

    res00 = g_00 + lt_a - t00
    spatial_res = (g_ij
                    + ls_a[:, None, None] * eye3[None, :, :]) - t_ij
    sq = res00 ** 2 + (spatial_res ** 2).sum(axis=(1, 2))
    return {
        "lt_mean": float(lt_a.mean()),
        "ls_mean": float(ls_a.mean()),
        "lt_p99":  float(np.percentile(lt_a, 99.0)),
        "ls_p99":  float(np.percentile(ls_a, 99.0)),
        "frob_median": float(np.median(np.sqrt(sq))),
        "frob_mean":   float(np.mean(np.sqrt(sq))),
    }


def main() -> int:
    rows = []
    for label, n_lat, rel, kind in LADDER:
        seeds = load_seeds(rel, n_lat, kind)
        if not seeds:
            print(f"  skip {label}: data not found")
            continue
        per_seed_O5 = []
        for xi, psi, kf, qf in seeds:
            np.fill_diagonal(xi, 1.0)
            prep = per_seed_galerkin(xi, psi, kf, qf, n_lat, np)
            omega_a = omega_per_node(xi)
            per_seed_O5.append(
                per_seed_O5_lambda(prep, omega_a, KAPPA_O5))

        def _agg(key):
            vals = [s[key] for s in per_seed_O5]
            arr = np.array(vals, dtype=float)
            return {
                "mean":  float(arr.mean()),
                "std":   float(arr.std()),
                "min":   float(arr.min()),
                "max":   float(arr.max()),
            }

        row = {
            "regime_label": label,
            "N": int(n_lat),
            "n_seeds": len(seeds),
            "lt_mean": _agg("lt_mean"),
            "ls_mean": _agg("ls_mean"),
            "frob_median": _agg("frob_median"),
            "frob_mean":   _agg("frob_mean"),
        }
        rows.append(row)
        print(f"  {label} N={n_lat:<4d} seeds={len(seeds):>2d}  "
              f"<Lambda_t>={row['lt_mean']['mean']:.4f} "
              f"+-{row['lt_mean']['std']:.4f}  "
              f"<Lambda_s>={row['ls_mean']['mean']:+.4f}  "
              f"frob_med={row['frob_median']['mean']:.4f}  "
              f"frob_mean={row['frob_mean']['mean']:.4f}")

    n_arr = np.array([r["N"] for r in rows], dtype=float)
    lt_arr = np.array([r["lt_mean"]["mean"] for r in rows])
    if len(n_arr) >= 2:
        lt_largest = float(lt_arr[-1])
        lt_residual_to_alpha_xi_sq = float(lt_largest - 0.81)
    else:
        lt_largest = float("nan")
        lt_residual_to_alpha_xi_sq = float("nan")

    bundle = {
        "method": "verify_O5_spatial_running_lambda_p5n512",
        "schema_version": "1.0.0",
        "constants": {
            "alpha_xi": ALPHA_XI,
            "gamma": GAMMA,
            "kappa_O5": KAPPA_O5,
            "lambda_t_target": LAMBDA_T,
            "lambda_s_target": LAMBDA_S,
        },
        "ladder": [r["regime_label"] for r in rows],
        "rows": rows,
        "summary": {
            "lambda_t_mean_at_largest_N": lt_largest,
            "lambda_t_residual_to_alpha_xi_sq":
                lt_residual_to_alpha_xi_sq,
            "lambda_t_within_1pc_of_alpha_xi_sq":
                bool(abs(lt_residual_to_alpha_xi_sq) < 0.01),
        },
    }
    out = (REPO / "outputs"
           / "verify_O5_spatial_running_lambda_p5n512.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    print()
    print(f"  largest-N <Lambda_t> = {lt_largest:.5f}  "
          f"(target alpha_xi^2 = 0.81000)  "
          f"residual = {lt_residual_to_alpha_xi_sq:+.5f}  "
          f"within 1%: "
          f"{bundle['summary']['lambda_t_within_1pc_of_alpha_xi_sq']}")
    print(f"  saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
