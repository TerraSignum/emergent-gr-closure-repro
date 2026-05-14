"""Stage 6h: shell-structure test on the Delta-residual core hierarchy.

Tests whether the layer-nested cores
  C^(95) >= C^(99) >= C^(99.5) >= C^(99.9) >= C^(sup)
form a quantitative shell-like hierarchy under five independent
diagnostics, and whether their relative spacings follow a
recognisable pattern (1/n, 1/n^2, exp(-lambda n), etc.).

Diagnostics per layer (cross-regime mean per phase):

  D1 |rho| coherence shift     (anti-coherence magnitude)
  D2 R_00 < 0 fraction          (signed-halo signature)
  D3 |psi|^2 mass fraction      (matter-mass on layer)
  D4 slaving residual           (per-node K-prediction error;
                                  high = unslaved defect node)
  D5 |T_00 - <T_00>| layer mean (matter-density extreme)

Slaving model (G4 reproducer):
  K_pred(a) = beta . feature_set(xi)(a)
where feature_set = (degree, laplacian^2 diag, Fiedler_1..3,
mean_edge_weight). Coefficients beta fit by ridge regression
on the full lattice; per-node residual |K_a - K_pred(a)| then
measures local unslaved-ness.

Shell-index assignment (innermost first):
  n=1: C^(sup)        core spine
  n=2: C^(99.9)       inner mantle
  n=3: C^(99.5)       outer mantle
  n=4: C^(99)         transition shell
  n=5: C^(95)         bulk-edge

For each diagnostic D and each layer n we compute the
cross-regime mean magnitude m_n^D and test whether the m_n
sequence aligns with simple shell models:
  - constant
  - 1/n
  - 1/n^2
  - 1/sqrt(n)
  - linear in n

Within-P5 ladder (8 regimes, P5N128 excluded):
  PRE  = P5(50), P5N64, P5N72, P5N84, P5N100      (5)
  POST = P5N200, P5N300, P5N512                    (3)

Output: outputs/stage6h_layer_shell_structure_test.json
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

# Within-P5 ladder
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
SHELL_INDEX = {"Csup": 1, "C99_9": 2, "C99_5": 3, "C99": 4, "C95": 5}


def chirality_phase(n_lat):
    x = math.log(n_lat / N_STAR) / math.log(D * N_GEN)
    th = math.atan(N_GEN ** (2 * x - 1))
    s2 = math.sin(th) ** 2
    return s2, ("PRE" if th < math.pi / 4 else "POST")


def _xi_features(xi_mat, n_lat):
    """Six per-node features for slaving reconstruction."""
    xi_off = xi_mat.copy()
    np.fill_diagonal(xi_off, 0.0)
    deg = xi_off.sum(axis=1)
    laplacian = np.diag(deg) - xi_off
    lap2 = np.diag(laplacian @ laplacian)
    try:
        eigvals, eigvecs = np.linalg.eigh(laplacian)
        f1 = eigvecs[:, 1] if n_lat > 1 else np.zeros(n_lat)
        f2 = eigvecs[:, 2] if n_lat > 2 else np.zeros(n_lat)
        f3 = eigvecs[:, 3] if n_lat > 3 else np.zeros(n_lat)
    except np.linalg.LinAlgError:
        f1 = np.zeros(n_lat); f2 = np.zeros(n_lat); f3 = np.zeros(n_lat)
    mean_edge = xi_off.mean(axis=1)
    feat = np.column_stack([deg, lap2, f1, f2, f3, mean_edge])
    # Standardise
    mu = feat.mean(axis=0)
    sd = feat.std(axis=0) + 1e-12
    return (feat - mu) / sd


def _slaving_predict(features, target):
    """Ridge regression: target ~ features. Returns predicted values."""
    a_mat = np.column_stack([np.ones(features.shape[0]), features])
    aTa = a_mat.T @ a_mat + RIDGE_LAMBDA * np.eye(a_mat.shape[1])
    aTy = a_mat.T @ target
    beta = np.linalg.solve(aTa, aTy)
    return a_mat @ beta


def _per_seed(xi_mat, psi, k_field, q_field, n_lat):
    """Compute per-node observables: delta, R_time, t00, K_diag, slaving residual."""
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

    # slaving: per-node K(a) target, regress on Xi-features
    if k_field.ndim == 2:
        eye = np.eye(n_lat, dtype=bool)
        k_target = np.where(eye, 0.0, k_field).sum(axis=1) / max(n_lat - 1, 1)
        q_target = np.where(eye, 0.0, q_field).sum(axis=1) / max(n_lat - 1, 1)
    else:
        k_target = np.full(n_lat, float(np.mean(k_field)))
        q_target = np.full(n_lat, float(np.mean(q_field)))

    feats = _xi_features(xi_mat, n_lat)
    k_pred = _slaving_predict(feats, k_target)
    q_pred = _slaving_predict(feats, q_target)
    slaving_res_K = np.abs(k_target - k_pred)
    slaving_res_Q = np.abs(q_target - q_pred)

    # phase coherence rho (xi-weighted real coherence to neighbours)
    xi_no = xi_mat.copy()
    np.fill_diagonal(xi_no, 0.0)
    w_sum = xi_no.sum(axis=1)
    safe = np.maximum(w_sum, 1e-12)
    eiphi = np.exp(1j * np.angle(psi))
    nbhd_eiphi = (xi_no @ eiphi) / safe
    rho = np.real(np.conj(eiphi) * nbhd_eiphi)

    t00_dev = np.abs(t00 - float(np.mean(t00)))

    return {
        "delta": delta, "R_time": R_time, "rho": rho,
        "slaving_K": slaving_res_K, "slaving_Q": slaving_res_Q,
        "psi_sq": psi.real ** 2 + psi.imag ** 2,
        "t00_dev": t00_dev,
    }


def _layer_indices(arr, percentile_or_max):
    if percentile_or_max == "max":
        return {int(np.argmax(arr))}
    thr = float(np.percentile(arr, percentile_or_max))
    return set(int(i) for i in np.nonzero(arr > thr)[0])


def _layer_diag(seeds_obs, layer_name, percentile_or_max):
    """Pool layer over seeds, compute the five diagnostics."""
    d_rho_lay = []
    d_rho_lat = []
    d_R_neg = []
    d_psi_sq_lay = []
    d_psi_sq_total = []
    d_slaving_K_lay = []
    d_slaving_K_lat = []
    d_t00_dev_lay = []
    d_t00_dev_lat = []
    for obs in seeds_obs:
        S = _layer_indices(obs["delta"], percentile_or_max)
        if not S:
            continue
        idx = np.array(sorted(S), dtype=int)
        d_rho_lay.append(float(np.mean(obs["rho"][idx])))
        d_rho_lat.append(float(np.mean(obs["rho"])))
        d_R_neg.append(float(np.mean(obs["R_time"][idx] < 0)))
        d_psi_sq_lay.append(float(np.sum(obs["psi_sq"][idx])))
        d_psi_sq_total.append(float(np.sum(obs["psi_sq"])))
        d_slaving_K_lay.append(float(np.mean(obs["slaving_K"][idx])))
        d_slaving_K_lat.append(float(np.mean(obs["slaving_K"])))
        d_t00_dev_lay.append(float(np.mean(obs["t00_dev"][idx])))
        d_t00_dev_lat.append(float(np.mean(obs["t00_dev"])))
    return {
        "layer_name": layer_name,
        "rho_layer_mean": float(np.mean(d_rho_lay)) if d_rho_lay else float("nan"),
        "rho_lattice_mean": float(np.mean(d_rho_lat)) if d_rho_lat else float("nan"),
        "rho_shift": (float(np.mean(d_rho_lay) - np.mean(d_rho_lat))
                       if d_rho_lay else float("nan")),
        "R_neg_fraction": float(np.mean(d_R_neg)) if d_R_neg else float("nan"),
        "psi_sq_fraction": (
            float(np.mean(np.array(d_psi_sq_lay)
                          / np.maximum(np.array(d_psi_sq_total), 1e-12)))
            if d_psi_sq_lay else float("nan")
        ),
        "slaving_K_layer": (float(np.mean(d_slaving_K_lay))
                              if d_slaving_K_lay else float("nan")),
        "slaving_K_lattice": (float(np.mean(d_slaving_K_lat))
                                if d_slaving_K_lat else float("nan")),
        "slaving_K_ratio": (
            float(np.mean(np.array(d_slaving_K_lay)
                          / np.maximum(np.array(d_slaving_K_lat), 1e-12)))
            if d_slaving_K_lay else float("nan")
        ),
        "t00_dev_layer": (float(np.mean(d_t00_dev_lay))
                            if d_t00_dev_lay else float("nan")),
        "t00_dev_lattice": (float(np.mean(d_t00_dev_lat))
                              if d_t00_dev_lat else float("nan")),
        "t00_dev_ratio": (
            float(np.mean(np.array(d_t00_dev_lay)
                          / np.maximum(np.array(d_t00_dev_lat), 1e-12)))
            if d_t00_dev_lay else float("nan")
        ),
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


def _shell_pattern_fit(values, model):
    """Score how well values match a model g(n) for n=1..len(values).

    Returns (R^2, normalised_amp).
    Normalised = compare unit-vectors (values/sum(values)) vs (model/sum(model)).
    """
    n_max = len(values)
    n_arr = np.arange(1, n_max + 1, dtype=float)
    mod = {
        "1/n": 1.0 / n_arr,
        "1/n^2": 1.0 / n_arr ** 2,
        "1/sqrt(n)": 1.0 / np.sqrt(n_arr),
        "n": n_arr,
        "n^2": n_arr ** 2,
        "constant": np.ones_like(n_arr),
        "exp(-n/2)": np.exp(-n_arr / 2),
    }[model]
    v = np.array(values, dtype=float)
    if np.any(np.isnan(v)) or np.sum(np.abs(v)) < 1e-12:
        return float("nan"), float("nan")
    # Pearson correlation between v and mod
    if v.std() < 1e-12 or mod.std() < 1e-12:
        return float("nan"), float("nan")
    r = float(np.corrcoef(v, mod)[0, 1])
    # Also normalised L1 distance of unit-vectors
    v_n = v / np.sum(np.abs(v))
    m_n = mod / np.sum(np.abs(mod))
    diff = float(np.sum(np.abs(v_n - m_n)) / 2.0)  # in [0, 1]
    return r, 1.0 - diff


def main() -> int:
    print("=" * 110)
    print("Stage 6h: layer shell-structure test (within-P5)")
    print("=" * 110)

    rows = []
    for reg, n_lat in LADDER:
        s2, phase = chirality_phase(n_lat)
        seeds_obs = _gather(reg, n_lat)
        if seeds_obs is None:
            continue
        layers = {}
        for ln, pc in LAYER_DEFS:
            layers[ln] = _layer_diag(seeds_obs, ln, pc)
        rows.append({
            "regime": reg, "N": n_lat, "n_seeds": len(seeds_obs),
            "sin2t": s2, "phase": phase, "layers": layers,
        })
        sup = layers["Csup"]
        c95 = layers["C95"]
        print(f"  {reg:<8s} N={n_lat:>4d} {phase}  "
              f"Csup: rho={sup['rho_shift']:+.3f} slaving={sup['slaving_K_ratio']:.2f}x "
              f"R<0={sup['R_neg_fraction']:.2f}  "
              f"|  C95: rho={c95['rho_shift']:+.3f} slaving={c95['slaving_K_ratio']:.2f}x")

    if len(rows) < 3:
        print("Not enough regimes")
        return 0

    # Aggregate per phase, build shell-pattern-fit per diagnostic
    print()
    print("=" * 110)
    print("Cross-regime aggregate per phase (shell sequence n=1..5):")
    print("  n=1 Csup, n=2 C99.9, n=3 C99.5, n=4 C99, n=5 C95")
    print("=" * 110)
    summary = {}
    for phase in ("PRE", "POST"):
        grp = [r for r in rows if r["phase"] == phase]
        n_grp = len(grp)
        summary[phase] = {"n_regimes": n_grp, "shells": {}}
        # Build value sequences ordered from n=1 (innermost) to n=5
        diag_names = ["|rho_shift|", "R<0_fraction", "psi_sq_fraction",
                       "slaving_K_ratio", "t00_dev_ratio"]
        diag_keys  = ["rho_shift",   "R_neg_fraction", "psi_sq_fraction",
                       "slaving_K_ratio", "t00_dev_ratio"]
        seqs = {dn: [] for dn in diag_names}
        for ln, _ in LAYER_DEFS:
            for dn, dk in zip(diag_names, diag_keys):
                vals = [r["layers"][ln][dk] for r in grp]
                v = float(np.mean(vals))
                if dn == "|rho_shift|":
                    v = abs(v)
                seqs[dn].append(v)
        summary[phase]["sequences"] = seqs

        print(f"\n{phase} (n={n_grp}):")
        print(f"  {'shell n':>8s} {'layer':>7s}  "
              f"{'|rho|':>8s} {'R<0':>6s} {'psi^2_frac':>10s} "
              f"{'slaving_K':>10s} {'T00dev_x':>9s}")
        for k, (ln, _) in enumerate(LAYER_DEFS):
            n = k + 1
            row = [seqs[dn][k] for dn in diag_names]
            print(f"  n={n:>4d}    {ln:>7s}  "
                  f"{row[0]:>8.4f} {row[1]:>6.3f} {row[2]:>10.4f} "
                  f"{row[3]:>10.3f} {row[4]:>9.3f}")

        # Pattern fits per diagnostic
        print(f"\n  Shell-pattern fits (Pearson r, unit-vec match) for {phase}:")
        print(f"  {'diagnostic':>16s}  "
              f"{'1/n':>10s} {'1/n^2':>10s} {'1/sqrt(n)':>12s} "
              f"{'n':>10s} {'constant':>10s}")
        summary[phase]["pattern_fits"] = {}
        for dn in diag_names:
            v = seqs[dn]
            scores = {}
            for model in ["1/n", "1/n^2", "1/sqrt(n)", "n", "constant"]:
                r, m = _shell_pattern_fit(v, model)
                scores[model] = {"pearson_r": r, "unit_match": m}
            summary[phase]["pattern_fits"][dn] = scores
            print(f"  {dn:>16s}  "
                  f"{scores['1/n']['pearson_r']:>+5.2f}/{scores['1/n']['unit_match']:>4.2f}  "
                  f"{scores['1/n^2']['pearson_r']:>+5.2f}/{scores['1/n^2']['unit_match']:>4.2f}  "
                  f"{scores['1/sqrt(n)']['pearson_r']:>+5.2f}/{scores['1/sqrt(n)']['unit_match']:>4.2f}  "
                  f"{scores['n']['pearson_r']:>+5.2f}/{scores['n']['unit_match']:>4.2f}  "
                  f"{scores['constant']['pearson_r']:>+5.2f}/{scores['constant']['unit_match']:>4.2f}")

    # Branching ratios pre vs post for each diagnostic at innermost layers
    print()
    print("=" * 110)
    print("Pre/Post branching ratios on innermost shells (n=1, 2, 3):")
    print("=" * 110)
    for dn in diag_names:
        print(f"  {dn:>16s}: ", end="")
        for k in (0, 1, 2):
            ln_name, _ = LAYER_DEFS[k]
            pre_v = summary["PRE"]["sequences"][dn][k]
            post_v = summary["POST"]["sequences"][dn][k]
            ratio = pre_v / post_v if abs(post_v) > 1e-12 else float("inf")
            print(f"  n={k+1}({ln_name}): PRE={pre_v:+.3f} POST={post_v:+.3f} ratio={ratio:+.2f}", end="")
        print()

    bundle = {
        "method": "stage6h_layer_shell_structure_test",
        "schema_version": "1.0.0",
        "lambda_t": LAMBDA_T, "lambda_s": LAMBDA_S,
        "n_flip": N_FLIP,
        "ladder_within_p5_only": [r for r, _ in LADDER],
        "shell_index": SHELL_INDEX,
        "regimes": rows,
        "phase_summary": summary,
    }
    out = REPO / "outputs" / "stage6h_layer_shell_structure_test.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print()
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
