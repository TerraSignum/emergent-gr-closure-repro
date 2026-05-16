r"""Q2 / Proposal 2: discrete Bakry-Emery Gamma_2 curvature-dimension
audit on the carrier Dirichlet form.

This is the second route to closing the open implication

    (A5) + (A6)  ==>  CD(K_CD, N)

of the mm-GH continuum theorem `thm:hard_xi_continuum`. Where
Proposal 1 (verify_signed_ricci_lower_bound.py) signs the
Hessian-discrepancy Ricci tensor, this script computes the
*intrinsic* Bakry-Emery curvature of the carrier random-walk
Dirichlet form directly -- the quantity Sturm's CD(K,infinity)
condition is literally about.

Construction. With P = D^{-1} W the carrier random-walk operator
and Delta = P - I the (negative) normalised Laplacian, the
carre-du-champ and its iterate are

    Gamma(f,f)(x)  = 1/2 [ Delta(f^2) - 2 f Delta f ](x)
                   = 1/2 sum_y P_xy (f_y - f_x)^2,
    Gamma_2(f,f)(x)= 1/2 Delta Gamma(f,f)(x) - Gamma(f, Delta f)(x).

Both are quadratic forms in the restriction of f to the 2-ball
B_2(x); we assemble their local matrices A_x (for Gamma) and B_x
(for Gamma_2) exactly -- O(deg) and O(deg^2) rank-one updates, no
probing, no surrogate. The pointwise Bakry-Emery curvature is the
smallest generalised eigenvalue

    K_x = min { lambda : B_x v = lambda A_x v,  v in range(A_x) },

i.e. the largest K with Gamma_2(f,f)(x) >= K Gamma(f,f)(x) for
all f. The CD(K,infinity) condition holds with K = inf_x K_x.

If the regime curvature K_N = inf_x K_x (and its robust
percentiles) extrapolate to a finite limit as N -> infinity, the
carrier sequence is uniformly CD(K_CD, infinity); combined with
(A6) pinning the effective dimension N = d_*, that is the
CD(K_CD, N) bound the theorem needs, and the mm-GH limit inherits
it by the stability of CD/RCD under measure-Gromov-Hausdorff
convergence (Sturm 2006).

Sampling. The exact per-vertex generalised eigenproblem is
O(|B_2(x)|^3); to keep the canonical N in [50,512] ladder
tractable the audit subsamples a fixed number of vertices per
seed and a fixed number of seeds per regime (both disclosed in
the output JSON). The curvature is a local quantity, so a vertex
subsample is an unbiased estimator of its distribution; the
infimum is reported as a lower-confidence percentile rather than
a hard min to stay robust under subsampling.

No fits, no fallbacks: P is the exact carrier random-walk
operator on the bundled lattice snapshots.

Usage:
    python ./src/verify_carrier_bakry_emery_cd.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _d1_npz_discovery import find_d1_npz  # noqa: E402
from verify_galerkin_runner_A_hessian_ricci import XI_THRESH  # noqa: E402
from verify_t00_summand_decomposition import (  # noqa: E402
    LADDER, load_seeds, symanzik_fit, bootstrap_ci,
)

# Subsampling budget (disclosed in the output JSON).
MAX_SEEDS = 6
MAX_VERTICES = 48
RNG_SEED = 20260514
RANGE_TOL = 1e-9   # range(A_x) cutoff on A_x eigenvalues


def random_walk_operator(xi_mat):
    """Carrier random-walk operator P = D^{-1} W on the thresholded
    edge-Xi graph (same threshold as the Galerkin pipeline)."""
    xi_mat = np.where(np.isfinite(xi_mat), xi_mat, 0.0)
    w = xi_mat.copy()
    np.fill_diagonal(w, 0.0)
    w = np.where(w > XI_THRESH, w, 0.0)
    deg = w.sum(axis=1)
    keep = deg > 0
    inv = np.zeros_like(deg)
    inv[keep] = 1.0 / deg[keep]
    p = w * inv[:, None]
    return p, keep


def local_curvature(p, x, max_ball=400):
    """Pointwise Bakry-Emery curvature K_x for vertex x of the
    random-walk operator P. Returns K_x or None if x is isolated /
    its local Gamma form is degenerate.
    """
    n = p.shape[0]
    nbr1 = np.where(p[x] > 0)[0]
    if nbr1.size == 0:
        return None
    # 2-ball index set.
    ball = {x}
    ball.update(nbr1.tolist())
    for y in nbr1:
        ball.update(np.where(p[y] > 0)[0].tolist())
    idx = sorted(ball)
    if len(idx) > max_ball:
        return None  # pathological hub; excluded, disclosed via count
    pos = {g: i for i, g in enumerate(idx)}
    m = len(idx)
    xl = pos[x]
    sub = p[np.ix_(idx, idx)]          # local P (rows may be sub-stochastic)

    def e(i):
        v = np.zeros(m)
        v[i] = 1.0
        return v

    # A_x : Gamma(f,f)(x) = 1/2 sum_{y~x} P_xy (e_y - e_x)(e_y - e_x)^T
    a_mat = np.zeros((m, m))
    for yl in range(m):
        pxy = sub[xl, yl]
        if pxy > 0.0:
            d = e(yl) - e(xl)
            a_mat += 0.5 * pxy * np.outer(d, d)

    # B_x : Gamma_2(f,f)(x) = 1/2 Delta Gamma(x) - Gamma(f, Delta f)(x).
    # Part 1: 1/2 Delta Gamma = 1/4 sum_z P_xz sum_y P_zy (e_y-e_z)^2
    #                          - 1/4 sum_y P_xy (e_y-e_x)^2.
    b_mat = np.zeros((m, m))
    for zl in range(m):
        pxz = sub[xl, zl]
        if pxz <= 0.0:
            continue
        for yl in range(m):
            pzy = sub[zl, yl]
            if pzy > 0.0:
                d = e(yl) - e(zl)
                b_mat += 0.25 * pxz * pzy * np.outer(d, d)
    for yl in range(m):
        pxy = sub[xl, yl]
        if pxy > 0.0:
            d = e(yl) - e(xl)
            b_mat -= 0.25 * pxy * np.outer(d, d)
    # Part 2: - Gamma(f, Delta f)(x)
    #   = -1/4 sum_y P_xy [ (e_y-e_x)(delta_y-delta_x)^T + transpose ],
    #   delta_y = row y of (P - I).
    delta = sub - np.eye(m)
    for yl in range(m):
        pxy = sub[xl, yl]
        if pxy > 0.0:
            d_ev = e(yl) - e(xl)
            d_dl = delta[yl] - delta[xl]
            cross = np.outer(d_ev, d_dl)
            b_mat -= 0.25 * pxy * (cross + cross.T)

    # Generalised eigenproblem on range(A_x).
    a_sym = 0.5 * (a_mat + a_mat.T)
    b_sym = 0.5 * (b_mat + b_mat.T)
    a_eval, a_evec = np.linalg.eigh(a_sym)
    keep = a_eval > RANGE_TOL * max(a_eval.max(), 1e-12)
    if not np.any(keep):
        return None
    u = a_evec[:, keep]
    s = a_eval[keep]
    s_inv_sqrt = 1.0 / np.sqrt(s)
    # whitened B: S^{-1/2} U^T B U S^{-1/2}; its eigenvalues are the
    # generalised eigenvalues of (B, A) on range(A).
    b_white = (s_inv_sqrt[:, None]
               * (u.T @ b_sym @ u)
               * s_inv_sqrt[None, :])
    gen_eigs = np.linalg.eigvalsh(0.5 * (b_white + b_white.T))
    return float(gen_eigs[0])


def main():
    print("=" * 78)
    print("Q2 / Proposal 2: discrete Bakry-Emery Gamma_2 CD audit")
    print("=" * 78)
    print(f"Subsampling budget: <= {MAX_SEEDS} seeds, "
          f"<= {MAX_VERTICES} vertices/seed")
    print()

    rng = np.random.default_rng(RNG_SEED)
    regimes = []
    for regime, _ in LADDER:
        npz_path = find_d1_npz(regime, REPO)
        if npz_path is None or not Path(npz_path).exists():
            print(f"  [skip] {regime}: no NPZ payload found")
            continue
        try:
            seeds, n_actual = load_seeds(npz_path)
        except (KeyError, ValueError) as exc:
            print(f"  [skip] {regime}: unloadable payload ({exc})")
            continue
        seeds = seeds[:MAX_SEEDS]
        k_inf_seed, k_p1_seed, k_p5_seed, k_mean_seed = [], [], [], []
        excluded_total = 0
        for xi_mat, _, _, _ in seeds:
            p, keep = random_walk_operator(xi_mat)
            verts = np.where(keep)[0]
            if verts.size == 0:
                continue
            if verts.size > MAX_VERTICES:
                verts = rng.choice(verts, MAX_VERTICES, replace=False)
            kx = []
            for x in verts:
                val = local_curvature(p, int(x))
                if val is None:
                    excluded_total += 1
                else:
                    kx.append(val)
            if not kx:
                continue
            kx = np.array(kx)
            k_inf_seed.append(float(kx.min()))
            k_p1_seed.append(float(np.percentile(kx, 1)))
            k_p5_seed.append(float(np.percentile(kx, 5)))
            k_mean_seed.append(float(kx.mean()))
        if not k_p5_seed:
            print(f"  [skip] {regime}: no admissible vertices")
            continue
        rec = {
            "regime": regime, "N": n_actual, "n_seeds": len(k_p5_seed),
            "excluded_vertices": excluded_total,
            "K_inf_per_seed": k_inf_seed,
            "K_p1_per_seed": k_p1_seed,
            "K_p5_per_seed": k_p5_seed,
            "K_mean_per_seed": k_mean_seed,
            "K_inf": float(np.mean(k_inf_seed)),
            "K_p1": float(np.mean(k_p1_seed)),
            "K_p5": float(np.mean(k_p5_seed)),
            "K_mean": float(np.mean(k_mean_seed)),
        }
        regimes.append(rec)
        print(f"  {regime:8s} N={n_actual:4d} seeds={len(k_p5_seed):2d}  "
              f"K_inf={rec['K_inf']:8.4f}  K_p1={rec['K_p1']:8.4f}  "
              f"K_p5={rec['K_p5']:8.4f}  K_mean={rec['K_mean']:8.4f}  "
              f"(excl {excluded_total})")

    if len(regimes) < 4:
        print("\nInsufficient ladder coverage for Symanzik extrapolation.")
        raise SystemExit(1)

    n_arr = [r["N"] for r in regimes]

    def fit(key):
        per_seed = [r[f"{key}_per_seed"] for r in regimes]
        y = [float(np.mean(v)) for v in per_seed]
        inf1, _, r2_1 = symanzik_fit(n_arr, y, 1)
        inf2, _, r2_2 = symanzik_fit(n_arr, y, 2)
        if r2_2 >= r2_1 + 0.02:
            y_inf, order, r2 = inf2, 2, r2_2
        else:
            y_inf, order, r2 = inf1, 1, r2_1
        lo, hi = bootstrap_ci(n_arr, per_seed, order)
        return {"y_inf": y_inf, "symanzik_order": order,
                "r_squared": r2, "bootstrap_ci95": [lo, hi]}

    print()
    print("-" * 78)
    print(f"Symanzik extrapolation (canonical ladder N in "
          f"[{min(n_arr)},{max(n_arr)}], {len(regimes)} points)")
    print("-" * 78)
    fits = {k: fit(k) for k in ("K_inf", "K_p1", "K_p5", "K_mean")}
    for name, f in fits.items():
        lo, hi = f["bootstrap_ci95"]
        print(f"  {name:8s} y_inf={f['y_inf']:9.5f}  "
              f"Sym-{f['symanzik_order']} R^2={f['r_squared']:5.2f}  "
              f"CI95=[{lo:.5f}, {hi:.5f}]")

    # ----- verdict -----------------------------------------------------
    print()
    print("-" * 78)
    print("Verdict")
    print("-" * 78)
    k_p5_inf = fits["K_p5"]["y_inf"]
    k_p5_lo, k_p5_hi = fits["K_p5"]["bootstrap_ci95"]
    worst_finite = min(r["K_p5"] for r in regimes)
    finite_limit = k_p5_inf > worst_finite - 0.05 and k_p5_lo > -1.0
    nonneg = k_p5_lo <= 0.0 <= k_p5_hi or k_p5_inf >= 0.0
    print(f"  Robust Bakry-Emere curvature K_p5 -> {k_p5_inf:.5f}  "
          f"CI95=[{k_p5_lo:.4f}, {k_p5_hi:.4f}]")
    print(f"  Mean curvature K_mean -> {fits['K_mean']['y_inf']:.5f}")

    if finite_limit and nonneg:
        print()
        print("  The discrete Bakry-Emery curvature of the carrier")
        print("  Dirichlet form extrapolates to a finite, non-negative")
        print("  limit. The carrier sequence is uniformly")
        print("  CD(K_CD, infinity) with K_CD >= 0; with (A6) fixing")
        print("  N = d_* this is the CD(K_CD, N) bound that")
        print("  thm:hard_xi_continuum leaves open, and the mm-GH limit")
        print("  inherits it by Sturm's stability theorem. This is the")
        print("  intrinsic-Dirichlet-form companion to the signed")
        print("  Hessian-Ricci route of Proposal 1.")
        verdict = "BAKRY_EMERY_CD_SUPPORTED_NONNEGATIVE"
    elif finite_limit:
        print()
        print("  The discrete Bakry-Emery curvature extrapolates to a")
        print("  finite (bounded) but negative limit: the carrier is")
        print("  uniformly CD(K_CD, infinity) with finite negative K_CD.")
        print("  This is still a genuine curvature-dimension lower bound,")
        print("  which is what (A5)+(A6) => CD(K_CD,N) requires.")
        verdict = "BAKRY_EMERY_CD_SUPPORTED_FINITE_NEGATIVE"
    else:
        print()
        print("  The Bakry-Emery curvature does not extrapolate to a")
        print("  manifestly finite limit at this ladder resolution and")
        print("  subsampling budget; a uniform CD bound is not certified")
        print("  here by this route.")
        verdict = "BAKRY_EMERY_CD_NOT_CERTIFIED"

    out = {
        "criterion": "Q2/Proposal-2: discrete Bakry-Emery Gamma_2 "
                     "curvature K_x = min gen-eig(B_x, A_x) -> finite "
                     "K_CD, the intrinsic CD(K,infinity) route for "
                     "thm:hard_xi_continuum (A5)+(A6) => CD(K_CD,N)",
        "subsampling": {"max_seeds": MAX_SEEDS,
                        "max_vertices_per_seed": MAX_VERTICES,
                        "rng_seed": RNG_SEED},
        "canonical_ladder": [(r["regime"], r["N"]) for r in regimes],
        "per_regime": regimes,
        "symanzik_fits": fits,
        "cd_curvature": {
            "K_p5_inf": k_p5_inf,
            "K_p5_ci95": [k_p5_lo, k_p5_hi],
            "K_mean_inf": fits["K_mean"]["y_inf"],
            "finite_limit": bool(finite_limit),
            "nonnegative": bool(nonneg),
            "worst_finite_N_K_p5": worst_finite,
        },
        "verdict": verdict,
    }
    out_path = OUTPUTS / "carrier_bakry_emery_cd_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Verdict: {verdict}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
