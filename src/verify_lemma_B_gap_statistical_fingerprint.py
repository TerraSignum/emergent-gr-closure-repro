r"""Lemma B: what determines the (SG) spectral gap?

GPU decomposition study. Question: is the carrier's unweighted
tau=0.10 skeleton spectral gap lambda_skel -> 7/24 a consequence of
the skeleton's graph *statistics* alone, or of deeper carrier-action
correlation structure?

Method.
  Part A -- carrier ground truth. From the bundled P5/P5N snapshots,
    build the tau=0.10 skeleton A_skel = 1[Xi > 0.10], extract its
    statistical profile (degree distribution, clustering, degree
    assortativity, per-node triangle count) and its normalised-
    Laplacian spectral gap lambda_2(L_skel). Symanzik-1 extrapolate
    -> lambda_skel^carrier.
  Part B -- synthetic ensembles, matching progressively more of the
    carrier's statistics, generated at N up to 64k and solved on GPU:
      E0  Erdos-Renyi          -- matches mean degree only
      E1  configuration model  -- matches the full degree distribution
      E2  Newman clustered     -- matches degree + per-node triangle
                                  count (clustering)
      E3  Watts-Strogatz       -- small-world: mean degree + clustering
                                  via the rewiring probability
    For each ensemble: generate (CPU, networkx), simplify + giant
    component, lambda_2(L) = 1 - mu_2(D^-1/2 A D^-1/2) via GPU
    eigsh(k=2, which='LA'). Symanzik-1 -> lambda_inf per ensemble.

Verdict.
  The minimal ensemble whose lambda_inf matches lambda_skel^carrier
  (~7/24) is the statistical fingerprint that determines the gap.
  If E0/E1/E2/E3 all miss, the gap is NOT a function of these
  statistics -> the analytic proof needs the full carrier-action
  correlation structure. Either outcome steers the Lemma-B route.

This is a research computation. Per the governing rule it produces a
JSON in outputs/ and a verdict; it enters a manuscript only on a
clean, decisive result.

Output: outputs/verify_lemma_B_gap_statistical_fingerprint.json
        (checkpointed incrementally)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
OUTPUTS = REPO / "outputs"
OUT = OUTPUTS / "verify_lemma_B_gap_statistical_fingerprint.json"

from _d1_npz_discovery import find_d1_npz  # noqa: E402

TAU = 0.10
RNG_SEED = 12345

# Carrier ladder (ground truth, N <= 512).
CARRIER_LADDER = [
    ("P5", 50), ("P5N64", 64), ("P5N72", 72), ("P5N84", 84),
    ("P5N100", 100), ("P5N128", 128), ("P5N200", 200),
    ("P5N256", 256), ("P5N300", 300), ("P5N512", 512),
]
MAX_CARRIER_SEEDS = 12

# Synthetic ladder + budget (tuned for ~12-16 h on an RTX 5070;
# the 64k/128k sparse eigsh(k=2) solves dominate the wall time).
# Results are checkpointed after every (ensemble, N) batch, so an
# over- or under-shoot of the time budget is harmless.
SYNTH_N = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
SYNTH_SEEDS = 250
ENSEMBLES = ("E0_erdos_renyi", "E1_configuration",
             "E2_newman_clustered", "E3_watts_strogatz",
             "E4_random_regular")

# Phase C: Watts-Strogatz rewiring-probability sweep -- traces the
# whole small-world -> random transition and pinpoints where
# lambda_inf crosses 7/24. (E3 was closest in pilot runs.)
PHASE_C_P = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4,
             0.55, 0.7, 0.85, 1.0]
PHASE_C_N = [8000, 16000, 32000, 64000]
PHASE_C_SEEDS = 100

TARGET_SKEL = 7.0 / 24.0   # 0.291667


# --------------------------------------------------------------------
# GPU spectral gap
# --------------------------------------------------------------------
def _gpu():
    import cupy as cp  # noqa: F401
    from cupyx.scipy.sparse import csr_matrix  # noqa: F401
    from cupyx.scipy.sparse.linalg import eigsh  # noqa: F401
    return cp, csr_matrix, eigsh


def lambda2_normalised_laplacian_gpu(adj_csr):
    """lambda_2(L) = 1 - mu_2 of the normalised adjacency
    D^-1/2 A D^-1/2, computed on the GPU. adj_csr is a scipy CSR
    (host); only the giant component should be passed in."""
    cp, csr_matrix, eigsh = _gpu()
    deg = np.asarray(adj_csr.sum(axis=1)).ravel()
    if np.any(deg <= 0):
        return None
    dis = 1.0 / np.sqrt(deg)
    a_norm = adj_csr.multiply(dis[:, None]).multiply(dis[None, :]).tocsr()
    a_norm = a_norm.astype(np.float64)
    n = a_norm.shape[0]
    try:
        if n <= 3000:
            # dense path -- fast and robust for small N
            ev = cp.linalg.eigvalsh(cp.asarray(a_norm.toarray()))
            mu2 = float(cp.asnumpy(ev)[-2])
        else:
            ag = csr_matrix(a_norm)
            ev = eigsh(ag, k=2, which="LA", return_eigenvectors=False,
                       maxiter=20000, tol=1e-7)
            mu2 = float(np.sort(cp.asnumpy(ev))[-2])
    except Exception as exc:  # noqa: BLE001
        print(f"      [eigsh failed: {exc}]")
        return None
    finally:
        cp.get_default_memory_pool().free_all_blocks()
    return 1.0 - mu2


# --------------------------------------------------------------------
# Carrier snapshot loading (both bundled NPZ layouts)
# --------------------------------------------------------------------
def load_xi_snapshots(regime, n_lat):
    p = find_d1_npz(regime, REPO)
    if p is None or not p.exists():
        return []
    d = np.load(p, allow_pickle=True)
    snaps = []
    if "edge_xi_snapshots" in d.files:
        arr = d["edge_xi_snapshots"]
        if arr.ndim == 4:
            for s in range(min(MAX_CARRIER_SEEDS, arr.shape[0])):
                xi = np.asarray(arr[s, -1], dtype=float)
                if xi.shape == (n_lat, n_lat):
                    snaps.append(xi)
        return snaps
    for s in range(MAX_CARRIER_SEEDS):
        key = f"xi_seed{s}"
        if key not in d.files:
            break
        xi = np.asarray(d[key], dtype=float)
        if xi.shape == (n_lat, n_lat):
            snaps.append(xi)
    return snaps


def skeleton_from_xi(xi):
    w = 0.5 * (xi + xi.T)
    np.fill_diagonal(w, 0.0)
    w = np.maximum(w, 0.0)
    return (w > TAU).astype(np.float64)


def carrier_profile():
    """Part A: carrier ground-truth gap + statistical profile."""
    print("=" * 78)
    print("Part A -- carrier ground truth (tau=0.10 skeleton)")
    print("=" * 78)
    import scipy.sparse as sp
    rows = []
    deg_pool, tri_pool, clus_pool, assort_pool = [], [], [], []
    for regime, n_lat in CARRIER_LADDER:
        snaps = load_xi_snapshots(regime, n_lat)
        if not snaps:
            print(f"  [skip {regime}: no snapshots]")
            continue
        lam2s = []
        for xi in snaps:
            skel = skeleton_from_xi(xi)
            G = nx.from_numpy_array(skel)
            G.remove_edges_from(nx.selfloop_edges(G))
            if G.number_of_edges() == 0:
                continue
            giant = G.subgraph(max(nx.connected_components(G),
                                   key=len)).copy()
            if giant.number_of_nodes() < 4:
                continue
            A = nx.to_scipy_sparse_array(giant, dtype=np.float64,
                                         format="csr")
            lam2 = lambda2_normalised_laplacian_gpu(A)
            if lam2 is None:
                continue
            lam2s.append(lam2)
            degs = np.array([d for _, d in giant.degree()])
            deg_pool.extend(degs.tolist())
            tri = nx.triangles(giant)
            tri_pool.extend(list(tri.values()))
            clus_pool.append(nx.transitivity(giant))
            try:
                assort_pool.append(
                    nx.degree_assortativity_coefficient(giant))
            except Exception:  # noqa: BLE001
                pass
        if lam2s:
            rows.append({"regime": regime, "N": n_lat,
                         "n_seeds": len(lam2s),
                         "lambda2_skel_mean": float(np.mean(lam2s))})
            print(f"  {regime:8s} N={n_lat:4d} n={len(lam2s):2d}  "
                  f"lambda2_skel = {np.mean(lam2s):.4f}")
    a, b, r2 = symanzik1([r["N"] for r in rows],
                         [r["lambda2_skel_mean"] for r in rows])
    profile = {
        "mean_degree": float(np.mean(deg_pool)) if deg_pool else None,
        "degree_cv": (float(np.std(deg_pool) / np.mean(deg_pool))
                      if deg_pool else None),
        "mean_triangles_per_node": (float(np.mean(tri_pool))
                                    if tri_pool else None),
        "transitivity": float(np.mean(clus_pool)) if clus_pool else None,
        "degree_assortativity": (float(np.mean(assort_pool))
                                 if assort_pool else None),
        "degree_pool": deg_pool,
        "triangle_pool": tri_pool,
    }
    print(f"  Symanzik-1: lambda_skel^carrier = {a!s:.6} "
          f"(b={b!s:.5}, R2={r2!s:.4})  vs 7/24={TARGET_SKEL:.5f}")
    print(f"  profile: <d>={profile['mean_degree']!s:.5}  "
          f"CV={profile['degree_cv']!s:.4}  "
          f"transitivity={profile['transitivity']!s:.4}  "
          f"assort={profile['degree_assortativity']!s:.4}")
    return {"per_regime": rows, "symanzik": {"a_inf": a, "b": b, "r2": r2},
            "profile": profile}


# --------------------------------------------------------------------
# Synthetic ensembles
# --------------------------------------------------------------------
def _sample_degrees(deg_pool, n, rng):
    """Resample n degrees from the empirical carrier degree pool;
    enforce even sum (configuration-model requirement)."""
    degs = rng.choice(deg_pool, size=n, replace=True).astype(int)
    degs = np.clip(degs, 1, n - 1)
    if degs.sum() % 2 == 1:
        i = int(rng.integers(n))
        degs[i] = min(degs[i] + 1, n - 1) if degs[i] < n - 1 else degs[i] - 1
    return degs


def _giant_simple_csr(G):
    """Simple-graph giant component as a scipy CSR adjacency."""
    import scipy.sparse  # noqa: F401
    G = nx.Graph(G)                       # collapse multi-edges
    G.remove_edges_from(nx.selfloop_edges(G))
    if G.number_of_nodes() < 4 or G.number_of_edges() == 0:
        return None
    giant = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    if giant.number_of_nodes() < 4:
        return None
    return nx.to_scipy_sparse_array(giant, dtype=np.float64, format="csr")


def make_ensemble(name, n, profile, rng):
    """Generate one synthetic graph of the named ensemble at size n."""
    mean_deg = profile["mean_degree"]
    deg_pool = np.asarray(profile["degree_pool"])
    if name == "E0_erdos_renyi":
        m = int(round(n * mean_deg / 2.0))
        G = nx.gnm_random_graph(n, m, seed=int(rng.integers(2**31)))
    elif name == "E1_configuration":
        degs = _sample_degrees(deg_pool, n, rng)
        G = nx.configuration_model(degs, seed=int(rng.integers(2**31)))
    elif name == "E2_newman_clustered":
        # joint (independent-degree, triangle-degree) sequence resampled
        # from the carrier per-node (degree, triangle-count) pairs.
        tri_pool = np.asarray(profile["triangle_pool"])
        m = min(len(deg_pool), len(tri_pool))
        idx = rng.integers(0, m, size=n)
        degs = np.clip(deg_pool[idx].astype(int), 1, n - 1)
        tris = np.clip(tri_pool[idx].astype(int), 0, None)
        d_indep = np.clip(degs - 2 * tris, 0, None).astype(int)
        d_tri = tris.astype(int)
        # parity fixes: sum(d_tri) % 3 == 0, sum(d_indep) % 2 == 0
        while d_tri.sum() % 3 != 0:
            d_tri[int(rng.integers(n))] += 1
        if d_indep.sum() % 2 == 1:
            d_indep[int(rng.integers(n))] += 1
        joint = list(zip(d_indep.tolist(), d_tri.tolist()))
        G = nx.random_clustered_graph(joint,
                                      seed=int(rng.integers(2**31)))
    elif name == "E3_watts_strogatz":
        k = max(2, int(round(mean_deg)))
        if k % 2 == 1:
            k += 1
        # rewiring probability calibrated so transitivity ~ carrier;
        # WS transitivity ~ (3(k-2)/(4(k-1)))*(1-p)^3, invert for p.
        c_target = profile["transitivity"] or 0.142
        c0 = 3.0 * (k - 2) / (4.0 * (k - 1)) if k > 2 else 0.5
        ratio = np.clip(c_target / c0, 1e-6, 1.0)
        p = float(np.clip(1.0 - ratio ** (1.0 / 3.0), 0.0, 1.0))
        G = nx.watts_strogatz_graph(n, k, p,
                                    seed=int(rng.integers(2**31)))
    elif name == "E4_random_regular":
        d = max(2, int(round(mean_deg)))
        if (n * d) % 2 == 1:
            d += 1
        G = nx.random_regular_graph(d, n, seed=int(rng.integers(2**31)))
    else:
        raise ValueError(name)
    return _giant_simple_csr(G)


def symanzik1(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    msk = np.isfinite(ys)
    if msk.sum() < 3:
        return None, None, None
    inv = 1.0 / xs[msk]
    y = ys[msk]
    A = np.column_stack([np.ones_like(inv), inv])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    ssr = float(((y - pred) ** 2).sum())
    r2 = 1.0 - ssr / sst if sst > 0 else 0.0
    return float(beta[0]), float(beta[1]), r2


def checkpoint(bundle):
    OUT.write_text(json.dumps(bundle, indent=2, default=float),
                   encoding="utf-8")


def main() -> int:
    t_start = time.time()
    rng = np.random.default_rng(RNG_SEED)

    carrier = carrier_profile()
    profile = carrier["profile"]
    if profile["mean_degree"] is None:
        print("ERROR: no carrier profile extracted")
        return 1

    bundle = {
        "method": "verify_lemma_B_gap_statistical_fingerprint",
        "stand": "2026-05-14",
        "question": ("is lambda_skel -> 7/24 determined by the "
                     "tau=0.10 skeleton graph statistics alone?"),
        "tau": TAU,
        "target_skel_7_24": TARGET_SKEL,
        "carrier_ground_truth": {
            "per_regime": carrier["per_regime"],
            "symanzik": carrier["symanzik"],
            "profile": {k: v for k, v in profile.items()
                        if k not in ("degree_pool", "triangle_pool")},
        },
        "synthetic": {},
        "verdict": "INCOMPLETE",
    }
    checkpoint(bundle)

    print()
    print("=" * 78)
    print(f"Part B -- synthetic ensembles (GPU), seeds={SYNTH_SEEDS}, "
          f"N={SYNTH_N}")
    print("=" * 78)
    for name in ENSEMBLES:
        ens_rows = []
        for n in SYNTH_N:
            lam2s = []
            t_n = time.time()
            for s in range(SYNTH_SEEDS):
                try:
                    A = make_ensemble(name, n, profile, rng)
                except Exception as exc:  # noqa: BLE001
                    print(f"    [{name} N={n} seed{s}: gen failed: {exc}]")
                    continue
                if A is None:
                    continue
                lam2 = lambda2_normalised_laplacian_gpu(A)
                if lam2 is not None:
                    lam2s.append(lam2)
            if lam2s:
                ens_rows.append({
                    "N": n, "n_seeds": len(lam2s),
                    "lambda2_mean": float(np.mean(lam2s)),
                    "lambda2_std": float(np.std(lam2s)),
                })
                print(f"  {name:22s} N={n:6d} n={len(lam2s):2d}  "
                      f"lambda2={np.mean(lam2s):.4f}+-{np.std(lam2s):.4f}  "
                      f"({time.time()-t_n:.0f}s, "
                      f"total {(time.time()-t_start)/3600:.1f}h)")
            a, b, r2 = symanzik1([r["N"] for r in ens_rows],
                                 [r["lambda2_mean"] for r in ens_rows])
            bundle["synthetic"][name] = {
                "per_N": ens_rows,
                "symanzik": {"a_inf": a, "b": b, "r2": r2},
            }
            checkpoint(bundle)
        if ens_rows and bundle["synthetic"][name]["symanzik"]["a_inf"]:
            a = bundle["synthetic"][name]["symanzik"]["a_inf"]
            print(f"  -> {name}: lambda_inf = {a:.5f}  "
                  f"(vs 7/24={TARGET_SKEL:.5f}, "
                  f"rel {100*(a-TARGET_SKEL)/TARGET_SKEL:+.1f}%)")

    # ---- Phase C: Watts-Strogatz rewiring-probability sweep -------
    print()
    print("=" * 78)
    print(f"Part C -- Watts-Strogatz rewiring-prob sweep "
          f"(seeds={PHASE_C_SEEDS}, N={PHASE_C_N})")
    print("=" * 78)
    k_ws = max(2, int(round(profile["mean_degree"])))
    if k_ws % 2 == 1:
        k_ws += 1
    bundle["phase_c_ws_sweep"] = {"k": k_ws, "by_p": {}}
    for p_rw in PHASE_C_P:
        rows = []
        t_p = time.time()
        for n in PHASE_C_N:
            lam2s = []
            for s in range(PHASE_C_SEEDS):
                try:
                    G = nx.watts_strogatz_graph(
                        n, k_ws, p_rw, seed=int(rng.integers(2**31)))
                except Exception:  # noqa: BLE001
                    continue
                A = _giant_simple_csr(G)
                if A is None:
                    continue
                lam2 = lambda2_normalised_laplacian_gpu(A)
                if lam2 is not None:
                    lam2s.append(lam2)
            if lam2s:
                rows.append({"N": n, "n_seeds": len(lam2s),
                             "lambda2_mean": float(np.mean(lam2s))})
        a, b, r2 = symanzik1([r["N"] for r in rows],
                             [r["lambda2_mean"] for r in rows])
        bundle["phase_c_ws_sweep"]["by_p"][f"{p_rw:.2f}"] = {
            "p": p_rw, "per_N": rows,
            "symanzik": {"a_inf": a, "b": b, "r2": r2},
        }
        checkpoint(bundle)
        print(f"  WS p={p_rw:.2f}  lambda_inf = {a!s:.6}  "
              f"(vs 7/24={TARGET_SKEL:.5f})  "
              f"({time.time()-t_p:.0f}s, total "
              f"{(time.time()-t_start)/3600:.1f}h)")

    # ---- verdict --------------------------------------------------
    carrier_inf = carrier["symanzik"]["a_inf"]
    matches = []
    for name in ENSEMBLES:
        sym = bundle["synthetic"].get(name, {}).get("symanzik", {})
        a = sym.get("a_inf")
        if a is None or carrier_inf is None:
            continue
        rel_to_carrier = abs(a - carrier_inf) / max(abs(carrier_inf), 1e-9)
        rel_to_724 = abs(a - TARGET_SKEL) / TARGET_SKEL
        if rel_to_carrier < 0.03 and rel_to_724 < 0.05:
            matches.append(name)
    # Phase C: which WS rewiring probability reproduces 7/24?
    c_match = []
    for key, rec in bundle.get("phase_c_ws_sweep", {}).get("by_p", {}).items():
        a = rec.get("symanzik", {}).get("a_inf")
        if a is None or carrier_inf is None:
            continue
        if abs(a - carrier_inf) / max(abs(carrier_inf), 1e-9) < 0.03:
            c_match.append((rec["p"], a))
    bundle["phase_c_matching_p"] = c_match

    if matches:
        verdict = (f"GAP_STATISTICALLY_DETERMINED: minimal matching "
                   f"ensemble(s) {matches} reproduce lambda_skel; the "
                   f"7/24 gap is a function of the skeleton statistics "
                   f"they match.")
    else:
        verdict = ("GAP_NOT_STATISTICALLY_DETERMINED: no ensemble "
                   "E0-E4 reproduces lambda_skel^carrier within 3%; "
                   "the gap depends on carrier-action correlation "
                   "structure beyond degree/clustering/assortativity/"
                   "regularity.")
    if c_match:
        verdict += (f" Phase-C: Watts-Strogatz reproduces it at "
                    f"rewiring p in {[round(p,2) for p,_ in c_match]}.")
    bundle["verdict"] = verdict
    bundle["wall_time_hours"] = (time.time() - t_start) / 3600.0
    checkpoint(bundle)
    print()
    print("=" * 78)
    print(f"VERDICT: {verdict}")
    print(f"wall time: {bundle['wall_time_hours']:.2f} h")
    print(f"Saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
