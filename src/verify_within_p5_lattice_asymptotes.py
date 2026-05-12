"""
verify_within_p5_lattice_asymptotes.py

Direct multi-N evaluation of five lattice-side observables on the
within-canonical-regime P5N ladder (N in {64, 72, 84, 100, 128, 200, 256, 300},
130 seeds total), bypassing the d1-aggregator scalar summary.

Reads NPZ snapshot files of the lattice run output and computes per-N:
  - <Var(Xi)>           (edge correlation variance)
  - <edge_xi> * N       (edge correlation density times lattice size)
  - <K>                 (factor field K mean)
  - <Q>                 (factor field Q mean)
  - <|psi|^2>           (scalar field amplitude squared)

Then performs Symanzik 1/N and 1/N^2 fits, reports the asymptote and rmse.
The output JSON is consumed by the P4 manuscript section
"Within-canonical-regime P5N validation of asymptotic rationals".

The script expects the snapshot NPZ files to live in
  $WORKSPACE/results_d1_p5n*_*seeds/P5N*.snapshots.npz
relative to the top-level Emergence directory. If running outside that
layout, set the environment variable EMERGENCE_RESULTS_ROOT to override.

Usage:
  python src/verify_within_p5_lattice_asymptotes.py
Outputs:
  data/within_p5_lattice_asymptotes.json
"""
import sys, io, os, glob, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np

ROOT_DEFAULT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
ROOT = os.environ.get('EMERGENCE_RESULTS_ROOT', ROOT_DEFAULT)

P5N_DIRS = {
    64:  ['results_d1_p5n64_24seeds',  'results_d1_p5n64'],
    72:  ['results_d1_p5n72_24seeds',  'results_d1_p5n72'],
    84:  ['results_d1_p5n84_24seeds',  'results_d1_p5n84'],
    100: ['results_d1_p5n100_24seeds', 'results_d1_p5n100'],
    128: ['results_d1_p5n128_kq_fixed', 'results_d1_p5n128'],
    200: ['results_d1_p5n200_12seeds', 'results_d1_p5n200_8seeds', 'results_d1_p5n200'],
    256: ['results_d1_p5n256_12seeds', 'results_d1_p5n256_8seeds', 'results_d1_p5n256'],
    300: ['results_d1_p5n300_12seeds', 'results_d1_p5n300'],
    512: ['results_d1_p5n512_12seeds'],
}

def find_npz(N):
    for d in P5N_DIRS[N]:
        cand = glob.glob(os.path.join(ROOT, d, 'P5N*.snapshots.npz'))
        if cand:
            return cand[0]
    return None

results = {}
print('=== P5N within-canonical-regime lattice asymptotes ===')
print('ROOT = ' + ROOT)
print(f'{"N":<5} {"seeds":<7} {"Var(Xi)":<14} {"<edge>*N":<12} {"<K>":<10} {"<Q>":<10} {"<|psi|^2>":<12}')
print('-' * 80)

for N in sorted(P5N_DIRS.keys()):
    path = find_npz(N)
    if path is None:
        print(f'N={N}: MISSING')
        continue
    d = np.load(path, allow_pickle=True)
    edge = d['edge_xi_snapshots'][:, -1, :, :]  # (S, N, N) - final snapshot
    S = edge.shape[0]
    edge_mean = float(np.mean(edge))
    var_xi_per_seed = np.var(edge.reshape(S, -1), axis=1)
    var_xi_mean = float(np.mean(var_xi_per_seed))
    var_xi_std = float(np.std(var_xi_per_seed))

    if 'k_snapshots' in d.files:
        k_snap = d['k_snapshots'][:, -1, :, :]
        k_mean = float(np.mean(k_snap))
    else:
        k_mean = None
    if 'q_snapshots' in d.files:
        q_mean = float(np.mean(d['q_snapshots'][:, -1, :, :]))
    else:
        q_mean = None

    psi_r = d.get('psi_real_snapshots')
    psi_i = d.get('psi_imag_snapshots')
    if psi_r is not None and psi_i is not None:
        psi_sq = float(np.mean(psi_r[:, -1, :]**2 + psi_i[:, -1, :]**2))
    else:
        psi_sq = None

    n_seeds = int(d['n_seeds'][0]) if 'n_seeds' in d.files else S
    edge_times_N = edge_mean * N

    results[N] = {
        'n_seeds': n_seeds,
        'var_xi_mean': var_xi_mean,
        'var_xi_std': var_xi_std,
        'edge_xi_mean': edge_mean,
        'edge_xi_times_N': edge_times_N,
        'K_mean': k_mean,
        'Q_mean': q_mean,
        'psi_sq_mean': psi_sq,
    }
    print(f'{N:<5} {n_seeds:<7} {var_xi_mean:<14.6f} {edge_times_N:<12.4f} '
          f'{(k_mean if k_mean is not None else 0):<10.4f} '
          f'{(q_mean if q_mean is not None else 0):<10.4f} '
          f'{(psi_sq if psi_sq is not None else 0):<12.6f}')

# Symanzik fits
print()
print('=== Symanzik 1/N and 1/N^2 fits ===')
Ns = sorted(results.keys())
N_arr = np.array(Ns, dtype=float)
fit_results = {}

for label, key in [('Var(Xi)','var_xi_mean'),
                   ('<edge_xi>*N','edge_xi_times_N'),
                   ('<K>','K_mean'),
                   ('<Q>','Q_mean'),
                   ('<|psi|^2>','psi_sq_mean')]:
    arr = np.array([results[N][key] for N in Ns])
    A1 = np.column_stack([np.ones_like(N_arr), 1.0/N_arr])
    c1, *_ = np.linalg.lstsq(A1, arr, rcond=None)
    a1, b1 = c1
    rmse1 = float(np.sqrt(np.mean((arr - A1 @ c1)**2)))
    A2 = np.column_stack([np.ones_like(N_arr), 1.0/N_arr**2])
    c2, *_ = np.linalg.lstsq(A2, arr, rcond=None)
    a2, b2 = c2
    rmse2 = float(np.sqrt(np.mean((arr - A2 @ c2)**2)))
    fit_results[label] = {
        'one_over_N':  {'a_inf': float(a1), 'b': float(b1), 'rmse': rmse1},
        'one_over_N2': {'a_inf': float(a2), 'b': float(b2), 'rmse': rmse2},
    }
    print(f'{label:<14}  1/N: a_inf={a1:+.6f}, b={b1:+.4f}, rmse={rmse1:.2e}  |  '
          f'1/N^2: a_inf={a2:+.6f}, b={b2:+.4f}, rmse={rmse2:.2e}')

# Rational identity table
print()
print('=== Rational identity table ===')
identity_targets = {
    'Var(Xi)':     {'val': 0.0,        'name': '0 (classical limit)'},
    '<edge_xi>*N': {'val': 4.5,        'name': '9/2'},
    '<K>':         {'val': 4.0/3.0,    'name': '4/3 (SU(3) Casimir C_F)'},
    '<Q>':         {'val': 0.25,       'name': '1/4 (BH = 1/d_spacetime)'},
    '<|psi|^2>':   {'val': 0.2,        'name': '1/5 (= 4*eps_sync2 = 2*gamma)'},
}
print(f'{"observable":<14} {"a_inf 1/N":<14} {"target":<35} {"rel diff":<12}')
for label in fit_results:
    a_inf = fit_results[label]['one_over_N']['a_inf']
    tgt = identity_targets[label]
    rel = (a_inf - tgt['val']) / tgt['val'] * 100 if tgt['val'] != 0 else None
    rel_str = f'{rel:+.3f}%' if rel is not None else 'trivial'
    print(f'{label:<14} {a_inf:<14.6f} {tgt["name"]:<35} {rel_str:<12}')

# Save bundle
out = {
    'bundle': 'within_p5_lattice_asymptotes',
    'date': '2026-05-04',
    'description': 'Five lattice-side observables on the within-P5 ladder, '
                   'extracted directly from snapshot NPZ files, with Symanzik '
                   '1/N and 1/N^2 extrapolations against framework-rational targets.',
    'N_sequence': Ns,
    'per_N': {str(N): results[N] for N in Ns},
    'symanzik_fits': fit_results,
    'rational_targets': identity_targets,
}
out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data'))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'within_p5_lattice_asymptotes.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f'\nSaved: {out_path}')
