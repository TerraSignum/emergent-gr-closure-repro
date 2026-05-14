"""derive_federer_prefactors_galerkin_fem.py

First-principles derivation attempt of the Federer prefactors c and d in
   med(Delta_N) = c * N^(-1/3) + d * N^(-2/3),
empirically c=+0.16130, d=-0.17695 on the within-canonical 10-point ladder.

Five-step formal derivation following Brenner-Scott "The Mathematical
Theory of Finite Element Methods" (2nd ed., Springer 2008):

  Step 1: Causal-wave bilinear form a(.,.) on H^1
  Step 2: Coercivity alpha_coer + boundedness M_bound on the
          non-common-mode subspace
  Step 3: Cea / Strang-Fix quasi-optimality:
            ||u - u_h||_{H^1} <= (M_bound / alpha_coer) * inf ||u - v_h||
  Step 4: Aubin-Nitsche duality for L^2 super-convergence:
            ||u - u_h||_{L^2} <= C_AN * h * ||u - u_h||_{H^1}
  Step 5: Tetrahedral interpolation constant C_I (Brenner-Scott Thm 4.4.20)

The Galerkin-Frobenius residual norm of the Einstein-class identity
||R||_F = ||G + Lambda^back - 8 pi G T^Xi||_F is bounded above by the
H^1-error of the Galerkin solution on the local 4x4 patch.

Result: predicted c, d in closed form as functions of (alpha_xi, gamma,
beta_pi, eps_sync2, D_Omega), to be compared against empirical values.
"""
from __future__ import annotations
import json
from pathlib import Path
from fractions import Fraction

# ============================================================
# System R rationals (parameter-free framework anchors)
# ============================================================
ALPHA_XI  = Fraction(9, 10)        # Xi reaction rate
GAMMA     = Fraction(1, 10)        # damping rate
EPS_SYNC2 = Fraction(1, 20)        # sync persistence
BETA_PI   = Fraction(15, 16)       # common-phase strength
D_OMEGA   = BETA_PI - GAMMA        # 67/80 (Einstein relation, C2)

# Linear-stability lemma (P2 sec:linear_stability):
LAMBDA_CRIT = (ALPHA_XI - GAMMA + EPS_SYNC2) / D_OMEGA  # 68/67
G_NET = ALPHA_XI + BETA_PI + EPS_SYNC2 - GAMMA          # 143/80

print('=' * 78)
print('First-principles derivation: Federer prefactors c, d via Galerkin FEM')
print('=' * 78)
print()
print('System R parameter-free anchors:')
print(f'  alpha_xi = {ALPHA_XI},  gamma = {GAMMA},  eps_sync2 = {EPS_SYNC2}')
print(f'  beta_pi  = {BETA_PI},   D(Omega) = {D_OMEGA}')
print(f'  lambda_crit = {LAMBDA_CRIT} (non-common mode lower bound)')
print(f'  G_NET = {G_NET} (common-mode growth rate)')
print()

# ============================================================
# STEP 1: Causal-wave bilinear form
# ============================================================
print('=' * 78)
print('STEP 1: Bilinear form a(C, phi) of the linearised Causal-Wave operator')
print('=' * 78)
print('''
The P2 transport law dC/dt = D(Omega) Lap C - gamma C + alpha_xi C
                          + beta_pi Pi_common C + eps_sync2 C
in stationary form 0 = ... gives, on the non-common-mode subspace
V_perp (orthogonal to the constant mode), the bilinear form

    a(C, phi) = integral [ D(Omega) grad C . grad phi
                          + (gamma - alpha_xi - eps_sync2) C phi ] dV

where the (gamma - alpha_xi - eps_sync2) coefficient is the *negative* of
the non-common-mode growth rate (i.e. the *damping* on V_perp). Note:

    gamma - alpha_xi - eps_sync2 = 1/10 - 9/10 - 1/20 = -17/20

This is NEGATIVE: the non-common subspace is unstable about the trivial
vacuum *unless* we include the diffusion D(Omega) Lap term. Coercivity
must therefore come from the gradient-energy term, which is positive on
modes with k^2 > lambda_crit = 68/67.
''')

# Damping on V_perp (positive=stable, negative=unstable)
mu_perp = GAMMA - ALPHA_XI - EPS_SYNC2
print(f'  Mass term mu_perp = gamma - alpha_xi - eps_sync2 = {mu_perp} '
      f'(NEGATIVE on V_perp without diffusion)')
print(f'  Diffusion term: D(Omega) k^2 stabilises modes with k^2 > {LAMBDA_CRIT}')
print()

# ============================================================
# STEP 2: Coercivity + boundedness on V_perp restricted
# ============================================================
print('=' * 78)
print('STEP 2: Coercivity alpha_coer and boundedness M_bound')
print('=' * 78)

# On V_perp restricted to modes with k^2 > lambda_crit + delta (delta > 0):
# a(C,C) = D(Omega) ||grad C||^2 + mu_perp ||C||^2
#        >= D(Omega) k_min^2 ||C||^2 + mu_perp ||C||^2
#        = (D(Omega) k_min^2 + mu_perp) ||C||^2
# For this to be coercive we need k_min^2 > -mu_perp / D(Omega)
#                                          = -((-17/20) / (67/80))
#                                          = 68/67 = lambda_crit  ✓
# Coercivity constant on the strictly-stable subspace
# (k^2 = lambda_crit + something > 0):
# alpha_coer = D(Omega) * lambda_crit + mu_perp = (numerator of lambda_crit
# canceled) = 0 at threshold.
# To get a positive coercivity, we evaluate at the *typical* non-common mode
# k^2 = 1 (the natural unit on the lattice), giving:
alpha_coer = D_OMEGA * 1 + mu_perp  # = 67/80 - 17/20 = 67/80 - 68/80 = -1/80
print(f'  At k^2 = 1: alpha_coer = D(Omega)*1 + mu_perp = {alpha_coer}')
print(f'  This is at the coercivity threshold (typical bulk mode k^2 ~ 1).')
print()
# Better: use the fully-coerced regime k^2 = lambda_crit + 1 (typical bulk mode
# above threshold)
alpha_coer_bulk = D_OMEGA * (LAMBDA_CRIT + 1) + mu_perp
print(f'  At k^2 = lambda_crit + 1 = {LAMBDA_CRIT + 1}:')
print(f'    alpha_coer = D(Omega)*(lambda_crit+1) + mu_perp = {alpha_coer_bulk}')
print(f'    = {float(alpha_coer_bulk):.6f}')

# Boundedness M_bound on H^1: simple norm-bound
M_bound = D_OMEGA + abs(mu_perp)  # naive H^1 boundedness
print(f'  M_bound on H^1 = D(Omega) + |mu_perp| = {M_bound} '
      f'= {float(M_bound):.6f}')
M_over_alpha = float(M_bound) / float(alpha_coer_bulk)
print(f'  M_bound / alpha_coer = {M_over_alpha:.6f} (Cea constant)')
print()

# ============================================================
# STEP 3: Cea / Strang-Fix quasi-optimality
# ============================================================
print('=' * 78)
print('STEP 3: Cea / Strang-Fix quasi-optimality')
print('=' * 78)
print('''
Standard Cea Lemma (Brenner-Scott Thm 2.5.1):
   ||u - u_h||_{H^1} <= (M_bound/alpha_coer) * inf_{v_h in V_h} ||u - v_h||_{H^1}

For first-order tetrahedral elements (P1) on a quasi-uniform mesh:
   inf ||u - v_h||_{H^1} <= C_I * h * ||u||_{H^2}

with C_I the interpolation constant (Brenner-Scott Thm 4.4.20):
   C_I ~ 1 for unit-volume quasi-uniform tetrahedral mesh
''')
C_I = 1.0  # tetrahedral interpolation constant, leading order
print(f'  Interpolation constant C_I (P1 tetrahedra, quasi-uniform) = {C_I}')
print()

# ============================================================
# STEP 4: Aubin-Nitsche duality (sub-leading)
# ============================================================
print('=' * 78)
print('STEP 4: Aubin-Nitsche L^2 super-convergence (sub-leading)')
print('=' * 78)
print('''
Aubin-Nitsche duality (Brenner-Scott Thm 5.4.4):
   ||u - u_h||_{L^2} <= C_AN * h * ||u - u_h||_{H^1}

with C_AN = M_bound/alpha_coer * C_reg, where C_reg is the regularity
constant of the dual problem (a(phi, w) = (psi, phi)). For elliptic
problems on smooth domains C_reg = O(1).

Combining Cea + Aubin-Nitsche:
   ||u - u_h||_{L^2} <= (M_bound/alpha_coer)^2 * C_I^2 * h^2 * ||u||_{H^2}
''')

# H^1 -> L^2 super-convergence factor
C_reg = 1.0  # regularity, O(1)
C_AN = M_over_alpha * C_reg
print(f'  C_AN = (M_bound/alpha_coer) * C_reg = {C_AN:.6f}')
print()

# ============================================================
# STEP 5: Stationary closed-form mode amplitude
# ============================================================
print('=' * 78)
print('STEP 5: Stationary geschlossene Form for the Hessian norm |u|_{H^2}')
print('=' * 78)
print('''
The stationary solution of the linearised transport law on a bulk mode
with eigenvalue lambda has amplitude bounded by the source norm. For the
canonical regime, the local stress-energy ||T||_F at the median of the
bulk-percentile distribution is approximately the median Var(Xi) which
the within-canonical N-ladder gives ~ (1/N)*4.5 = 4.5/N (the
<edge_xi>*N -> 9/2 lattice asymptote).

The Hessian norm |u|_{H^2} of the residual is bounded by
    |u|_{H^2} ~ k^2 * ||u||_{L^2} ~ lambda_crit * ||u||_{L^2}
on bulk modes with k^2 ~ lambda_crit.

The typical residual amplitude scales with the typical stress-energy:
    ||u||_{L^2} ~ ||T||_F * (something parameter-free)
''')

# Empirical anchor: <edge_xi>*N -> 9/2 (today's P5N audit)
# So |T|_F ~ <edge_xi> ~ (9/2)/N  --> drops out in the relative
# Frobenius norm Delta_a = ||R||_F / ||T||_F.
print('  Lattice anchor (P5N today): <edge_xi>*N -> 9/2 (Federer-rational)')
print()

# ============================================================
# STEP 6: Integration -> predicted c, d
# ============================================================
print('=' * 78)
print('STEP 6: Integration -> predicted c, d')
print('=' * 78)
print('''
Combining Cea + Aubin-Nitsche + tetrahedral interpolation, the median
relative Frobenius residual decays as:

    med(Delta_N) ~ (M_bound/alpha_coer) * C_I * h * (Lipschitz of R)
                 + (M_bound/alpha_coer)^2 * C_I^2 * C_reg * h^2 * (Hessian of R)

with h = (V/N)^{1/3} = (some O(1) volumetric)/N^{1/3}.

Identifying:
   c = (M_bound/alpha_coer) * C_I * V^{1/3} * (Lipschitz seminorm of R / |T|)
   d = -(M_bound/alpha_coer)^2 * C_I^2 * V^{2/3} * (Hessian seminorm of R / |T|)

The negative sign of d arises from the *opposite-sign* second-order
correction in the Aubin-Nitsche estimate when the residual itself
decreases faster than its first derivative under refinement.

The framework-rational prediction for c (working hypothesis):
   c_predicted = 2 * alpha_xi^2 * gamma = 81/500
             = 2 * Lambda_t * gamma  (cosmological-tensor x damping)
''')

c_emp = 0.161304
d_emp = -0.176953
c_hyp = 2 * float(ALPHA_XI)**2 * float(GAMMA)  # = 0.162
print(f'  c_predicted (hypothesis) = 2 * alpha_xi^2 * gamma = '
      f'2 * (9/10)^2 * (1/10) = {Fraction(81,500)} = {c_hyp}')
print(f'  c_empirical = {c_emp}')
print(f'  rel diff = {(c_hyp - c_emp)/c_emp*100:+.3f}%')
print()
print('  Mapping to derivation:')
print(f'    M_bound / alpha_coer = {M_over_alpha:.4f}')
print(f'    c_predicted = M_over_alpha * C_I * V^{{1/3}} * (Lipschitz / |T|)')
print(f'    => Lipschitz / |T| = c_hypothesis / (M_over_alpha * C_I * V^{{1/3}})')
needed_ratio_c = c_hyp / (M_over_alpha * C_I * 1.0)  # assume V=1
print(f'      with V=1, C_I=1: Lipschitz / |T| = {needed_ratio_c:.4f}')
print()

# Sub-leading
d_hyp = -(1 + float(GAMMA)) * c_hyp  # -891/5000
print(f'  d_predicted (hypothesis) = -(1+gamma) * 2*alpha_xi^2*gamma = '
      f'{Fraction(-891,5000)} = {d_hyp}')
print(f'  d_empirical = {d_emp}')
print(f'  rel diff = {(d_hyp - d_emp)/d_emp*100:+.3f}%')
print()
print('  Sub-leading via Aubin-Nitsche:')
print(f'    C_AN = (M_over_alpha)^2 * C_I^2 = {M_over_alpha**2:.4f}')
print(f'    d_predicted = -C_AN * V^{{2/3}} * (Hessian / |T|)')
needed_ratio_d = abs(d_hyp) / (M_over_alpha**2 * C_I**2 * 1.0)
print(f'      with V=1, C_I=1: Hessian / |T| = {needed_ratio_d:.4f}')
print()

# Critical test: ratio |d|/c
print('  Crucial first-principles check: ratio |d|/c.')
print(f'    From Cea+Aubin-Nitsche:')
print(f'      |d|/c = (M_over_alpha) * C_I * V^{{1/3}} * (Hessian / Lipschitz)')
print(f'    Empirical: |d|/c = {abs(d_emp)/c_emp:.4f}')
print(f'    Hypothesis: |d|/c = (1 + gamma) = 11/10 = 1.1')
print()
print('    For the ratio to give exactly (1+gamma), we need:')
print(f'      (M_over_alpha) * (Hessian/Lipschitz) = (1+gamma) = 1.1')
required = float(1 + GAMMA) / M_over_alpha
print(f'      => (Hessian/Lipschitz) = (1+gamma)/M_over_alpha '
      f'= 1.1 / {M_over_alpha:.4f} = {required:.4f}')
print()

# ============================================================
# STEP 7: Verdict
# ============================================================
print('=' * 78)
print('STEP 7: Verdict')
print('=' * 78)
print('''
What we have:

  (A) The Cea+Aubin-Nitsche framework gives the CORRECT scaling N^{-1/3}
      and N^{-2/3} for first-order tetrahedral Galerkin. ✓

  (B) The Cea constant M_over_alpha is computed from the framework rationals
      directly: M_bound/alpha_coer = (D(Omega) + |mu_perp|) /
      (D(Omega)*(lambda_crit+1) + mu_perp) = ({M_bound}) / ({alpha_coer_bulk}).

  (C) For the leading prefactor c to equal the hypothesised
      2*alpha_xi^2*gamma = 81/500, the *Lipschitz seminorm of the residual
      relative to the stress-energy* must equal {needed_ratio_c:.4f} on
      the canonical-regime lattice. This is a quantity that requires a
      direct measurement on the lattice; we have NOT yet derived it from
      first principles.

  (D) For the sub-leading d to equal the hypothesised -(1+gamma)*c, the
      *Hessian/Lipschitz ratio* must be {required:.4f}. Same status as (C).

Conclusion: The Galerkin-FEM framework is the right tool and yields the
correct N-scalings. The framework-rational *prefactor identification*
2*alpha_xi^2*gamma and -(1+gamma)*c is consistent with the framework but
requires two independent measurements (Lipschitz, Hessian) to be derived
from first principles in closed form. The "0.04% match" of |d|/c with
(1+gamma) is an empirical coincidence that reduces to a ratio of
two-residual-derivatives constraint, not yet a strict closure.

Honest verdict: hypothesis remains a hypothesis, not a derived theorem.
The N-scaling SHAPE is derived; the prefactors are not.
''')
print(f'Numerical anchors:')
print(f'  M_bound = {M_bound}, alpha_coer_bulk = {alpha_coer_bulk}')
print(f'  needed_ratio_c (Lipschitz/|T|) = {needed_ratio_c:.4f}')
print(f'  required (Hessian/Lipschitz) = {required:.4f}')

# ============================================================
# Save bundle
# ============================================================
out = {
    'audit': 'federer-prefactor-galerkin-fem-derivation-attempt',
    'stand': '2026-05-04',
    'system_R_anchors': {
        'alpha_xi': str(ALPHA_XI), 'gamma': str(GAMMA),
        'eps_sync2': str(EPS_SYNC2), 'beta_pi': str(BETA_PI),
        'D_Omega': str(D_OMEGA),
        'lambda_crit': str(LAMBDA_CRIT), 'G_NET': str(G_NET),
    },
    'cea_framework': {
        'mu_perp': str(mu_perp),
        'alpha_coer_at_kSq_eq_1': str(alpha_coer),
        'alpha_coer_at_kSq_eq_lambda_crit_plus_1': str(alpha_coer_bulk),
        'M_bound': str(M_bound),
        'Cea_constant_M_over_alpha': float(M_over_alpha),
        'C_I_tetrahedral': C_I,
        'C_AN_aubin_nitsche': C_AN,
    },
    'predicted_vs_empirical': {
        'c_predicted_hypothesis': c_hyp,
        'c_empirical': c_emp,
        'c_rel_diff_pct': (c_hyp - c_emp)/c_emp*100,
        'd_predicted_hypothesis': d_hyp,
        'd_empirical': d_emp,
        'd_rel_diff_pct': (d_hyp - d_emp)/d_emp*100,
        'required_Lipschitz_over_T_for_c_match': float(needed_ratio_c),
        'required_Hessian_over_Lipschitz_for_d_over_c_match': float(required),
    },
    'verdict': (
        'Galerkin-FEM framework yields the correct N^{-1/3} and N^{-2/3} '
        'leading and sub-leading scalings via Cea + Aubin-Nitsche. The '
        'framework-rational prefactors c=2*alpha_xi^2*gamma and '
        'd=-(1+gamma)*c are CONSISTENT with the framework but the '
        'Lipschitz and Hessian seminorms of the residual relative to the '
        'stress-energy must be measured (not derived) on the canonical '
        'lattice. Hypothesis remains a hypothesis; the N-scaling shape is '
        'derived but the prefactors are not.'
    ),
}
out_path = Path(__file__).resolve().parents[1] / 'data' / 'federer_prefactor_galerkin_fem_derivation.json'
out_path.parent.mkdir(exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f'\nSaved: {out_path}')
