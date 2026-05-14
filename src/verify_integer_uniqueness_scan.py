"""Verify the integer-uniqueness theorem: only (N_gen=3, d=4) satisfies
the carrier-transport closure system within tolerance |Delta| <= 2.5%
across the integer scan N_gen in {1..5}, d in {2..6}.

Constraint system (companion causal-wave paper):
  C1: alpha_xi + gamma = 1
  C2: D(Omega) = beta_pi - gamma
  C3: epsilon_sync^2 = gamma / 2
  C4: gamma = 1 / (N_gen^2 + 1)               (Pythagoras-complementarity)
  C5: beta_pi = (2^d - 1) / 2^d                (Clifford-projector identity)

Lattice-measured targets (causal_wave_transport_equation_probe_bundle):
  alpha_xi   = 0.900819
  gamma      = 0.100206
  beta_pi    = 0.937913
  D(Omega)   = 0.839964
  epsilon_sync^2 = 0.050000

For each (N_gen, d), derive the C1-C5 predictions and report the
maximum relative residual against the lattice readouts.

Writes:
  data/integer_uniqueness_scan.json
"""
from __future__ import annotations
import json
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent

LATTICE = {
    "alpha_xi":      0.900819,
    "gamma":         0.100206,
    "beta_pi":       0.937913,
    "D_omega":       0.839964,
    "epsilon_sync2": 0.050000,
}
TOLERANCE = 0.025  # 2.5%


def predict(n_gen: int, d: int) -> dict:
    """C1-C5 algebraic predictions for given (N_gen, d)."""
    gamma = 1.0 / (n_gen ** 2 + 1)              # C4
    alpha_xi = 1.0 - gamma                      # C1
    beta_pi = (2 ** d - 1) / (2 ** d)           # C5
    D_omega = beta_pi - gamma                   # C2
    eps2 = gamma / 2.0                          # C3
    return {
        "alpha_xi": alpha_xi, "gamma": gamma,
        "beta_pi": beta_pi, "D_omega": D_omega,
        "epsilon_sync2": eps2,
    }


def max_residual(pred: dict, target: dict) -> tuple[float, str]:
    worst = 0.0
    worst_key = ""
    for k in target:
        if target[k] == 0:
            continue
        rel = abs(pred[k] - target[k]) / abs(target[k])
        if rel > worst:
            worst = rel
            worst_key = k
    return worst, worst_key


def main() -> int:
    print("=" * 84)
    print("Integer-uniqueness scan: (N_gen, d) admissibility under C1-C5")
    print("=" * 84)
    print()
    print("Lattice readouts (causal-wave-measured targets):")
    for k, v in LATTICE.items():
        print(f"  {k:<18} = {v:.6f}")
    print()
    print(f"Tolerance: |Delta_rel| <= {TOLERANCE * 100:.1f}%")
    print()

    rows = []
    print(f"{'N_gen':>5} {'d':>3} {'alpha_xi':>10} {'gamma':>9} "
          f"{'beta_pi':>9} {'D_omega':>9} {'eps^2':>8} {'max_rel':>9} "
          f"{'verdict':>10}")
    print("-" * 90)
    for n_gen in range(1, 6):  # 1..5
        for d in range(2, 7):  # 2..6
            p = predict(n_gen, d)
            res, key = max_residual(p, LATTICE)
            verdict = "PASS" if res <= TOLERANCE else "FAIL"
            print(f"{n_gen:>5} {d:>3} "
                  f"{p['alpha_xi']:>10.4f} {p['gamma']:>9.4f} "
                  f"{p['beta_pi']:>9.4f} {p['D_omega']:>9.4f} "
                  f"{p['epsilon_sync2']:>8.4f} "
                  f"{res:>9.4f} {verdict:>10}")
            rows.append({
                "N_gen": n_gen, "d": d,
                "predicted": p,
                "max_relative_residual": res,
                "worst_key": key,
                "passes_tolerance": bool(res <= TOLERANCE),
            })

    n_pass = sum(1 for r in rows if r["passes_tolerance"])
    pass_rows = [r for r in rows if r["passes_tolerance"]]
    print()
    print(f"Total admissible (N_gen, d) under tolerance {TOLERANCE*100:.1f}%: "
          f"{n_pass} / {len(rows)}")
    if n_pass == 1:
        unique = pass_rows[0]
        print(f"  UNIQUE solution: (N_gen={unique['N_gen']}, d={unique['d']})")
    elif n_pass > 0:
        for r in pass_rows:
            print(f"  Admissible: (N_gen={r['N_gen']}, d={r['d']}) "
                  f"max_rel = {r['max_relative_residual']:.4f}")
    else:
        print("  No (N_gen, d) admissible under given tolerance.")

    bundle = {
        "method": "integer_uniqueness_scan",
        "tolerance_relative": TOLERANCE,
        "lattice_targets": LATTICE,
        "constraint_system": {
            "C1": "alpha_xi + gamma = 1",
            "C2": "D(Omega) = beta_pi - gamma",
            "C3": "epsilon_sync^2 = gamma / 2",
            "C4": "gamma = 1 / (N_gen^2 + 1)",
            "C5": "beta_pi = (2^d - 1) / 2^d",
        },
        "scan": rows,
        "n_admissible": n_pass,
        "unique": (n_pass == 1),
        "unique_solution": (
            {"N_gen": pass_rows[0]["N_gen"], "d": pass_rows[0]["d"]}
            if n_pass == 1 else None
        ),
    }
    out_path = REPO / "data" / "integer_uniqueness_scan.json"
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
