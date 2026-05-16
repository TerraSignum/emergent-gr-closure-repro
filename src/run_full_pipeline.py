r"""Full auto-pipeline driver: regenerate everything when new N data lands.

Usage:
  cd emergent-gr-closure-repro
  python src/run_full_pipeline.py [--skip-compile]

Steps (each cheap; full run ~3 min):
  1. derive_skeleton_weighted_lift_chain.py  (auto-discovered ladder)
  2. derive_skeleton_eigenvector_mode_audit.py
  3. derive_branch_resolved_audit.py
  4. derive_bayesian_rational_identification.py
  5. run_adaptive_pipeline.py
  6. generate_sg_ladder_tex.py
  7. (optional) tectonic compile manuscript.tex
  8. (optional) compendium rebuild

After completion:
  - paper/generated/*.tex regenerated
  - outputs/derive_*.json regenerated
  - paper/manuscript.pdf rebuilt (if --skip-compile not set)
  - compendium/compendium.pdf rebuilt (if --skip-compile not set)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
REPO = SRC.parent
EMERGENCE_ROOT = REPO.parent
COMPENDIUM_DIR = EMERGENCE_ROOT / "compendium"
MANUSCRIPTS_PDF = EMERGENCE_ROOT / "manuscripts_pdf"

PY = sys.executable
TECTONIC = "C:/Users/user/.local/bin/tectonic.exe"


def run_step(name: str, cmd: list[str], cwd: Path) -> bool:
    """Run one pipeline step, report timing."""
    print(f"\n--- {name} ---")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    elapsed = time.time() - t0
    print(f"  elapsed: {elapsed:.1f}s, exit code: {r.returncode}")
    if r.returncode != 0:
        print(f"  STDERR: {r.stderr[-500:]}")
        return False
    # Show key output lines
    for line in r.stdout.splitlines()[-6:]:
        print(f"  | {line}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-compile", action="store_true",
                          help="skip pdflatex compile + compendium rebuild")
    parser.add_argument("--skip-compendium", action="store_true",
                          help="skip just the compendium rebuild")
    args = parser.parse_args()

    print("=" * 78)
    print("Adaptive Carrier Pipeline — Full Auto-Regeneration")
    print("=" * 78)
    t_total = time.time()

    steps = [
        ("derive_skeleton_weighted_lift_chain",
         [PY, "src/derive_skeleton_weighted_lift_chain.py"]),
        ("derive_skeleton_eigenvector_mode_audit",
         [PY, "src/derive_skeleton_eigenvector_mode_audit.py"]),
        ("derive_branch_resolved_audit",
         [PY, "src/derive_branch_resolved_audit.py"]),
        ("derive_bayesian_rational_identification",
         [PY, "src/derive_bayesian_rational_identification.py"]),
        ("run_adaptive_pipeline (Tier 1-4 orchestrator)",
         [PY, "src/run_adaptive_pipeline.py"]),
        ("generate_sg_ladder_tex (auto-tex macros)",
         [PY, "src/generate_sg_ladder_tex.py"]),
    ]

    failed = []
    for name, cmd in steps:
        ok = run_step(name, cmd, REPO)
        if not ok:
            failed.append(name)

    if failed:
        print(f"\n  WARNING: {len(failed)} steps failed: {failed}")
        return 1

    if not args.skip_compile:
        # Compile P4 manuscript
        ok = run_step("tectonic P4 compile",
                       [TECTONIC, "manuscript.tex"], REPO / "paper")
        if ok and (REPO / "paper" / "manuscript.pdf").is_file():
            shutil.copy(REPO / "paper" / "manuscript.pdf",
                         MANUSCRIPTS_PDF / "P4_emergent_gr.pdf")
            print(f"  Copied to {MANUSCRIPTS_PDF / 'P4_emergent_gr.pdf'}")

        if not args.skip_compendium:
            # Rebuild compendium
            ok = run_step("compendium rebuild",
                           [PY, "make_compendium.py"], COMPENDIUM_DIR)
            if ok and (COMPENDIUM_DIR / "compendium.pdf").is_file():
                shutil.copy(COMPENDIUM_DIR / "compendium.pdf",
                             MANUSCRIPTS_PDF / "00_compendium.pdf")
                print(f"  Copied to {MANUSCRIPTS_PDF / '00_compendium.pdf'}")

    total = time.time() - t_total
    print(f"\n{'=' * 78}")
    print(f"FULL PIPELINE COMPLETE in {total:.1f}s")
    print("=" * 78)
    print("\nKey output files:")
    print(f"  paper/generated/sg_asymptotes.tex")
    print(f"  paper/generated/sg_ladder_rows.tex")
    print(f"  paper/generated/adaptive_macros.tex")
    print(f"  paper/generated/branch_resolved_macros.tex")
    print(f"  outputs/adaptive_pipeline_report.json")
    print(f"  outputs/derive_branch_resolved_audit.json")
    if not args.skip_compile:
        print(f"  {REPO}/paper/manuscript.pdf")
        if not args.skip_compendium:
            print(f"  {COMPENDIUM_DIR}/compendium.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
