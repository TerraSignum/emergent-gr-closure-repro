"""Auto-discover D1 snapshot ladder data.

The canonical ladder for (SG) closure analysis is defined dynamically by
whatever ``results_d1_p5n*_*/P5N*.snapshots.npz`` files are present in the
repo's external parent directory. This module returns the sorted list of
(regime, N, npz_path) tuples discovered, so downstream analyses don't need
hardcoded LADDER lists.

When a new high-N production run lands (e.g. P5N2048, P5N4096), it is
automatically picked up on the next analysis run, without any manual
script update.

Priority order for duplicate regimes (multiple snapshot-runs of same N):
  kq_fixed > 12seeds > 24seeds > 8seeds > trial1seed > _v2 > default
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

_RE_REGIME = re.compile(r"p5n(\d+)")

_PRIORITY_TAGS = (
    "kq_fixed",
    "12seeds",
    "24seeds",
    "8seeds",
    "canonical",
    "_v2",
    "trial1seed",
)


def _priority_index(dirname: str) -> int:
    """Return priority rank (lower is better) for a results_d1_* dirname."""
    lower = dirname.lower()
    for idx, tag in enumerate(_PRIORITY_TAGS):
        if tag in lower:
            return idx
    return 999  # unknown / plain results_d1_p5nX/


def discover_d1_ladder(repo_root: Path,
                        regimes: Iterable[str] | None = None
                        ) -> list[tuple[str, int, Path]]:
    """Discover all available D1 snapshot files via glob.

    Args:
      repo_root: Path to the *_repro repository root. The function looks
                 at ``repo_root.parent`` (i.e. Emergence/) for
                 ``results_d1_*/*.snapshots.npz``.
      regimes:   Optional iterable of regime names (e.g. {"P5N512", "P5N1024"})
                 to restrict discovery. If None, returns all discovered.

    Returns:
      Sorted list of (regime, N, npz_path) tuples, ascending by N.
    """
    external = repo_root.parent
    seen: dict[str, tuple[int, Path, int]] = {}  # regime -> (N, path, priority)
    for results_dir in external.glob("results_d1_*"):
        if not results_dir.is_dir():
            continue
        m = _RE_REGIME.search(results_dir.name.lower())
        if m is None:
            continue
        n = int(m.group(1))
        regime = f"P5N{n}"
        if regimes is not None and regime not in regimes:
            continue
        snap_name = f"{regime}.snapshots.npz"
        npz_path = results_dir / snap_name
        if not npz_path.is_file():
            continue
        priority = _priority_index(results_dir.name)
        existing = seen.get(regime)
        if existing is None or priority < existing[2]:
            seen[regime] = (n, npz_path, priority)

    return sorted([(r, n, p) for r, (n, p, _pr) in seen.items()],
                   key=lambda x: x[1])


def discover_ladder_n_values(repo_root: Path) -> list[int]:
    """Return just the N-values of the discovered ladder, sorted ascending."""
    return [n for _, n, _ in discover_d1_ladder(repo_root)]


def discover_max_n(repo_root: Path) -> int:
    """Return the largest N currently available, or 0 if no ladder data."""
    ns = discover_ladder_n_values(repo_root)
    return max(ns) if ns else 0


if __name__ == "__main__":
    # Command-line: print discovered ladder.
    import sys
    here = Path(__file__).resolve().parent
    repo = here.parent
    print(f"Discovering D1 ladder from {repo.parent} ...")
    found = discover_d1_ladder(repo)
    if not found:
        print("  no D1 snapshot files found")
        sys.exit(1)
    print(f"  {len(found)} regimes discovered:")
    for regime, n, p in found:
        rel = p.relative_to(repo.parent) if repo.parent in p.parents else p
        print(f"    {regime:>10s}  N={n:>5d}  {rel}")
