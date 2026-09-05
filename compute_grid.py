#!/usr/bin/env python3
"""Compute and display grid cell centres from grid_config.yaml.

Bilinear interpolation maps four physical corners (C1–C4) onto an
nx × ny grid.  Cell (i, j) sits at the centre of the i-th column
(0-indexed, C1→C2 direction) and j-th row (0-indexed, C1→C4 direction).

Usage:
    python compute_grid.py
    python compute_grid.py --grid-file my_grid.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


def compute_grid_map(
    config_path: str | Path,
) -> tuple[dict[tuple[int, int], tuple[float, float]], float]:
    """Return ({(i,j): (x,y)}, z_height) for every cell centre.

    Corners in the YAML must be ordered:
        C1 = (i=0,    j=0)     near-left
        C2 = (i=nx-1, j=0)     near-right
        C3 = (i=nx-1, j=ny-1)  far-right
        C4 = (i=0,    j=ny-1)  far-left
    """
    cfg = yaml.safe_load(Path(config_path).read_text())

    corners = cfg["corners"]
    C1 = np.array(corners["C1"], dtype=float)
    C2 = np.array(corners["C2"], dtype=float)
    C3 = np.array(corners["C3"], dtype=float)
    C4 = np.array(corners["C4"], dtype=float)

    nx: int = cfg["grid"]["nx"]
    ny: int = cfg["grid"]["ny"]
    z_height: float = float(cfg.get("z_height", 0.10))

    grid_map: dict[tuple[int, int], tuple[float, float]] = {}
    for j in range(ny):
        for i in range(nx):
            # s ∈ (0,1) along C1→C2 axis, t ∈ (0,1) along C1→C4 axis
            s = (i + 0.5) / nx
            t = (j + 0.5) / ny
            pt = (
                (1 - s) * (1 - t) * C1
                + s * (1 - t) * C2
                + s * t * C3
                + (1 - s) * t * C4
            )
            grid_map[(i, j)] = (float(pt[0]), float(pt[1]))

    return grid_map, z_height


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print grid cell centres from grid_config.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--grid-file", default="grid_config.yaml")
    args = parser.parse_args()

    path = Path(args.grid_file)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        sys.exit(f"Grid file not found: {path}")

    cfg = yaml.safe_load(path.read_text())
    nx: int = cfg["grid"]["nx"]
    ny: int = cfg["grid"]["ny"]

    grid_map, z_height = compute_grid_map(path)

    print(f"Grid: {nx} cols × {ny} rows   z_height={z_height} m\n")
    print(f"  {'Cell (i,j)':<12}  {'x':>9}  {'y':>9}")
    print("  " + "-" * 34)

    for j in range(ny - 1, -1, -1):   # top row first so layout matches bird's-eye view
        for i in range(nx):
            x, y = grid_map[(i, j)]
            print(f"  ({i}, {j})          {x:+.4f}   {y:+.4f}")
        print()

    print("Hashmap  (i, j) → (x, y):")
    for (i, j), (x, y) in sorted(grid_map.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"  ({i}, {j}): ({x:+.6f}, {y:+.6f})")


if __name__ == "__main__":
    main()
