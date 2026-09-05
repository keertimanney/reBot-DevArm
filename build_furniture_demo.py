#!/usr/bin/env python3
"""Build a bed or an L-shaped sofa using the same tool stack as the voice agent.

This is a non-LLM driver: it calls the same executor methods that the realtime
voice loop dispatches. Useful for:
  - demonstrating the tool-call sequence for multi-block furniture
  - testing the executor (occupancy, gripper state, IK fallback) end-to-end
    without needing the OpenAI Realtime API
  - regression checks after touching executor logic

Usage:
    # Dry-run (no hardware needed) — prints what the model would do:
    python build_furniture_demo.py bed --dry-run
    python build_furniture_demo.py l_sofa --dry-run

    # Real arm (pauses for ENTER between blocks so you can reload pickup zone):
    python build_furniture_demo.py bed
    python build_furniture_demo.py l_sofa

    # List the named recipes:
    python build_furniture_demo.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

from run_voice_agent_reachy import (  # noqa: E402
    DEFAULT_GRID_FILE,
    DEFAULT_POSITIONS_FILE,
    DEFAULT_RECIPES_FILE,
    DEFAULT_STATE_FILE,
    DryRunGraspExecutor,
    ReBotGraspExecutor,
    load_recipes,
)


def build(
    executor: DryRunGraspExecutor | ReBotGraspExecutor,
    recipes: dict[str, dict[str, Any]],
    recipe_name: str,
    pause_between: bool,
    grip_style: str,
    reset_first: bool,
) -> None:
    recipe = recipes[recipe_name]
    cells  = recipe["cells"]

    print(f"\n=== Building '{recipe_name}' ===")
    print(f"    {recipe['description']}")
    print(f"    Cells (in placement order): {cells}\n")

    if reset_first:
        print("[reset_occupancy]")
        print("  ->", executor.execute_tool("reset_occupancy", {}))

    for n, (i, j) in enumerate(cells, start=1):
        print(f"\n--- block {n}/{len(cells)}  ->  cell ({i}, {j}) ---")
        if pause_between:
            input(f"    Load a block onto the pickup zone and press ENTER... ")

        for tool, args in (
            ("move_to_pickup_zone", {}),
            ("grab_block",          {"grip_style": grip_style}),
            ("move_to_grid_cell",   {"i": i, "j": j}),
            ("drop_block",          {"grip_style": grip_style}),
        ):
            result = executor.execute_tool(tool, args)
            ok = not (result.startswith("error") or result.startswith("IK failed"))
            print(f"  {tool:<22} args={args}")
            print(f"    -> {'OK ' if ok else 'FAIL'} {result}")
            if not ok:
                print("\n[aborting recipe: tool returned an error]")
                return

    print(f"\n=== '{recipe_name}' complete ===")
    print(f"    Final occupancy: {executor.occupancy_snapshot()}\n")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("recipe", nargs="?",
                   help="Which furniture to build (use --list to see options).")
    p.add_argument("--list", action="store_true",
                   help="List recipes and their cell layouts, then exit.")
    p.add_argument("--recipes-file", default=DEFAULT_RECIPES_FILE,
                   help=f"YAML recipe library (default: {DEFAULT_RECIPES_FILE!r}).")
    p.add_argument("--dry-run", action="store_true",
                   help="No hardware — exercise the executor only.")
    p.add_argument("--pause-between", action="store_true",
                   help="Pause for ENTER between blocks (default: on for real arm, "
                        "off for dry-run).")
    p.add_argument("--no-pause", action="store_true",
                   help="Disable the per-block ENTER prompt even on real arm.")
    p.add_argument("--grip-style", default="edge", choices=["face", "edge"],
                   help="Grip style for grab_block / drop_block.")
    p.add_argument("--no-reset", action="store_true",
                   help="Skip reset_occupancy at the start (append to current layout).")
    p.add_argument("--positions-file", default=DEFAULT_POSITIONS_FILE)
    p.add_argument("--grid-file",      default=DEFAULT_GRID_FILE)
    p.add_argument("--state-file",     default=DEFAULT_STATE_FILE)
    p.add_argument("--arm-config",     default=None)
    p.add_argument("--duration",       type=float, default=2.0)
    args = p.parse_args()

    recipes_path = Path(args.recipes_file)
    if not recipes_path.is_absolute():
        recipes_path = _REPO_ROOT / recipes_path
    recipes = load_recipes(recipes_path)

    if args.list:
        for name, recipe in recipes.items():
            print(f"\n{name}:")
            print(f"  {recipe['description'].strip()}")
            print(f"  cells: {recipe['cells']}")
        return 0

    if not args.recipe:
        p.error("recipe is required (or use --list).")
    if args.recipe not in recipes:
        p.error(f"unknown recipe {args.recipe!r}. Available: {list(recipes)}")

    if args.dry_run:
        executor = DryRunGraspExecutor(nx=7, ny=4)
        default_pause = False
    else:
        executor = ReBotGraspExecutor(
            positions_file=_REPO_ROOT / args.positions_file,
            grid_file=_REPO_ROOT / args.grid_file,
            arm_config=args.arm_config,
            duration=args.duration,
            state_file=_REPO_ROOT / args.state_file,
        )
        default_pause = True

    pause = default_pause
    if args.pause_between:
        pause = True
    if args.no_pause:
        pause = False

    try:
        build(
            executor,
            recipes=recipes,
            recipe_name=args.recipe,
            pause_between=pause,
            grip_style=args.grip_style,
            reset_first=not args.no_reset,
        )
    finally:
        executor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
