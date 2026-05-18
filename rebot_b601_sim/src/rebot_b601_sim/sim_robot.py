from __future__ import annotations

import argparse
import time
from pathlib import Path

from .joints import SIM_JOINT_NAMES
from .pinocchio_meshcat import B601MeshcatSim, default_control_repo, default_workcell_config


def parse_joint_degrees(values: list[float]) -> dict[str, float]:
    return {f"{name}.pos": values[idx] for idx, name in enumerate(SIM_JOINT_NAMES[: len(values)])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a B601-DM MeshCat simulation.")
    parser.add_argument("--control-repo", type=Path, default=default_control_repo())
    parser.add_argument("--robot-config", type=Path)
    parser.add_argument("--urdf")
    parser.add_argument("--workcell", type=Path, default=default_workcell_config())
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--q-deg", nargs="*", type=float, default=[0, 0, 0, 0, 0, 0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim = B601MeshcatSim(
        control_repo=args.control_repo,
        robot_config=args.robot_config,
        urdf=args.urdf,
        open_browser=not args.no_browser,
    )
    sim.load_workcell(args.workcell)
    sim.display_degrees_action(parse_joint_degrees(args.q_deg))
    print(f"MeshCat is running. URDF: {sim.urdf_path}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

