from __future__ import annotations

import argparse
import time
from pathlib import Path

from .leader import import_leader_classes
from .pinocchio_meshcat import B601MeshcatSim, default_control_repo, default_workcell_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive the simulated B601 follower from the reBot Arm 102 leader.")
    parser.add_argument("--leader-port", required=True, help="Leader serial port, e.g. /dev/cu.usbserial-0001")
    parser.add_argument("--leader-id", default="leader_sim")
    parser.add_argument("--leader-baudrate", type=int, default=1_000_000)
    parser.add_argument("--control-repo", type=Path, default=default_control_repo())
    parser.add_argument("--robot-config", type=Path)
    parser.add_argument("--urdf")
    parser.add_argument("--workcell", type=Path, default=default_workcell_config())
    parser.add_argument("--interval", type=float, default=0.03)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RebotArm102Leader, RebotArm102LeaderConfig = import_leader_classes()

    leader = RebotArm102Leader(
        RebotArm102LeaderConfig(
            id=args.leader_id,
            port=args.leader_port,
            baudrate=args.leader_baudrate,
        )
    )
    sim = B601MeshcatSim(
        control_repo=args.control_repo,
        robot_config=args.robot_config,
        urdf=args.urdf,
        open_browser=not args.no_browser,
    )
    sim.load_workcell(args.workcell)

    leader.connect(calibrate=True)
    print("Streaming leader joints into MeshCat. Press Ctrl+C to stop.")
    try:
        while True:
            action = leader.get_action()
            sim.display_degrees_action(action)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        leader.disconnect()


if __name__ == "__main__":
    main()
