#!/usr/bin/env python3
"""Run teleoperation from dev_config.yaml."""

import subprocess
import sys
from pathlib import Path

import yaml


def main():
    config_path = Path(__file__).parent / "dev_config.yaml"
    cfg = yaml.safe_load(config_path.read_text())

    leader = cfg["leader"]
    follower = cfg["follower"]

    cmd = [
        "lerobot-teleoperate",
        f"--teleop.type={leader['type']}",
        f"--teleop.id={leader['id']}",
        f"--teleop.port={leader['port']}",
        f"--robot.type={follower['type']}",
        f"--robot.id={follower['id']}",
        f"--robot.port={follower['port']}",
        f"--robot.can_adapter={follower['can_adapter']}",
    ]

    cameras = follower.get("cameras") or cfg.get("cameras")
    if cameras:
        cam_str = str(cameras).replace("'", '"')
        cmd.append(f"--robot.cameras={cam_str}")

    print("Running:", " ".join(cmd))
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
