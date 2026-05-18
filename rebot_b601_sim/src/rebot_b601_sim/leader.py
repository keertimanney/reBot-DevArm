from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def add_repo_packages_to_path(repo_root: Path) -> None:
    leader_pkg = repo_root / "lerobot-teleoperator-rebot-arm-102"
    if leader_pkg.exists():
        sys.path.insert(0, str(leader_pkg))


def import_leader_classes() -> tuple[type[Any], type[Any]]:
    repo_root = Path(__file__).resolve().parents[3]
    add_repo_packages_to_path(repo_root)
    try:
        from lerobot_teleoperator_rebot_arm_102 import RebotArm102Leader, RebotArm102LeaderConfig
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        raise SystemExit(
            f"Missing Python package: {missing}\n\n"
            "Install the leader-sim dependencies into the environment that runs this command:\n"
            "  python -m pip install -e ./lerobot -e ./lerobot-teleoperator-rebot-arm-102\n"
            "  python -m pip install -e rebot_b601_sim\n\n"
            "Then re-run with your leader port, for example:\n"
            "  rebot-b601-leader-sim --leader-port /dev/cu.usbserial-0001"
        ) from exc
    return RebotArm102Leader, RebotArm102LeaderConfig

