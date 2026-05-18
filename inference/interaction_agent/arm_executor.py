from __future__ import annotations

import time
from abc import ABC, abstractmethod

from .models import Pose, SkillResult


class ArmExecutor(ABC):
    @abstractmethod
    def move_block(self, from_position: str, to_position: str, source_pose: Pose, target_pose: Pose) -> SkillResult:
        """Move a block between named tabletop positions."""

    @abstractmethod
    def go_home(self) -> SkillResult:
        """Move the arm to a neutral home pose."""

    @abstractmethod
    def stop(self) -> SkillResult:
        """Stop current motion as quickly as the backend supports."""


class MockArmExecutor(ArmExecutor):
    """Non-hardware executor that prints the intended arm actions."""

    def move_block(self, from_position: str, to_position: str, source_pose: Pose, target_pose: Pose) -> SkillResult:
        print(
            "[mock-arm] move_block "
            f"from={from_position} "
            f"source_pose=({source_pose.x:.2f}, {source_pose.y:.2f}, {source_pose.z:.2f}) "
            f"to={to_position} "
            f"target_pose=({target_pose.x:.2f}, {target_pose.y:.2f}, {target_pose.z:.2f})"
        )
        time.sleep(0.4)
        return SkillResult(success=True, message=f"Moved the block from {from_position} to {to_position}.")

    def go_home(self) -> SkillResult:
        print("[mock-arm] go_home")
        return SkillResult(success=True, message="Arm returned home.")

    def stop(self) -> SkillResult:
        print("[mock-arm] stop")
        return SkillResult(success=True, message="Motion stopped.")
