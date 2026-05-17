from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .arm_executor import ArmExecutor
from .models import Scene, SkillResult, display_position_name


SkillHandler = Callable[[dict[str, Any], Scene], SkillResult]


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    handler: SkillHandler


class SkillRegistry:
    def __init__(self, arm: ArmExecutor, min_confidence: float = 0.7):
        self._arm = arm
        self._min_confidence = min_confidence
        self._skills = {
            "move_block": SkillSpec(
                name="move_block",
                description="Move a block from one named tabletop position to another.",
                handler=self._move_block,
            ),
            "go_home": SkillSpec(
                name="go_home",
                description="Move the robot arm to its neutral home pose.",
                handler=self._go_home,
            ),
            "stop": SkillSpec(
                name="stop",
                description="Stop robot motion.",
                handler=self._stop,
            ),
        }

    def schemas(self) -> list[dict[str, str]]:
        return [
            {"name": spec.name, "description": spec.description}
            for spec in self._skills.values()
        ]

    def realtime_tools(self) -> list[dict[str, Any]]:
        positions = ["top_left", "top_right", "bottom_left", "bottom_right", "center"]
        return [
            {
                "type": "function",
                "name": "move_block",
                "description": (
                    "Move the block from one calibrated tabletop position to another. "
                    "Use this when the user asks to move a block between named positions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_position": {
                            "type": "string",
                            "description": "The source position containing the block.",
                            "enum": positions,
                        },
                        "to_position": {
                            "type": "string",
                            "description": "The destination position for the block.",
                            "enum": positions,
                        },
                    },
                    "required": ["from_position", "to_position"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "go_home",
                "description": "Move the robot arm to its neutral home pose.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "stop",
                "description": "Stop robot motion immediately.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_scene",
                "description": "Report the calibrated tabletop positions and currently tracked block state.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        ]

    def call(self, name: str, arguments: dict[str, Any], scene: Scene) -> SkillResult:
        spec = self._skills.get(name)
        if spec is None:
            return SkillResult(
                success=False,
                message=f"Unknown skill: {name}",
                recoverable=True,
            )
        return spec.handler(arguments, scene)

    def _move_block(self, arguments: dict[str, Any], scene: Scene) -> SkillResult:
        from_query = str(arguments.get("from_position", "")).strip()
        to_query = str(arguments.get("to_position", "")).strip()
        if not from_query or not to_query:
            return SkillResult(
                success=False,
                message="The move_block skill needs both a source position and destination position.",
                recoverable=True,
            )

        from_position = scene.resolve_position(from_query)
        if from_position is None:
            return SkillResult(
                success=False,
                message=f"I do not know the source position {from_query}.",
                recoverable=True,
            )

        to_position = scene.resolve_position(to_query)
        if to_position is None:
            return SkillResult(
                success=False,
                message=f"I do not know the destination position {to_query}.",
                recoverable=True,
            )

        if from_position == to_position:
            return SkillResult(
                success=False,
                message="The source and destination positions are the same.",
                recoverable=True,
            )

        source_pose = scene.pose_for(from_position)
        target_pose = scene.pose_for(to_position)
        if source_pose is None or target_pose is None:
            return SkillResult(
                success=False,
                message="One of those positions does not have a calibrated pose.",
                recoverable=True,
            )

        if not self._is_inside_workspace(target_pose.x, target_pose.y, target_pose.z):
            return SkillResult(
                success=False,
                message=f"The destination {display_position_name(to_position)} is outside the arm workspace.",
                recoverable=True,
            )

        return self._arm.move_block(
            from_position=display_position_name(from_position),
            to_position=display_position_name(to_position),
            source_pose=source_pose,
            target_pose=target_pose,
        )

    def _go_home(self, arguments: dict[str, Any], scene: Scene) -> SkillResult:
        return self._arm.go_home()

    def _stop(self, arguments: dict[str, Any], scene: Scene) -> SkillResult:
        return self._arm.stop()

    @staticmethod
    def _is_inside_workspace(x: float, y: float, z: float) -> bool:
        return 0.05 <= x <= 0.65 and -0.40 <= y <= 0.40 and 0.0 <= z <= 0.35
