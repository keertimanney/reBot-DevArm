from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class Pose:
    """Cartesian pose in meters. Rotation is intentionally optional for MVP skills."""

    x: float
    y: float
    z: float
    yaw: float = 0.0


@dataclass(frozen=True)
class Scene:
    positions: dict[str, Pose] = field(default_factory=dict)
    occupied_positions: dict[str, str] = field(default_factory=dict)

    def resolve_position(self, value: str) -> str | None:
        normalized = normalize_position_name(value)
        if normalized in self.positions:
            return normalized
        return None

    def pose_for(self, position: str) -> Pose | None:
        resolved = self.resolve_position(position)
        if resolved is None:
            return None
        return self.positions[resolved]

    def describe(self) -> str:
        positions = ", ".join(display_position_name(name) for name in self.positions)
        occupied = ", ".join(
            f"{block} at {display_position_name(position)}"
            for position, block in self.occupied_positions.items()
        )
        if not occupied:
            occupied = "no tracked blocks"
        return f"Available positions: {positions}. I currently track {occupied}."


POSITION_ALIASES = {
    "top left": "top_left",
    "upper left": "top_left",
    "top-left": "top_left",
    "top_left": "top_left",
    "top right": "top_right",
    "upper right": "top_right",
    "top-right": "top_right",
    "top_right": "top_right",
    "bottom left": "bottom_left",
    "lower left": "bottom_left",
    "bottom-left": "bottom_left",
    "bottom_left": "bottom_left",
    "bottom right": "bottom_right",
    "lower right": "bottom_right",
    "bottom-right": "bottom_right",
    "bottom_right": "bottom_right",
    "center": "center",
    "centre": "center",
    "middle": "center",
}


def normalize_position_name(value: str) -> str:
    cleaned = " ".join(value.lower().replace("_", " ").replace("-", " ").split())
    return POSITION_ALIASES.get(cleaned, cleaned.replace(" ", "_"))


def display_position_name(value: str) -> str:
    return normalize_position_name(value).replace("_", " ")


class DecisionType(str, Enum):
    SPEAK = "speak"
    CALL_SKILL = "call_skill"
    ASK_CLARIFICATION = "ask_clarification"


@dataclass(frozen=True)
class AgentDecision:
    type: DecisionType
    text: str = ""
    skill_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillResult:
    success: bool
    message: str
    recoverable: bool = False
    data: dict[str, Any] = field(default_factory=dict)
