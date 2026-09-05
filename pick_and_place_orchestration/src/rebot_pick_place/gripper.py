from __future__ import annotations

import time
from dataclasses import dataclass

from .constants import DEFAULT_GRIPPER_MAX_WIDTH_M, GRIPPER_CLOSED_DEG, GRIPPER_OPEN_DEG
from .types import GripResult


@dataclass
class Gripper:
    robot: object | None = None
    max_width_m: float = DEFAULT_GRIPPER_MAX_WIDTH_M
    hold_torque_threshold: float = 0.03
    command_settle_s: float = 0.25

    def __post_init__(self) -> None:
        self._angle_deg = GRIPPER_OPEN_DEG
        self._last_effort: float | None = None

    def open(self) -> None:
        self._send(GRIPPER_OPEN_DEG)

    def close(self, force_limit: float | None = None) -> GripResult:
        self._send(GRIPPER_CLOSED_DEG, force_limit=force_limit)
        contact = self.is_holding()
        return GripResult(
            success=contact,
            contact_detected=contact,
            achieved_width=self.width(),
            effort=self._last_effort,
            message="" if contact else "No current/torque contact detected.",
        )

    def is_holding(self) -> bool:
        if self.robot is None:
            return False
        obs = self.robot.get_observation()
        effort = float(obs.get("gripper.torque", 0.0))
        self._last_effort = effort
        return abs(effort) >= self.hold_torque_threshold and self.width() > 0.002

    def width(self) -> float:
        if self.robot is not None:
            obs = self.robot.get_observation()
            self._angle_deg = float(obs.get("gripper.pos", self._angle_deg))
        open_fraction = min(1.0, max(0.0, abs(self._angle_deg) / abs(GRIPPER_OPEN_DEG)))
        return open_fraction * self.max_width_m

    def _send(self, angle_deg: float, force_limit: float | None = None) -> None:
        self._angle_deg = angle_deg
        if self.robot is not None:
            if force_limit is not None and hasattr(self.robot.config, "force_pos_torque_ration"):
                self.robot.config.force_pos_torque_ration = float(force_limit)
            self.robot.send_action({"gripper.pos": angle_deg})
        time.sleep(self.command_settle_s)

