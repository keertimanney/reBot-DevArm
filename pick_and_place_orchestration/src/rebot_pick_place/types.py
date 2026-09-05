from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class MotionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    IK_INFEASIBLE = "IK_INFEASIBLE"
    COLLISION = "COLLISION"
    JOINT_LIMIT = "JOINT_LIMIT"
    TIMEOUT = "TIMEOUT"
    EXECUTION_ERROR = "EXECUTION_ERROR"


@dataclass(frozen=True)
class MotionResult:
    status: MotionStatus
    message: str = ""
    q_target: np.ndarray | None = None
    T_target: np.ndarray | None = None
    q_final: np.ndarray | None = None
    T_final: np.ndarray | None = None

    @property
    def ok(self) -> bool:
        return self.status == MotionStatus.SUCCESS


@dataclass(frozen=True)
class GripResult:
    success: bool
    contact_detected: bool
    achieved_width: float
    effort: float | None = None
    message: str = ""


@dataclass(frozen=True)
class ComplianceSettings:
    stiffness_xyz: tuple[float, float, float]
    stiffness_rpy: tuple[float, float, float]
    damping_xyz: tuple[float, float, float] | None = None
    damping_rpy: tuple[float, float, float] | None = None

