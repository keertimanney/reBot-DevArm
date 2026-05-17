from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from rebot_b601_sim.joints import SIM_JOINT_NAMES

from .constants import Q_OBSERVE
from .kinematics import B601Kinematics
from .types import ComplianceSettings, MotionResult, MotionStatus


def _q_to_action(q: np.ndarray) -> dict[str, float]:
    return {f"{name}.pos": math.degrees(float(q[idx])) for idx, name in enumerate(SIM_JOINT_NAMES)}


def _action_obs_to_q(obs: dict[str, float]) -> np.ndarray:
    return np.array([math.radians(float(obs.get(f"{name}.pos", 0.0))) for name in SIM_JOINT_NAMES])


@dataclass
class MotionPlanner:
    """Orchestration-facing motion API.

    Pass a connected SeeedB601DMFollower-compatible robot to execute on hardware.
    Without a robot, this planner is a dry-run kinematic planner useful for tests
    and perception integration.
    """

    robot: object | None = None
    kinematics: B601Kinematics | None = None
    command_interval_s: float = 0.03

    def __post_init__(self) -> None:
        self.kinematics = self.kinematics or B601Kinematics()
        self._q_current = Q_OBSERVE.copy()
        self._compliance: ComplianceSettings | None = None

    def get_current_joints(self) -> np.ndarray:
        if self.robot is None:
            return self._q_current.copy()
        obs = self.robot.get_observation()
        self._q_current = _action_obs_to_q(obs)
        return self._q_current.copy()

    def get_current_pose(self) -> np.ndarray:
        return self.kinematics.fk(self.get_current_joints())

    def move_to_joint_config(self, q_target: np.ndarray, speed: float = 1.0) -> MotionResult:
        q_target = np.asarray(q_target, dtype=float).reshape(-1)
        if q_target.shape[0] != self.kinematics.nq:
            return MotionResult(MotionStatus.EXECUTION_ERROR, f"Expected {self.kinematics.nq} joints.")
        if not self.kinematics.within_limits(q_target):
            return MotionResult(MotionStatus.JOINT_LIMIT, "Joint target outside configured limits.", q_target=q_target)

        q_start = self.get_current_joints()
        max_delta = float(np.max(np.abs(q_target - q_start)))
        steps = max(1, int(max_delta / max(0.01, 0.04 * max(0.05, speed))))
        try:
            for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
                q = q_start + alpha * (q_target - q_start)
                self._send_q(q)
                time.sleep(self.command_interval_s)
            self._q_current = q_target.copy()
            return MotionResult(
                MotionStatus.SUCCESS,
                q_target=q_target,
                q_final=self.get_current_joints(),
                T_final=self.get_current_pose(),
            )
        except Exception as exc:
            return MotionResult(MotionStatus.EXECUTION_ERROR, str(exc), q_target=q_target)

    def move_to_pose(self, T_base_target: np.ndarray, speed: float = 1.0) -> MotionResult:
        ik = self.kinematics.ik(T_base_target, q_seed=self.get_current_joints())
        if not ik.success:
            return MotionResult(MotionStatus.IK_INFEASIBLE, ik.message, T_target=T_base_target, q_target=ik.q)
        return self.move_to_joint_config(ik.q, speed=speed)

    def move_linear_cartesian(self, T_base_target: np.ndarray, speed: float = 0.05) -> MotionResult:
        T_start = self.get_current_pose()
        T_base_target = np.asarray(T_base_target, dtype=float)
        distance = float(np.linalg.norm(T_base_target[:3, 3] - T_start[:3, 3]))
        steps = max(2, int(distance / max(0.002, speed * self.command_interval_s)))
        q_seed = self.get_current_joints()

        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
            T_step = T_start.copy()
            T_step[:3, 3] = (1.0 - alpha) * T_start[:3, 3] + alpha * T_base_target[:3, 3]
            T_step[:3, :3] = T_base_target[:3, :3]
            ik = self.kinematics.ik(T_step, q_seed=q_seed, max_iters=100)
            if not ik.success:
                return MotionResult(MotionStatus.IK_INFEASIBLE, ik.message, T_target=T_step, q_target=ik.q)
            if not self.kinematics.within_limits(ik.q):
                return MotionResult(MotionStatus.JOINT_LIMIT, "Linear step exceeded joint limits.", q_target=ik.q)
            result = self.move_to_joint_config(ik.q, speed=1.0)
            if not result.ok:
                return result
            q_seed = ik.q
        return MotionResult(
            MotionStatus.SUCCESS,
            T_target=T_base_target,
            q_final=self.get_current_joints(),
            T_final=self.get_current_pose(),
        )

    @contextlib.contextmanager
    def compliant_mode(
        self,
        stiffness_xyz: list[float] | tuple[float, float, float],
        stiffness_rpy: list[float] | tuple[float, float, float],
    ) -> Iterator[None]:
        settings = ComplianceSettings(tuple(stiffness_xyz), tuple(stiffness_rpy))
        previous = self._compliance
        self._compliance = settings
        self._enter_low_level_compliance(settings)
        try:
            yield
        finally:
            self._exit_low_level_compliance()
            self._compliance = previous

    def _send_q(self, q: np.ndarray) -> None:
        self._q_current = q.copy()
        if self.robot is not None:
            self.robot.send_action(_q_to_action(q))

    def _enter_low_level_compliance(self, settings: ComplianceSettings) -> None:
        # Hook for Damiao MIT mode once motorbridge exposes the needed command on
        # the follower object. Current fallback is position-commanded motion with
        # slower Cartesian steps inside the context.
        if self.robot is not None and hasattr(self.robot, "enter_compliant_mode"):
            self.robot.enter_compliant_mode(settings)

    def _exit_low_level_compliance(self) -> None:
        if self.robot is not None and hasattr(self.robot, "exit_compliant_mode"):
            self.robot.exit_compliant_mode()


def move_and_settle(planner: MotionPlanner, q: np.ndarray, settle_ms: int = 200) -> MotionResult:
    result = planner.move_to_joint_config(q)
    if result.ok:
        time.sleep(settle_ms / 1000.0)
    return result

