from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rebot_b601_sim.pinocchio_meshcat import (
    default_control_repo,
    find_urdf,
)

from .constants import JOINT_LIMITS_RAD, JOINT_ORDER, Q_OBSERVE


def _rotation_error(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    import pinocchio as pin

    return pin.log3(R_current.T @ R_target)


@dataclass
class IKResult:
    success: bool
    q: np.ndarray
    error_norm: float
    iterations: int
    message: str = ""


class B601Kinematics:
    def __init__(
        self,
        control_repo: Path | None = None,
        urdf: str | None = None,
        tool_frame: str = "end_link",
    ) -> None:
        import pinocchio as pin

        self.pin = pin
        self.control_repo = (control_repo or default_control_repo()).resolve()
        self.urdf_path = find_urdf(self.control_repo, urdf)
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self.tool_frame = tool_frame
        self.tool_frame_id = self.model.getFrameId(tool_frame)
        if self.tool_frame_id >= len(self.model.frames):
            raise ValueError(f"Tool frame {tool_frame!r} not found in {self.urdf_path}")

    @property
    def nq(self) -> int:
        return self.model.nq

    def neutral(self) -> np.ndarray:
        return np.array(self.pin.neutral(self.model), dtype=float)

    def fk(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != self.model.nq:
            raise ValueError(f"Expected q with {self.model.nq} joints, got {q.shape[0]}.")
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        return np.array(self.data.oMf[self.tool_frame_id].homogeneous)

    def within_limits(self, q: np.ndarray) -> bool:
        for idx, joint_name in enumerate(JOINT_ORDER):
            lo, hi = JOINT_LIMITS_RAD[joint_name]
            if not lo <= float(q[idx]) <= hi:
                return False
        return True

    def clip_to_limits(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).copy()
        for idx, joint_name in enumerate(JOINT_ORDER):
            lo, hi = JOINT_LIMITS_RAD[joint_name]
            q[idx] = np.clip(q[idx], lo, hi)
        return q

    def ik(
        self,
        T_base_target: np.ndarray,
        q_seed: np.ndarray | None = None,
        max_iters: int = 200,
        tol: float = 1e-4,
        dt: float = 0.35,
        damping: float = 1e-6,
    ) -> IKResult:
        T_base_target = np.asarray(T_base_target, dtype=float)
        if T_base_target.shape != (4, 4):
            raise ValueError("T_base_target must be 4x4.")

        q = np.array(q_seed if q_seed is not None else Q_OBSERVE, dtype=float).reshape(-1)
        q = self.clip_to_limits(q)
        target_t = T_base_target[:3, 3]
        target_R = T_base_target[:3, :3]

        last_error_norm = float("inf")
        for it in range(max_iters):
            self.pin.forwardKinematics(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)
            placement = self.data.oMf[self.tool_frame_id]
            current_t = np.array(placement.translation)
            current_R = np.array(placement.rotation)
            err = np.concatenate((target_t - current_t, _rotation_error(current_R, target_R)))
            last_error_norm = float(np.linalg.norm(err))
            if last_error_norm < tol:
                return IKResult(True, q, last_error_norm, it)

            J = self.pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.tool_frame_id,
                self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + damping * np.eye(6), err)
            q = self.clip_to_limits(q + dt * dq[: self.model.nv])

        return IKResult(False, q, last_error_norm, max_iters, "IK did not converge.")


_DEFAULT_KINEMATICS: B601Kinematics | None = None


def default_kinematics() -> B601Kinematics:
    global _DEFAULT_KINEMATICS
    if _DEFAULT_KINEMATICS is None:
        _DEFAULT_KINEMATICS = B601Kinematics()
    return _DEFAULT_KINEMATICS


def fk(q: np.ndarray) -> np.ndarray:
    return default_kinematics().fk(q)
