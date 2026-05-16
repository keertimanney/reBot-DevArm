"""
Embodiment translation layer: reBot <-> DROID (Franka Panda)

Converts between reBot and Franka joint spaces using end-effector pose
as the common intermediate representation.

Pipeline:
  Observation: reBot joints → reBot FK → EE pose → Franka IK → Franka joints → pi0.5
  Action:      pi0.5 → Franka joint delta → Franka FK → EE pose → reBot IK → reBot joints
"""

import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from typing import Optional, Tuple

from robot_descriptions.loaders.pinocchio import load_robot_description


@dataclass
class EEPose:
    """End-effector pose in Cartesian space."""
    position: np.ndarray   # (3,) xyz in meters
    rotation: np.ndarray   # (3,3) rotation matrix


class FrankaModel:
    """Franka Panda kinematics via Pinocchio."""

    # Franka joint limits (radians)
    JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    # DROID home position (from libero/openpi)
    HOME_Q = np.array([0.0, -0.161, 0.0, -2.445, 0.0, 2.227, 0.785])

    def __init__(self):
        self.model = load_robot_description("panda_description")
        self.data = self.model.createData()

        # Find end-effector frame (panda_link8 or similar)
        self.ee_frame_id = None
        for i in range(self.model.nframes):
            name = self.model.frames[i].name
            if "link8" in name or name == "panda_hand":
                self.ee_frame_id = i
                break
        if self.ee_frame_id is None:
            # Fallback: use last frame
            self.ee_frame_id = self.model.nframes - 1

    def fk(self, q: np.ndarray) -> EEPose:
        """Forward kinematics: 7 joint angles → end-effector pose."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.ee_frame_id]
        return EEPose(
            position=oMf.translation.copy(),
            rotation=oMf.rotation.copy(),
        )

    def ik(self, target: EEPose, q_init: Optional[np.ndarray] = None,
           max_iter: int = 1000, tol: float = 1e-4, damping: float = 1e-6) -> Tuple[np.ndarray, bool]:
        """Inverse kinematics: end-effector pose → 7 joint angles.
        Returns (q, success)."""
        if q_init is None:
            q_init = self.HOME_Q.copy()

        oMdes = pin.SE3(target.rotation, target.position)
        q = q_init.copy()

        for i in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            oMf = self.data.oMf[self.ee_frame_id]

            err = pin.log6(oMf.actInv(oMdes)).vector
            if np.linalg.norm(err) < tol:
                return q, True

            J = pin.computeFrameJacobian(self.model, self.data, q,
                                         self.ee_frame_id, pin.LOCAL)
            # Damped least squares
            JtJ = J.T @ J + damping * np.eye(self.model.nv)
            dq = np.linalg.solve(JtJ, J.T @ err)
            q = pin.integrate(self.model, q, dq)

            # Clamp to joint limits
            q = np.clip(q, self.JOINT_LOWER, self.JOINT_UPPER)

        return q, False


class ReBotModel:
    """reBot B601 DM kinematics via Pinocchio."""

    # reBot joint limits (radians, from URDF)
    JOINT_LOWER = np.array([-2.80, -3.14, -3.14, -1.87, -1.57, -3.14])
    JOINT_UPPER = np.array([2.80, 0.00, 0.00, 1.57, 1.57, 3.14])

    HOME_Q = np.zeros(6)

    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # Find end-effector frame
        self.ee_frame_id = None
        for i in range(self.model.nframes):
            name = self.model.frames[i].name
            if "end_link" in name:
                self.ee_frame_id = i
                break
        if self.ee_frame_id is None:
            self.ee_frame_id = self.model.nframes - 1

    def fk(self, q: np.ndarray) -> EEPose:
        """Forward kinematics: 6 joint angles → end-effector pose."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.ee_frame_id]
        return EEPose(
            position=oMf.translation.copy(),
            rotation=oMf.rotation.copy(),
        )

    def ik(self, target: EEPose, q_init: Optional[np.ndarray] = None,
           max_iter: int = 1000, tol: float = 1e-4, damping: float = 1e-6) -> Tuple[np.ndarray, bool]:
        """Inverse kinematics: end-effector pose → 6 joint angles.
        Returns (q, success)."""
        if q_init is None:
            q_init = self.HOME_Q.copy()

        oMdes = pin.SE3(target.rotation, target.position)
        q = q_init.copy()

        for i in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            oMf = self.data.oMf[self.ee_frame_id]

            err = pin.log6(oMf.actInv(oMdes)).vector
            if np.linalg.norm(err) < tol:
                return q, True

            J = pin.computeFrameJacobian(self.model, self.data, q,
                                         self.ee_frame_id, pin.LOCAL)
            JtJ = J.T @ J + damping * np.eye(self.model.nv)
            dq = np.linalg.solve(JtJ, J.T @ err)
            q = pin.integrate(self.model, q, dq)

            # Clamp to joint limits
            q = np.clip(q, self.JOINT_LOWER, self.JOINT_UPPER)

        return q, False


class EmbodimentTranslator:
    """
    Translates between reBot and Franka (DROID) joint spaces via end-effector pose.

    Usage:
        translator = EmbodimentTranslator(rebot_urdf_path="path/to/reBot.urdf")

        # Observation: reBot → Franka (for sending to pi0.5)
        franka_joints = translator.rebot_to_franka(rebot_joint_positions)

        # Action: Franka → reBot (for executing pi0.5 output)
        rebot_joints = translator.franka_to_rebot(franka_joint_targets, current_rebot_joints)
    """

    def __init__(self, rebot_urdf_path: str):
        self.franka = FrankaModel()
        self.rebot = ReBotModel(rebot_urdf_path)

        # Compute workspace scaling: ratio of arm reaches
        # Franka reach ~0.855m, reBot reach ~0.650m
        franka_home_ee = self.franka.fk(self.franka.HOME_Q)
        rebot_home_ee = self.rebot.fk(self.rebot.HOME_Q)
        self._franka_home_pos = franka_home_ee.position
        self._rebot_home_pos = rebot_home_ee.position

        # Scale factor: map reBot workspace → Franka workspace
        self._scale_rebot_to_franka = 0.855 / 0.650
        self._scale_franka_to_rebot = 0.650 / 0.855

        # Track last known states for IK seeding
        self._last_franka_q = self.franka.HOME_Q.copy()
        self._last_rebot_q = self.rebot.HOME_Q.copy()

    def _scale_position(self, pos: np.ndarray, source_home: np.ndarray,
                        target_home: np.ndarray, scale: float) -> np.ndarray:
        """Scale a position from one robot's workspace to another's."""
        offset = pos - source_home
        return target_home + offset * scale

    def rebot_to_franka(self, rebot_q: np.ndarray) -> np.ndarray:
        """
        Convert reBot joint positions → Franka joint positions.
        Used for building observations to send to pi0.5.

        reBot joints (6) → reBot FK → EE pose → scale to Franka workspace → Franka IK → Franka joints (7)
        """
        # reBot FK
        ee_pose = self.rebot.fk(rebot_q)

        # Scale position from reBot workspace to Franka workspace
        scaled_pos = self._scale_position(
            ee_pose.position, self._rebot_home_pos,
            self._franka_home_pos, self._scale_rebot_to_franka,
        )
        target = EEPose(position=scaled_pos, rotation=ee_pose.rotation)

        # Franka IK
        franka_q, success = self.franka.ik(target, q_init=self._last_franka_q)
        if success:
            self._last_franka_q = franka_q.copy()
        else:
            print(f"Warning: Franka IK failed, using last known state")

        return franka_q

    def franka_to_rebot(self, franka_q: np.ndarray,
                        current_rebot_q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert Franka joint positions → reBot joint positions.
        Used for executing pi0.5 action outputs on the reBot arm.

        Franka joints (7) → Franka FK → EE pose → scale to reBot workspace → reBot IK → reBot joints (6)
        """
        # Franka FK
        ee_pose = self.franka.fk(franka_q)

        # Scale position from Franka workspace to reBot workspace
        scaled_pos = self._scale_position(
            ee_pose.position, self._franka_home_pos,
            self._rebot_home_pos, self._scale_franka_to_rebot,
        )
        target = EEPose(position=scaled_pos, rotation=ee_pose.rotation)

        # reBot IK, seeded from current position
        seed = current_rebot_q if current_rebot_q is not None else self._last_rebot_q
        rebot_q, success = self.rebot.ik(target, q_init=seed)
        if success:
            self._last_rebot_q = rebot_q.copy()
        else:
            print(f"Warning: reBot IK failed, using last known state")

        return rebot_q

    def translate_action_chunk(self, franka_actions: np.ndarray,
                               franka_state: np.ndarray,
                               current_rebot_q: np.ndarray) -> np.ndarray:
        """
        Translate a full action chunk from pi0.5 (Franka deltas) to reBot joint targets.

        Args:
            franka_actions: (horizon, 8) — delta joint positions from pi0.5 (7 joints + 1 gripper)
            franka_state: (7,) — current Franka joint state (used as base for deltas)
            current_rebot_q: (6,) — current reBot joint positions

        Returns:
            rebot_targets: (horizon, 7) — reBot joint targets (6 joints + 1 gripper)
        """
        horizon = franka_actions.shape[0]
        rebot_targets = np.zeros((horizon, 7))  # 6 joints + 1 gripper

        franka_q = franka_state.copy()
        rebot_q = current_rebot_q.copy()

        for i in range(horizon):
            # Apply delta to get absolute Franka position
            franka_q = franka_q + franka_actions[i, :7]

            # Translate to reBot
            rebot_q = self.franka_to_rebot(franka_q, current_rebot_q=rebot_q)

            rebot_targets[i, :6] = rebot_q
            rebot_targets[i, 6] = franka_actions[i, 7]  # pass gripper through

        return rebot_targets
