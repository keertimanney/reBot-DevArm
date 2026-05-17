from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np

from .constants import Q_OBSERVE
from .gripper import Gripper
from .kinematics import fk
from .planner import MotionPlanner
from .types import MotionStatus


class PoseEstimatorProtocol(Protocol):
    def estimate(
        self,
        rgb_top: np.ndarray,
        rgb_wrist: np.ndarray,
        T_base_tool: np.ndarray,
        query: str,
        candidate_cad_ids: list[str],
    ) -> list[Any]:
        ...

    def refine_close_range(
        self,
        rgb_wrist: np.ndarray,
        T_base_tool: np.ndarray,
        prior_T_base_obj: np.ndarray,
        cad_id: str,
    ) -> Any:
        ...


class CameraProviderProtocol(Protocol):
    def read_top(self) -> np.ndarray:
        ...

    def read_wrist(self) -> np.ndarray:
        ...


class PickPlaceStatus(str, Enum):
    SUCCESS = "SUCCESS"
    NO_DETECTION = "NO_DETECTION"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MOTION_FAILED = "MOTION_FAILED"
    GRASP_FAILED = "GRASP_FAILED"
    VERIFY_FAILED = "VERIFY_FAILED"


@dataclass(frozen=True)
class PickPlaceResult:
    status: PickPlaceStatus
    message: str = ""
    detected_object: Any | None = None
    refined_object: Any | None = None
    motion_status: MotionStatus | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == PickPlaceStatus.SUCCESS


def approach_offset(distance_m: float) -> np.ndarray:
    T = np.eye(4)
    T[2, 3] = distance_m
    return T


class PickAndPlaceSkill:
    """Main orchestration shell for perception -> motion -> gripper execution."""

    def __init__(
        self,
        pose_estimator: PoseEstimatorProtocol,
        camera_provider: CameraProviderProtocol,
        planner: MotionPlanner,
        gripper: Gripper,
        min_detection_confidence: float = 0.35,
    ) -> None:
        self.pose_estimator = pose_estimator
        self.camera_provider = camera_provider
        self.planner = planner
        self.gripper = gripper
        self.min_detection_confidence = min_detection_confidence

    def pick_and_place(
        self,
        query: str,
        T_base_place: np.ndarray,
        candidate_cad_ids: list[str],
        grasp_site_offset: np.ndarray | None = None,
        approach_distance_m: float = 0.10,
    ) -> PickPlaceResult:
        grasp_site_offset = np.eye(4) if grasp_site_offset is None else grasp_site_offset
        approach = approach_offset(approach_distance_m)

        result = self.planner.move_to_joint_config(Q_OBSERVE)
        if not result.ok:
            return PickPlaceResult(PickPlaceStatus.MOTION_FAILED, result.message, motion_status=result.status)

        rgb_top = self.camera_provider.read_top()
        rgb_wrist = self.camera_provider.read_wrist()
        detections = self.pose_estimator.estimate(
            rgb_top,
            rgb_wrist,
            fk(self.planner.get_current_joints()),
            query,
            candidate_cad_ids,
        )
        if not detections:
            return PickPlaceResult(PickPlaceStatus.NO_DETECTION, "No object detection returned.")

        obj = max(detections, key=lambda detection: float(detection.confidence))
        if float(obj.confidence) < self.min_detection_confidence:
            return PickPlaceResult(PickPlaceStatus.LOW_CONFIDENCE, "Best detection below confidence threshold.", obj)

        T_pre_grasp = obj.T_base_obj @ grasp_site_offset @ approach
        result = self.planner.move_to_pose(T_pre_grasp)
        if not result.ok:
            return PickPlaceResult(PickPlaceStatus.MOTION_FAILED, result.message, obj, motion_status=result.status)

        obj_refined = self.pose_estimator.refine_close_range(
            self.camera_provider.read_wrist(),
            fk(self.planner.get_current_joints()),
            obj.T_base_obj,
            obj.cad_id,
        )
        T_grasp = obj_refined.T_base_obj @ grasp_site_offset
        T_pre_grasp = obj_refined.T_base_obj @ grasp_site_offset @ approach

        with self.planner.compliant_mode(stiffness_xyz=(200, 200, 50), stiffness_rpy=(10, 10, 5)):
            result = self.planner.move_linear_cartesian(T_grasp)
        if not result.ok:
            return PickPlaceResult(PickPlaceStatus.MOTION_FAILED, result.message, obj, obj_refined, result.status)

        grip = self.gripper.close()
        if not grip.success:
            return PickPlaceResult(
                PickPlaceStatus.GRASP_FAILED,
                grip.message,
                obj,
                obj_refined,
                metadata={"grip": grip},
            )

        result = self.planner.move_linear_cartesian(T_pre_grasp)
        if not result.ok:
            return PickPlaceResult(PickPlaceStatus.MOTION_FAILED, result.message, obj, obj_refined, result.status)

        T_pre_place = T_base_place @ approach
        result = self.planner.move_to_pose(T_pre_place)
        if not result.ok:
            return PickPlaceResult(PickPlaceStatus.MOTION_FAILED, result.message, obj, obj_refined, result.status)

        with self.planner.compliant_mode(stiffness_xyz=(200, 200, 50), stiffness_rpy=(10, 10, 5)):
            result = self.planner.move_linear_cartesian(T_base_place)
        if not result.ok:
            return PickPlaceResult(PickPlaceStatus.MOTION_FAILED, result.message, obj, obj_refined, result.status)

        self.gripper.open()
        result = self.planner.move_linear_cartesian(T_pre_place)
        if not result.ok:
            return PickPlaceResult(PickPlaceStatus.MOTION_FAILED, result.message, obj, obj_refined, result.status)

        return PickPlaceResult(PickPlaceStatus.SUCCESS, detected_object=obj, refined_object=obj_refined)

