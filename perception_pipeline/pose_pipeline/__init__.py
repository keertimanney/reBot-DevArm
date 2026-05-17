"""pose_pipeline — 6D pose estimation for CAD-known blocks in robot base frame."""

from pose_pipeline.pose_estimator import PoseEstimator
from pose_pipeline.types import DetectedObject, PoseEstimate

__all__ = ["PoseEstimator", "DetectedObject", "PoseEstimate"]
__version__ = "0.0.1"
