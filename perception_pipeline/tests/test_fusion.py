"""Tests for cross-view association and SE(3) fusion."""
from __future__ import annotations

import numpy as np
import pytest

from pose_pipeline.types import PoseEstimate
from pose_pipeline.fusion import associate_across_views, fuse_pose_estimates


def _T(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def test_association_pairs_same_object():
    est_t = PoseEstimate(T_cam_obj=np.eye(4), confidence=0.7, view="top")
    est_w = PoseEstimate(T_cam_obj=np.eye(4), confidence=0.9, view="wrist")
    top = [("brick_2x4_red", _T(0.30, 0.10, 0.02), est_t)]
    wrist = [("brick_2x4_red", _T(0.301, 0.101, 0.021), est_w)]

    assocs = associate_across_views(top, wrist, distance_threshold_m=0.05)
    assert len(assocs) == 1
    assert assocs[0].T_base_top is not None
    assert assocs[0].T_base_wrist is not None


def test_association_separates_different_objects():
    est_t = PoseEstimate(T_cam_obj=np.eye(4), confidence=0.7, view="top")
    est_w = PoseEstimate(T_cam_obj=np.eye(4), confidence=0.9, view="wrist")
    top = [("brick_2x4_red", _T(0.30, 0.10), est_t)]
    wrist = [("brick_2x4_red", _T(0.30, 0.50), est_w)]  # too far

    assocs = associate_across_views(top, wrist, distance_threshold_m=0.05)
    assert len(assocs) == 2


def test_fusion_prefers_high_confidence_view():
    est_t = PoseEstimate(T_cam_obj=np.eye(4), confidence=0.3, view="top")
    est_w = PoseEstimate(T_cam_obj=np.eye(4), confidence=0.95, view="wrist")
    from pose_pipeline.fusion import Association
    assoc = Association(
        cad_id="x", T_base_top=_T(0.1, 0.2),
        T_base_wrist=_T(0.105, 0.205),
        conf_top=0.3, conf_wrist=0.95,
        per_view=[est_t, est_w],
    )
    T, conf, src = fuse_pose_estimates(assoc)
    assert src == "wrist"
    assert np.allclose(T[:3, 3], [0.105, 0.205, 0.0])
