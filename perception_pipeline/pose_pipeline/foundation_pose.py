"""FoundationPose wrapper.

Model-based mode: given an RGB image, depth, mask, CAD mesh, and intrinsics,
returns a 4x4 SE(3) pose in camera frame plus a confidence score.

Thin wrapper around https://github.com/NVlabs/FoundationPose.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


class FoundationPoseWrapper:
    """Lazy-loaded FoundationPose model.

    First call to register() loads the network and compiles the CUDA renderer.
    Subsequent calls are fast.
    """

    def __init__(self, iterations: int = 5, device: str = "cuda"):
        self.iterations = iterations
        self.device = device
        self._estimator = None
        self._cached_meshes: dict[str, object] = {}

    def _ensure_loaded(self):
        if self._estimator is not None:
            return
        # TODO: from foundationpose import FoundationPose as FP
        # self._estimator = FP(...)
        raise NotImplementedError(
            "Wire up FoundationPose. Their API is roughly:\n"
            "  estimator = FoundationPose(model_pts=..., model_normals=..., \n"
            "                             scorer=..., refiner=..., glctx=...)\n"
            "  pose = estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,\n"
            "                            iteration=self.iterations)\n"
            "  score = estimator.score (or via estimator.pose_last)\n"
            "See their run_demo.py for the canonical init pattern."
        )

    def register(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        mesh: object,                  # trimesh.Trimesh
        K: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Estimate pose of `mesh` in camera frame.

        Returns:
            (T_cam_obj 4x4, score in roughly [0, 1]).
        """
        self._ensure_loaded()
        raise NotImplementedError

    def refine(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        mesh: object,
        K: np.ndarray,
        prior_pose_cam: np.ndarray,
        depth: np.ndarray | None = None,
        iterations: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """Refine an existing pose estimate. Used by PoseEstimator.refine_close_range().

        At close range with a known mesh, depth can be None and the refiner
        uses RGB only.
        """
        self._ensure_loaded()
        raise NotImplementedError
