"""Monocular metric depth.

Depth Anything V2 outputs relative depth (up to scale and shift). We recover
metric depth by fitting the dominant ground plane to a known z-value in robot
base frame.
"""
from __future__ import annotations

import numpy as np


class MonoDepth:
    """Lazy-loaded Depth Anything V2."""

    def __init__(self, model: str = "depth-anything/Depth-Anything-V2-Base", device: str = "cuda"):
        self.model_name = model
        self.device = device
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        # TODO: load via transformers `pipeline("depth-estimation", model=...)`
        # or the standalone DepthAnythingV2 class from the official repo.
        raise NotImplementedError("Wire up Depth Anything V2 loading.")

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Return H x W float32 relative depth (larger = farther OR closer
        depending on the model variant — check before use)."""
        self._ensure_loaded()
        raise NotImplementedError


def fit_table_plane_scale(
    rel_depth: np.ndarray,
    object_masks: list[np.ndarray],
    K: np.ndarray,
    T_base_cam: np.ndarray,
    table_z_in_base: float,
    ransac_iters: int = 200,
    ransac_thresh: float = 0.01,
) -> np.ndarray:
    """Scale relative depth to metric using the known table plane.

    Strategy:
      1. Mask out object pixels (keep only table).
      2. Lift remaining pixels into the camera frame using `rel_depth` as
         placeholder depth (gives a point cloud up to scale).
      3. Fit a plane via RANSAC.
      4. Transform the plane into base frame; the offset along base z should be
         `table_z_in_base`. Solve for the scale factor that makes this true.
      5. Multiply rel_depth by that scale → metric depth in camera frame.

    Args:
        rel_depth: H x W relative depth from Depth Anything V2.
        object_masks: list of H x W bool masks to exclude (objects above table).
        K: 3x3 camera intrinsics.
        T_base_cam: 4x4 camera pose in base frame.
        table_z_in_base: z of the table surface in base frame (e.g. 0.0 if
            base origin sits at table height; -0.05 if base is 5cm above table).
        ransac_iters / ransac_thresh: RANSAC params for plane fit.

    Returns:
        H x W float32 metric depth in camera frame.
    """
    # TODO: implement.
    # Key steps to fill in:
    #   - build a "not object" mask: ~ logical_or(*object_masks)
    #   - sample pixels uniformly on the table region
    #   - back-project: p_cam = depth_rel * K^-1 @ [u, v, 1].T
    #   - RANSAC for plane: ax + by + cz + d = 0 in camera frame
    #   - scale = table_z_in_base / (T_base_cam @ plane_point_at_unit_depth).z
    #     (work out the algebra cleanly — there's a closed form)
    raise NotImplementedError(
        "Implement table-plane scaling. The unit test in tests/test_depth.py "
        "(TODO) should verify on a synthetic image with known plane."
    )
