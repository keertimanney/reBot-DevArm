"""Overlay detection masks and pose axes on captured frames for debugging."""
from __future__ import annotations

# TODO: implement.
#   - draw bounding boxes + mask alpha overlay
#   - project the object's local axes through K and overlay
#   - save side-by-side image (top | wrist) with annotations
#   - useful when debugging "the pose looks ~right but the gripper missed by 1cm"

def draw_pose_axes(image, T_cam_obj, K, length_m=0.03):
    """Draw the XYZ axes of T_cam_obj projected through K onto `image` in-place."""
    raise NotImplementedError


def overlay_detection(image, mask, color=(0, 255, 0), alpha=0.4):
    """Blend a mask onto an image with the given color and alpha."""
    raise NotImplementedError
