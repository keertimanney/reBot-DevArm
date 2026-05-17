"""Public dataclasses returned by the pose estimator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np


@dataclass
class PoseEstimate:
    """Single-view pose result from one camera, before fusion."""
    T_cam_obj: np.ndarray              # 4x4, in camera frame
    confidence: float                  # FoundationPose ranking score, [0, 1] ish
    view: Literal["top", "wrist"]
    mask: Optional[np.ndarray] = None  # H x W bool
    bbox_xyxy: Optional[tuple] = None  # (x1, y1, x2, y2)


@dataclass
class DetectedObject:
    """Final pose in robot base frame, what the rest of the system consumes."""
    label: str                         # e.g. "red_2x4_brick" — passed through from query
    cad_id: str                        # which CAD mesh matched best
    T_base_obj: np.ndarray             # 4x4 SE(3) in robot base frame
    confidence: float                  # fused / best-view confidence
    source_view: Literal["top", "wrist", "fused"]
    per_view: list[PoseEstimate] = field(default_factory=list)

    def __repr__(self) -> str:
        t = self.T_base_obj[:3, 3]
        return (
            f"DetectedObject(label={self.label!r}, cad_id={self.cad_id!r}, "
            f"xyz=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}), "
            f"conf={self.confidence:.2f}, src={self.source_view})"
        )
