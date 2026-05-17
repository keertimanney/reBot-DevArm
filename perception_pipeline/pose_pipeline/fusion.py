"""Cross-view association and SE(3) fusion.

When both top and wrist cameras see the same object, fuse their pose estimates
using a weighted mean in the log-SE(3) tangent space. Weight on FoundationPose
confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import numpy as np

from pose_pipeline.types import PoseEstimate


@dataclass
class Association:
    """One physical object as seen across views."""
    cad_id: str
    T_base_top: np.ndarray | None = None
    T_base_wrist: np.ndarray | None = None
    conf_top: float = 0.0
    conf_wrist: float = 0.0
    per_view: list[PoseEstimate] = field(default_factory=list)


def associate_across_views(
    top: list[tuple[str, np.ndarray, PoseEstimate]],
    wrist: list[tuple[str, np.ndarray, PoseEstimate]],
    distance_threshold_m: float = 0.05,
) -> list[Association]:
    """Pair up top and wrist detections that point at the same physical object.

    Strategy: for each top detection, find the wrist detection of the same
    cad_id whose base-frame translation is closest, within `distance_threshold_m`.
    Unmatched detections on either side are kept as solo associations.
    """
    associations: list[Association] = []
    wrist_used = [False] * len(wrist)

    for cad_id_t, T_base_t, est_t in top:
        best_j = -1
        best_dist = np.inf
        for j, (cad_id_w, T_base_w, est_w) in enumerate(wrist):
            if wrist_used[j] or cad_id_w != cad_id_t:
                continue
            d = float(np.linalg.norm(T_base_t[:3, 3] - T_base_w[:3, 3]))
            if d < best_dist:
                best_dist = d
                best_j = j

        if best_j >= 0 and best_dist < distance_threshold_m:
            cad_id_w, T_base_w, est_w = wrist[best_j]
            wrist_used[best_j] = True
            associations.append(Association(
                cad_id=cad_id_t,
                T_base_top=T_base_t,
                T_base_wrist=T_base_w,
                conf_top=est_t.confidence,
                conf_wrist=est_w.confidence,
                per_view=[est_t, est_w],
            ))
        else:
            associations.append(Association(
                cad_id=cad_id_t, T_base_top=T_base_t,
                conf_top=est_t.confidence, per_view=[est_t],
            ))

    for j, used in enumerate(wrist_used):
        if not used:
            cad_id_w, T_base_w, est_w = wrist[j]
            associations.append(Association(
                cad_id=cad_id_w, T_base_wrist=T_base_w,
                conf_wrist=est_w.confidence, per_view=[est_w],
            ))

    return associations


def fuse_pose_estimates(
    assoc: Association,
) -> tuple[np.ndarray, float, Literal["top", "wrist", "fused"]]:
    """Return (T_base_obj, confidence, source_view) from an association.

    Rules:
        - Only one view present → return that one.
        - Both present, wrist conf >> top conf → trust wrist outright.
        - Otherwise → weighted SE(3) mean.
    """
    if assoc.T_base_top is None:
        return assoc.T_base_wrist, assoc.conf_wrist, "wrist"
    if assoc.T_base_wrist is None:
        return assoc.T_base_top, assoc.conf_top, "top"

    if assoc.conf_wrist > 2.0 * assoc.conf_top:
        return assoc.T_base_wrist, assoc.conf_wrist, "wrist"

    w_top = assoc.conf_top ** 2
    w_wrist = assoc.conf_wrist ** 2
    fused_T = se3_weighted_mean(
        [(assoc.T_base_top, w_top), (assoc.T_base_wrist, w_wrist)]
    )
    fused_conf = max(assoc.conf_top, assoc.conf_wrist)
    return fused_T, fused_conf, "fused"


def se3_weighted_mean(weighted: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """Weighted mean in the log-SE(3) tangent space, iterated to convergence.

    Translations average linearly. Rotations average in so(3) via the
    Karcher mean on SO(3): iteratively pick a reference, average log-maps
    around it, exp back. Two or three iterations are enough for our case.
    """
    # TODO: implement Karcher mean. Sketch:
    #   t = sum(w_i * t_i) / sum(w_i)
    #   R_ref = R_0
    #   for _ in range(3):
    #       deltas = [w_i * log_so3(R_ref.T @ R_i) for ...]
    #       R_ref = R_ref @ exp_so3(sum(deltas) / sum(weights))
    #   stack into T
    raise NotImplementedError(
        "Implement Karcher mean on SO(3). scipy.spatial.transform.Rotation "
        "gives you log/exp via Rotation.from_matrix / .as_rotvec / .from_rotvec."
    )
