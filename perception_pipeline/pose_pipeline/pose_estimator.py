"""PoseEstimator — the public skill.

Orchestrates: Grounded-SAM 2 → Depth Anything V2 → FoundationPose → frame
transform → optional fusion across top and wrist views.

This file is the public surface. Heavy lifting lives in sibling modules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np

from pose_pipeline.types import DetectedObject, PoseEstimate
from pose_pipeline.grounding import GroundedSAM, PROMPT_GENERIC
from pose_pipeline.depth import MonoDepth, fit_table_plane_scale
from pose_pipeline.foundation_pose import FoundationPoseWrapper
from pose_pipeline.cad import CADLibrary
from pose_pipeline.fusion import associate_across_views, fuse_pose_estimates


class PoseEstimator:
    """6D pose estimation for CAD-known blocks, output in robot base frame.

    The class is stateful only in that it holds loaded models and the fixed
    extrinsics. Each call to `estimate()` or `refine_close_range()` is
    independent.
    """

    def __init__(
        self,
        cad_library_path: str | Path,
        T_base_top: np.ndarray,
        T_tool_wrist: np.ndarray,
        K_top: np.ndarray,
        K_wrist: np.ndarray,
        config: dict,
    ):
        assert T_base_top.shape == (4, 4)
        assert T_tool_wrist.shape == (4, 4)
        assert K_top.shape == (3, 3) and K_wrist.shape == (3, 3)

        self.T_base_top = T_base_top
        self.T_tool_wrist = T_tool_wrist
        self.K_top = K_top
        self.K_wrist = K_wrist
        self.config = config

        self.cad = CADLibrary(cad_library_path)
        self.grounding = GroundedSAM(
            gdino_model=config["models"]["grounding_dino"],
            sam_model=config["models"]["sam2"],
            box_threshold=config["thresholds"]["grounding_box"],
            text_threshold=config["thresholds"]["grounding_text"],
        )
        self.depth = MonoDepth(model=config["models"]["depth_anything"])
        self.foundation_pose = FoundationPoseWrapper(
            iterations=config["foundation_pose"]["refine_iters"],
        )

    # ---------- public API ----------

    def estimate(
        self,
        rgb_top: np.ndarray,
        rgb_wrist: np.ndarray,
        T_base_tool: np.ndarray,
        query: str,
        candidate_cad_ids: list[str],
        prompt_override: Optional[str] = None,
    ) -> list[DetectedObject]:
        """Wide-view scan from the observe pose.

        Args:
            rgb_top:   H×W×3 uint8 image from the fixed top camera.
            rgb_wrist: H×W×3 uint8 image from the wrist camera (at observe pose).
            T_base_tool: 4×4 tool flange pose from FK at the moment of capture.
            query: Natural language description, passes through to label field
                   and is used as a fallback prompt when prompt_override is None.
            candidate_cad_ids: Which CAD meshes to try matching. Keep this short
                   (3-10) for sensible latency.
            prompt_override: Replace the generic prompt with a custom one. Use
                   sparingly; the default generic prompt is preferred.

        Returns:
            List of DetectedObject in robot base frame. Empty list if nothing
            grounded, never raises.
        """
        prompt = prompt_override or PROMPT_GENERIC

        # 1. Ground objects in both views.
        det_top = self.grounding.detect(rgb_top, prompt)
        det_wrist = self.grounding.detect(rgb_wrist, prompt)

        if not det_top and not det_wrist:
            return []

        # 2. Metric depth for each view (used by FoundationPose).
        depth_top = self._metric_depth_top(rgb_top, [d.mask for d in det_top])
        depth_wrist = self._metric_depth_wrist(rgb_wrist, [d.mask for d in det_wrist])

        # 3. Per-view pose estimation: try each candidate CAD against each mask,
        #    keep the best CAD per mask.
        estimates_top = self._estimate_poses(
            rgb_top, depth_top, det_top, candidate_cad_ids,
            K=self.K_top, view="top",
        )
        estimates_wrist = self._estimate_poses(
            rgb_wrist, depth_wrist, det_wrist, candidate_cad_ids,
            K=self.K_wrist, view="wrist",
        )

        # 4. Lift each per-view pose into base frame.
        T_base_wrist = T_base_tool @ self.T_tool_wrist
        per_view_base_top = [
            (cad_id, self.T_base_top @ est.T_cam_obj, est)
            for cad_id, est in estimates_top
        ]
        per_view_base_wrist = [
            (cad_id, T_base_wrist @ est.T_cam_obj, est)
            for cad_id, est in estimates_wrist
        ]

        # 5. Associate same-object detections across views and fuse.
        return self._fuse_and_package(
            per_view_base_top, per_view_base_wrist, query=query,
        )

    def refine_close_range(
        self,
        rgb_wrist: np.ndarray,
        T_base_tool: np.ndarray,
        prior_T_base_obj: np.ndarray,
        cad_id: str,
    ) -> DetectedObject:
        """High-precision refinement after the arm has approached the object.

        Uses only the wrist camera. Skips Depth Anything (at close range with a
        known mesh, FoundationPose's RGB mode is fine). Initializes from prior.
        """
        raise NotImplementedError("TODO: implement using FoundationPose refine() with prior")

    # ---------- internal ----------

    def _metric_depth_top(self, rgb: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
        """Run Depth Anything V2 on top view and scale via table plane fit."""
        rel = self.depth.predict(rgb)
        return fit_table_plane_scale(
            rel_depth=rel,
            object_masks=masks,
            K=self.K_top,
            T_base_cam=self.T_base_top,
            table_z_in_base=self.config["workspace"]["table_z"],
        )

    def _metric_depth_wrist(self, rgb: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
        """Depth for wrist view at observe pose.

        Simpler than top: assume the workspace is at a known z and the wrist is
        looking roughly downward. For close-range refine() we skip depth.
        """
        raise NotImplementedError(
            "TODO: choose strategy — table plane fit (if wrist sees ground) "
            "or skip + RGB-only FoundationPose."
        )

    def _estimate_poses(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        detections: list,
        candidate_cad_ids: list[str],
        K: np.ndarray,
        view: str,
    ) -> list[tuple[str, PoseEstimate]]:
        """For each detection mask, register every candidate CAD; keep the best."""
        results: list[tuple[str, PoseEstimate]] = []
        for det in detections:
            best: Optional[tuple[str, PoseEstimate]] = None
            for cad_id in candidate_cad_ids:
                mesh = self.cad.mesh(cad_id)
                pose_cam, score = self.foundation_pose.register(
                    rgb=rgb,
                    depth=depth,
                    mask=det.mask,
                    mesh=mesh,
                    K=K,
                )
                est = PoseEstimate(
                    T_cam_obj=pose_cam,
                    confidence=score,
                    view=view,  # type: ignore[arg-type]
                    mask=det.mask,
                    bbox_xyxy=det.bbox,
                )
                if best is None or score > best[1].confidence:
                    best = (cad_id, est)
            if best is not None:
                results.append(best)
        return results

    def _fuse_and_package(
        self,
        top: list[tuple[str, np.ndarray, PoseEstimate]],
        wrist: list[tuple[str, np.ndarray, PoseEstimate]],
        query: str,
    ) -> list[DetectedObject]:
        """Cross-view association + SE(3) weighted fusion, then wrap into DetectedObject."""
        associations = associate_across_views(top, wrist)
        out: list[DetectedObject] = []
        for assoc in associations:
            fused_T, fused_conf, src = fuse_pose_estimates(assoc)
            out.append(
                DetectedObject(
                    label=query,
                    cad_id=assoc.cad_id,
                    T_base_obj=fused_T,
                    confidence=fused_conf,
                    source_view=src,
                    per_view=assoc.per_view,
                )
            )
        return out
