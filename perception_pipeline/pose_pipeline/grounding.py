"""Grounded-SAM 2 wrapper.

Wraps Grounding DINO (open-vocabulary detection from a text prompt) followed by
SAM 2 (segmentation). Prompts here are intentionally generic — identity comes
from CAD matching downstream, not language.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


# ---------- prompt templates ----------
# Period-separated phrases is Grounding DINO's multi-class query format.
# Keep these generic so the same prompt works for any block library.

PROMPT_GENERIC = "block. brick. cube. cuboid. piece. part."
"""Default. Returns every blocklike thing; CAD matching disambiguates."""

PROMPT_ON_TABLE = "object on table. block. brick."
"""When the workspace has clutter that isn't blocks, scoping with 'on table' helps."""


def prompt_by_color(color: str) -> str:
    """Build a color-scoped prompt. Use only when the agent gives you a color."""
    return f"{color} block. {color} brick. {color} cube."


# ---------- data class ----------

@dataclass
class Detection:
    bbox: tuple              # (x1, y1, x2, y2)
    mask: np.ndarray         # H x W bool
    phrase: str              # the grounded phrase
    score: float             # grounding score


# ---------- wrapper ----------

class GroundedSAM:
    """Lazy-loaded wrapper around Grounding DINO + SAM 2.

    We import inside __init__ so that this module is importable without the
    heavy deps installed (useful for tests and the calibration scripts).
    """

    def __init__(
        self,
        gdino_model: str = "IDEA-Research/grounding-dino-base",
        sam_model: str = "facebook/sam2-hiera-large",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        device: str = "cuda",
    ):
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.gdino_model_name = gdino_model
        self.sam_model_name = sam_model
        self._gdino = None
        self._sam = None
        # Models load lazily on first detect() call to keep startup fast.

    def _ensure_loaded(self):
        if self._gdino is not None:
            return
        # TODO: load Grounding DINO via transformers AutoProcessor / AutoModel
        # TODO: load SAM 2 from the grounded-sam-2 repo or huggingface
        raise NotImplementedError(
            "Wire up actual model loading. See Grounded-SAM-2 README for the "
            "supported init pattern. Recommended: use their `GroundedSAM2` "
            "wrapper class directly and adapt its output to Detection."
        )

    def detect(self, rgb: np.ndarray, prompt: str) -> list[Detection]:
        """Run grounding + segmentation.

        Args:
            rgb: H x W x 3 uint8.
            prompt: Period-separated multi-class query (see PROMPT_* constants).

        Returns:
            List of Detection. Empty if nothing grounded above thresholds.
        """
        self._ensure_loaded()
        # 1. Grounding DINO → boxes + phrases + scores
        # 2. SAM 2 with each box as a prompt → masks
        # 3. Filter by thresholds, package as Detection
        raise NotImplementedError
