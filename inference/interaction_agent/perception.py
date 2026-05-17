from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Pose, Scene


class PerceptionProvider(ABC):
    @abstractmethod
    def read_scene(self) -> Scene:
        """Return the latest structured scene state."""


class MockPerception(PerceptionProvider):
    """Static five-slot tabletop scene for developing the orchestration loop."""

    def read_scene(self) -> Scene:
        return Scene(
            positions={
                "top_left": Pose(x=0.35, y=0.20, z=0.03),
                "top_right": Pose(x=0.35, y=-0.20, z=0.03),
                "center": Pose(x=0.45, y=0.00, z=0.03),
                "bottom_left": Pose(x=0.55, y=0.20, z=0.03),
                "bottom_right": Pose(x=0.55, y=-0.20, z=0.03),
            },
            occupied_positions={
                "center": "block",
            },
        )
