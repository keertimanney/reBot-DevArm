"""PerceptionProvider backed by a recorded_arm_positions.yaml file."""
from __future__ import annotations

from pathlib import Path

import yaml

from .models import Pose, Scene
from .perception import PerceptionProvider


class RecordedPerception(PerceptionProvider):
    """Builds a Scene from the positions stored in the YAML recording.

    The xyz from each position's end_effector field is used as the Pose so
    the existing Scene API works. The arm executor looks up positions by
    label name, not xyz, so the exact values here are only used for describe().
    """

    def __init__(self, positions_file: str | Path) -> None:
        data = yaml.safe_load(Path(positions_file).read_text())
        self._positions: dict[str, Pose] = {}
        for label, rec in data["positions"].items():
            xyz = rec["end_effector"]["xyz"]
            self._positions[label] = Pose(x=xyz[0], y=xyz[1], z=xyz[2])

    def read_scene(self) -> Scene:
        return Scene(positions=self._positions, occupied_positions={})

    @property
    def position_names(self) -> list[str]:
        return list(self._positions.keys())
