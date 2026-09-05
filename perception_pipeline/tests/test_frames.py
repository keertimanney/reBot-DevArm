"""Tests for frame conversions and extrinsics IO."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from pose_pipeline.frames import load_extrinsics, save_extrinsics, invert_transform


def test_invert_transform_identity():
    T = np.eye(4)
    assert np.allclose(invert_transform(T), np.eye(4))


def test_invert_transform_round_trip():
    rng = np.random.default_rng(0)
    # build a random SE(3)
    from scipy.spatial.transform import Rotation as R  # type: ignore
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(rng.standard_normal(3)).as_matrix()
    T[:3, 3] = rng.standard_normal(3)
    assert np.allclose(T @ invert_transform(T), np.eye(4), atol=1e-10)


def test_extrinsics_round_trip(tmp_path: Path):
    path = tmp_path / "ext.yaml"
    save_extrinsics(
        path,
        T_base_top=np.eye(4),
        T_tool_wrist=np.eye(4),
        K_top=np.eye(3),
        K_wrist=np.eye(3),
    )
    loaded = load_extrinsics(path)
    assert np.allclose(loaded["T_base_top"], np.eye(4))
    assert np.allclose(loaded["K_top"], np.eye(3))
