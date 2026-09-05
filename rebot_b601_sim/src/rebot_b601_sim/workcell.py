from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _rotation_from_quat_xyzw(quat: list[float]) -> np.ndarray:
    x, y, z, w = quat
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("Quaternion norm must be non-zero.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_wxyz_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def quat_wxyz_from_pose(pose: dict[str, Any]) -> tuple[float, float, float, float]:
    if "rpy" in pose:
        return quat_wxyz_from_rpy(*[float(v) for v in pose["rpy"]])
    if "quat_xyzw" in pose:
        x, y, z, w = [float(v) for v in pose["quat_xyzw"]]
        return (w, x, y, z)
    return (1.0, 0.0, 0.0, 0.0)


def xyz_from_pose(pose: dict[str, Any]) -> tuple[float, float, float]:
    xyz = pose.get("xyz", [0.0, 0.0, 0.0])
    if len(xyz) != 3:
        raise ValueError("pose.xyz must contain 3 values.")
    return tuple(float(v) for v in xyz)


def transform_matrix(pose: dict[str, Any]) -> np.ndarray:
    translation = xyz_from_pose(pose)

    if "rpy" in pose:
        rotation = _rotation_from_rpy(*[float(v) for v in pose["rpy"]])
    elif "quat_xyzw" in pose:
        rotation = _rotation_from_quat_xyzw([float(v) for v in pose["quat_xyzw"]])
    else:
        rotation = np.eye(3)

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.array(translation, dtype=float)
    return transform


def load_workcell(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if "objects" not in config or not isinstance(config["objects"], list):
        raise ValueError(f"{path} must contain an objects list.")
    return config


def resolve_mesh_path(mesh: str, config_path: Path) -> Path:
    mesh_path = Path(mesh)
    if not mesh_path.is_absolute():
        mesh_path = (config_path.parent / mesh_path).resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh does not exist: {mesh_path}")
    return mesh_path


def load_objects_into_viewer(viewer: Any, config_path: Path, root: str = "workcell") -> None:
    import meshcat.geometry as g

    config = load_workcell(config_path)
    viewer[root].delete()
    for obj in config["objects"]:
        name = obj["name"]
        mesh_path = resolve_mesh_path(obj["mesh"], config_path)
        color = int(str(obj.get("color", "0xcccccc")), 16)
        opacity = float(obj.get("opacity", 1.0))
        material = g.MeshLambertMaterial(color=color, opacity=opacity)
        viewer[f"{root}/{name}"].set_object(g.StlMeshGeometry.from_file(str(mesh_path)), material)
        viewer[f"{root}/{name}"].set_transform(transform_matrix(obj.get("pose", {})))
