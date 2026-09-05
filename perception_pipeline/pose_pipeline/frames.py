"""Frame transforms — extrinsics loading and a pluggable FK function.

Keeps Pinocchio (or any other robotics framework) at arm's length. The rest of
the pipeline only ever sees 4x4 numpy matrices.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional
import numpy as np
import yaml


def load_extrinsics(path: str | Path) -> dict:
    """Load T_base_top, T_tool_wrist, K_top, K_wrist from YAML.

    The YAML format is plain nested lists; we convert to numpy here so the
    caller never has to remember.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {
        "T_base_top": np.array(raw["T_base_top"], dtype=np.float64),
        "T_tool_wrist": np.array(raw["T_tool_wrist"], dtype=np.float64),
        "K_top": np.array(raw["K_top"], dtype=np.float64),
        "K_wrist": np.array(raw["K_wrist"], dtype=np.float64),
        "dist_top": np.array(raw.get("dist_top", [0, 0, 0, 0, 0]), dtype=np.float64),
        "dist_wrist": np.array(raw.get("dist_wrist", [0, 0, 0, 0, 0]), dtype=np.float64),
    }


def save_extrinsics(path: str | Path, **transforms) -> None:
    """Write extrinsics back to YAML. Pass any subset of named matrices."""
    out: dict = {}
    for k, v in transforms.items():
        out[k] = v.tolist() if isinstance(v, np.ndarray) else v
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(out, f, default_flow_style=None)


FKFunction = Callable[[np.ndarray], np.ndarray]
"""A function mapping joint angles (N,) to tool-flange pose in base frame (4x4)."""


def make_fk(
    urdf_path: Optional[str | Path] = None,
    tool_link: str = "gripper_tool",
) -> FKFunction:
    """Factory returning an FK callable.

    Tries Pinocchio first; if unavailable, raises with instructions. You can
    skip this entirely and pass your own FKFunction into PoseEstimator.estimate().
    """
    try:
        import pinocchio as pin  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Pinocchio not installed. Either `pip install pin` or supply your "
            "own callable that takes joint angles and returns T_base_tool."
        ) from e

    if urdf_path is None:
        raise ValueError("urdf_path is required for the default Pinocchio FK backend.")

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    try:
        tool_id = model.getFrameId(tool_link)
    except Exception as e:
        raise ValueError(f"Tool link {tool_link!r} not found in URDF.") from e

    def fk(q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, tool_id)
        T = np.eye(4)
        T[:3, :3] = data.oMf[tool_id].rotation
        T[:3, 3] = data.oMf[tool_id].translation
        return T

    return fk


# ---------- SE(3) helpers used elsewhere ----------

def invert_transform(T: np.ndarray) -> np.ndarray:
    """Fast SE(3) inverse — transpose the rotation, rotate-negate the translation."""
    Ti = np.eye(4)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti
