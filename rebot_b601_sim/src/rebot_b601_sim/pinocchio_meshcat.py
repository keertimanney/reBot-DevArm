from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .joints import SIM_JOINT_NAMES, degrees_action_to_q
from .workcell import load_objects_into_viewer

UPSTREAM_REPO = "https://github.com/vectorBH6/reBotArm_control_py.git"


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_control_repo() -> Path:
    env_path = os.environ.get("REBOTARM_CONTROL_PY")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return package_root().parent / "reBotArm_control_py"


def default_workcell_config() -> Path:
    return package_root() / "configs" / "workcell_two_cubes.json"


def load_robot_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        path = package_root() / "configs" / "robot_sim.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_urdf(control_repo: Path, preferred: str | None = None) -> Path:
    if preferred:
        urdf_path = Path(preferred).expanduser()
        if not urdf_path.is_absolute():
            urdf_path = control_repo / urdf_path
        if urdf_path.exists():
            return urdf_path.resolve()
        raise FileNotFoundError(f"Preferred URDF does not exist: {urdf_path}")

    candidates = sorted(control_repo.glob("urdf/**/*.urdf")) + sorted(control_repo.glob("**/*.urdf"))
    if not candidates:
        raise FileNotFoundError(
            f"No URDF found under {control_repo}. Run setup or pass --control-repo/--urdf."
        )
    b601_candidates = [p for p in candidates if "b601" in p.name.lower() or "rebot" in p.name.lower()]
    return (b601_candidates or candidates)[0].resolve()


def _package_names_in_urdf(urdf_path: Path) -> set[str]:
    text = urdf_path.read_text(encoding="utf-8")
    return set(re.findall(r"package://([^/]+)/", text))


def _actual_mesh_package_dirs(control_repo: Path) -> list[Path]:
    return sorted(
        p.parent
        for p in control_repo.glob("urdf/**/meshes")
        if p.is_dir()
    )


def _ensure_package_aliases(control_repo: Path, urdf_path: Path) -> Path | None:
    """Create package:// aliases for upstream URDF package-name typos.

    The current Seeed URDF references package://reBot-DevArm_description_fixend,
    while the checked-out directory is named reBot-DevArm_fixend_description.
    Pinocchio resolves package URLs by directory name, so we create a local
    alias instead of editing upstream files.
    """
    requested_packages = _package_names_in_urdf(urdf_path)
    existing_roots = [control_repo, control_repo / "urdf", urdf_path.parent.parent]
    existing_package_names = {
        child.name
        for root in existing_roots
        if root.exists()
        for child in root.iterdir()
        if child.is_dir()
    }
    missing_packages = requested_packages - existing_package_names
    if not missing_packages:
        return None

    mesh_package_dirs = _actual_mesh_package_dirs(control_repo)
    if len(mesh_package_dirs) != 1:
        return None

    alias_root = package_root() / ".cache" / "pinocchio_package_aliases"
    alias_root.mkdir(parents=True, exist_ok=True)
    actual_package = mesh_package_dirs[0]
    for package_name in missing_packages:
        alias_path = alias_root / package_name
        if alias_path.exists():
            continue
        try:
            alias_path.symlink_to(actual_package, target_is_directory=True)
        except OSError:
            shutil.copytree(actual_package, alias_path)
    return alias_root


def package_dirs_for_urdf(control_repo: Path, urdf_path: Path) -> list[str]:
    package_dirs = [
        control_repo,
        control_repo / "urdf",
        urdf_path.parent,
        urdf_path.parent.parent,
    ]
    alias_root = _ensure_package_aliases(control_repo, urdf_path)
    if alias_root is not None:
        package_dirs.insert(0, alias_root)
    return [str(path) for path in package_dirs if path.exists()]


class B601MeshcatSim:
    def __init__(
        self,
        control_repo: Path | None = None,
        robot_config: Path | None = None,
        urdf: str | None = None,
        open_browser: bool = True,
        root_node: str = "rebot_b601",
    ) -> None:
        import pinocchio as pin
        from pinocchio.visualize import MeshcatVisualizer

        self.pin = pin
        self.control_repo = (control_repo or default_control_repo()).resolve()
        self.config = load_robot_config(robot_config)
        self.joint_order = tuple(self.config.get("joint_order", SIM_JOINT_NAMES))
        self.urdf_path = find_urdf(self.control_repo, urdf or self.config.get("urdf"))
        self.root_node = root_node
        self.gripper_node = f"{root_node}/visual_gripper"

        package_dirs = package_dirs_for_urdf(self.control_repo, self.urdf_path)
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            str(self.urdf_path),
            package_dirs,
        )
        self.neutral_q = np.array(pin.neutral(self.model), dtype=float)
        self.current_q = self.neutral_q.copy()
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        self.viz.initViewer(open=open_browser)
        self.viz.loadViewerModel(rootNodeName=root_node)
        self._end_frame_id = self._find_end_frame_id()
        self._load_visual_gripper()
        self.display_q(self.neutral_q)

    @property
    def viewer(self) -> Any:
        return self.viz.viewer

    def display_q(self, q: np.ndarray | list[float]) -> None:
        self.current_q = np.array(q, dtype=float)
        self.viz.display(self.current_q)

    def display_degrees_action(self, action: dict[str, float]) -> None:
        q = degrees_action_to_q(action, self.neutral_q, self.joint_order)
        self.display_q(q)
        self.display_gripper(float(action.get("gripper.pos", action.get("gripper", 0.0))))

    def load_workcell(self, path: Path | None = None) -> None:
        load_objects_into_viewer(self.viewer, path or default_workcell_config())

    def _find_end_frame_id(self) -> int:
        for frame_name in ("end_link", "link6", self.model.frames[-1].name):
            frame_id = self.model.getFrameId(frame_name)
            if frame_id < len(self.model.frames):
                return frame_id
        return len(self.model.frames) - 1

    def _load_visual_gripper(self) -> None:
        import meshcat.geometry as g

        material = g.MeshLambertMaterial(color=0x202020, opacity=1.0)
        palm_material = g.MeshLambertMaterial(color=0x707070, opacity=1.0)
        self.viewer[f"{self.gripper_node}/palm"].set_object(g.Box([0.018, 0.07, 0.025]), palm_material)
        self.viewer[f"{self.gripper_node}/left_finger"].set_object(g.Box([0.075, 0.012, 0.018]), material)
        self.viewer[f"{self.gripper_node}/right_finger"].set_object(g.Box([0.075, 0.012, 0.018]), material)
        self.display_gripper(0.0)

    def _end_link_transform(self) -> np.ndarray:
        self.pin.forwardKinematics(self.model, self.viz.data, self.current_q)
        self.pin.updateFramePlacements(self.model, self.viz.data)
        return np.array(self.viz.data.oMf[self._end_frame_id].homogeneous)

    @staticmethod
    def _translation(x: float, y: float, z: float) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, 3] = [x, y, z]
        return transform

    def display_gripper(self, gripper_degrees: float) -> None:
        """Draw a simple visual gripper from the leader gripper command.

        The upstream URDF only exposes a fixed end link, so this is visual only:
        it makes leader gripper motion visible in MeshCat but does not add a real
        kinematic joint or contact geometry.
        """
        open_fraction = max(0.0, min(1.0, abs(gripper_degrees) / 270.0))
        half_gap = 0.018 + 0.035 * open_fraction
        base = self._end_link_transform()
        self.viewer[f"{self.gripper_node}/palm"].set_transform(base @ self._translation(0.025, 0.0, 0.0))
        self.viewer[f"{self.gripper_node}/left_finger"].set_transform(
            base @ self._translation(0.072, half_gap, 0.0)
        )
        self.viewer[f"{self.gripper_node}/right_finger"].set_transform(
            base @ self._translation(0.072, -half_gap, 0.0)
        )
