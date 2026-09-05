from __future__ import annotations

import re
import struct
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .pinocchio_meshcat import default_control_repo, find_urdf
from .workcell import load_workcell, quat_wxyz_from_pose, resolve_mesh_path, xyz_from_pose


def _as_vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{v:.9g}" for v in values)


def _is_ascii_stl(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(5).lower() == b"solid"


def _ascii_stl_triangles(path: Path) -> list[tuple[tuple[float, float, float], list[tuple[float, float, float]]]]:
    triangles = []
    normal = (0.0, 0.0, 0.0)
    vertices: list[tuple[float, float, float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if not parts:
            continue
        if parts[:2] == ["facet", "normal"]:
            normal = tuple(float(v) for v in parts[2:5])  # type: ignore[assignment]
            vertices = []
        elif parts[0] == "vertex":
            vertices.append(tuple(float(v) for v in parts[1:4]))  # type: ignore[arg-type]
        elif parts[0] == "endfacet" and len(vertices) == 3:
            triangles.append((normal, vertices.copy()))
    if not triangles:
        raise ValueError(f"No triangles found in ASCII STL: {path}")
    return triangles


def _write_binary_stl(source: Path, target: Path) -> None:
    triangles = _ascii_stl_triangles(source)
    with target.open("wb") as f:
        header = f"converted from {source.name}".encode("ascii", errors="ignore")[:80]
        f.write(header.ljust(80, b"\0"))
        f.write(struct.pack("<I", len(triangles)))
        for normal, vertices in triangles:
            f.write(struct.pack("<3f", *normal))
            for vertex in vertices:
                f.write(struct.pack("<3f", *vertex))
            f.write(struct.pack("<H", 0))


def _mujoco_mesh_path(mesh_path: Path, asset_dir: Path) -> Path:
    if not _is_ascii_stl(mesh_path):
        return mesh_path
    target = asset_dir / f"{mesh_path.stem}_binary.stl"
    _write_binary_stl(mesh_path, target)
    return target


def _patch_package_urls(text: str, control_repo: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        package_name = match.group(1)
        suffix = match.group(2)
        candidates = [
            control_repo / package_name / suffix,
            control_repo / "urdf" / package_name / suffix,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        return match.group(0)

    return re.sub(r"package://([^/]+)/([^\"<]+)", replace, text)


def _clear_visual_names(root: ET.Element) -> None:
    for tag in ("visual", "collision"):
        for elem in root.findall(f".//{tag}"):
            if elem.get("name") == "":
                elem.attrib.pop("name")


def _ensure_world_root(root: ET.Element, old_root_link: str) -> None:
    world_link = ET.Element("link", {"name": "world"})
    floor_link = ET.Element("link", {"name": "workcell_floor"})
    collision = ET.SubElement(floor_link, "collision", {"name": "workcell_floor_collision"})
    ET.SubElement(collision, "origin", {"xyz": "0 0 -0.01", "rpy": "0 0 0"})
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "box", {"size": "1.4 1.4 0.02"})
    visual = ET.SubElement(floor_link, "visual", {"name": "workcell_floor_visual"})
    ET.SubElement(visual, "origin", {"xyz": "0 0 -0.01", "rpy": "0 0 0"})
    visual_geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(visual_geometry, "box", {"size": "1.4 1.4 0.02"})

    world_to_base = ET.Element(
        "joint",
        {"name": "world_to_robot_base", "type": "fixed"},
    )
    ET.SubElement(world_to_base, "parent", {"link": "world"})
    ET.SubElement(world_to_base, "child", {"link": old_root_link})
    ET.SubElement(world_to_base, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})

    floor_joint = ET.Element(
        "joint",
        {"name": "world_to_workcell_floor", "type": "fixed"},
    )
    ET.SubElement(floor_joint, "parent", {"link": "world"})
    ET.SubElement(floor_joint, "child", {"link": "workcell_floor"})
    ET.SubElement(floor_joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})

    root.insert(0, floor_joint)
    root.insert(0, floor_link)
    root.insert(0, world_to_base)
    root.insert(0, world_link)


def _add_inertial(link: ET.Element, mass_value: float, inertia: float = 0.0001) -> None:
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": f"{mass_value:.9g}"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": f"{inertia:.9g}",
            "ixy": "0",
            "ixz": "0",
            "iyy": f"{inertia:.9g}",
            "iyz": "0",
            "izz": f"{inertia:.9g}",
        },
    )


def _add_box_link(
    root: ET.Element,
    name: str,
    size: tuple[float, float, float],
    mass: float,
    color_rgba: str = "0.08 0.08 0.08 1",
) -> ET.Element:
    link = ET.Element("link", {"name": name})
    _add_inertial(link, mass, inertia=0.00001)
    for tag in ("visual", "collision"):
        elem = ET.SubElement(link, tag, {"name": f"{name}_{tag}"})
        ET.SubElement(elem, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(elem, "geometry")
        ET.SubElement(geometry, "box", {"size": _as_vec(size)})
        if tag == "visual":
            material = ET.SubElement(elem, "material", {"name": f"{name}_material"})
            ET.SubElement(material, "color", {"rgba": color_rgba})
    root.append(link)
    return link


def _add_fixed_joint(
    root: ET.Element,
    name: str,
    parent: str,
    child: str,
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    joint = ET.Element("joint", {"name": name, "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "origin", {"xyz": _as_vec(xyz), "rpy": _as_vec(rpy)})
    root.append(joint)


def _add_prismatic_joint(
    root: ET.Element,
    name: str,
    parent: str,
    child: str,
    xyz: tuple[float, float, float],
    axis: tuple[float, float, float],
    lower: float,
    upper: float,
) -> None:
    joint = ET.Element("joint", {"name": name, "type": "prismatic"})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "origin", {"xyz": _as_vec(xyz), "rpy": "0 0 0"})
    ET.SubElement(joint, "axis", {"xyz": _as_vec(axis)})
    ET.SubElement(
        joint,
        "limit",
        {
            "lower": f"{lower:.9g}",
            "upper": f"{upper:.9g}",
            "effort": "20",
            "velocity": "1",
        },
    )
    root.append(joint)


def _add_synthetic_gripper(root: ET.Element) -> None:
    """Add a simple MuJoCo-only gripper because the upstream URDF end_joint is fixed."""
    _add_box_link(root, "sim_gripper_palm", (0.018, 0.08, 0.024), 0.05, "0.45 0.45 0.45 1")
    _add_box_link(root, "sim_left_finger", (0.09, 0.012, 0.018), 0.03)
    _add_box_link(root, "sim_right_finger", (0.09, 0.012, 0.018), 0.03)
    _add_fixed_joint(
        root,
        "end_link_to_sim_gripper_palm",
        "end_link",
        "sim_gripper_palm",
        (0.035, 0.0, 0.0),
    )
    _add_prismatic_joint(
        root,
        "sim_left_finger_slide",
        "sim_gripper_palm",
        "sim_left_finger",
        (0.055, 0.018, 0.0),
        (0.0, 1.0, 0.0),
        0.0,
        0.04,
    )
    _add_prismatic_joint(
        root,
        "sim_right_finger_slide",
        "sim_gripper_palm",
        "sim_right_finger",
        (0.055, -0.018, 0.0),
        (0.0, -1.0, 0.0),
        0.0,
        0.04,
    )


def _add_workcell_objects(root: ET.Element, config_path: Path, asset_dir: Path) -> None:
    config = load_workcell(config_path)
    for obj in config["objects"]:
        name = obj["name"]
        pose = obj.get("pose", {})
        mesh_path = _mujoco_mesh_path(resolve_mesh_path(obj["mesh"], config_path), asset_dir)
        mass = float(obj.get("mass", 0.1))
        xyz = xyz_from_pose(pose)
        quat_wxyz = quat_wxyz_from_pose(pose)

        link = ET.Element("link", {"name": name})
        _add_inertial(link, mass)

        for tag in ("visual", "collision"):
            elem = ET.SubElement(link, tag, {"name": f"{name}_{tag}"})
            ET.SubElement(elem, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            geometry = ET.SubElement(elem, "geometry")
            ET.SubElement(geometry, "mesh", {"filename": str(mesh_path)})

        joint = ET.Element("joint", {"name": f"world_to_{name}", "type": "floating"})
        ET.SubElement(joint, "parent", {"link": "world"})
        ET.SubElement(joint, "child", {"link": name})
        ET.SubElement(
            joint,
            "origin",
            {
                "xyz": _as_vec(xyz),
                "rpy": "0 0 0",
            },
        )

        root.append(link)
        root.append(joint)

        # Store MuJoCo-friendly quaternion metadata in a custom tag. The compiler
        # ignores unknown URDF tags, and we set the qpos after loading.
        root.append(
            ET.Comment(
                f"mujoco_free_body_quat {name} {_as_vec(quat_wxyz)}"
            )
        )


def make_mujoco_urdf(
    control_repo: Path | None,
    workcell_config: Path,
    urdf: str | None = None,
) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    control_repo = (control_repo or default_control_repo()).resolve()
    source_urdf = find_urdf(control_repo, urdf)
    patched_text = _patch_package_urls(source_urdf.read_text(encoding="utf-8"), control_repo)
    root = ET.fromstring(patched_text)
    tmpdir = tempfile.TemporaryDirectory(prefix="rebot_b601_mujoco_")
    tmp_path = Path(tmpdir.name)
    old_root_link = root.find("link").get("name")  # type: ignore[union-attr]
    _clear_visual_names(root)
    _ensure_world_root(root, old_root_link)
    _add_synthetic_gripper(root)
    _add_workcell_objects(root, workcell_config, tmp_path)

    out_path = tmp_path / "scene.urdf"
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path, tmpdir


def initial_free_body_poses(config_path: Path) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]]:
    config = load_workcell(config_path)
    poses: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}
    for obj in config["objects"]:
        pose = obj.get("pose", {})
        poses[obj["name"]] = (xyz_from_pose(pose), quat_wxyz_from_pose(pose))
    return poses
