from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from .mujoco_macos import ensure_mjpython_for_viewer
from .pinocchio_meshcat import default_control_repo, default_workcell_config
from .mujoco_scene import initial_free_body_poses, make_mujoco_urdf

MUJOCO_JOINT_NAMES: tuple[str, ...] = (
    "joint1",
    "joint2",
    "join3",
    "joint4",
    "joint5",
    "joint6",
)
MUJOCO_GRIPPER_JOINT_NAMES: tuple[str, ...] = (
    "sim_left_finger_slide",
    "sim_right_finger_slide",
)
MUJOCO_ALL_CONTROLLED_JOINTS: tuple[str, ...] = MUJOCO_JOINT_NAMES + MUJOCO_GRIPPER_JOINT_NAMES
GRIPPER_MAX_OPEN_M = 0.04

ACTION_TO_MUJOCO_JOINT: dict[str, str] = {
    "shoulder_pan": "joint1",
    "shoulder_lift": "joint2",
    "elbow_flex": "join3",
    "wrist_flex": "joint4",
    "wrist_yaw": "joint5",
    "wrist_roll": "joint6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a contact/dynamics MuJoCo sim for the B601 workcell.")
    parser.add_argument("--control-repo", type=Path, default=default_control_repo())
    parser.add_argument("--urdf")
    parser.add_argument("--workcell", type=Path, default=default_workcell_config())
    parser.add_argument("--q-deg", nargs="*", type=float, default=[0, 0, 0, 0, 0, 0])
    parser.add_argument("--gripper-deg", type=float, default=0.0, help="Standalone synthetic gripper command: 0 closed, -270 open.")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--duration", type=float, help="Run for N seconds, then exit.")
    parser.add_argument("--kp", type=float, default=35.0, help="Joint-space PD stiffness.")
    parser.add_argument("--kd", type=float, default=2.0, help="Joint-space PD damping.")
    return parser.parse_args()


def import_mujoco():
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "MuJoCo is not installed in this Python environment. Install with:\n"
            "  python -m pip install -e 'rebot_b601_sim[mujoco]'\n"
            "or:\n"
            "  python -m pip install mujoco"
        ) from exc
    return mujoco


def set_initial_robot_qpos(mujoco, model, data, q_deg: list[float]) -> np.ndarray:
    target = np.zeros(model.nv)
    for idx, joint_name in enumerate(MUJOCO_JOINT_NAMES[: len(q_deg)]):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        qadr = model.jnt_qposadr[joint_id]
        dadr = model.jnt_dofadr[joint_id]
        value = math.radians(q_deg[idx])
        data.qpos[qadr] = value
        target[dadr] = value
    set_gripper_target(mujoco, model, target, 0.0)
    return target


def set_gripper_target(mujoco, model, target_q: np.ndarray, gripper_degrees: float) -> None:
    open_fraction = max(0.0, min(1.0, abs(float(gripper_degrees)) / 270.0))
    target = open_fraction * GRIPPER_MAX_OPEN_M
    for joint_name in MUJOCO_GRIPPER_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        dadr = model.jnt_dofadr[joint_id]
        target_q[dadr] = target


def set_target_from_leader_action(mujoco, model, target_q: np.ndarray, action: dict[str, float]) -> None:
    for action_joint, mujoco_joint in ACTION_TO_MUJOCO_JOINT.items():
        key = f"{action_joint}.pos"
        if key not in action:
            continue
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mujoco_joint)
        if joint_id < 0:
            continue
        dadr = model.jnt_dofadr[joint_id]
        target_q[dadr] = math.radians(float(action[key]))
    if "gripper.pos" in action:
        set_gripper_target(mujoco, model, target_q, float(action["gripper.pos"]))


def set_initial_object_poses(mujoco, model, data, workcell_path: Path) -> None:
    for body_name, (xyz, quat_wxyz) in initial_free_body_poses(workcell_path).items():
        joint_name = f"world_to_{body_name}"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        qadr = model.jnt_qposadr[joint_id]
        data.qpos[qadr : qadr + 3] = np.array(xyz)
        data.qpos[qadr + 3 : qadr + 7] = np.array(quat_wxyz)


def apply_robot_pd(mujoco, model, data, target_q: np.ndarray, kp: float, kd: float) -> None:
    data.qfrc_applied[:] = 0.0
    for joint_name in MUJOCO_ALL_CONTROLLED_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        qadr = model.jnt_qposadr[joint_id]
        dadr = model.jnt_dofadr[joint_id]
        data.qfrc_applied[dadr] = kp * (target_q[dadr] - data.qpos[qadr]) - kd * data.qvel[dadr]


def configure_visuals(model, data=None) -> None:
    try:
        model.vis.headlight.active = 1
        model.vis.headlight.ambient[:] = [0.35, 0.35, 0.35]
        model.vis.headlight.diffuse[:] = [0.75, 0.75, 0.75]
        model.vis.headlight.specular[:] = [0.25, 0.25, 0.25]
    except Exception:
        pass


def configure_viewer_camera(viewer) -> None:
    viewer.cam.lookat[:] = [0.28, 0.0, 0.22]
    viewer.cam.distance = 1.1
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -25


def configure_viewer_lighting(viewer) -> None:
    if not hasattr(viewer, "user_scn"):
        return
    scene = viewer.user_scn
    if len(scene.lights) == 0:
        return
    scene.nlight = max(scene.nlight, 1)
    light = scene.lights[0]
    light.pos[:] = [0.35, -0.55, 1.25]
    light.dir[:] = [-0.25, 0.35, -1.0]
    light.ambient[:] = [0.25, 0.25, 0.25]
    light.diffuse[:] = [0.9, 0.9, 0.85]
    light.specular[:] = [0.25, 0.25, 0.25]
    light.attenuation[:] = [1.0, 0.0, 0.0]
    light.castshadow = 1


def run_headless(mujoco, model, data, target_q: np.ndarray, duration: float, kp: float, kd: float) -> None:
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        apply_robot_pd(mujoco, model, data, target_q, kp, kd)
        mujoco.mj_step(model, data)


def run_viewer(mujoco, model, data, target_q: np.ndarray, duration: float | None, kp: float, kd: float) -> None:
    import mujoco.viewer

    start = time.monotonic()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_viewer_camera(viewer)
        configure_viewer_lighting(viewer)
        while viewer.is_running():
            if duration is not None and time.monotonic() - start >= duration:
                break
            step_start = time.monotonic()
            apply_robot_pd(mujoco, model, data, target_q, kp, kd)
            mujoco.mj_step(model, data)
            viewer.sync()
            sleep_time = model.opt.timestep - (time.monotonic() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


def main() -> None:
    args = parse_args()
    ensure_mjpython_for_viewer("rebot_b601_sim.mujoco_sim", args.no_viewer)
    mujoco = import_mujoco()

    scene_urdf, tmpdir = make_mujoco_urdf(args.control_repo, args.workcell, args.urdf)
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_urdf))
        data = mujoco.MjData(model)
        configure_visuals(model, data)
        target_q = set_initial_robot_qpos(mujoco, model, data, args.q_deg)
        set_gripper_target(mujoco, model, target_q, args.gripper_deg)
        set_initial_object_poses(mujoco, model, data, args.workcell)
        mujoco.mj_forward(model, data)

        print(f"MuJoCo scene: {scene_urdf}")
        print(f"Workcell: {args.workcell}")
        if args.no_viewer:
            run_headless(mujoco, model, data, target_q, args.duration or 5.0, args.kp, args.kd)
        else:
            run_viewer(mujoco, model, data, target_q, args.duration, args.kp, args.kd)
    finally:
        tmpdir.cleanup()


if __name__ == "__main__":
    main()
