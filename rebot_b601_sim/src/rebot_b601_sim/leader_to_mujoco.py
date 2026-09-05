from __future__ import annotations

import argparse
import time
from pathlib import Path

from .leader import import_leader_classes
from .mujoco_macos import ensure_mjpython_for_viewer
from .mujoco_scene import make_mujoco_urdf
from .mujoco_sim import (
    apply_robot_pd,
    configure_viewer_camera,
    configure_viewer_lighting,
    configure_visuals,
    import_mujoco,
    set_initial_object_poses,
    set_initial_robot_qpos,
    set_target_from_leader_action,
)
from .pinocchio_meshcat import default_control_repo, default_workcell_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive the MuJoCo B601 follower from the reBot Arm 102 leader.")
    parser.add_argument("--leader-port", required=True, help="Leader serial port, e.g. /dev/cu.usbserial-0001")
    parser.add_argument("--leader-id", default="leader_mujoco")
    parser.add_argument("--leader-baudrate", type=int, default=1_000_000)
    parser.add_argument("--control-repo", type=Path, default=default_control_repo())
    parser.add_argument("--urdf")
    parser.add_argument("--workcell", type=Path, default=default_workcell_config())
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--duration", type=float, help="Run for N seconds, then exit.")
    parser.add_argument("--leader-interval", type=float, default=0.03, help="Seconds between leader reads.")
    parser.add_argument("--kp", type=float, default=35.0, help="Joint-space PD stiffness.")
    parser.add_argument("--kd", type=float, default=2.0, help="Joint-space PD damping.")
    return parser.parse_args()


def run_loop(mujoco, model, data, target_q, leader, args, viewer=None) -> None:
    start = time.monotonic()
    next_leader_read = 0.0
    while viewer is None or viewer.is_running():
        now = time.monotonic()
        if args.duration is not None and now - start >= args.duration:
            break
        if now >= next_leader_read:
            action = leader.get_action()
            set_target_from_leader_action(mujoco, model, target_q, action)
            next_leader_read = now + args.leader_interval

        step_start = time.monotonic()
        apply_robot_pd(mujoco, model, data, target_q, args.kp, args.kd)
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        sleep_time = model.opt.timestep - (time.monotonic() - step_start)
        if sleep_time > 0:
            time.sleep(sleep_time)


def main() -> None:
    args = parse_args()
    ensure_mjpython_for_viewer("rebot_b601_sim.leader_to_mujoco", args.no_viewer)
    mujoco = import_mujoco()
    RebotArm102Leader, RebotArm102LeaderConfig = import_leader_classes()

    leader = RebotArm102Leader(
        RebotArm102LeaderConfig(
            id=args.leader_id,
            port=args.leader_port,
            baudrate=args.leader_baudrate,
        )
    )
    scene_urdf, tmpdir = make_mujoco_urdf(args.control_repo, args.workcell, args.urdf)
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_urdf))
        data = mujoco.MjData(model)
        configure_visuals(model, data)
        target_q = set_initial_robot_qpos(mujoco, model, data, [0, 0, 0, 0, 0, 0])
        set_initial_object_poses(mujoco, model, data, args.workcell)
        mujoco.mj_forward(model, data)

        leader.connect(calibrate=True)
        print(f"MuJoCo scene: {scene_urdf}")
        print("Streaming leader joints into MuJoCo. Press Ctrl+C to stop.")
        try:
            if args.no_viewer:
                run_loop(mujoco, model, data, target_q, leader, args)
            else:
                import mujoco.viewer

                with mujoco.viewer.launch_passive(model, data) as viewer:
                    configure_viewer_camera(viewer)
                    configure_viewer_lighting(viewer)
                    run_loop(mujoco, model, data, target_q, leader, args, viewer)
        except KeyboardInterrupt:
            pass
        finally:
            leader.disconnect()
    finally:
        tmpdir.cleanup()


if __name__ == "__main__":
    main()
