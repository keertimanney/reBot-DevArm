#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class JointCfg:
    name: str
    motor_id: int
    feedback_id: int
    model: str
    vlim: float
    vel_kp: float
    vel_ki: float
    pos_kp: float
    pos_ki: float


@dataclass(frozen=True)
class RecordedPose:
    label: str
    xyz: list[float]
    rpy_rad: list[float]
    joints_rad: list[float]


def add_control_package_to_path(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root / "reBotArm_control_py"))


def load_arm_config(path: Path) -> tuple[str, list[JointCfg]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    joints = []
    for item in data["joints"]:
        pos_vel = item.get("POS_VEL", {})
        joints.append(
            JointCfg(
                name=str(item["name"]),
                motor_id=int(item["motor_id"]),
                feedback_id=int(item["feedback_id"]),
                model=str(item.get("model", "4340P")),
                vlim=float(pos_vel.get("vlim", 1.0)),
                vel_kp=float(pos_vel.get("vel_kp", 0.0)),
                vel_ki=float(pos_vel.get("vel_ki", 0.0)),
                pos_kp=float(pos_vel.get("pos_kp", 0.0)),
                pos_ki=float(pos_vel.get("pos_ki", 0.0)),
            )
        )
    return str(data["channel"]), joints


def load_recorded_poses(path: Path) -> dict[str, RecordedPose]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    poses = {}
    for label, record in (data.get("positions") or {}).items():
        ee = (record or {}).get("end_effector") or {}
        xyz = ee.get("xyz")
        rpy = ee.get("rpy_rad")
        if not isinstance(xyz, list) or len(xyz) != 3:
            raise ValueError(f"{label!r} is missing end_effector.xyz")
        if not isinstance(rpy, list) or len(rpy) != 3:
            raise ValueError(f"{label!r} is missing end_effector.rpy_rad")
        poses[str(label)] = RecordedPose(
            label=str(label),
            xyz=[float(v) for v in xyz],
            rpy_rad=[float(v) for v in rpy],
            joints_rad=[float(v) for v in (record.get("joints_rad") or [])],
        )
    return poses


def request_state(ctrl: Any, motors: list[Any]) -> np.ndarray:
    for motor in motors:
        motor.request_feedback()
    ctrl.poll_feedback_once()
    q = []
    for motor in motors:
        state = motor.get_state()
        q.append(float(state.pos if state is not None else 0.0))
    return np.asarray(q, dtype=np.float64)


def ensure_pos_vel(motors: list[Any], joints: list[JointCfg], mode: Any) -> None:
    for motor, joint in zip(motors, joints):
        motor.write_register_f32(25, joint.vel_kp)
        motor.write_register_f32(26, joint.vel_ki)
        motor.write_register_f32(27, joint.pos_kp)
        motor.write_register_f32(28, joint.pos_ki)
        time.sleep(0.02)
        try:
            motor.ensure_mode(mode, 1000)
        except Exception as exc:
            print(f"[warn] ensure POS_VEL failed for {joint.name}: {exc}")


def plan_to_pose(pose: RecordedPose, q_start: np.ndarray, duration: float) -> list[np.ndarray]:
    import pinocchio as pin
    from reBotArm_control_py.kinematics import (
        compute_fk,
        get_end_effector_frame_id,
        load_robot_model,
        pos_rot_to_se3,
    )
    from reBotArm_control_py.kinematics.inverse_kinematics import IKParams, solve_ik
    from reBotArm_control_py.trajectory import (
        IKParams as ClikIKParams,
        TrajPlanParams,
        TrajProfile,
        plan_cartesian_geodesic_trajectory,
        track_trajectory,
    )

    model = load_robot_model()
    data = model.createData()
    frame_id = get_end_effector_frame_id(model)
    roll, pitch, yaw = pose.rpy_rad
    x, y, z = pose.xyz

    q_seed = np.asarray(pose.joints_rad, dtype=np.float64)
    if q_seed.shape != q_start.shape:
        q_seed = q_start

    target = pos_rot_to_se3(np.array([x, y, z]), roll=roll, pitch=pitch, yaw=yaw)
    ik_result = solve_ik(
        model,
        data,
        frame_id,
        target,
        q_seed,
        IKParams(max_iter=500, tolerance=1e-4, step_size=0.5, damping=1e-6),
    )
    if not ik_result.success:
        raise RuntimeError(
            f"IK failed for target {pose.label} ({x:.3f}, {y:.3f}, {z:.3f}); "
            f"final error={ik_result.error:.4e}"
        )

    _, _, h_start = compute_fk(model, q_start)
    _, _, h_end = compute_fk(model, ik_result.q)
    cart = plan_cartesian_geodesic_trajectory(
        h_start,
        h_end,
        duration,
        TrajPlanParams(dt=0.02, profile=TrajProfile.MIN_JERK),
    )
    joint_traj = track_trajectory(
        model,
        frame_id,
        cart.trajectory,
        q_start,
        ClikIKParams(max_iter=200, tolerance=1e-4, damping=1e-6, step_size=0.8),
        null_gain=0.1,
    )
    if not joint_traj:
        raise RuntimeError(f"Trajectory tracking failed for target {pose.label}")
    return [pt.q.copy() for pt in joint_traj]


def stream_points(
    motors: list[Any],
    joints: list[JointCfg],
    points: list[np.ndarray],
    duration: float,
    label: str,
) -> None:
    if not points:
        return
    interval = duration / len(points)
    vlims = [joint.vlim for joint in joints]
    print(f"[move:{label}] start streaming {len(points)} points, dt={interval:.3f}s")
    for idx, point in enumerate(points, start=1):
        for motor, target, vlim in zip(motors, point, vlims):
            motor.send_pos_vel(float(target), float(vlim))
        if idx == 1 or idx == len(points) or idx % 25 == 0:
            print(
                f"[move:{label}] point {idx:03d}/{len(points)} "
                f"q={[round(float(v), 3) for v in point]}"
            )
        time.sleep(interval)
    print(f"[move:{label}] done")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move through recorded poses using IK planning and direct motorbridge POS_VEL streaming."
    )
    parser.add_argument("--recorded", default="recorded_arm_positions.yaml")
    parser.add_argument("--arm-config", default="/tmp/rebot_arm_mac.yaml")
    parser.add_argument("--sequence", nargs="+", default=["A", "C", "B", "A"])
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--serial-baud", type=int, default=921600)
    parser.add_argument("--no-disable", action="store_true", help="Leave motors enabled at the end.")
    parser.add_argument(
        "--seed",
        choices=["recorded", "live"],
        default="recorded",
        help="IK seed source for the first target. Use recorded to avoid stale/bad live feedback.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    add_control_package_to_path(repo_root)

    from motorbridge import Controller, Mode

    poses = load_recorded_poses(Path(args.recorded))
    missing = [label for label in args.sequence if label not in poses]
    if missing:
        raise SystemExit(f"Missing recorded poses: {missing}. Available: {sorted(poses)}")

    channel, joints = load_arm_config(Path(args.arm_config))
    print(f"Connecting to {channel}")
    print("=" * 72)
    print("Recorded IK/POS_VEL sequence")
    print("Sequence:", " -> ".join(args.sequence))
    print(f"Duration per segment: {args.duration:.2f}s")
    print("=" * 72)

    ctrl = Controller.from_dm_serial(channel, args.serial_baud)
    motors = []
    try:
        for joint in joints:
            motors.append(ctrl.add_damiao_motor(joint.motor_id, joint.feedback_id, joint.model))

        ctrl.enable_all()
        time.sleep(0.3)
        ensure_pos_vel(motors, joints, Mode.POS_VEL)

        q_live = request_state(ctrl, motors)
        print("Live q:", [round(float(v), 4) for v in q_live])

        first_pose = poses[args.sequence[0]]
        if args.seed == "recorded":
            if len(first_pose.joints_rad) != len(joints):
                raise RuntimeError(f"Recorded pose {first_pose.label} does not include {len(joints)} joint angles")
            q_curr = np.asarray(first_pose.joints_rad, dtype=np.float64)
            print("IK seed q from recorded first pose:", [round(float(v), 4) for v in q_curr])
        else:
            q_curr = q_live

        for label in args.sequence:
            pose = poses[label]
            if len(pose.joints_rad) == len(joints):
                recorded_joints = [round(float(v), 3) for v in pose.joints_rad]
            else:
                recorded_joints = []
            print(
                f"\n[target:{label}] "
                f"xyz={tuple(round(v, 4) for v in pose.xyz)} "
                f"rpy={tuple(round(v, 4) for v in pose.rpy_rad)} "
                f"recorded_q={recorded_joints}"
            )
            points = plan_to_pose(pose, q_curr, args.duration)
            print(
                f"[target:{label}] planned {len(points)} points: "
                f"q_start={[round(float(v), 3) for v in points[0]]} "
                f"q_end={[round(float(v), 3) for v in points[-1]]}"
            )
            stream_points(motors, joints, points, args.duration, label=label)
            q_curr = points[-1].copy()
            time.sleep(0.25)

        print("\n" + "=" * 72)
        print("Sequence complete:", " -> ".join(args.sequence))
        print("=" * 72)
    finally:
        if not args.no_disable:
            try:
                ctrl.disable_all()
                time.sleep(0.2)
            except Exception as exc:
                print(f"[warn] disable_all failed: {exc}")
        for motor in motors:
            try:
                motor.close()
            except Exception:
                pass
        try:
            ctrl.shutdown()
        except Exception:
            pass
        ctrl.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
