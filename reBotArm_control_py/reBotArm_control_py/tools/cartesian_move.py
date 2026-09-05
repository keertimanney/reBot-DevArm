"""cartesian_move — Claude API tool for Cartesian trajectory generation and execution.

Generates a smooth SE(3) geodesic trajectory from the current end-effector
pose to a target (x, y, z) position and executes it in one of two modes:

  sim   MuJoCo physics simulation with an interactive viewer window.
  real  Stream joint positions to the physical robot via POS_VEL control.

CLI usage (run as a module so relative imports work):

    python -m reBotArm_control_py.tools.cartesian_move \\
        --mode sim -x 0.30 -y 0.0 -z 0.25

    python -m reBotArm_control_py.tools.cartesian_move \\
        --mode real -x 0.30 -y 0.0 -z 0.25 --duration 3.0

Programmatic (Claude API tool call) usage:

    from reBotArm_control_py.tools import TOOL_DEFINITION, move_to_cartesian

    # Pass TOOL_DEFINITION to the Anthropic tools= list.
    # When Claude returns a tool_use block, dispatch it:
    result = move_to_cartesian(**tool_use_block.input)
"""

from __future__ import annotations

import sys
import time
import argparse
from typing import Any

import numpy as np

# ── Claude API tool schema ─────────────────────────────────────────────────────

TOOL_DEFINITION: dict[str, Any] = {
    "name": "move_to_cartesian",
    "description": (
        "Generate a smooth SE(3) geodesic trajectory and execute it to move the "
        "reBot-DevArm end-effector to a target Cartesian position (x, y, z) with "
        "optional orientation (roll, pitch, yaw in radians). "
        "Use mode='sim' for a MuJoCo physics simulation with an interactive viewer, "
        "or mode='real' to stream joint positions to the physical robot via "
        "POS_VEL control at 500 Hz."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "Target end-effector x position in metres.",
            },
            "y": {
                "type": "number",
                "description": "Target end-effector y position in metres.",
            },
            "z": {
                "type": "number",
                "description": "Target end-effector z position in metres.",
            },
            "roll": {
                "type": "number",
                "description": "Target end-effector roll angle in radians (default 0.0).",
            },
            "pitch": {
                "type": "number",
                "description": "Target end-effector pitch angle in radians (default 0.0).",
            },
            "yaw": {
                "type": "number",
                "description": "Target end-effector yaw angle in radians (default 0.0).",
            },
            "duration": {
                "type": "number",
                "description": (
                    "Desired motion duration in seconds (default 2.0). "
                    "Pass 0 or a negative value to auto-size based on Cartesian distance "
                    "(≈ distance / 0.1 m·s⁻¹, minimum 1 s)."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["sim", "real"],
                "description": (
                    "'sim': run in MuJoCo physics simulation with a live viewer window. "
                    "'real': stream joint trajectory to the physical robot."
                ),
            },
        },
        "required": ["x", "y", "z", "mode"],
    },
}

# ── Shared: trajectory planning ────────────────────────────────────────────────

def _plan(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    duration: float,
    q_init: np.ndarray | None = None,
) -> tuple[list[np.ndarray], float]:
    """Solve IK and plan a SE(3) geodesic trajectory.

    Returns
    -------
    joint_pts : list of (nq,) arrays, one per trajectory step (dt = 0.02 s).
    duration  : actual motion duration used (seconds).
    """
    import pinocchio as pin
    from ..kinematics import (
        load_robot_model,
        get_end_effector_frame_id,
        pos_rot_to_se3,
        compute_fk,
    )
    from ..kinematics.inverse_kinematics import solve_ik, IKParams
    from ..trajectory import (
        TrajPlanParams,
        TrajProfile,
        IKParams as ClikIKParams,
        plan_cartesian_geodesic_trajectory,
        track_trajectory,
    )

    model = load_robot_model()
    data = model.createData()
    frame_id = get_end_effector_frame_id(model)

    if q_init is None:
        q_init = pin.neutral(model)

    T_target = pos_rot_to_se3(np.array([x, y, z]), roll=roll, pitch=pitch, yaw=yaw)

    ik_result = solve_ik(
        model, data, frame_id, T_target, q_init,
        IKParams(max_iter=500, tolerance=1e-4, step_size=0.5, damping=1e-6),
    )
    if not ik_result.success:
        raise RuntimeError(
            f"IK failed for target ({x:.3f}, {y:.3f}, {z:.3f}): "
            f"final error = {ik_result.error:.4e} rad (tolerance 1e-4)."
        )

    q_end = ik_result.q
    pos_start, _, H_start = compute_fk(model, q_init)
    _, _, H_end = compute_fk(model, q_end)

    if duration <= 0:
        dist = float(np.linalg.norm(np.array([x, y, z]) - pos_start))
        duration = max(1.0, dist / 0.1)

    params = TrajPlanParams(dt=0.02, profile=TrajProfile.MIN_JERK)
    clik = ClikIKParams(max_iter=200, tolerance=1e-4, damping=1e-6, step_size=0.8)

    # H_start / H_end are (4,4) homogeneous matrices; sampler accepts ndarray or SE3.
    cart = plan_cartesian_geodesic_trajectory(H_start, H_end, duration, params)
    joint_traj = track_trajectory(model, frame_id, cart.trajectory, q_init, clik, null_gain=0.1)

    if not joint_traj:
        raise RuntimeError("Trajectory tracking returned no points — check IK convergence.")

    return [pt.q.copy() for pt in joint_traj], duration


# ── Sim backend ────────────────────────────────────────────────────────────────

def _run_sim(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    duration: float,
    kp: float = 35.0,
    kd: float = 2.0,
) -> dict[str, Any]:
    """Execute trajectory in MuJoCo physics simulation with a viewer."""
    try:
        import mujoco
        import mujoco.viewer as mj_viewer
    except ModuleNotFoundError:
        return {
            "success": False,
            "error": "mujoco not installed. Run: pip install mujoco",
        }

    try:
        from rebot_b601_sim.mujoco_scene import make_mujoco_urdf
        from rebot_b601_sim.mujoco_sim import (
            MUJOCO_JOINT_NAMES,
            apply_robot_pd,
            configure_visuals,
            configure_viewer_camera,
            configure_viewer_lighting,
        )
        from rebot_b601_sim.pinocchio_meshcat import (
            default_control_repo,
            default_workcell_config,
        )
    except ModuleNotFoundError:
        return {
            "success": False,
            "error": (
                "rebot_b601_sim not installed. "
                "Run: pip install -e rebot_b601_sim[mujoco]"
            ),
        }

    print(f"[sim] Planning trajectory → ({x:.3f}, {y:.3f}, {z:.3f}) …")
    joint_pts, duration = _plan(x, y, z, roll, pitch, yaw, duration, q_init=None)
    n_pts = len(joint_pts)
    print(f"[sim] {n_pts} trajectory points over {duration:.2f} s")

    control_repo = default_control_repo()
    workcell = default_workcell_config()
    scene_urdf, tmpdir = make_mujoco_urdf(control_repo, workcell, None)

    try:
        model = mujoco.MjModel.from_xml_path(str(scene_urdf))
        data = mujoco.MjData(model)
        configure_visuals(model, data)
        mujoco.mj_forward(model, data)

        # Build a map: pinocchio joint index → MuJoCo DOF address
        joint_dadr: list[int | None] = []
        for jname in MUJOCO_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            joint_dadr.append(model.jnt_dofadr[jid] if jid >= 0 else None)

        target_q = np.zeros(model.nv)

        def _set_target(q6: np.ndarray) -> None:
            for i, dadr in enumerate(joint_dadr):
                if dadr is not None and i < len(q6):
                    target_q[dadr] = float(q6[i])

        # Initialise simulation at the start configuration
        _set_target(joint_pts[0])
        mujoco.mj_forward(model, data)

        # Number of MuJoCo physics steps per trajectory waypoint
        dt_traj = duration / n_pts
        steps_per_pt = max(1, round(dt_traj / model.opt.timestep))

        with mj_viewer.launch_passive(model, data) as viewer:
            configure_viewer_camera(viewer)
            configure_viewer_lighting(viewer)
            print("[sim] Viewer open — executing trajectory …")

            for pt_q in joint_pts:
                _set_target(pt_q)
                for _ in range(steps_per_pt):
                    if not viewer.is_running():
                        break
                    t0 = time.monotonic()
                    apply_robot_pd(mujoco, model, data, target_q, kp, kd)
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    slack = model.opt.timestep - (time.monotonic() - t0)
                    if slack > 0:
                        time.sleep(slack)

            if viewer.is_running():
                print("[sim] Trajectory complete — close the viewer window to exit.")
                while viewer.is_running():
                    t0 = time.monotonic()
                    apply_robot_pd(mujoco, model, data, target_q, kp, kd)
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    slack = model.opt.timestep - (time.monotonic() - t0)
                    if slack > 0:
                        time.sleep(slack)
    finally:
        tmpdir.cleanup()

    return {
        "success": True,
        "mode": "sim",
        "target": {"x": x, "y": y, "z": z, "roll": roll, "pitch": pitch, "yaw": yaw},
        "duration": duration,
        "n_points": n_pts,
    }


# ── Real backend ───────────────────────────────────────────────────────────────

def _run_real(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    duration: float,
) -> dict[str, Any]:
    """Generate a trajectory and stream joint positions to the physical robot."""
    from ..actuator import RobotArm
    from ..controllers import ArmEndPos

    arm = RobotArm()
    ctrl = ArmEndPos(arm)
    ctrl.start()

    try:
        q_curr, _, _ = arm.get_state()
        print(f"[real] Current joints (rad): {[f'{v:+.3f}' for v in q_curr]}")
        print(f"[real] Planning trajectory → ({x:.3f}, {y:.3f}, {z:.3f}) …")

        # _plan builds the full trajectory from the current joint state so that
        # the geodesic starts at the actual end-effector pose, not neutral.
        joint_pts, duration = _plan(x, y, z, roll, pitch, yaw, duration, q_init=q_curr)
        n_pts = len(joint_pts)
        print(f"[real] {n_pts} trajectory points over {duration:.2f} s")

        # Stream joint positions directly into the running POS_VEL control loop.
        # The 500 Hz loop in RobotArm sends ctrl._q_target to the motors every tick.
        dt_interval = duration / n_pts
        print("[real] Streaming joint positions …")
        for pt_q in joint_pts:
            ctrl._q_target[:] = pt_q
            time.sleep(dt_interval)

        # Hold final position briefly before homing
        print("[real] Trajectory complete — holding final position.")
        time.sleep(0.5)

    finally:
        ctrl.end()  # safe_home() + disconnect

    return {
        "success": True,
        "mode": "real",
        "target": {"x": x, "y": y, "z": z, "roll": roll, "pitch": pitch, "yaw": yaw},
        "duration": duration,
        "n_points": n_pts,
    }


# ── Tool executor (Claude API dispatcher) ──────────────────────────────────────

def move_to_cartesian(
    x: float,
    y: float,
    z: float,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    duration: float = 2.0,
    mode: str = "sim",
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute a move_to_cartesian tool call dispatched by Claude.

    Parameters
    ----------
    x, y, z   : target end-effector position in metres.
    roll, pitch, yaw : target orientation in radians (default 0).
    duration  : motion time in seconds; ≤ 0 = auto from distance.
    mode      : 'sim' or 'real'.
    **kwargs  : extra keyword args forwarded to the backend
                (e.g. kp, kd for sim PD gains).

    Returns
    -------
    dict with at minimum a boolean ``success`` key.
    """
    try:
        if mode == "sim":
            return _run_sim(
                x, y, z, roll, pitch, yaw, duration,
                kp=float(kwargs.get("kp", 35.0)),
                kd=float(kwargs.get("kd", 2.0)),
            )
        elif mode == "real":
            return _run_real(x, y, z, roll, pitch, yaw, duration)
        else:
            return {"success": False, "error": f"Unknown mode {mode!r}. Use 'sim' or 'real'."}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ── CLI entry point ────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate and execute a Cartesian trajectory for the reBot-DevArm.\n\n"
            "Example (sim):  python -m reBotArm_control_py.tools.cartesian_move "
            "--mode sim -x 0.30 -y 0.0 -z 0.25\n"
            "Example (real): python -m reBotArm_control_py.tools.cartesian_move "
            "--mode real -x 0.30 -y 0.0 -z 0.25 --duration 3.0"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["sim", "real"], default="sim",
        help="'sim' = MuJoCo physics sim with viewer; 'real' = physical robot",
    )
    p.add_argument("-x", type=float, required=True, help="Target x (m)")
    p.add_argument("-y", type=float, required=True, help="Target y (m)")
    p.add_argument("-z", type=float, required=True, help="Target z (m)")
    p.add_argument("--roll",     type=float, default=0.0,  help="End-effector roll  (rad)")
    p.add_argument("--pitch",    type=float, default=0.0,  help="End-effector pitch (rad)")
    p.add_argument("--yaw",      type=float, default=0.0,  help="End-effector yaw   (rad)")
    p.add_argument("--duration", type=float, default=2.0,
                   help="Motion duration in seconds (0 = auto-size from distance)")
    p.add_argument("--kp",  type=float, default=35.0, help="[sim] PD stiffness")
    p.add_argument("--kd",  type=float, default=2.0,  help="[sim] PD damping")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    result = move_to_cartesian(
        x=args.x, y=args.y, z=args.z,
        roll=args.roll, pitch=args.pitch, yaw=args.yaw,
        duration=args.duration,
        mode=args.mode,
        kp=args.kp, kd=args.kd,
    )
    if result.get("success"):
        print(f"\n[OK] {result}")
    else:
        print(f"\n[FAIL] {result.get('error', result)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
