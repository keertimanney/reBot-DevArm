"""Pose-to-pose robot motion with end-effector-down constraint.

Supports both MeshCAT (visual FK) and MuJoCo (physics) backends.
Trajectory is planned with Pinocchio SE(3) geodesic CLIK.

EEF-down orientation: roll=π, pitch=0, yaw=user-supplied.
This aligns the local tool Z-axis with world -Z (pointing at the floor).

Usage (Python API)
------------------
    from rebot_b601_sim.robot_mover import RobotMover

    # MeshCAT - interactive, immediate playback
    mover = RobotMover(backend="meshcat")
    mover.move_to(0.35, 0.0, 0.15)
    mover.move_to(0.40, 0.10, 0.10, yaw=0.3)

    # MuJoCo - queue moves, then launch viewer (blocking)
    mover = RobotMover(backend="mujoco")
    mover.move_to(0.35, 0.0, 0.15)
    mover.move_to(0.40, 0.10, 0.10)
    mover.run_mujoco()           # opens viewer, runs physics, blocks until closed
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pinocchio as pin


# ── import bootstrap ──────────────────────────────────────────────────────────

def _ensure_rebotarm_importable() -> None:
    """Make reBotArm_control_py.kinematics/trajectory importable without hardware deps.

    The package __init__.py imports `actuator` which requires `motorbridge`
    (a hardware driver not installed in sim environments).  We locate the real
    package directory, add it to sys.path, and install a lightweight stub as
    the top-level package so Python resolves submodule imports without executing
    __init__.py and triggering the missing hardware dependency.
    """
    import types

    # Find the reBotArm_control_py package directory from the repo layout.
    here = Path(__file__).resolve()
    pkg_dir: Path | None = None
    for parent in here.parents:
        candidate = parent / "reBotArm_control_py" / "reBotArm_control_py"
        if (candidate / "__init__.py").exists():
            pkg_dir = candidate
            break

    if pkg_dir is None:
        # Not found in repo — maybe it's properly installed; check directly.
        try:
            from reBotArm_control_py.kinematics import robot_model  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Cannot find reBotArm_control_py alongside rebot_b601_sim. "
                "Set PYTHONPATH or install the package."
            ) from exc
        return

    pkg_root = str(pkg_dir.parent)
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

    # Install (or replace) the top-level package stub so __init__.py is skipped.
    # A previously-failed import may have left a broken namespace-package entry.
    existing = sys.modules.get("reBotArm_control_py")
    need_stub = existing is None or list(getattr(existing, "__path__", [])) != [str(pkg_dir)]
    if need_stub:
        stub = types.ModuleType("reBotArm_control_py")
        stub.__path__ = [str(pkg_dir)]
        stub.__package__ = "reBotArm_control_py"
        sys.modules["reBotArm_control_py"] = stub


_ensure_rebotarm_importable()


# ── orientation helper ────────────────────────────────────────────────────────

def _eef_down_rot(yaw: float = 0.0) -> np.ndarray:
    """3×3 rotation with EEF pointing down (local X → world -Z).

    The gripper extends along the local X-axis (confirmed by the MeshCAT visual
    in pinocchio_meshcat.py: palm/fingers placed at +X offsets from end_link).
    pitch=π/2 maps local X to world -Z, i.e. the gripper faces straight down.
    yaw rotates the gripper about the world Z-axis (spin in place while still
    pointing down).
    """
    return pin.rpy.rpyToMatrix(0.0, math.pi / 2, yaw)


# ── parameters ────────────────────────────────────────────────────────────────

@dataclass
class MoveParams:
    """Trajectory and controller tuning knobs."""
    dt: float = 0.02            # trajectory sample period (s)
    linear_speed: float = 0.10  # m/s for auto-duration
    min_duration: float = 1.0   # minimum move time (s)
    ik_max_iter: int = 200
    ik_tolerance: float = 1e-4
    ik_damping: float = 1e-6
    ik_step_size: float = 0.8
    null_gain: float = 0.10     # null-space joint-limit avoidance gain
    kp: float = 35.0            # MuJoCo PD stiffness
    kd: float = 2.0             # MuJoCo PD damping


# ── core class ────────────────────────────────────────────────────────────────

class RobotMover:
    """Move the reBot arm pose-to-pose keeping the end effector pointed down.

    Parameters
    ----------
    backend : ``"meshcat"`` (default) or ``"mujoco"``
    control_repo : path to reBotArm_control_py (auto-detected when None)
    open_browser : open MeshCAT browser tab on init (meshcat backend only)
    workcell : path to workcell JSON to load into the scene (optional)
    params : MoveParams instance for trajectory tuning
    """

    def __init__(
        self,
        backend: str = "meshcat",
        control_repo: Path | None = None,
        open_browser: bool = True,
        workcell: Path | None = None,
        params: MoveParams | None = None,
    ) -> None:
        from reBotArm_control_py.kinematics.robot_model import (
            load_robot_model,
            get_end_effector_frame_id,
        )
        from reBotArm_control_py.trajectory.sampler import TrajPlanParams, TrajProfile
        from reBotArm_control_py.trajectory.clik_tracker import IKParams

        self.backend = backend.lower()
        self.params = params or MoveParams()

        self.model = load_robot_model()
        self.end_frame_id = get_end_effector_frame_id(self.model)
        self.q = pin.neutral(self.model).copy()

        self._ik_params = IKParams(
            max_iter=self.params.ik_max_iter,
            tolerance=self.params.ik_tolerance,
            damping=self.params.ik_damping,
            step_size=self.params.ik_step_size,
        )
        self._traj_params = TrajPlanParams(
            dt=self.params.dt,
            profile=TrajProfile.MIN_JERK,
            accel_ratio=0.25,
        )

        if self.backend == "meshcat":
            self._init_meshcat(control_repo, open_browser, workcell)
        elif self.backend == "mujoco":
            self._pending_trajs: list[tuple] = []
            self._init_mujoco(control_repo, workcell)
        else:
            raise ValueError(f"Unknown backend {backend!r}. Choose 'meshcat' or 'mujoco'.")

    # ── initialisation helpers ────────────────────────────────────────────────

    def _init_meshcat(self, control_repo, open_browser, workcell) -> None:
        from .pinocchio_meshcat import B601MeshcatSim, default_control_repo
        ctrl = (control_repo or default_control_repo()).resolve()
        self._sim = B601MeshcatSim(
            control_repo=ctrl,
            open_browser=open_browser,
        )
        self.q = self._sim.neutral_q.copy()
        if workcell:
            self._sim.load_workcell(workcell)

    def _init_mujoco(self, control_repo, workcell) -> None:
        from .pinocchio_meshcat import default_control_repo, default_workcell_config
        from .mujoco_scene import make_mujoco_urdf
        from .mujoco_sim import (
            import_mujoco, configure_visuals,
            set_initial_robot_qpos, set_initial_object_poses,
        )
        ctrl = (control_repo or default_control_repo()).resolve()
        self._mj = import_mujoco()
        wc_path = workcell or default_workcell_config()
        scene_urdf, self._tmpdir = make_mujoco_urdf(ctrl, wc_path)
        self._mj_model = self._mj.MjModel.from_xml_path(str(scene_urdf))
        self._mj_data = self._mj.MjData(self._mj_model)
        configure_visuals(self._mj_model, self._mj_data)
        self._mj_target_q = set_initial_robot_qpos(
            self._mj, self._mj_model, self._mj_data, [0] * 6
        )
        set_initial_object_poses(self._mj, self._mj_model, self._mj_data, wc_path)
        self._mj.mj_forward(self._mj_model, self._mj_data)

    # ── public API ────────────────────────────────────────────────────────────

    def target_pose(self, x: float, y: float, z: float, yaw: float = 0.0) -> pin.SE3:
        """Return a pin.SE3 at (x, y, z) with end-effector pointing down."""
        return pin.SE3(_eef_down_rot(yaw), np.array([x, y, z], dtype=float))

    def move_to(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float = 0.0,
        duration: float | None = None,
        gripper: float = 0.0,
    ) -> bool:
        """Move the end effector to (x, y, z) keeping it pointed at the floor.

        Parameters
        ----------
        x, y, z : target position in metres
        yaw : rotation of the gripper about world Z (radians)
        duration : move time in seconds — auto-computed from distance when None
        gripper : gripper open command in degrees (0 = closed, -270 = fully open)

        Returns True if IK found a solution.
        """
        target = self.target_pose(x, y, z, yaw)
        dur = duration if duration is not None else self._auto_duration(target.translation)

        q_end, ok = self._solve_ik(target)
        if not ok:
            print(f"[RobotMover] IK failed for pos=({x:.3f}, {y:.3f}, {z:.3f})")
            return False

        traj = self._plan(q_end, dur)
        if self.backend == "meshcat":
            self._play_meshcat(traj, gripper)
        else:
            self._pending_trajs.append((traj, gripper))

        self.q = traj[-1].q.copy()
        return True

    def run_waypoints(
        self,
        waypoints: Sequence[tuple],
        yaw: float = 0.0,
        duration: float | None = None,
        gripper: float = 0.0,
    ) -> None:
        """Move through a list of (x, y, z) or (x, y, z, yaw) waypoints.

        For the MuJoCo backend call ``run_mujoco()`` afterwards to launch the viewer.
        """
        for wp in waypoints:
            wp_yaw = float(wp[3]) if len(wp) >= 4 else yaw
            self.move_to(float(wp[0]), float(wp[1]), float(wp[2]), wp_yaw, duration, gripper)

    def home(self) -> None:
        """Return to the neutral (zero) configuration via a smooth trajectory."""
        q_home = pin.neutral(self.model).copy()
        T_cur = self._fk(self.q)
        T_home = self._fk(q_home)
        dist = float(np.linalg.norm(T_home.translation - T_cur.translation))
        dur = max(self.params.min_duration, dist / self.params.linear_speed)
        traj = self._plan_se3(T_cur, T_home, dur, self.q)
        if self.backend == "meshcat":
            self._play_meshcat(traj, 0.0)
        else:
            self._pending_trajs.append((traj, 0.0))
        self.q = q_home

    def current_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (position_xyz, rpy_rad) of the current end-effector pose."""
        T = self._fk(self.q)
        return T.translation.copy(), pin.rpy.matrixToRpy(T.rotation).copy()

    def run_mujoco(self) -> None:
        """Launch the MuJoCo viewer and execute all pending moves (blocking).

        Call this after queuing moves via ``move_to()`` / ``run_waypoints()``.
        On macOS this must run under ``mjpython`` — the function re-execs
        automatically when invoked via the CLI script.
        """
        if self.backend != "mujoco":
            raise RuntimeError("run_mujoco() is only valid for the 'mujoco' backend.")
        from .mujoco_macos import ensure_mjpython_for_viewer
        from .mujoco_sim import (
            apply_robot_pd, set_gripper_target,
            configure_viewer_camera, configure_viewer_lighting,
            MUJOCO_JOINT_NAMES,
        )
        ensure_mjpython_for_viewer("rebot_b601_sim.robot_mover", no_viewer=False)
        import mujoco.viewer

        mj = self._mj
        model = self._mj_model
        data = self._mj_data
        target_q = self._mj_target_q
        pending = list(self._pending_trajs)
        self._pending_trajs.clear()

        # Flatten to a single time-ordered list of (point, gripper) pairs.
        flat: list[tuple] = []
        offset = 0.0
        for traj, gripper in pending:
            for pt in traj:
                flat.append((pt.time + offset, pt.q, gripper))
            if traj:
                offset += traj[-1].time + self.params.dt

        def _set_target(q_joints, gripper):
            for i, jname in enumerate(MUJOCO_JOINT_NAMES):
                if i >= len(q_joints):
                    break
                jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    continue
                target_q[model.jnt_dofadr[jid]] = q_joints[i]
            set_gripper_target(mj, model, target_q, gripper)

        step_dt = float(model.opt.timestep)
        sim_time = 0.0
        pt_idx = 0

        with mujoco.viewer.launch_passive(model, data) as viewer:
            configure_viewer_camera(viewer)
            configure_viewer_lighting(viewer)
            while viewer.is_running():
                step_start = time.monotonic()
                # Advance trajectory index when simulation time reaches next waypoint.
                while pt_idx < len(flat) and sim_time >= flat[pt_idx][0] - step_dt * 0.5:
                    _, q_joints, gripper = flat[pt_idx]
                    _set_target(q_joints, gripper)
                    pt_idx += 1
                apply_robot_pd(mj, model, data, target_q, self.params.kp, self.params.kd)
                mj.mj_step(model, data)
                viewer.sync()
                sim_time += step_dt
                sleep = step_dt - (time.monotonic() - step_start)
                if sleep > 0:
                    time.sleep(sleep)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _fk(self, q: np.ndarray) -> pin.SE3:
        data = self.model.createData()
        pin.forwardKinematics(self.model, data, q)
        pin.updateFramePlacements(self.model, data)
        return data.oMf[self.end_frame_id].copy()

    def _auto_duration(self, pos: np.ndarray) -> float:
        dist = float(np.linalg.norm(pos - self._fk(self.q).translation))
        return max(self.params.min_duration, dist / self.params.linear_speed)

    def _solve_ik(self, target: pin.SE3) -> tuple[np.ndarray, bool]:
        from reBotArm_control_py.kinematics.inverse_kinematics import solve_ik_with_retry
        data = self.model.createData()
        result = solve_ik_with_retry(
            self.model, data, self.end_frame_id, target, self.q.copy(),
            self._ik_params, max_retries=8,
        )
        return result.q, result.success

    def _plan(self, q_end: np.ndarray, duration: float):
        return self._plan_se3(self._fk(self.q), self._fk(q_end), duration, self.q)

    def _plan_se3(self, T_start: pin.SE3, T_end: pin.SE3, duration: float, q_init: np.ndarray):
        from reBotArm_control_py.trajectory.sampler import plan_cartesian_geodesic_trajectory
        from reBotArm_control_py.trajectory.clik_tracker import track_trajectory
        cart = plan_cartesian_geodesic_trajectory(T_start, T_end, duration, self._traj_params)
        return track_trajectory(
            self.model, self.end_frame_id,
            cart.trajectory, q_init.copy(),
            self._ik_params, self.params.null_gain,
        )

    def _play_meshcat(self, traj, gripper: float) -> None:
        times = [pt.time for pt in traj]
        for i, pt in enumerate(traj):
            self._sim.display_q(pt.q)
            self._sim.display_gripper(gripper)
            if i < len(times) - 1:
                time.sleep(max(0.002, times[i + 1] - times[i]))

    def __del__(self) -> None:
        if hasattr(self, "_tmpdir"):
            try:
                self._tmpdir.cleanup()
            except Exception:
                pass
