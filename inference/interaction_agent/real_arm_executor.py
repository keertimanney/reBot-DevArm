"""Real hardware ArmExecutor driven by recorded joint-space positions."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

# Make reBotArm importable from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "reBotArm_control_py"))

from reBotArm_control_py.actuator import RobotArm
from reBotArm_control_py.controllers import ArmEndPos

from .arm_executor import ArmExecutor
from .models import Pose, SkillResult


_DEFAULT_SPEED = 0.5   # rad/s — used to auto-compute move duration
_MIN_DURATION  = 2.0   # seconds minimum per segment
_DT            = 0.02  # trajectory sample period (s)


# ── trajectory primitives ──────────────────────────────────────────────────────

def _min_jerk(tau: float) -> float:
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def _auto_duration(q_from: np.ndarray, q_to: np.ndarray, speed: float) -> float:
    max_delta = float(np.max(np.abs(q_to - q_from)))
    return max(_MIN_DURATION, max_delta / speed)


def _move_joints(
    controller: ArmEndPos,
    arm: RobotArm,
    q_target: np.ndarray,
    duration: float,
) -> None:
    """Min-jerk joint-space trajectory. Blocks until complete or stopped.

    Respects controller._stop_send so stop() interrupts mid-trajectory.
    Registers the send thread as controller._send_thread so the ArmEndPos
    infrastructure can join/cancel it correctly.
    """
    q_start, _, _ = arm.get_state()
    n = max(2, int(duration / _DT))

    # Cancel any prior trajectory.
    controller._stop_send.set()
    if controller._send_thread is not None:
        controller._send_thread.join(timeout=1.0)
    controller._stop_send.clear()

    done = threading.Event()

    def _send() -> None:
        interval = duration / n
        for i in range(n):
            if controller._stop_send.is_set():
                break
            tau = i / (n - 1)
            s = _min_jerk(tau)
            controller._q_target[:] = q_start + s * (q_target - q_start)
            time.sleep(interval)
        controller._q_target[:] = q_target
        done.set()

    t = threading.Thread(target=_send, daemon=True)
    controller._send_thread = t   # register so stop() can join it
    t.start()
    done.wait(timeout=duration + 2.0)


# ── executor ───────────────────────────────────────────────────────────────────

class RecordedArmExecutor(ArmExecutor):
    """ArmExecutor that replays joint-space positions from a YAML recording.

    move_block(from, to): moves arm to from_position, then to to_position.
    go_home():            moves to home_label (default 'zero_act').
    stop():               cancels the in-progress trajectory immediately.
    """

    def __init__(
        self,
        positions_file: str | Path,
        home_label: str = "zero_act",
        arm_cfg: str | None = None,
        speed: float = _DEFAULT_SPEED,
    ) -> None:
        data = yaml.safe_load(Path(positions_file).read_text())
        self._positions: dict[str, np.ndarray] = {
            label: np.array(rec["joints_rad"], dtype=np.float64)
            for label, rec in data["positions"].items()
        }
        self._home_label = home_label
        self._speed = speed
        self._stopped = False

        self._arm = RobotArm(cfg_path=arm_cfg)
        self._controller = ArmEndPos(self._arm)

        print("Connecting and enabling motors …")
        self._controller.start()
        print("Motors ready.")

        if home_label in self._positions:
            print(f"Moving to home position '{home_label}' …")
            self._go_to(home_label)
            print("At home.")
        else:
            print(f"WARNING: home label '{home_label}' not in {positions_file}. Skipping home move.")

    # ── ArmExecutor interface ──────────────────────────────────────────────────

    def move_block(
        self,
        from_position: str,
        to_position: str,
        source_pose: Pose,
        target_pose: Pose,
    ) -> SkillResult:
        self._stopped = False

        from_key = self._resolve(from_position)
        to_key   = self._resolve(to_position)

        if from_key is None:
            return SkillResult(
                success=False,
                message=f"Position '{from_position}' is not in the recorded positions.",
                recoverable=True,
            )
        if to_key is None:
            return SkillResult(
                success=False,
                message=f"Position '{to_position}' is not in the recorded positions.",
                recoverable=True,
            )
        if from_key == to_key:
            return SkillResult(
                success=False,
                message="Source and destination are the same position.",
                recoverable=True,
            )

        print(f"[arm] {from_key!r} → {to_key!r}")

        if not self._go_to(from_key):
            return SkillResult(success=False, message=f"Could not reach {from_key}.")

        if self._stopped:
            return SkillResult(success=False, message="Motion stopped.", recoverable=True)

        if not self._go_to(to_key):
            return SkillResult(success=False, message=f"Could not reach {to_key}.")

        return SkillResult(
            success=True,
            message=f"Moved from {from_position} to {to_position}.",
        )

    def go_home(self) -> SkillResult:
        self._stopped = False
        if self._go_to(self._home_label):
            return SkillResult(success=True, message="Arm returned home.")
        return SkillResult(
            success=False,
            message=f"Home position '{self._home_label}' not found.",
        )

    def stop(self) -> SkillResult:
        self._stopped = True
        self._controller._stop_send.set()
        return SkillResult(success=True, message="Motion stopped.")

    def shutdown(self) -> None:
        """Home the arm and disable motors. Call on program exit."""
        try:
            if self._home_label in self._positions:
                self._go_to(self._home_label)
        except Exception:
            pass
        self._controller.end()

    @property
    def position_names(self) -> list[str]:
        return list(self._positions.keys())

    # ── internal helpers ───────────────────────────────────────────────────────

    def _resolve(self, name: str) -> str | None:
        if name in self._positions:
            return name
        lower = name.lower().replace(" ", "_")
        for key in self._positions:
            if key.lower() == lower:
                return key
        return None

    def _go_to(self, label: str) -> bool:
        if label not in self._positions:
            print(f"[arm] '{label}' not in YAML")
            return False
        q_target = self._positions[label]
        q_now, _, _ = self._arm.get_state()
        dur = _auto_duration(q_now, q_target, self._speed)
        print(f"[arm] → '{label}'  dur={dur:.2f}s")
        _move_joints(self._controller, self._arm, q_target, dur)
        return True
