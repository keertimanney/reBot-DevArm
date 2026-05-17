#!/usr/bin/env python3
"""Interactive gripper test terminal.

Run ALONE (arm must not be connected) so the serial port is free:
    python release_arm.py
    python gripper_test.py

Commands:
    s          show current state (pos, vel, torque, mode)
    z          set current position as zero
    m          switch control mode  (MIT / POS_VEL / VEL)
    c          send a control command (prompts for values)
    h          show this help
    q          stop loop → disable → exit
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "reBotArm_control_py"))

from reBotArm_control_py.actuator.gripper import Gripper

HELP = """
Commands
--------
  s          show current state
  z          set zero position
  m          switch mode  (MIT / POS_VEL / VEL)
  c          send control command
  h          help
  q          quit (disables motor)
"""


class GripperTerminal:
    def __init__(self):
        self.g = Gripper()
        print("Enabling gripper motor ...")
        ok = self.g.enable()
        print(f"Enable {'OK' if ok else 'FAILED (check status below)'}  mode={self.g.mode}")
        self._show_state()

        self._target_pos = 0.0
        self._target_vel = 0.0
        self._target_tau = 0.0

        self.g.start_control_loop(self._loop, rate=100.0)
        print(f"Control loop running at {self.g._rate} Hz\n")

    def _loop(self, gripper, dt: float) -> None:
        if self.g.mode == "mit":
            self.g.mit(pos=self._target_pos, vel=self._target_vel, tau=self._target_tau)
        elif self.g.mode == "pos_vel":
            self.g.pos_vel(pos=self._target_pos)
        elif self.g.mode == "vel":
            self.g.set_vel(vel=self._target_vel)

    def _show_state(self) -> None:
        pos, vel, torq = self.g.get_state(request=True)
        print(f"  pos={pos:+.4f} rad  vel={vel:+.4f} rad/s  torque={torq:+.4f} Nm  mode={self.g.mode}")

    def run(self) -> None:
        print(HELP)
        while True:
            try:
                cmd = input("gripper> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                cmd = "q"

            if not cmd:
                continue

            if cmd == "q":
                print("Stopping control loop and disabling motor ...")
                self.g.stop_control_loop()
                self.g.disable()
                self.g.disconnect()
                break

            elif cmd == "h":
                print(HELP)

            elif cmd == "s":
                self._show_state()

            elif cmd == "z":
                print("Setting zero position (will disable motor briefly) ...")
                self.g.set_zero()
                print("Zero set. Re-enabling ...")
                self.g.enable()
                self._show_state()

            elif cmd == "m":
                print(f"Current mode: {self.g.mode}")
                print("  0 = MIT      (torque + position + velocity gains)")
                print("  1 = POS_VEL  (position with velocity limit)")
                print("  2 = VEL      (velocity only)")
                sel = input("  Select [0/1/2]: ").strip()
                if sel == "0":
                    self.g.mode_mit()
                    print("Switched to MIT mode.")
                elif sel == "1":
                    self.g.mode_pos_vel()
                    print("Switched to POS_VEL mode.")
                elif sel == "2":
                    self.g.mode_vel()
                    print("Switched to VEL mode.")
                else:
                    print("Invalid selection.")

            elif cmd == "c":
                if self.g.mode == "mit":
                    try:
                        p   = float(input("  target pos (rad): ").strip() or "0.0")
                        v   = float(input("  target vel (rad/s) [0.0]: ").strip() or "0.0")
                        tau = float(input("  feedforward torque (Nm) [0.0]: ").strip() or "0.0")
                        self._target_pos, self._target_vel, self._target_tau = p, v, tau
                        print(f"  → pos={p:+.4f}  vel={v:+.4f}  tau={tau:+.4f}")
                    except ValueError:
                        print("  Invalid input.")
                elif self.g.mode == "pos_vel":
                    try:
                        p = float(input("  target pos (rad): ").strip() or "0.0")
                        self._target_pos = p
                        print(f"  → pos={p:+.4f}")
                    except ValueError:
                        print("  Invalid input.")
                elif self.g.mode == "vel":
                    try:
                        v = float(input("  target vel (rad/s): ").strip() or "0.0")
                        self._target_vel = v
                        print(f"  → vel={v:+.4f}")
                    except ValueError:
                        print("  Invalid input.")
                self._show_state()

            else:
                print(f"Unknown command '{cmd}'. Type h for help.")


if __name__ == "__main__":
    GripperTerminal().run()
