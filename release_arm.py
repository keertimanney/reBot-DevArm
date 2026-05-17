#!/usr/bin/env python3
"""Disable all arm + gripper motors so the arm can be moved freely.

The gripper shares the same serial bus as the arm, so we register it on
the arm's existing controller instead of opening a second connection.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "reBotArm_control_py"))

from reBotArm_control_py.actuator import RobotArm

arm = RobotArm()

# Register the gripper motor (0x07 / feedback 0x17) on the arm's existing
# controller so disable_all() covers it without opening a second serial port.
for ctrl in arm._ctrl_map.values():
    try:
        ctrl.add_damiao_motor(0x07, 0x17, "4310")
    except Exception:
        pass  # already registered or not supported — disable_all still broadcasts

arm.disable()
arm.disconnect()
print("All motors disabled — arm and gripper are free to move.")
