"""
Move reBot arm to a grid coordinate on the workspace.

Grid is 10x10 (0-9), mapped to the best-fit rectangle over positions A,B,C,D.
  x=0..9: left to right
  y=0..9: far (D/C) to near (B/A)

Usage:
    python goto_grid.py 5 5          # go to center
    python goto_grid.py 0 0          # go to D corner (far-left)
    python goto_grid.py 9 9          # go to A corner (near-right)
    python goto_grid.py 3 7 --speed 0.5  # slower movement
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "inference" / "shared"))

from embodiment_translator import ReBotModel, EEPose
from reBotArm_control_py.actuator import RobotArm


# ---------------------------------------------------------------------------
# Grid definition (best-fit rectangle corners from positions.yaml)
# ---------------------------------------------------------------------------
# Corners in XYZ (meters), projected onto best-fit plane
CORNER_D = np.array([0.18119221, 0.26799131, 0.07723731])  # grid (0,0)
CORNER_C = np.array([0.53194975, 0.29724024, 0.09572378])  # grid (9,0)
CORNER_B = np.array([0.22372723, -0.25021837, 0.09009009]) # grid (0,9)
CORNER_A = np.array([0.57448477, -0.22096944, 0.10857656]) # grid (9,9)

# Default URDF path
DEFAULT_URDF = str(
    Path(__file__).parent / "inference" / "shared" / "reBotArm_control_py" / "urdf"
    / "reBot-DevArm_fixend_description" / "urdf" / "reBot-DevArm_fixend.urdf"
)

# Known good joint config near center (for IK seed)
CENTER_Q = np.array([
    0.01201629638671875, -1.9308385848999023, -1.3460369110107422,
    0.7040128707885742, -0.18101024627685547, -0.049782752990722656
])

CONTROL_HZ = 50
INTERPOLATION_STEPS = 100  # steps to interpolate between current and target


def grid_to_xyz(gx: int, gy: int) -> np.ndarray:
    """Convert grid coordinate (0-9, 0-9) to XYZ position in meters."""
    u = gx / 9.0  # 0=left(D/B), 1=right(C/A)
    v = gy / 9.0  # 0=far(D/C), 1=near(B/A)
    return (1-u)*(1-v)*CORNER_D + u*(1-v)*CORNER_C + (1-u)*v*CORNER_B + u*v*CORNER_A


def move_to_position(arm: RobotArm, model: ReBotModel, target_xyz: np.ndarray,
                     speed: float = 1.0):
    """Move arm to target XYZ using IK with interpolation."""
    # Get current joint state
    current_q, _, _ = arm.get_state()
    current_q = np.array(current_q[:6])

    # Get current EE pose for rotation reference
    current_ee = model.fk(current_q)

    # Target pose: desired XYZ with current rotation (keep end-effector orientation)
    target_ee = EEPose(position=target_xyz, rotation=current_ee.rotation)

    # Solve IK from current position
    target_q, success = model.ik(target_ee, q_init=current_q)
    if not success:
        # Try from center config
        target_q, success = model.ik(target_ee, q_init=CENTER_Q)
        if not success:
            print(f"WARNING: IK did not converge for target {target_xyz}, using best estimate")

    # Interpolate joint trajectory
    steps = int(INTERPOLATION_STEPS / speed)
    dt = 1.0 / CONTROL_HZ

    print(f"Moving to target ({steps} steps at {CONTROL_HZ}Hz)...")
    for i in range(1, steps + 1):
        alpha = i / steps
        # Smooth interpolation (cosine ease)
        alpha_smooth = 0.5 * (1 - np.cos(alpha * np.pi))
        q_cmd = current_q + alpha_smooth * (target_q - current_q)
        arm.mit(pos=q_cmd)
        time.sleep(dt)

    print("Arrived.")


def main():
    parser = argparse.ArgumentParser(description="Move reBot arm to grid coordinate")
    parser.add_argument("x", type=int, help="Grid X coordinate (0-9, left to right)")
    parser.add_argument("y", type=int, help="Grid Y coordinate (0-9, far to near)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speed multiplier (default 1.0, lower=slower)")
    parser.add_argument("--urdf", type=str, default=DEFAULT_URDF,
                        help="Path to reBot URDF file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print target position without moving")
    args = parser.parse_args()

    # Validate grid coordinates
    if not (0 <= args.x <= 9 and 0 <= args.y <= 9):
        print("Error: x and y must be between 0 and 9")
        sys.exit(1)

    # Compute target
    target_xyz = grid_to_xyz(args.x, args.y)
    print(f"Grid ({args.x}, {args.y}) -> XYZ: [{target_xyz[0]:.4f}, {target_xyz[1]:.4f}, {target_xyz[2]:.4f}] m")

    if args.dry_run:
        print("(dry run, not moving)")
        return

    # Initialize model
    print(f"Loading kinematics model...")
    model = ReBotModel(args.urdf)

    # Initialize arm
    print("Connecting to arm...")
    arm = RobotArm()
    arm.connect()
    arm.enable()
    arm.mode_mit()
    print("Arm ready.")

    try:
        move_to_position(arm, model, target_xyz, speed=args.speed)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        # Hold position briefly then release
        time.sleep(0.5)


if __name__ == "__main__":
    main()
