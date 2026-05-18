from __future__ import annotations

import numpy as np

from rebot_pick_place import MotionPlanner, Q_OBSERVE, fk, move_and_settle


def capture_calibration_images():
    raise NotImplementedError("Wire this to the top/wrist camera capture stack.")


def main() -> None:
    planner = MotionPlanner()
    calibration_poses = [
        Q_OBSERVE,
        Q_OBSERVE + np.array([0.25, 0.0, 0.0, 0.0, 0.2, 0.0]),
        Q_OBSERVE + np.array([-0.25, 0.1, 0.0, -0.15, 0.0, 0.2]),
    ]
    samples = []
    for q in calibration_poses:
        result = move_and_settle(planner, q, settle_ms=200)
        if not result.ok:
            raise RuntimeError(result.message)
        T_base_tool = fk(planner.get_current_joints())
        images = capture_calibration_images()
        samples.append((T_base_tool, images))
    print(f"Captured {len(samples)} calibration samples.")


if __name__ == "__main__":
    main()

