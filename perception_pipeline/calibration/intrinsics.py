"""Camera intrinsics calibration via OpenCV checkerboard.

Usage:
    python calibration/intrinsics.py --camera top --pattern 9x6 --square 0.025
    python calibration/intrinsics.py --camera wrist --pattern 9x6 --square 0.025

Captures ~20 images of a checkerboard at varied poses, solves for K and
distortion, writes to configs/extrinsics.yaml under the appropriate key.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# TODO: implement.
#   - open the camera (cv2.VideoCapture or your driver)
#   - on keypress, try to detect cv2.findChessboardCornersSB
#   - collect ~20 sets of corner points
#   - cv2.calibrateCamera(...)
#   - merge into configs/extrinsics.yaml without clobbering other keys

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", choices=["top", "wrist"], required=True)
    p.add_argument("--pattern", default="9x6", help="inner corners, WxH")
    p.add_argument("--square", type=float, default=0.025, help="square size in meters")
    p.add_argument("--out", default="configs/extrinsics.yaml")
    args = p.parse_args()
    raise NotImplementedError(
        "Implement using cv2.findChessboardCornersSB + cv2.calibrateCamera. "
        "See OpenCV camera_calibration tutorial; this is a 30-line script."
    )


if __name__ == "__main__":
    sys.exit(main())
