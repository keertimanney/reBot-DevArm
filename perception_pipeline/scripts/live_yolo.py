"""
Live YOLO11-seg on a camera feed (or a video file).

Drop-in standalone script — no dependencies on the perception_pipeline.
Defaults to USB camera index 0, which works for plain UVC webcams AND the RGB
stream of an Intel RealSense (D435i / D405) or Orbbec Gemini when no special
driver is needed.

Usage:
    python live_yolo.py                       # webcam 0
    python live_yolo.py --camera 1            # webcam 1
    python live_yolo.py --video wrist.mp4     # test against a file
    python live_yolo.py --save out.mp4        # also record annotated video
    python live_yolo.py --model yolo11n-seg.pt  # smaller/faster
    python live_yolo.py --classes 39 41 67    # filter to bottle, cup, cell phone

Press 'q' or ESC in the window to quit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0, help="cv2 camera index")
    src.add_argument("--video", type=str, help="path to a video file (overrides --camera)")
    p.add_argument("--model", default="yolo11s-seg.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="optional list of COCO class IDs to keep")
    p.add_argument("--save", type=str, default=None, help="path to save annotated mp4")
    p.add_argument("--no-window", action="store_true",
                   help="headless mode (skip cv2.imshow) — use with --save")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device}  model={args.model}")

    source = args.video if args.video else args.camera
    print(f"[init] source={source!r}")

    model = YOLO(args.model)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source!r}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    print(f"[init] frame={w}x{h} src_fps={fps_src:.1f}")

    writer = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h)
        )
        print(f"[init] saving to {args.save}")

    stream = model.track(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        tracker=args.tracker,
        classes=args.classes,
        persist=True,
        stream=True,
        verbose=False,
    )

    n, t0 = 0, time.time()
    try:
        for result in stream:
            annotated = result.plot()

            n += 1
            if n % 10 == 0:
                fps = n / (time.time() - t0)
                cv2.putText(annotated, f"{fps:.1f} FPS",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

            if writer is not None:
                writer.write(annotated)

            if not args.no_window:
                cv2.imshow("yolo11-seg (q/ESC to quit)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - t0
        if n:
            print(f"[done] {n} frames in {elapsed:.1f}s  ({n/elapsed:.1f} FPS)")


if __name__ == "__main__":
    main()
