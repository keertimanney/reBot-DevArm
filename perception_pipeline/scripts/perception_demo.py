"""Standalone perception demo for Apple Silicon (MPS).

Runs Grounding DINO + SAM 2 + Depth Anything V2 on one image, all local.
This is a proof-of-life for the perception models used by pose_pipeline; it
does NOT touch the (currently stubbed) pose_pipeline package or FoundationPose.

Usage:
    python scripts/perception_demo.py --image media/v1.0.png --prompt "robot. arm. gripper."
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    pipeline,
    Sam2Model,
    Sam2Processor,
)


GDINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM2_ID = "facebook/sam2.1-hiera-tiny"
DEPTH_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_grounding(image: Image.Image, prompt: str, device: str,
                  box_threshold: float, text_threshold: float):
    processor = AutoProcessor.from_pretrained(GDINO_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to(device).eval()

    # GDino expects lowercased, period-separated phrases.
    text = prompt.lower().strip()
    if not text.endswith("."):
        text += "."

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    dt = time.time() - t0

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],  # (H, W)
    )[0]
    boxes = results["boxes"].cpu().numpy()       # (N, 4) xyxy
    scores = results["scores"].cpu().numpy()     # (N,)
    labels = results.get("labels") or results.get("text_labels") or [""] * len(boxes)
    print(f"  grounding: {len(boxes)} boxes in {dt:.2f}s")
    for b, s, l in zip(boxes, scores, labels):
        print(f"    [{s:.2f}] {l}  bbox={b.round(1).tolist()}")
    return boxes, scores, list(labels)


def run_sam2(image: Image.Image, boxes: np.ndarray, device: str):
    if len(boxes) == 0:
        return np.zeros((0, image.height, image.width), dtype=bool)

    processor = Sam2Processor.from_pretrained(SAM2_ID)
    model = Sam2Model.from_pretrained(SAM2_ID).to(device).eval()

    # SAM2 expects input_boxes shaped (batch=1, num_boxes, 4) in xyxy pixel coords.
    inputs = processor(
        images=image,
        input_boxes=[boxes.tolist()],
        return_tensors="pt",
    ).to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    dt = time.time() - t0

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        original_sizes=inputs["original_sizes"].cpu(),
    )
    # `masks` is a list (length=batch=1) of tensors shaped (num_boxes, num_masks, H, W)
    masks = masks[0].numpy()
    if masks.ndim == 4:
        masks = masks[:, 0]  # take the single mask per box
    print(f"  sam2: {masks.shape[0]} masks in {dt:.2f}s")
    return masks.astype(bool)


def run_depth(image: Image.Image, device: str):
    pipe = pipeline("depth-estimation", model=DEPTH_ID, device=device)
    t0 = time.time()
    out = pipe(image)
    dt = time.time() - t0
    depth = np.array(out["depth"]).astype(np.float32)
    print(f"  depth: {depth.shape} in {dt:.2f}s "
          f"(min={depth.min():.1f}, max={depth.max():.1f}, relative units)")
    return depth


def visualize(image: Image.Image, boxes, scores, labels, masks, depth, out_path: Path):
    arr = np.array(image)
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    axes[0].imshow(arr); axes[0].set_title("input"); axes[0].axis("off")

    axes[1].imshow(arr); axes[1].set_title(f"Grounding DINO ({len(boxes)} boxes)")
    axes[1].axis("off")
    for box, s, l in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        axes[1].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    fill=False, edgecolor="lime", linewidth=2))
        axes[1].text(x1, max(y1 - 4, 0), f"{l} {s:.2f}",
                     color="black", backgroundcolor="lime", fontsize=8)

    overlay = arr.copy()
    rng = np.random.default_rng(0)
    for m in masks:
        color = rng.integers(80, 255, size=3)
        overlay[m] = (0.5 * overlay[m] + 0.5 * color).astype(np.uint8)
    axes[2].imshow(overlay); axes[2].set_title(f"SAM 2 masks ({len(masks)})")
    axes[2].axis("off")

    axes[3].imshow(depth, cmap="inferno")
    axes[3].set_title("Depth Anything V2 (relative)")
    axes[3].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--prompt", default="block. brick. cube. object.",
                    help="Period-separated Grounding DINO query")
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--output", type=Path, default=Path("perception_demo_out.png"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or pick_device()
    print(f"device: {device}")
    image = Image.open(args.image).convert("RGB")
    print(f"image: {args.image} ({image.size[0]}x{image.size[1]})")

    print("→ grounding dino")
    boxes, scores, labels = run_grounding(
        image, args.prompt, device, args.box_threshold, args.text_threshold,
    )

    print("→ sam2")
    masks = run_sam2(image, boxes, device)

    print("→ depth anything v2")
    depth = run_depth(image, device)

    print("→ visualize")
    visualize(image, boxes, scores, labels, masks, depth, args.output)


if __name__ == "__main__":
    main()
