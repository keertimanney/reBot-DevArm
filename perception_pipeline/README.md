# pose_pipeline

Zero-shot 6D pose estimation for CAD-known blocks, output in robot base frame.
Designed for the Seeed reBot B601 arm with two RGB cameras: one fixed top-down,
one wrist-mounted on the gripper. Motion (Pinocchio IK, Motorbridge SDK) is
deliberately out of scope and integrated later.

## What this is

A perception skill. Given:

- a language description of what to find ("the red 2x4 brick")
- a top-down RGB image
- a wrist RGB image
- the current tool flange pose `T_base_tool` from forward kinematics
- a set of candidate CAD meshes to match against

it returns a list of `DetectedObject` with a 4x4 `T_base_obj` pose, a confidence
score, and per-view masks. The upstream reasoning agent decides which object to
pick and where to place it; this pipeline only answers "where is it, exactly, in
the robot's frame."

## Pipeline

```
LANGUAGE ───┐
            ▼
       ┌─────────────────┐
TOP ──▶│ Grounded-SAM 2  │──▶ mask + class
       └─────────────────┘         │
                                   ▼
       ┌─────────────────┐    ┌──────────────────┐
       │ Depth Anything  │───▶│ FoundationPose   │──▶ T_cam_obj
       │      V2         │    │ (CAD mesh)       │
       └─────────────────┘    └──────────────────┘         │
                                                            ▼
                              ┌──────────────────┐    T_base_cam (known)
                              │ T_base_cam @     │◀──────────┘
                              │ T_cam_obj        │
                              └──────────────────┘
                                       │
                                       ▼
                                T_base_obj (output)

WRIST ─▶ (same pipeline, T_base_cam = T_base_tool @ T_tool_wrist)
```

Two entry points:

- `estimate(rgb_top, rgb_wrist, T_base_tool, query, candidate_cad_ids)` — wide
  scan from the observe pose, returns all matching objects in base frame.
- `refine_close_range(rgb_wrist, T_base_tool, prior_T_base_obj, cad_id)` —
  high-precision pose after the arm has approached, called by your motion code.

## Frames

Everything in the pipeline either is or produces an SE(3) transform. The four
that matter:

| Transform        | Meaning                                           | Source                                                    |
|------------------|---------------------------------------------------|-----------------------------------------------------------|
| `T_base_top`     | Top camera pose in robot base frame               | One-time **eye-to-hand** calibration (AprilTag on gripper)|
| `T_tool_wrist`   | Wrist camera pose relative to tool flange         | One-time **eye-in-hand** calibration (AprilTag in world)  |
| `T_base_tool(q)` | Tool flange pose given joint angles               | Pinocchio FK on B601 URDF, computed at observation time   |
| `T_base_wrist(q)`| Wrist camera in base frame                        | `T_base_tool(q) @ T_tool_wrist`                           |

`T_base_top` and `T_tool_wrist` are calibrated once and saved to
`configs/extrinsics.yaml`. `T_base_tool` is recomputed each call.

## Observe pose

A fixed joint configuration `q_observe` chosen so both cameras see the workspace
clearly. Stored in `configs/default.yaml`. The pipeline doesn't choose or care
about this — your motion stack moves the arm to `q_observe`, captures both
frames, calls `estimate(...)`. Recommended starting point: arm folded so the
wrist camera looks roughly along the same axis as the top camera, ~30cm above
the table.

## Vision prompts are general

Grounded-SAM 2 prompts describe *kinds of things in the scene*, not specific
blocks. Identity comes from CAD geometry matching, not language. This means:

- Adding a new block type → add its mesh to `cad_library/`, no code change
- The reasoning agent's "red 2x4" passes through; FoundationPose verifies which
  CAD it actually is

See `pose_pipeline/grounding.py` for the prompt templates.

## Directory layout

```
pose_pipeline/
├── README.md
├── pyproject.toml
├── pose_pipeline/                 # the library
│   ├── __init__.py
│   ├── pose_estimator.py          # PoseEstimator class, public API
│   ├── types.py                   # DetectedObject, PoseEstimate dataclasses
│   ├── frames.py                  # extrinsics loader, FK passthrough
│   ├── grounding.py               # Grounded-SAM 2 wrapper + prompt constants
│   ├── depth.py                   # Depth Anything V2 + metric scaling
│   ├── foundation_pose.py         # FoundationPose wrapper
│   ├── fusion.py                  # SE(3) weighted mean, cross-view association
│   └── cad.py                     # CAD library loader
├── calibration/
│   ├── intrinsics.py              # OpenCV checkerboard for K_top, K_wrist
│   ├── hand_eye_top.py            # eye-to-hand for T_base_top
│   ├── hand_eye_wrist.py          # eye-in-hand for T_tool_wrist
│   └── validate_extrinsics.py     # sanity check: AprilTag at known base pose
├── cad_library/
│   ├── manifest.yaml              # cad_id -> mesh path, color hint, symmetries
│   └── meshes/                    # .obj / .stl per block type
├── configs/
│   ├── default.yaml               # model names, thresholds, observe pose
│   └── extrinsics.yaml            # T_base_top, T_tool_wrist (written by calib)
├── scripts/
│   ├── run_estimate.py            # CLI: load images + FK, call estimate()
│   └── visualize.py               # overlay masks + axes on captured frames
└── tests/
    ├── test_frames.py
    ├── test_fusion.py
    └── fixtures/                  # tiny synthetic mesh + image pair
```

## Install

```bash
# main env (Python 3.10+ recommended for FoundationPose compatibility)
conda create -n pose python=3.10 -y
conda activate pose
pip install -e .

# heavy deps (install separately; each has its own CUDA notes)
# 1. Grounded-SAM 2
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git ../Grounded-SAM-2
pip install -e ../Grounded-SAM-2

# 2. FoundationPose
git clone https://github.com/NVlabs/FoundationPose.git ../FoundationPose
# follow their build instructions; uses CUDA kernels for rendering

# 3. Depth Anything V2
pip install depth-anything-v2  # or clone from github
```

On a 4090: SAM 2 (~20-40 Hz at 720p), Depth Anything V2 Base (~30 Hz), and
FoundationPose (~30 Hz inference, slower one-shot register) all run concurrently
with room to spare.

## Calibration workflow (do this first)

Pose estimation is no better than your extrinsics. Sequence:

1. **Intrinsics for both cameras** — `calibration/intrinsics.py` with a
   checkerboard. Writes `K_top`, `K_wrist`, distortion coeffs to
   `configs/extrinsics.yaml`.
2. **Top camera eye-to-hand** — `calibration/hand_eye_top.py`. Mount an AprilTag
   to the gripper. Move the arm through ~20 varied poses, capture (image,
   `T_base_tool`) pairs. Solve via OpenCV `calibrateHandEye`. Writes
   `T_base_top`.
3. **Wrist camera eye-in-hand** — `calibration/hand_eye_wrist.py`. Fix an
   AprilTag to the table. Move the arm through ~20 poses looking at it from
   different angles. Solve. Writes `T_tool_wrist`.
4. **Validate** — `calibration/validate_extrinsics.py`. Put an AprilTag at a
   known, measured position on the table. The script estimates its pose through
   both cameras and reports the error in base frame. Aim for <3mm translation,
   <1° rotation. If worse, redo the calibration with more poses or better
   coverage.

Do not skip step 4. A 5mm extrinsic error looks identical to a 5mm
FoundationPose error, and debugging the wrong layer wastes days.

## Quick usage

```python
from pose_pipeline import PoseEstimator
import yaml, numpy as np, cv2

cfg = yaml.safe_load(open("configs/default.yaml"))
ext = yaml.safe_load(open("configs/extrinsics.yaml"))

estimator = PoseEstimator(
    cad_library_path="cad_library/manifest.yaml",
    T_base_top=np.array(ext["T_base_top"]),
    T_tool_wrist=np.array(ext["T_tool_wrist"]),
    K_top=np.array(ext["K_top"]),
    K_wrist=np.array(ext["K_wrist"]),
    config=cfg,
)

# at runtime, after moving to q_observe:
rgb_top   = cv2.cvtColor(cap_top.read()[1],   cv2.COLOR_BGR2RGB)
rgb_wrist = cv2.cvtColor(cap_wrist.read()[1], cv2.COLOR_BGR2RGB)
T_base_tool = my_fk_function(current_joint_angles)   # 4x4

detections = estimator.estimate(
    rgb_top=rgb_top,
    rgb_wrist=rgb_wrist,
    T_base_tool=T_base_tool,
    query="red 2x4 brick",
    candidate_cad_ids=["brick_2x4_red", "brick_2x4_blue", "brick_2x2_red"],
)

for d in detections:
    print(d.label, d.confidence)
    print(d.T_base_obj)   # 4x4 in robot base frame, ready for IK
```

## Integration with Pinocchio (later)

`frames.py` exposes a `make_fk(urdf_path)` factory that returns a function
`q -> T_base_tool`. Default uses Pinocchio if available, falls back to a
user-provided callable. Your motion stack imports that and passes the FK
function in:

```python
from pose_pipeline.frames import make_fk
fk = make_fk("/path/to/rebot_b601.urdf", tool_link="gripper_tool")
T_base_tool = fk(q_current)
```

Everything downstream stays framework-agnostic.

## What this pipeline does NOT do

- Choose objects to pick — that's the reasoning agent.
- Plan motion or solve IK — that's Pinocchio / curobo / MoveIt.
- Track objects over time — single-shot per call. FoundationPose has a tracker
  mode; wire it up if you need closed-loop visual servoing during assembly.
- Handle objects without a CAD mesh — model-free FoundationPose is possible but
  not in this version. Add later if you need it.

## Failure modes the API surfaces

`estimate()` returns an empty list or partial results rather than raising. Each
`DetectedObject` carries a `confidence` and `source_view`. Common diagnostics:

| What you see                              | Likely cause                                |
|-------------------------------------------|---------------------------------------------|
| Empty list, query was reasonable          | Grounding DINO didn't fire — drop the box/text thresholds, or use the generic prompt |
| Detections exist but low confidence       | Wrong CAD candidates, or extrinsics off     |
| Top and wrist disagree by >1cm            | Hand-eye calibration suspect; re-run step 4 |
| Pose flips 180° on symmetric blocks       | Expected; add symmetries to `manifest.yaml` so FoundationPose canonicalises |
| Mask correct, pose drifts as block moves  | Depth scaling broken; check table-plane fit |

## Status

Skeleton with stubs and TODOs. Calibration scripts are the first thing to
flesh out — until you trust the extrinsics, nothing else matters.
