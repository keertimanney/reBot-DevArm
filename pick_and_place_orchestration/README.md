# reBot Pick-And-Place Orchestration Interfaces

This package is the stable boundary between the pick-and-place reasoning layer,
`pose_pipeline`, and the B601 motion/control stack.

It exposes:

- `fk(q) -> T_base_tool`
- `MotionPlanner`
- `Gripper`
- `Q_OBSERVE`
- `move_and_settle(planner, q, settle_ms=200)`
- `PickAndPlaceSkill`, a thin orchestration shell matching the runtime state machine

Install from the repo root:

```bash
python -m pip install -e pick_and_place_orchestration
```

Example imports:

```python
from rebot_pick_place import MotionPlanner, Gripper, Q_OBSERVE, fk
```

The planner can run as a dry-run kinematic planner without hardware:

```python
planner = MotionPlanner()
planner.move_to_joint_config(Q_OBSERVE)
T_base_tool = fk(planner.get_current_joints())
```

For hardware, pass a connected `SeeedB601DMFollower`-compatible robot:

```python
planner = MotionPlanner(robot=follower)
gripper = Gripper(robot=follower)
```

Main orchestration shell:

```python
from rebot_pick_place import PickAndPlaceSkill

skill = PickAndPlaceSkill(
    pose_estimator=pose_estimator,
    camera_provider=camera_provider,
    planner=planner,
    gripper=gripper,
)
result = skill.pick_and_place(
    query="red 2x4 block",
    T_base_place=T_base_place,
    candidate_cad_ids=["red_2x4"],
)
```

## Current Limits

- FK and IK use Pinocchio on the upstream B601 URDF.
- `move_to_pose` uses iterative IK and then joint-space execution.
- `move_linear_cartesian` samples straight Cartesian translation and solves IK at each step.
- Collision checking is not implemented yet; collision status is reserved in the result enum.
- The `compliant_mode(...)` context is a stable API hook. It calls
  `robot.enter_compliant_mode(settings)` / `robot.exit_compliant_mode()` if those
  methods are present. The current follower does not yet expose true Damiao MIT
  Cartesian compliance through Motorbridge, so this is ready for the low-level
  implementation but currently falls back to position-commanded motion.
