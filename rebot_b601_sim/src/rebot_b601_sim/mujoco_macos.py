from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def ensure_mjpython_for_viewer(module_name: str, no_viewer: bool) -> None:
    """Re-launch under mjpython on macOS when using MuJoCo's passive viewer."""
    if no_viewer or sys.platform != "darwin" or os.environ.get("MUJOCO_MJPYTHON_REEXEC") == "1":
        return

    executable = Path(sys.executable).name
    if executable == "mjpython":
        return

    mjpython = shutil.which("mjpython")
    if mjpython is None:
        candidate = Path(sys.executable).with_name("mjpython")
        if candidate.exists():
            mjpython = str(candidate)
    if mjpython is None:
        raise SystemExit(
            "MuJoCo's macOS viewer requires mjpython.\n"
            "Run the command with --no-viewer, or use:\n"
            f"  mjpython -m {module_name} {' '.join(sys.argv[1:])}"
        )

    env = os.environ.copy()
    env["MUJOCO_MJPYTHON_REEXEC"] = "1"
    cmd = [mjpython, "-m", module_name, *sys.argv[1:]]
    raise SystemExit(subprocess.run(cmd, env=env).returncode)

