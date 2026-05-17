#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rebot_b601_sim.mujoco_sim import main


if __name__ == "__main__":
    main()

