#!/usr/bin/env python3
"""Move the reBot arm between poses in simulation (script entry point).

See rebot_b601_sim/src/rebot_b601_sim/move_robot.py for full docs.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rebot_b601_sim.move_robot import main

if __name__ == "__main__":
    main()
