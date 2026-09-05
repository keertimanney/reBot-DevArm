"""Make Reachy Mini wave hi: nod head + play 'hi' sound through its speaker."""

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini

HI_WAV = "/tmp/hi.wav"


def main() -> None:
    with ReachyMini(connection_mode="auto") as reachy:
        print("connected — saying hi")

        reachy.media.play_sound(HI_WAV)

        nod_down = np.eye(4)
        nod_down[:3, :3] = R.from_euler("xyz", [0, 15, 0], degrees=True).as_matrix()
        nod_up = np.eye(4)
        nod_up[:3, :3] = R.from_euler("xyz", [0, -5, 0], degrees=True).as_matrix()

        reachy.goto_target(nod_down, antennas=[0.6, -0.6], duration=0.25)
        reachy.goto_target(nod_up, antennas=[-0.4, 0.4], duration=0.25)
        reachy.goto_target(np.eye(4), antennas=[0.0, 0.0], duration=0.3)

        time.sleep(0.4)
        print("done")


if __name__ == "__main__":
    main()
