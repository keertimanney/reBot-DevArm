"""Boot up the Reachy Mini and play a few emotes via the SDK."""

import time

from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves

EMOTES_TO_PLAY = ["happy1", "curious1", "sad1"]


def main() -> None:
    with ReachyMini(connection_mode="auto") as reachy:
        print("connected — waking up")
        reachy.wake_up()
        time.sleep(0.3)

        print("loading emotions library from HuggingFace cache")
        library = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
        available = library.list_moves()
        print(f"available emotes ({len(available)}): {available}")

        chosen = [name for name in EMOTES_TO_PLAY if name in available]
        if not chosen:
            chosen = available[:3]
        print(f"playing: {chosen}")

        for name in chosen:
            print(f"  -> {name}")
            reachy.play_move(library.get(name), initial_goto_duration=0.5)
            time.sleep(0.2)

        print("going to sleep")
        reachy.goto_sleep()


if __name__ == "__main__":
    main()
