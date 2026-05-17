from __future__ import annotations

import argparse

from .arm_executor import MockArmExecutor
from .perception import MockPerception
from .realtime_client import DEFAULT_MODEL, RealtimeToolClient
from .skills import SkillRegistry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a local text console backed by the OpenAI Realtime API and robot-arm tool calls."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Realtime model name")
    parser.add_argument("--verbose", action="store_true", help="Print raw Realtime events")
    args = parser.parse_args()

    perception = MockPerception()
    skills = SkillRegistry(arm=MockArmExecutor())
    client = RealtimeToolClient(
        perception=perception,
        skills=skills,
        model=args.model,
        verbose=args.verbose,
    )

    print("Connecting to OpenAI Realtime API...")
    client.connect()
    print("Connected. Type a command, or type quit.")

    try:
        while True:
            user_text = input("user> ").strip()
            if user_text.lower() in {"quit", "exit"}:
                break
            if not user_text:
                continue

            print("reachy> ", end="", flush=True)
            client.send_user_text(user_text)
    finally:
        client.close()
        print("Realtime loop stopped.")


if __name__ == "__main__":
    main()

