from __future__ import annotations

from .agent import InteractionAgent
from .arm_executor import MockArmExecutor
from .models import DecisionType
from .perception import MockPerception
from .skills import SkillRegistry
from .voice import ConsoleSpeaker


def main() -> None:
    perception = MockPerception()
    speaker = ConsoleSpeaker()
    agent = InteractionAgent()
    skills = SkillRegistry(arm=MockArmExecutor())

    speaker.say("Interaction loop ready. Type a command, or type quit.")

    while True:
        try:
            user_text = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_text.lower() in {"quit", "exit"}:
            break

        scene = perception.read_scene()
        decision = agent.decide(user_text=user_text, scene=scene)

        if decision.text:
            speaker.say(decision.text)

        if decision.type == DecisionType.CALL_SKILL:
            result = skills.call(decision.skill_name, decision.arguments, scene)
            speaker.say(result.message)

    speaker.say("Interaction loop stopped.")


if __name__ == "__main__":
    main()

