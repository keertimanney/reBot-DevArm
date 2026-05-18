from __future__ import annotations

import re

from .models import AgentDecision, DecisionType, Scene, display_position_name, normalize_position_name


class InteractionAgent:
    """First-pass rule planner for the Reachy interaction loop.

    This is deliberately simple so the rest of the architecture can be tested
    before adding an LLM planner.
    """

    def decide(self, user_text: str, scene: Scene) -> AgentDecision:
        text = user_text.strip().lower()
        if not text:
            return AgentDecision(
                type=DecisionType.SPEAK,
                text="I did not hear a command.",
            )

        if text in {"stop", "halt", "emergency stop"}:
            return AgentDecision(
                type=DecisionType.CALL_SKILL,
                text="Stopping now.",
                skill_name="stop",
            )

        if "home" in text:
            return AgentDecision(
                type=DecisionType.CALL_SKILL,
                text="Returning the arm home.",
                skill_name="go_home",
            )

        if "what do you see" in text or "what can you see" in text:
            return AgentDecision(
                type=DecisionType.SPEAK,
                text=scene.describe(),
            )

        move_decision = self._parse_move(text)
        if move_decision is not None:
            return move_decision

        return AgentDecision(
            type=DecisionType.ASK_CLARIFICATION,
            text="I can move a block between top left, top right, bottom left, bottom right, and center. Which positions should I use?",
        )

    @staticmethod
    def _parse_move(text: str) -> AgentDecision | None:
        patterns = [
            r"move (?:the )?(?:block )?from (?P<from>.+?) to (?P<to>.+)$",
            r"move from (?P<from>.+?) to (?P<to>.+)$",
            r"put (?:the )?(?:block )?from (?P<from>.+?) to (?P<to>.+)$",
            r"place (?:the )?(?:block )?from (?P<from>.+?) to (?P<to>.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                from_position = normalize_position_name(_clean_label(match.group("from")))
                to_position = normalize_position_name(_clean_label(match.group("to")))
                return AgentDecision(
                    type=DecisionType.CALL_SKILL,
                    text=(
                        f"Moving the block from {display_position_name(from_position)} "
                        f"to {display_position_name(to_position)}."
                    ),
                    skill_name="move_block",
                    arguments={
                        "from_position": from_position,
                        "to_position": to_position,
                    },
                )
        return None


def _clean_label(value: str) -> str:
    value = re.sub(r"\bplease\b", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .")
