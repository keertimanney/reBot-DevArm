from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

import websocket

from .perception import PerceptionProvider
from .skills import SkillRegistry


DEFAULT_MODEL = "gpt-realtime-2"
REALTIME_URL = "wss://api.openai.com/v1/realtime"

INSTRUCTIONS = """
You are Reachy's local interaction brain for a tabletop robot arm.
You can move one block between exactly five calibrated positions:
top_left, top_right, bottom_left, bottom_right, and center.

When the user asks to move the block from one position to another, call move_block.
Use get_scene if the user asks what positions are available or what you can see.
Use go_home when asked to return home. Use stop for stop, halt, or emergency stop.

Speak briefly. After a tool call, summarize the result in one sentence.
If a requested position is ambiguous, ask a short clarification before calling a tool.
Never invent positions outside the five calibrated names.
"""


class RealtimeToolClient:
    def __init__(
        self,
        perception: PerceptionProvider,
        skills: SkillRegistry,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        verbose: bool = False,
    ):
        self.perception = perception
        self.skills = skills
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.verbose = verbose
        self.ws: websocket.WebSocket | None = None

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required to use the Realtime API")

    def connect(self) -> None:
        url = f"{REALTIME_URL}?model={self.model}"
        self.ws = websocket.create_connection(
            url,
            header=[
                f"Authorization: Bearer {self.api_key}",
                "OpenAI-Safety-Identifier: local-reachy-rebot-dev",
            ],
            timeout=30,
        )
        self._send(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": self.model,
                    "output_modalities": ["text"],
                    "instructions": INSTRUCTIONS.strip(),
                    "tools": self.skills.realtime_tools(),
                    "tool_choice": "auto",
                },
            }
        )
        self._wait_for_session_update()

    def close(self) -> None:
        if self.ws is not None:
            self.ws.close()
            self.ws = None

    def send_user_text(self, text: str) -> str:
        self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
        self._send({"type": "response.create"})
        return self._drain_until_text_response()

    def _drain_until_text_response(self) -> str:
        final_text = ""

        while True:
            event = self._recv()
            event_type = event.get("type")

            if event_type in {"response.output_text.delta", "response.text.delta"}:
                delta = event.get("delta", "")
                final_text += delta
                print(delta, end="", flush=True)
                continue

            if event_type == "response.done":
                tool_calls = self._extract_tool_calls(event)
                if not tool_calls:
                    if final_text:
                        print()
                    return final_text.strip()

                for tool_call in tool_calls:
                    self._handle_tool_call(tool_call)
                final_text = ""
                self._send({"type": "response.create"})
                continue

            if event_type == "error":
                error = event.get("error", event)
                raise RuntimeError(f"Realtime API error: {json.dumps(error)}")

    def _handle_tool_call(self, tool_call: dict[str, Any]) -> None:
        name = str(tool_call.get("name", ""))
        call_id = str(tool_call.get("call_id", ""))
        raw_arguments = tool_call.get("arguments") or "{}"

        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            result = {
                "success": False,
                "message": f"Invalid tool arguments: {exc}",
            }
        else:
            result = self._execute_tool(name, arguments)

        self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                },
            }
        )

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        scene = self.perception.read_scene()
        if name == "get_scene":
            return {
                "success": True,
                "message": scene.describe(),
                "positions": sorted(scene.positions),
                "occupied_positions": scene.occupied_positions,
            }

        result = self.skills.call(name, arguments, scene)
        payload = asdict(result)
        payload["tool"] = name
        return payload

    def _wait_for_session_update(self) -> None:
        while True:
            event = self._recv()
            event_type = event.get("type")
            if event_type == "session.updated":
                return
            if event_type == "error":
                error = event.get("error", event)
                raise RuntimeError(f"Realtime API error during session setup: {json.dumps(error)}")

    def _extract_tool_calls(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        response = event.get("response", {})
        output = response.get("output", [])
        return [
            item
            for item in output
            if item.get("type") == "function_call" and item.get("status") == "completed"
        ]

    def _send(self, event: dict[str, Any]) -> None:
        if self.verbose:
            print(f"\n>>> {json.dumps(event, indent=2)}")
        if self.ws is None:
            raise RuntimeError("Realtime client is not connected")
        self.ws.send(json.dumps(event))

    def _recv(self) -> dict[str, Any]:
        if self.ws is None:
            raise RuntimeError("Realtime client is not connected")
        raw = self.ws.recv()
        event = json.loads(raw)
        if self.verbose:
            print(f"\n<<< {json.dumps(event, indent=2)}")
        return event

