#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import signal
import sys
import threading
from typing import Any

import sounddevice as sd
import websocket


REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-realtime-2"
SAMPLE_RATE = 24000
POSITIONS = ["top_left", "top_right", "bottom_left", "bottom_right", "center"]

INSTRUCTIONS = """
You are a voice-controlled local robot arm assistant.

The table has exactly five calibrated positions:
top_left, top_right, bottom_left, bottom_right, and center.

Your only tool is move_object. Use it whenever the user asks to move the object
or block from one position to another. The tool always returns whether the
movement command was accepted.

Normalize natural speech to the enum values:
- "top left" or "upper left" -> top_left
- "top right" or "upper right" -> top_right
- "bottom left" or "lower left" -> bottom_left
- "bottom right" or "lower right" -> bottom_right
- "center", "centre", or "middle" -> center

After calling the tool, briefly tell the user what happened.
If either position is missing or unclear, ask one short clarification question.
Do not call any tool other than move_object.
"""


def move_object(from_position: str, to_position: str) -> dict[str, Any]:
    if from_position not in POSITIONS:
        return {
            "success": False,
            "message": f"Unknown source position: {from_position}",
            "valid_positions": POSITIONS,
        }
    if to_position not in POSITIONS:
        return {
            "success": False,
            "message": f"Unknown destination position: {to_position}",
            "valid_positions": POSITIONS,
        }
    if from_position == to_position:
        return {
            "success": True,
            "message": "Source and destination are the same, so no movement was needed.",
            "from_position": from_position,
            "to_position": to_position,
        }

    return {
        "success": True,
        "message": f"Move command accepted from {from_position} to {to_position}.",
        "from_position": from_position,
        "to_position": to_position,
    }


def tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "move_object",
            "description": "Move the object from one calibrated table position to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_position": {
                        "type": "string",
                        "description": "The source position where the object currently is.",
                        "enum": POSITIONS,
                    },
                    "to_position": {
                        "type": "string",
                        "description": "The destination position where the object should move.",
                        "enum": POSITIONS,
                    },
                },
                "required": ["from_position", "to_position"],
                "additionalProperties": False,
            },
        }
    ]


class RealtimeVoiceToolLoop:
    def __init__(self, api_key: str, model: str, voice: str, verbose: bool):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.verbose = verbose
        self.ws: websocket.WebSocket | None = None
        self.stop_event = threading.Event()
        self.audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=100)
        self.output_stream: sd.RawOutputStream | None = None

    def run(self) -> None:
        self._connect()

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=1200,
            callback=self._on_input_audio,
        ), sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=1200,
        ) as output_stream:
            self.output_stream = output_stream

            sender = threading.Thread(target=self._send_audio_loop, daemon=True)
            receiver = threading.Thread(target=self._receive_loop, daemon=True)
            sender.start()
            receiver.start()

            print("Voice loop running. Speak into your microphone.")
            print("Try: 'move the object from top left to center'")
            print("Press Ctrl+C to stop.")

            while not self.stop_event.wait(0.2):
                pass

    def close(self) -> None:
        self.stop_event.set()
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None

    def _connect(self) -> None:
        url = f"{REALTIME_URL}?model={self.model}"
        self.ws = websocket.create_connection(
            url,
            header=[
                f"Authorization: Bearer {self.api_key}",
                "OpenAI-Safety-Identifier: local-rebot-voice-tool-test",
            ],
            timeout=30,
        )
        self._send(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": self.model,
                    "output_modalities": ["audio"],
                    "instructions": INSTRUCTIONS.strip(),
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                            "turn_detection": {"type": "semantic_vad"},
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                            "voice": self.voice,
                        },
                    },
                    "tools": tool_schema(),
                    "tool_choice": "auto",
                },
            }
        )

        while True:
            event = self._recv()
            if event.get("type") == "session.updated":
                print("Connected to OpenAI Realtime API.")
                return
            if event.get("type") == "error":
                raise RuntimeError(json.dumps(event.get("error", event), indent=2))

    def _on_input_audio(self, indata: bytes, frames: int, time: Any, status: sd.CallbackFlags) -> None:
        if status and self.verbose:
            print(f"[audio-input-status] {status}", file=sys.stderr)
        if self.stop_event.is_set():
            return
        try:
            self.audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def _send_audio_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            encoded = base64.b64encode(chunk).decode("ascii")
            try:
                self._send({"type": "input_audio_buffer.append", "audio": encoded})
            except Exception as exc:
                print(f"[send-error] {exc}", file=sys.stderr)
                self.stop_event.set()
                return

    def _receive_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                event = self._recv()
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"[receive-error] {exc}", file=sys.stderr)
                    self.stop_event.set()
                return

            event_type = event.get("type")

            if event_type in {"response.output_audio.delta", "response.audio.delta"}:
                self._play_audio_delta(event.get("delta", ""))
                continue

            if event_type in {
                "response.output_audio_transcript.done",
                "response.audio_transcript.done",
            }:
                transcript = event.get("transcript")
                if transcript:
                    print(f"[assistant] {transcript}")
                continue

            if event_type == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript")
                if transcript:
                    print(f"[user] {transcript}")
                continue

            if event_type == "response.done":
                self._handle_response_done(event)
                continue

            if event_type == "error":
                print(f"[realtime-error] {json.dumps(event.get('error', event), indent=2)}", file=sys.stderr)
                continue

            if self.verbose:
                print(f"[event] {json.dumps(event, indent=2)}")

    def _handle_response_done(self, event: dict[str, Any]) -> None:
        response = event.get("response", {})
        for item in response.get("output", []):
            if item.get("type") != "function_call":
                continue

            name = item.get("name")
            call_id = item.get("call_id")
            raw_args = item.get("arguments") or "{}"

            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                result = {"success": False, "message": f"Invalid JSON arguments: {exc}"}
            else:
                if name == "move_object":
                    print(f"[tool-call] move_object {json.dumps(args, sort_keys=True)}")
                    result = move_object(
                        from_position=str(args.get("from_position", "")),
                        to_position=str(args.get("to_position", "")),
                    )
                else:
                    result = {"success": False, "message": f"Unknown tool: {name}"}

            print(f"[tool-result] {json.dumps(result, sort_keys=True)}")
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
            self._send({"type": "response.create"})

    def _play_audio_delta(self, encoded: str) -> None:
        if not encoded or self.output_stream is None:
            return
        audio = base64.b64decode(encoded)
        self.output_stream.write(audio)

    def _send(self, event: dict[str, Any]) -> None:
        if self.verbose and event.get("type") != "input_audio_buffer.append":
            print(f"[send] {json.dumps(event, indent=2)}")
        if self.ws is None:
            raise RuntimeError("WebSocket is not connected")
        self.ws.send(json.dumps(event))

    def _recv(self) -> dict[str, Any]:
        if self.ws is None:
            raise RuntimeError("WebSocket is not connected")
        event = json.loads(self.ws.recv())
        if self.verbose and event.get("type") not in {
            "response.output_audio.delta",
            "response.audio.delta",
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
        }:
            print(f"[recv] {json.dumps(event, indent=2)}")
        return event


def main() -> int:
    parser = argparse.ArgumentParser(description="Voice Realtime API loop with one move_object tool.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--voice", default="marin")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY before running this script.", file=sys.stderr)
        return 2

    loop = RealtimeVoiceToolLoop(
        api_key=api_key,
        model=args.model,
        voice=args.voice,
        verbose=args.verbose,
    )

    def handle_signal(signum: int, frame: Any) -> None:
        loop.close()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        loop.run()
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
