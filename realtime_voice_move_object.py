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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sounddevice as sd
import websocket
import yaml


REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-realtime-2"
SAMPLE_RATE = 24000
DEFAULT_POSITIONS_FILE = "realtime_positions.yaml"

INSTRUCTIONS = """
You are a voice-controlled local robot arm assistant.

The robot has a YAML file of calibrated positions. Position names are the only
valid source and destination values.

Your only tool is move_object. Use it whenever the user asks to move the object
or block from one named position to another. If the user says "position A",
use "A"; if they say "position B", use "B"; and so on.

After calling the tool, briefly tell the user what happened.
If either position is missing or unclear, ask one short clarification question.
Do not call any tool other than move_object.
"""


@dataclass(frozen=True)
class CalibratedPosition:
    name: str
    x: float
    y: float
    z: float
    yaw: float = 0.0


@dataclass(frozen=True)
class MotionConfig:
    positions: dict[str, CalibratedPosition]
    roll: float = 0.0
    pitch: float = 1.5707963267948966
    duration: float = 2.0
    approach_height: float = 0.05
    clearance_z: float = 0.18

    @property
    def position_names(self) -> list[str]:
        return list(self.positions.keys())

    def resolve(self, value: str) -> str | None:
        cleaned = value.strip()
        if cleaned in self.positions:
            return cleaned
        lower_map = {name.lower(): name for name in self.positions}
        return lower_map.get(cleaned.lower())


def load_motion_config(path: str | Path) -> MotionConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_positions = data.get("positions") or {}
    if not isinstance(raw_positions, dict) or not raw_positions:
        raise ValueError(f"{path} must define a non-empty 'positions' mapping")

    positions: dict[str, CalibratedPosition] = {}
    for name, raw in raw_positions.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Position {name!r} must be a mapping")
        xyz = raw.get("xyz")
        if not isinstance(xyz, list) or len(xyz) != 3:
            raise ValueError(f"Position {name!r} must define xyz: [x, y, z]")
        positions[str(name)] = CalibratedPosition(
            name=str(name),
            x=float(xyz[0]),
            y=float(xyz[1]),
            z=float(xyz[2]),
            yaw=float(raw.get("yaw", 0.0)),
        )

    motion = data.get("motion") or {}
    return MotionConfig(
        positions=positions,
        roll=float(motion.get("roll", 0.0)),
        pitch=float(motion.get("pitch", 1.5707963267948966)),
        duration=float(motion.get("duration", 2.0)),
        approach_height=float(motion.get("approach_height", 0.05)),
        clearance_z=float(motion.get("clearance_z", 0.18)),
    )


class ObjectMotionExecutor:
    def move_object(self, from_position: str, to_position: str) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class DryRunObjectMotionExecutor(ObjectMotionExecutor):
    def __init__(self, config: MotionConfig):
        self.config = config

    def move_object(self, from_position: str, to_position: str) -> dict[str, Any]:
        resolved_from = self.config.resolve(from_position)
        resolved_to = self.config.resolve(to_position)
        validation_error = _validate_positions(self.config, resolved_from, resolved_to, from_position, to_position)
        if validation_error is not None:
            return validation_error

        if resolved_from == resolved_to:
            return {
                "success": True,
                "message": "Source and destination are the same, so no movement was needed.",
                "from_position": resolved_from,
                "to_position": resolved_to,
                "dry_run": True,
            }

        assert resolved_from is not None
        assert resolved_to is not None
        return {
            "success": True,
            "message": f"Dry run accepted move from {resolved_from} to {resolved_to}.",
            "from_position": resolved_from,
            "to_position": resolved_to,
            "from_xyz": _position_to_xyz(self.config.positions[resolved_from]),
            "to_xyz": _position_to_xyz(self.config.positions[resolved_to]),
            "dry_run": True,
        }


class ReBotIKObjectMotionExecutor(ObjectMotionExecutor):
    def __init__(
        self,
        config: MotionConfig,
        repo_root: Path,
        arm_config: str | None = None,
    ):
        self.config = config
        self.repo_root = repo_root
        self.arm_config = arm_config
        self._controller: Any | None = None

    def move_object(self, from_position: str, to_position: str) -> dict[str, Any]:
        resolved_from = self.config.resolve(from_position)
        resolved_to = self.config.resolve(to_position)
        validation_error = _validate_positions(self.config, resolved_from, resolved_to, from_position, to_position)
        if validation_error is not None:
            return validation_error

        if resolved_from == resolved_to:
            return {
                "success": True,
                "message": "Source and destination are the same, so no movement was needed.",
                "from_position": resolved_from,
                "to_position": resolved_to,
            }

        assert resolved_from is not None
        assert resolved_to is not None
        source = self.config.positions[resolved_from]
        target = self.config.positions[resolved_to]

        try:
            controller = self._ensure_controller()
            waypoints = self._build_waypoints(source, target)
            for waypoint in waypoints:
                ok = controller.move_to_traj(
                    x=waypoint.x,
                    y=waypoint.y,
                    z=waypoint.z,
                    roll=self.config.roll,
                    pitch=self.config.pitch,
                    yaw=waypoint.yaw,
                    duration=self.config.duration,
                )
                if not ok:
                    return {
                        "success": False,
                        "message": (
                            f"IK failed while moving from {resolved_from} to {resolved_to} "
                            f"at waypoint {waypoint.name}."
                        ),
                        "from_position": resolved_from,
                        "to_position": resolved_to,
                        "failed_waypoint": waypoint.name,
                    }
                self._wait_for_motion(controller)
        except Exception as exc:
            return {
                "success": False,
                "message": f"Motion execution failed: {exc}",
                "from_position": resolved_from,
                "to_position": resolved_to,
            }

        return {
            "success": True,
            "message": f"Moved object from {resolved_from} to {resolved_to}.",
            "from_position": resolved_from,
            "to_position": resolved_to,
            "from_xyz": _position_to_xyz(source),
            "to_xyz": _position_to_xyz(target),
        }

    def close(self) -> None:
        if self._controller is not None:
            self._controller.end()
            self._controller = None

    def _ensure_controller(self) -> Any:
        if self._controller is not None:
            return self._controller

        control_repo = self.repo_root / "reBotArm_control_py"
        sys.path.insert(0, str(control_repo))

        try:
            import pinocchio  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "pinocchio is required for IK. Install it in this Python environment before running hardware motion."
            ) from exc

        try:
            from reBotArm_control_py.actuator import RobotArm
            from reBotArm_control_py.controllers import ArmEndPos
        except ImportError as exc:
            raise RuntimeError(
                "Could not import reBotArm_control_py hardware control modules. "
                "Make sure motorbridge and the vendored package dependencies are installed."
            ) from exc

        arm = RobotArm(cfg_path=self.arm_config)
        controller = ArmEndPos(arm)
        controller.start()
        self._controller = controller
        return controller

    def _build_waypoints(
        self,
        source: CalibratedPosition,
        target: CalibratedPosition,
    ) -> list[CalibratedPosition]:
        source_clearance = max(source.z + self.config.approach_height, self.config.clearance_z)
        target_clearance = max(target.z + self.config.approach_height, self.config.clearance_z)
        return [
            CalibratedPosition(f"{source.name}_approach", source.x, source.y, source_clearance, source.yaw),
            source,
            CalibratedPosition(f"{source.name}_retreat", source.x, source.y, source_clearance, source.yaw),
            CalibratedPosition(f"{target.name}_approach", target.x, target.y, target_clearance, target.yaw),
            target,
            CalibratedPosition(f"{target.name}_retreat", target.x, target.y, target_clearance, target.yaw),
        ]

    def _wait_for_motion(self, controller: Any) -> None:
        deadline = time.monotonic() + max(5.0, self.config.duration + 5.0)
        while getattr(controller, "_moving", False):
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for ArmEndPos trajectory to finish")
            time.sleep(0.05)


def _validate_positions(
    config: MotionConfig,
    resolved_from: str | None,
    resolved_to: str | None,
    raw_from: str,
    raw_to: str,
) -> dict[str, Any] | None:
    if resolved_from is None:
        return {
            "success": False,
            "message": f"Unknown source position: {raw_from}",
            "valid_positions": config.position_names,
        }
    if resolved_to is None:
        return {
            "success": False,
            "message": f"Unknown destination position: {raw_to}",
            "valid_positions": config.position_names,
        }
    return None


def _position_to_xyz(position: CalibratedPosition) -> list[float]:
    return [position.x, position.y, position.z]


def tool_schema(position_names: list[str]) -> list[dict[str, Any]]:
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
                        "enum": position_names,
                    },
                    "to_position": {
                        "type": "string",
                        "description": "The destination position where the object should move.",
                        "enum": position_names,
                    },
                },
                "required": ["from_position", "to_position"],
                "additionalProperties": False,
            },
        }
    ]


class RealtimeVoiceToolLoop:
    def __init__(
        self,
        api_key: str,
        model: str,
        voice: str,
        verbose: bool,
        motion_config: MotionConfig,
        executor: ObjectMotionExecutor,
    ):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.verbose = verbose
        self.motion_config = motion_config
        self.executor = executor
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
            print(f"Loaded positions: {', '.join(self.motion_config.position_names)}")
            print("Try: 'move the object from A to B'")
            print("Press Ctrl+C to stop.")

            while not self.stop_event.wait(0.2):
                pass

    def close(self) -> None:
        self.stop_event.set()
        self.executor.close()
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
                    "instructions": (
                        INSTRUCTIONS.strip()
                        + "\n\nValid position names: "
                        + ", ".join(self.motion_config.position_names)
                        + "."
                    ),
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
                    "tools": tool_schema(self.motion_config.position_names),
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
                    result = self.executor.move_object(
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
    parser.add_argument(
        "--positions-file",
        default=DEFAULT_POSITIONS_FILE,
        help="YAML file defining calibrated position names and XYZ coordinates.",
    )
    parser.add_argument(
        "--arm-config",
        default=None,
        help="Optional reBotArm_control_py arm YAML config path. Defaults to the vendored package config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not connect to the robot arm; print successful tool results only.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY before running this script.", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parent
    positions_path = Path(args.positions_file)
    if not positions_path.is_absolute():
        positions_path = repo_root / positions_path
    motion_config = load_motion_config(positions_path)
    if args.dry_run:
        executor: ObjectMotionExecutor = DryRunObjectMotionExecutor(motion_config)
        print("Motion backend: dry-run")
    else:
        executor = ReBotIKObjectMotionExecutor(
            config=motion_config,
            repo_root=repo_root,
            arm_config=args.arm_config,
        )
        print("Motion backend: reBotArm_control_py ArmEndPos IK")
        print("Hardware will initialize on the first move_object tool call.")

    loop = RealtimeVoiceToolLoop(
        api_key=api_key,
        model=args.model,
        voice=args.voice,
        verbose=args.verbose,
        motion_config=motion_config,
        executor=executor,
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
