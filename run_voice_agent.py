#!/usr/bin/env python3
"""Voice-controlled arm agent using the OpenAI Realtime API.

Positions and motion parameters are loaded from realtime_positions.yaml.
The agent listens continuously (semantic VAD) and moves the arm on voice command.

Setup:
    export OPENAI_API_KEY=sk-...

Usage:
    python run_voice_agent.py                        # real hardware
    python run_voice_agent.py --dry-run              # no hardware
    python run_voice_agent.py --verbose              # print raw events
    python run_voice_agent.py --positions-file realtime_positions.yaml
"""

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

import numpy as np
import sounddevice as sd
import websocket
import yaml


REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-realtime-2"
SAMPLE_RATE = 24_000
DEFAULT_POSITIONS_FILE = "realtime_positions.yaml"

INSTRUCTIONS = """
You are a voice-controlled local robot arm assistant.

The robot has a set of calibrated positions. Position names are the only
valid source and destination values.

Your only tool is move_object. Use it whenever the user asks to move the object
or block from one named position to another. If the user says "position A",
use "A"; if they say "position B", use "B"; and so on.

After calling the tool, briefly tell the user what happened.
If either position is missing or unclear, ask one short clarification question.
Do not call any tool other than move_object.
"""


# ── data classes ───────────────────────────────────────────────────────────────

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
    pitch: float = 1.5707963267948966   # π/2 — EEF pointing down
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


# ── config loader ──────────────────────────────────────────────────────────────

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


# ── executors ──────────────────────────────────────────────────────────────────

class ObjectMotionExecutor:
    def move_object(self, from_position: str, to_position: str) -> dict[str, Any]:  # noqa: ARG002
        raise NotImplementedError

    def close(self) -> None:
        pass


class DryRunObjectMotionExecutor(ObjectMotionExecutor):
    def __init__(self, config: MotionConfig) -> None:
        self.config = config

    def move_object(self, from_position: str, to_position: str) -> dict[str, Any]:
        resolved_from = self.config.resolve(from_position)
        resolved_to   = self.config.resolve(to_position)
        err = _validate_positions(self.config, resolved_from, resolved_to, from_position, to_position)
        if err:
            return err
        assert resolved_from and resolved_to
        return {
            "success": True,
            "message": f"Dry run: would move from {resolved_from} to {resolved_to}.",
            "from_position": resolved_from,
            "to_position": resolved_to,
            "dry_run": True,
        }


class ReBotIKObjectMotionExecutor(ObjectMotionExecutor):
    """Executes moves via ArmEndPos.move_to_traj (IK from xyz)."""

    def __init__(
        self,
        config: MotionConfig,
        repo_root: Path,
        arm_config: str | None = None,
    ) -> None:
        self.config = config
        self.repo_root = repo_root
        self.arm_config = arm_config
        self._controller: Any | None = None
        self._ensure_controller()   # connect and home on startup

    def move_object(self, from_position: str, to_position: str) -> dict[str, Any]:
        resolved_from = self.config.resolve(from_position)
        resolved_to   = self.config.resolve(to_position)
        err = _validate_positions(self.config, resolved_from, resolved_to, from_position, to_position)
        if err:
            return err
        assert resolved_from and resolved_to

        if resolved_from == resolved_to:
            return {
                "success": True,
                "message": "Source and destination are the same — no movement needed.",
                "from_position": resolved_from,
                "to_position": resolved_to,
            }

        source = self.config.positions[resolved_from]
        target = self.config.positions[resolved_to]

        try:
            ctrl = self._ensure_controller()
            for wp in self._build_waypoints(source, target):
                ok = ctrl.move_to_traj(
                    x=wp.x, y=wp.y, z=wp.z,
                    roll=self.config.roll,
                    pitch=self.config.pitch,
                    yaw=wp.yaw,
                    duration=self.config.duration,
                )
                if not ok:
                    return {
                        "success": False,
                        "message": f"IK failed at waypoint '{wp.name}'.",
                        "from_position": resolved_from,
                        "to_position": resolved_to,
                    }
                self._wait_for_motion(ctrl)
        except Exception as exc:
            return {
                "success": False,
                "message": f"Motion failed: {exc}",
                "from_position": resolved_from,
                "to_position": resolved_to,
            }

        return {
            "success": True,
            "message": f"Moved from {resolved_from} to {resolved_to}.",
            "from_position": resolved_from,
            "to_position": resolved_to,
        }

    def close(self) -> None:
        if self._controller is not None:
            self._controller.end()
            self._controller = None

    def _ensure_controller(self) -> Any:
        if self._controller is not None:
            return self._controller

        sys.path.insert(0, str(self.repo_root / "reBotArm_control_py"))
        from reBotArm_control_py.actuator import RobotArm
        from reBotArm_control_py.controllers import ArmEndPos

        arm = RobotArm(cfg_path=self.arm_config)
        ctrl = ArmEndPos(arm)
        ctrl.start()
        print("Moving to zero_act (home) …")
        self._go_to_zero(ctrl, arm)
        print("At home. Ready.")
        self._controller = ctrl
        return ctrl

    @staticmethod
    def _go_to_zero(ctrl: Any, arm: Any) -> None:
        """Drive all joints to 0 rad via a min-jerk trajectory.

        Uses the same approach as replay_arm_positions._move_joints so the
        full trajectory is played out regardless of motor feedback state.
        """
        q_start, _, _ = arm.get_state()
        q_target = np.zeros(arm.num_joints)

        # Auto-duration: allow 0.5 rad/s max joint speed, minimum 2 s.
        max_delta = float(np.max(np.abs(q_start - q_target)))
        duration = max(2.0, max_delta / 0.5)

        n = max(2, int(duration / 0.02))
        done = threading.Event()

        def _send() -> None:
            interval = duration / n
            for i in range(n):
                if ctrl._stop_send.is_set():
                    break
                tau = i / (n - 1)
                s = 10*tau**3 - 15*tau**4 + 6*tau**5
                ctrl._q_target[:] = q_start + s * (q_target - q_start)
                time.sleep(interval)
            ctrl._q_target[:] = q_target
            done.set()

        t = threading.Thread(target=_send, daemon=True)
        ctrl._send_thread = t
        t.start()
        done.wait(timeout=duration + 3.0)

    def _build_waypoints(
        self,
        source: CalibratedPosition,
        target: CalibratedPosition,
    ) -> list[CalibratedPosition]:
        src_z = max(source.z + self.config.approach_height, self.config.clearance_z)
        tgt_z = max(target.z + self.config.approach_height, self.config.clearance_z)
        return [
            CalibratedPosition(f"{source.name}_approach", source.x, source.y, src_z, source.yaw),
            source,
            CalibratedPosition(f"{source.name}_retreat",  source.x, source.y, src_z, source.yaw),
            CalibratedPosition(f"{target.name}_approach", target.x, target.y, tgt_z, target.yaw),
            target,
            CalibratedPosition(f"{target.name}_retreat",  target.x, target.y, tgt_z, target.yaw),
        ]

    def _wait_for_motion(self, ctrl: Any) -> None:
        deadline = time.monotonic() + max(5.0, self.config.duration + 5.0)
        while getattr(ctrl, "_moving", False):
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for trajectory to finish")
            time.sleep(0.05)


# ── helpers ────────────────────────────────────────────────────────────────────

def _validate_positions(
    config: MotionConfig,
    resolved_from: str | None,
    resolved_to:   str | None,
    raw_from: str,
    raw_to:   str,
) -> dict[str, Any] | None:
    if resolved_from is None:
        return {"success": False, "message": f"Unknown source position: {raw_from}",
                "valid_positions": config.position_names}
    if resolved_to is None:
        return {"success": False, "message": f"Unknown destination position: {raw_to}",
                "valid_positions": config.position_names}
    return None


def _tool_schema(position_names: list[str]) -> list[dict[str, Any]]:
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
                        "description": "The destination position for the object.",
                        "enum": position_names,
                    },
                },
                "required": ["from_position", "to_position"],
                "additionalProperties": False,
            },
        }
    ]


# ── realtime loop ──────────────────────────────────────────────────────────────

class RealtimeVoiceToolLoop:
    def __init__(
        self,
        api_key: str,
        model: str,
        voice: str,
        verbose: bool,
        motion_config: MotionConfig,
        executor: ObjectMotionExecutor,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.verbose = verbose
        self.motion_config = motion_config
        self.executor = executor
        self.ws: websocket.WebSocket | None = None
        self.stop_event = threading.Event()
        self.audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=100)
        self._playback_buf = bytearray()
        self._playback_lock = threading.Lock()
        self._input_stream:  sd.RawInputStream  | None = None
        self._output_stream: sd.RawOutputStream | None = None

    def _output_callback(
        self, outdata: bytes, _frames: int, _time: Any, _status: sd.CallbackFlags
    ) -> None:
        """Called by sounddevice to fill each output block from the playback buffer."""
        n = len(outdata)
        with self._playback_lock:
            available = len(self._playback_buf)
            if available >= n:
                outdata[:] = bytes(self._playback_buf[:n])
                del self._playback_buf[:n]
            else:
                outdata[:available] = bytes(self._playback_buf)
                outdata[available:] = b"\x00" * (n - available)
                self._playback_buf.clear()

    def run(self) -> None:
        self._connect()

        self._input_stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=1200,
            callback=self._on_input_audio,
        )
        self._output_stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=1200,
            callback=self._output_callback,
        )

        try:
            self._input_stream.start()
            self._output_stream.start()

            sender   = threading.Thread(target=self._send_audio_loop,  daemon=True)
            receiver = threading.Thread(target=self._receive_loop,      daemon=True)
            sender.start()
            receiver.start()

            print(f"Positions: {', '.join(self.motion_config.position_names)}")
            print("Speak into your microphone. Try: 'move the object from A to B'")
            print("Press Ctrl+C to stop.\n")

            while not self.stop_event.wait(0.2):
                pass
        finally:
            self._release_devices()

    def close(self) -> None:
        self.stop_event.set()
        self._release_devices()
        self.executor.close()
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None

    def _release_devices(self) -> None:
        for stream in (self._input_stream, self._output_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
        self._input_stream = None
        self._output_stream = None

    # ── connection ─────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        url = f"{REALTIME_URL}?model={self.model}"
        self.ws = websocket.create_connection(
            url,
            header=[f"Authorization: Bearer {self.api_key}"],
            timeout=30,
        )
        self._send({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self.model,
                "output_modalities": ["audio"],
                "instructions": (
                    INSTRUCTIONS.strip()
                    + "\n\nValid position names: "
                    + ", ".join(self.motion_config.position_names) + "."
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
                "tools": _tool_schema(self.motion_config.position_names),
                "tool_choice": "auto",
            },
        })

        while True:
            event = self._recv()
            if event.get("type") == "session.updated":
                print("Connected to OpenAI Realtime API.")
                return
            if event.get("type") == "error":
                raise RuntimeError(json.dumps(event.get("error", event), indent=2))

    # ── audio I/O ──────────────────────────────────────────────────────────────

    def _on_input_audio(
        self, indata: bytes, _frames: int, _time: Any, status: sd.CallbackFlags
    ) -> None:
        if status and self.verbose:
            print(f"[audio-in] {status}", file=sys.stderr)
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

    def _collect_audio_delta(self, encoded: str) -> None:
        """Append a decoded audio chunk directly into the playback buffer."""
        if encoded:
            with self._playback_lock:
                self._playback_buf.extend(base64.b64decode(encoded))

    # ── event receiver ─────────────────────────────────────────────────────────

    def _receive_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                event = self._recv()
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"[receive-error] {exc}", file=sys.stderr)
                    self.stop_event.set()
                return

            etype = event.get("type")

            if etype in {"response.output_audio.delta", "response.audio.delta"}:
                self._collect_audio_delta(event.get("delta", ""))

            elif etype in {
                "response.output_audio_transcript.done",
                "response.audio_transcript.done",
            }:
                transcript = event.get("transcript", "").strip()
                if transcript:
                    print(f"[agent] {transcript}")

            elif etype == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript", "").strip()
                if transcript:
                    print(f"[you]   {transcript}")

            elif etype == "input_audio_buffer.speech_started":
                # Barge-in: clear playback buffer so agent stops speaking immediately
                with self._playback_lock:
                    self._playback_buf.clear()

            elif etype == "response.output_item.done":
                item = event.get("item", {})
                itype = item.get("type")
                print(f"[item]  type={itype!r}  name={item.get('name')!r}")
                if itype == "function_call":
                    self._dispatch_tool_call(item)

            elif etype == "response.done":
                # Safety net: also check response.output in case items weren't
                # delivered via response.output_item.done.
                for item in event.get("response", {}).get("output", []):
                    if item.get("type") == "function_call":
                        self._dispatch_tool_call(item)

            elif etype == "error":
                print(
                    f"[api-error] {json.dumps(event.get('error', event), indent=2)}",
                    file=sys.stderr,
                )

            elif self.verbose:
                print(f"[event] {json.dumps(event, indent=2)}")

    def _dispatch_tool_call(self, item: dict[str, Any]) -> None:
        name    = item.get("name", "")
        call_id = item.get("call_id", "")
        try:
            args = json.loads(item.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            result: dict[str, Any] = {"success": False, "message": f"Bad JSON args: {exc}"}
        else:
            if name == "move_object":
                print(f"[tool] move_object({json.dumps(args, sort_keys=True)})")
                result = self.executor.move_object(
                    from_position=str(args.get("from_position", "")),
                    to_position=str(args.get("to_position", "")),
                )
            else:
                result = {"success": False, "message": f"Unknown tool: {name}"}

        print(f"[tool] ← {json.dumps(result, sort_keys=True)}")
        self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
            },
        })
        self._send({"type": "response.create"})

    # ── ws helpers ─────────────────────────────────────────────────────────────

    def _send(self, event: dict[str, Any]) -> None:
        if self.verbose and event.get("type") != "input_audio_buffer.append":
            print(f"[send] {json.dumps(event, indent=2)}")
        if self.ws is None:
            raise RuntimeError("WebSocket not connected")
        self.ws.send(json.dumps(event))

    def _recv(self) -> dict[str, Any]:
        if self.ws is None:
            raise RuntimeError("WebSocket not connected")
        event = json.loads(self.ws.recv())
        if self.verbose and event.get("type") not in {
            "response.output_audio.delta",
            "response.audio.delta",
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
        }:
            print(f"[recv] {json.dumps(event, indent=2)}")
        return event


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Voice arm agent — OpenAI Realtime API + reBotArm IK motion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment:
  OPENAI_API_KEY  required

Examples:
  python run_voice_agent.py
  python run_voice_agent.py --dry-run
  python run_voice_agent.py --positions-file realtime_positions.yaml
  python run_voice_agent.py --verbose
        """,
    )
    parser.add_argument("--model",          default=DEFAULT_MODEL)
    parser.add_argument("--voice",          default="marin")
    parser.add_argument("--positions-file", default=DEFAULT_POSITIONS_FILE)
    parser.add_argument("--arm-config",     default=None,
                        help="Path to reBotArm arm.yaml (auto-detected when omitted).")
    parser.add_argument("--dry-run",        action="store_true",
                        help="No hardware — print tool results only.")
    parser.add_argument("--verbose",        action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY before running.", file=sys.stderr)
        return 2

    repo_root     = Path(__file__).resolve().parent
    positions_path = Path(args.positions_file)
    if not positions_path.is_absolute():
        positions_path = repo_root / positions_path

    motion_config = load_motion_config(positions_path)

    if args.dry_run:
        executor: ObjectMotionExecutor = DryRunObjectMotionExecutor(motion_config)
        print("Backend: dry-run (no hardware)")
    else:
        executor = ReBotIKObjectMotionExecutor(
            config=motion_config,
            repo_root=repo_root,
            arm_config=args.arm_config,
        )
        print("Backend: reBotArm IK (hardware initialises on first move)")

    loop = RealtimeVoiceToolLoop(
        api_key=api_key,
        model=args.model,
        voice=args.voice,
        verbose=args.verbose,
        motion_config=motion_config,
        executor=executor,
    )

    def _handle_signal(signum: int, frame: Any) -> None:
        loop.close()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        loop.run()
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
