#!/usr/bin/env python3
"""Voice-controlled arm agent using the OpenAI Realtime API.

Positions are loaded from recorded_arm_positions.yaml; grid from grid_config.yaml.
The agent listens continuously (semantic VAD) and executes arm tool calls on voice command.

Available tools: move_to_pickup_zone, grab_block, move_to_grid_cell, drop_block,
move_to_named_position_up.

Setup:
    export OPENAI_API_KEY=sk-...

Usage:
    python run_voice_agent.py                        # real hardware
    python run_voice_agent.py --dry-run              # no hardware
    python run_voice_agent.py --verbose              # print raw events
    python run_voice_agent.py --positions-file recorded_arm_positions.yaml
    python run_voice_agent.py --grid-file grid_config.yaml
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
from pathlib import Path
from typing import Any

from abc import ABC, abstractmethod

import numpy as np
import sounddevice as sd
import websocket
import yaml


_REPO_ROOT = Path(__file__).resolve().parent

REALTIME_URL           = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL          = "gpt-realtime"
SAMPLE_RATE            = 24_000
DEFAULT_POSITIONS_FILE = "recorded_arm_positions.yaml"
DEFAULT_GRID_FILE      = "grid_config.yaml"

INSTRUCTIONS = """
You are a voice-controlled robot arm assistant.

You have exactly five tools. Whenever the user asks you to move, pick up, place, or
manipulate the arm or a block, call the appropriate tool immediately — do not just
describe what you would do.

Tools and when to call them:
  home                           — use when user says reset, go home, or return to zero
  move_to_pickup_zone            — use to go above position A before grabbing
  grab_block  grip_style         — use to grab a block (face=flat, edge=upright)
  move_to_grid_cell  i  j        — use to move above a grid cell before dropping
  drop_block  grip_style         — use to release a block at current position
  move_to_named_position_up name — use to transit above any other named position

Standard pick-and-place order: move_to_pickup_zone → grab_block → move_to_grid_cell → drop_block.
After each tool call, briefly say what happened.
If a grid cell or position is unclear, ask one short clarification question.
Do not call any tool not listed above.
"""


# ── schema conversion ──────────────────────────────────────────────────────────

def _to_openai_tools(agent_tools: list[dict]) -> list[dict]:
    """Convert Anthropic-format tool list to OpenAI Realtime API tool format."""
    result = []
    for t in agent_tools:
        params = {**t["input_schema"], "additionalProperties": False}
        result.append({
            "type": "function",
            "name": t["name"],
            "description": t["description"],
            "parameters": params,
        })
    return result


# ── executors ──────────────────────────────────────────────────────────────────

class GraspExecutor(ABC):
    @abstractmethod
    def execute_tool(self, name: str, tool_input: dict) -> str: ...

    def info(self) -> str:
        return ""

    def close(self) -> None:
        pass


class DryRunGraspExecutor(GraspExecutor):
    def execute_tool(self, name: str, tool_input: dict) -> str:
        args_str = ", ".join(f"{k}={v!r}" for k, v in tool_input.items())
        return f"dry-run: {name}({args_str})"

    def info(self) -> str:
        return "Backend: dry-run (no hardware)"


class ReBotGraspExecutor(GraspExecutor):
    def __init__(
        self,
        positions_file: Path,
        grid_file: Path,
        arm_config: str | None,
        duration: float,
    ) -> None:
        sys.path.insert(0, str(_REPO_ROOT))
        sys.path.insert(0, str(_REPO_ROOT / "reBotArm_control_py"))

        from move_repl_xy import (
            dispatch_tool,
            _make_gripper_handle, _move_joints,
            _MIN_DURATION, _DEFAULT_SPEED, _HOME_LABEL, _JOINT_LABELS,
            zero_gripper,
        )
        from reBotArm_control_py.actuator import RobotArm
        from reBotArm_control_py.controllers import ArmEndPos
        from compute_grid import compute_grid_map

        self._dispatch      = dispatch_tool
        self._move_joints   = _move_joints
        self._JOINT_LABELS  = _JOINT_LABELS
        self.duration       = duration

        # Load positions
        data = yaml.safe_load(positions_file.read_text()) or {}
        raw  = data.get("positions") or {}
        if not raw:
            raise ValueError(f"No 'positions' found in {positions_file}")
        if _HOME_LABEL not in raw:
            raise ValueError(f"'{_HOME_LABEL}' not in {positions_file}")
        self._raw  = raw
        self.lookup = {name.lower(): (name, entry) for name, entry in raw.items()}

        # Load grid
        self.grid_map: dict[tuple[int, int], tuple[float, float]] = {}
        self.grid_z  = 0.15
        self.grid_nx = self.grid_ny = 0
        if grid_file.exists():
            self.grid_map, self.grid_z = compute_grid_map(grid_file)
            gcfg = yaml.safe_load(grid_file.read_text())
            self.grid_nx = gcfg["grid"]["nx"]
            self.grid_ny = gcfg["grid"]["ny"]

        # Initialise hardware
        self.arm     = RobotArm(cfg_path=arm_config)
        self.ctrl    = ArmEndPos(self.arm)
        self.gripper = _make_gripper_handle(self.arm)

        # Clear any latched motor errors before enabling (mirrors gripper start())
        print("Clearing motor errors …")
        for _ctrl in self.arm._ctrl_map.values():
            for _mot in getattr(_ctrl, '_motor_map', {}).values():
                try:
                    _mot.clear_error()
                except Exception:
                    pass
        time.sleep(0.2)

        print("Connecting and enabling motors …")
        self.ctrl.start()
        if self.gripper is not None:
            self.gripper.start()
            print("Zeroing gripper …")
            zero_gripper(self.gripper)

        q_home       = np.array(raw[_HOME_LABEL]["joints_rad"], dtype=np.float64)
        q_now, _, _  = self.arm.get_state()
        home_dur     = max(_MIN_DURATION, float(np.max(np.abs(q_home - q_now))) / _DEFAULT_SPEED)
        print(f"Moving to '{_HOME_LABEL}' ({home_dur:.1f}s) …")
        _move_joints(self.ctrl, self.arm, q_home, home_dur)
        self._q_home   = q_home
        self._home_dur = home_dur
        print("At home. Ready.")

    def execute_tool(self, name: str, tool_input: dict) -> str:
        return self._dispatch(
            name, tool_input,
            ctrl=self.ctrl,
            arm=self.arm,
            gripper=self.gripper,
            grid_map=self.grid_map,
            grid_nx=self.grid_nx,
            grid_ny=self.grid_ny,
            grid_z=self.grid_z,
            lookup=self.lookup,
            duration=self.duration,
        )

    def info(self) -> str:
        pos_names = [n for n in self._raw if n.lower() not in self._JOINT_LABELS]
        grid_info = (
            f"grid {self.grid_nx}×{self.grid_ny}" if self.grid_nx > 0 else "no grid loaded"
        )
        return f"positions: {', '.join(pos_names)}  |  {grid_info}"

    def close(self) -> None:
        try:
            self._move_joints(self.ctrl, self.arm, self._q_home, self._home_dur)
        except Exception:
            pass
        self.ctrl.end()
        if self.gripper is not None:
            self.gripper.disconnect()


# ── realtime loop ──────────────────────────────────────────────────────────────

class RealtimeVoiceToolLoop:
    def __init__(
        self,
        api_key: str,
        model: str,
        voice: str,
        verbose: bool,
        executor: GraspExecutor,
        openai_tools: list[dict],
        context_info: str,
    ) -> None:
        self.api_key      = api_key
        self.model        = model
        self.voice        = voice
        self.verbose      = verbose
        self.executor     = executor
        self.openai_tools = openai_tools
        self.context_info = context_info
        self.ws: websocket.WebSocket | None = None
        self.stop_event   = threading.Event()
        self.audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=100)
        self._dispatched_calls: set[str] = set()   # dedup tool calls across events
        self._playback_buf  = bytearray()
        self._playback_lock = threading.Lock()
        self._input_stream:  sd.RawInputStream  | None = None
        self._output_stream: sd.RawOutputStream | None = None

    def _output_callback(
        self, outdata: bytes, _frames: int, _time: Any, _status: sd.CallbackFlags
    ) -> None:
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
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=1200, callback=self._on_input_audio,
        )
        self._output_stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=1200, callback=self._output_callback,
        )

        try:
            self._input_stream.start()
            self._output_stream.start()

            sender   = threading.Thread(target=self._send_audio_loop, daemon=True)
            receiver = threading.Thread(target=self._receive_loop,    daemon=True)
            sender.start()
            receiver.start()

            print(self.context_info)
            print("Speak into your microphone. Try: 'pick up the block from A and place it in grid cell 0 0'")
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
        self._input_stream  = None
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
                "instructions": INSTRUCTIONS.strip() + "\n\n" + self.context_info,
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
                "tools": self.openai_tools,
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
                with self._playback_lock:
                    self._playback_buf.clear()
                self._dispatched_calls.clear()

            elif etype == "response.output_item.done":
                item  = event.get("item", {})
                itype = item.get("type")
                if itype == "function_call":
                    print(f"[tool-call]  name={item.get('name')!r}  args={item.get('arguments')!r}")
                    self._dispatch_tool_call(item)
                elif self.verbose:
                    print(f"[item]  type={itype!r}  name={item.get('name')!r}")

            elif etype == "response.done":
                output = event.get("response", {}).get("output", [])
                for item in output:
                    if item.get("type") == "function_call":
                        self._dispatch_tool_call(item)
                if self.verbose:
                    print(f"[response.done]  {len(output)} output item(s)")

            elif etype == "response.function_call_arguments.done":
                print(f"[args-done]  name={event.get('name')!r}  args={event.get('arguments')!r}")

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
        if call_id in self._dispatched_calls:
            return
        self._dispatched_calls.add(call_id)
        try:
            args = json.loads(item.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            result_str = f"error: bad JSON args: {exc}"
        else:
            print(f"[tool→exec]  {name}({json.dumps(args, sort_keys=True)})")
            result_str = self.executor.execute_tool(name, args)

        success = not result_str.startswith("error")
        result  = {"success": success, "message": result_str}
        print(f"[tool←result]  {json.dumps(result, sort_keys=True)}")
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
        description="Voice arm agent — OpenAI Realtime API + reBotArm grasp tools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment:
  OPENAI_API_KEY  required

Examples:
  python run_voice_agent.py
  python run_voice_agent.py --dry-run
  python run_voice_agent.py --grid-file grid_config.yaml
  python run_voice_agent.py --verbose
        """,
    )
    parser.add_argument("--model",          default=DEFAULT_MODEL)
    parser.add_argument("--voice",          default="marin")
    parser.add_argument("--positions-file", default=DEFAULT_POSITIONS_FILE)
    parser.add_argument("--grid-file",      default=DEFAULT_GRID_FILE)
    parser.add_argument("--arm-config",     default=None,
                        help="Path to reBotArm arm.yaml (auto-detected when omitted).")
    parser.add_argument("--duration",       type=float, default=2.0,
                        help="Default motion duration in seconds.")
    parser.add_argument("--dry-run",        action="store_true",
                        help="No hardware — print tool results only.")
    parser.add_argument("--verbose",        action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY before running.", file=sys.stderr)
        return 2

    positions_path = Path(args.positions_file)
    if not positions_path.is_absolute():
        positions_path = _REPO_ROOT / positions_path

    grid_path = Path(args.grid_file)
    if not grid_path.is_absolute():
        grid_path = _REPO_ROOT / grid_path

    sys.path.insert(0, str(_REPO_ROOT))
    from move_repl_xy import AGENT_TOOLS  # noqa: PLC0415
    openai_tools = _to_openai_tools(AGENT_TOOLS)

    if args.dry_run:
        executor: GraspExecutor = DryRunGraspExecutor()
        print("Backend: dry-run (no hardware)")
    else:
        executor = ReBotGraspExecutor(
            positions_file=positions_path,
            grid_file=grid_path,
            arm_config=args.arm_config,
            duration=args.duration,
        )

    context_info = executor.info()

    loop = RealtimeVoiceToolLoop(
        api_key=api_key,
        model=args.model,
        voice=args.voice,
        verbose=args.verbose,
        executor=executor,
        openai_tools=openai_tools,
        context_info=context_info,
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
