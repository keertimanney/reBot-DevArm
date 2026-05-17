#!/usr/bin/env python3
"""Text-mode LLM agent driving the reBot arm via gpt-5.5.

Same tool stack as run_voice_agent_reachy.py — same DryRun/ReBot executor,
same arm tools, same furniture recipe library, same gripper-state guards,
same loop-breaker, same occupancy snapshot, same next_recommended hints —
but text in / text out through the standard Chat Completions API. No mic,
no speaker, no realtime websocket.

Use this when you want deterministic tool sequencing and clean logs.

Usage:
    export OPENAI_API_KEY=sk-...
    python run_text_agent.py                        # real arm
    python run_text_agent.py --dry-run              # no hardware
    python run_text_agent.py --event-log /tmp/x.jsonl

At the prompt, type commands like:
    > build a bed
    > put down the L sofa
    > clear the room
    > quit
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

from openai import OpenAI

from run_voice_agent_reachy import (  # noqa: E402
    ARM_TOOL_ERROR_EMOTE,
    ARM_TOOL_SUCCESS_EMOTE,
    DEFAULT_GRID_FILE,
    DEFAULT_POSITIONS_FILE,
    DEFAULT_RECIPES_FILE,
    DEFAULT_STATE_FILE,
    DryRunGraspExecutor,
    GraspExecutor,
    INSTRUCTIONS_TEMPLATE,
    PLAY_EMOTE_TOOL_NAME,
    ReBotGraspExecutor,
    ReachyEmoteController,
    _build_play_emote_tool,
    _to_openai_tools,
    format_recipe_library,
    load_recipes,
)

DEFAULT_MODEL = "gpt-5.5"


def _reset_hardware(executor: GraspExecutor) -> None:
    """Belt-and-braces hardware reset using move_repl_xy helpers.

    ReBotGraspExecutor already runs clear_error + zero_gripper during __init__,
    but if a previous session left the arm in a wedged state (e.g. joint3
    failing to disable cleanly), a second pass here gives a clean slate before
    the chat loop opens. Safe to call on a healthy arm — clear_error and
    zero_gripper are idempotent.
    """
    if not isinstance(executor, ReBotGraspExecutor):
        return  # nothing to do in dry-run

    from move_repl_xy import zero_gripper  # noqa: PLC0415

    print("[reset] clearing latched motor errors …")
    for ctrl in executor.arm._ctrl_map.values():
        for mot in getattr(ctrl, "_motor_map", {}).values():
            try:
                mot.clear_error()
            except Exception as exc:
                print(f"[reset] clear_error skipped: {exc}")
    time.sleep(0.2)

    if executor.gripper is not None:
        print("[reset] re-zeroing gripper …")
        try:
            zero_gripper(executor.gripper)
        except Exception as exc:
            print(f"[reset] zero_gripper skipped: {exc}")

    print("[reset] hardware reset complete.")


def _realtime_tools_to_chat(tools: list[dict]) -> list[dict]:
    """Convert Realtime-API tool schemas to Chat Completions schemas.

    Realtime:        {"type": "function", "name": ..., "description": ..., "parameters": ...}
    Chat Completions:{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    """
    out = []
    for t in tools:
        if t.get("type") != "function":
            out.append(t)
            continue
        out.append({
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t.get("description", ""),
                "parameters":  t.get("parameters", {"type": "object", "properties": {}}),
            },
        })
    return out


# ── per-call hint mirror of RealtimeVoiceToolLoop._recommended_next ────────────

def _recommended_next(name: str, success: bool, holding: bool) -> str | None:
    if not success:
        return None
    if name == "move_to_pickup_zone":
        if holding:
            return ("Unexpected: gripper holding after move_to_pickup_zone. "
                    "Call drop_block to clear.")
        return ("Call grab_block({\"grip_style\": \"edge\"}) NEXT to pick up "
                "the block from the pickup zone.")
    if name == "grab_block":
        return ("Call move_to_grid_cell({\"i\": <col>, \"j\": <row>}) NEXT "
                "with the target cell from the recipe you are building. After "
                "that, drop_block.")
    if name == "move_to_grid_cell":
        if holding:
            return ("Call drop_block({\"grip_style\": \"edge\"}) NEXT to "
                    "place the block at this cell. Do NOT call "
                    "move_to_grid_cell again.")
        return ("You are above this cell but holding nothing. Call "
                "move_to_pickup_zone NEXT, then grab_block.")
    if name == "drop_block":
        return ("Block placed. If the recipe has more cells, call "
                "move_to_pickup_zone NEXT to start the next block. If the "
                "recipe is complete, say a short \"done\" and stop calling "
                "tools.")
    if name == "move_block_in_grid":
        return ("Move complete (gripper is empty again). If the recipe has "
                "more cells to rearrange, call move_block_in_grid again for "
                "the next pair; otherwise stop.")
    if name == "move_to_named_position_up":
        if holding:
            return "Call drop_block NEXT to place the block here."
        return "Call grab_block NEXT if this is a pickup spot."
    if name == "find_empty_cell":
        return ("Call move_to_grid_cell with the (i, j) just returned, then "
                "drop_block.")
    if name == "reset_occupancy":
        return ("Room cleared. Wait for the next instruction unless you were "
                "already mid-recipe.")
    return None


# ── dispatcher with the same guards as the realtime loop ───────────────────────

class TextToolDispatcher:
    """Executes tool calls against the executor and returns the result dict
    that gets fed back to the LLM (as a `tool` message).

    Mirrors the safety guards from RealtimeVoiceToolLoop:
      - loop-breaker after 3 consecutive identical calls
      - next_recommended in every result
      - occupancy snapshot
    """

    LOOP_THRESHOLD = 3

    def __init__(
        self,
        executor: GraspExecutor,
        emote_controller: "ReachyEmoteController | None" = None,
        event_log_fp: Any = None,
    ) -> None:
        self.executor          = executor
        self.emote_controller  = emote_controller
        self.event_log_fp      = event_log_fp
        self._last_key:    str | None  = None
        self._last_count:  int         = 0
        self._last_hint:   str | None  = None

    def _log(self, direction: str, payload: dict) -> None:
        if self.event_log_fp is None:
            return
        try:
            self.event_log_fp.write(json.dumps({
                "t":   time.time(),
                "dir": direction,
                "ev":  payload,
            }) + "\n")
        except Exception:
            pass

    def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        call_key = f"{name}|{json.dumps(args, sort_keys=True)}"

        if self._last_key == call_key and self._last_count >= self.LOOP_THRESHOLD:
            directive = self._last_hint or (
                "Read the 'occupancy' snapshot — especially 'holding' — and "
                "pick a DIFFERENT next tool."
            )
            result_str = (
                "error: REPEAT-CALL ABORTED. You have just called this exact "
                f"tool with these arguments {self._last_count} times in a row "
                f"with no other progress. STOP repeating. Required next "
                f"action: {directive}"
            )
            print(f"[loop-breaker] {call_key} after {self._last_count} "
                  f"consecutive calls")
        else:
            print(f"[tool→exec]  {name}({json.dumps(args, sort_keys=True)})")
            if name == PLAY_EMOTE_TOOL_NAME:
                if self.emote_controller is None:
                    result_str = "error: reachy emote controller not available"
                else:
                    result_str = self.emote_controller.play(args.get("name", ""))
            else:
                result_str = self.executor.execute_tool(name, args)

        success = not (
            result_str.startswith("error") or result_str.startswith("IK failed")
        )

        if self._last_key == call_key:
            self._last_count += 1
        else:
            self._last_key = call_key
            self._last_count = 1

        result: dict[str, Any] = {"success": success, "message": result_str}
        snapshot = self.executor.occupancy_snapshot()
        if snapshot is not None:
            result["occupancy"] = snapshot
        holding = snapshot.get("holding", False) if snapshot else False
        hint = _recommended_next(name, success, holding)
        if hint is not None:
            result["next_recommended"] = hint
            self._last_hint = hint
        return result


# ── main loop ──────────────────────────────────────────────────────────────────

def _build_messages(instructions: str, context_info: str) -> list[dict]:
    return [{
        "role": "system",
        "content": instructions.strip() + "\n\n" + context_info,
    }]


def _tool_call_args(tc: Any) -> dict[str, Any]:
    try:
        return json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        return {"_bad_json": str(exc), "_raw": tc.function.arguments}


def chat_loop(
    client: OpenAI,
    model: str,
    instructions: str,
    context_info: str,
    openai_tools: list[dict],
    dispatcher: TextToolDispatcher,
    verbose: bool,
) -> None:
    messages = _build_messages(instructions, context_info)
    print(context_info)
    print("Type your command (or 'quit').\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line.lower() in {"quit", "exit", "q"}:
            return

        messages.append({"role": "user", "content": line})

        # Inner loop: keep going while the assistant returns tool_calls.
        for hop in range(50):  # hard ceiling against runaway loops
            dispatcher._log("send", {
                "model": model, "messages_len": len(messages),
                "tools_len": len(openai_tools),
            })
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
            )
            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            dispatcher._log("recv", {
                "finish_reason": response.choices[0].finish_reason,
                "tool_calls":    len(msg.tool_calls or []),
                "content":       (msg.content or "")[:200],
            })

            if msg.content and msg.content.strip():
                print(f"agent> {msg.content.strip()}")

            tool_calls = msg.tool_calls or []
            if not tool_calls:
                break

            for tc in tool_calls:
                name = tc.function.name
                args = _tool_call_args(tc)
                if "_bad_json" in args:
                    result = {"success": False,
                              "message": f"error: bad JSON args: {args['_bad_json']}"}
                else:
                    if verbose:
                        print(f"[tool-call] id={tc.id!r} name={name!r} args={args}")
                    result = dispatcher.dispatch(name, args)
                if verbose:
                    print(f"[tool←result] {json.dumps(result, sort_keys=True)}")
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(result),
                })
        else:
            print("[warn] hop ceiling (50) reached — stopping this turn")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model",          default=DEFAULT_MODEL)
    p.add_argument("--positions-file", default=DEFAULT_POSITIONS_FILE)
    p.add_argument("--grid-file",      default=DEFAULT_GRID_FILE)
    p.add_argument("--state-file",     default=DEFAULT_STATE_FILE)
    p.add_argument("--recipes-file",   default=DEFAULT_RECIPES_FILE)
    p.add_argument("--arm-config",     default=None)
    p.add_argument("--duration",       type=float, default=2.0)
    p.add_argument("--dry-run",        action="store_true",
                   help="No hardware — print tool results only.")
    p.add_argument("--event-log",      default=None,
                   help="Write structured tool-call events (sent + received) "
                        "to this JSONL file for debugging.")
    p.add_argument("--no-emotes",      action="store_true",
                   help="Disable the play_emote tool (skip Reachy SDK + emote "
                        "library load).")
    p.add_argument("--verbose",        action="store_true")
    args = p.parse_args()

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
    state_path = Path(args.state_file)
    if not state_path.is_absolute():
        state_path = _REPO_ROOT / state_path
    recipes_path = Path(args.recipes_file)
    if not recipes_path.is_absolute():
        recipes_path = _REPO_ROOT / recipes_path

    recipes = load_recipes(recipes_path)
    print(f"Loaded {len(recipes)} recipe(s) from {recipes_path.name}: "
          f"{list(recipes)}")

    sys.path.insert(0, str(_REPO_ROOT))
    from move_repl_xy import AGENT_TOOLS  # noqa: PLC0415
    openai_tools = _realtime_tools_to_chat(_to_openai_tools(AGENT_TOOLS))

    if args.dry_run:
        executor: GraspExecutor = DryRunGraspExecutor(nx=7, ny=4)
        print("Backend: dry-run (no hardware)")
    else:
        executor = ReBotGraspExecutor(
            positions_file=positions_path,
            grid_file=grid_path,
            arm_config=args.arm_config,
            duration=args.duration,
            state_file=state_path,
        )
        # Second-pass clear_error + zero_gripper to recover from any wedged
        # state left by a previous session (e.g. joint3 failing to disable).
        _reset_hardware(executor)

    emote_controller: ReachyEmoteController | None = None
    emote_names: list[str] = []
    if not args.no_emotes:
        emote_controller = ReachyEmoteController()
        emote_names = emote_controller.available
        play_emote_tool_realtime = _build_play_emote_tool(emote_names)
        openai_tools.extend(_realtime_tools_to_chat([play_emote_tool_realtime]))

    instructions = INSTRUCTIONS_TEMPLATE.format(
        emote_names=", ".join(emote_names) if emote_names else "(emotes disabled)",
        recipe_library=format_recipe_library(recipes),
    )
    context_info = executor.info()

    event_log_fp = None
    if args.event_log:
        event_log_path = Path(args.event_log)
        if not event_log_path.is_absolute():
            event_log_path = _REPO_ROOT / event_log_path
        event_log_fp = open(event_log_path, "a", buffering=1)
        print(f"Logging tool events to {event_log_path}")

    dispatcher = TextToolDispatcher(
        executor,
        emote_controller=emote_controller,
        event_log_fp=event_log_fp,
    )
    client = OpenAI(api_key=api_key)

    def _handle_signal(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        chat_loop(
            client=client,
            model=args.model,
            instructions=instructions,
            context_info=context_info,
            openai_tools=openai_tools,
            dispatcher=dispatcher,
            verbose=args.verbose,
        )
    finally:
        executor.close()
        if emote_controller is not None:
            emote_controller.close()
        if event_log_fp is not None:
            try:
                event_log_fp.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
