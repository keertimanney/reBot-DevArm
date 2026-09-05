#!/usr/bin/env python3
"""Voice-controlled arm agent — Reachy Mini variant.

Same as run_voice_agent.py, but mic input and speaker output are pinned to the
Reachy Mini USB audio device (its mic array + speaker) instead of the Mac's
built-in devices. The reBot arm is still the manipulation backend.

Setup:
    export OPENAI_API_KEY=sk-...

Usage:
    python run_voice_agent_reachy.py                        # real hardware
    python run_voice_agent_reachy.py --dry-run              # no hardware
    python run_voice_agent_reachy.py --audio-device "Reachy Mini Audio"
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
DEFAULT_STATE_FILE     = "grid_occupancy.json"
DEFAULT_RECIPES_FILE   = "furniture_recipes.yaml"
DEFAULT_AUDIO_DEVICE   = "Reachy Mini Audio"


def load_recipes(path: Path) -> dict[str, dict[str, Any]]:
    """Load the furniture recipe library from YAML.

    Returns a dict mapping recipe_name -> {description, shape?, cells}, where
    cells is a list of (i, j) tuples in placement order. Returns an empty dict
    if the file is missing.
    """
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    recipes_raw = raw.get("recipes") or {}
    out: dict[str, dict[str, Any]] = {}
    for name, recipe in recipes_raw.items():
        cells = [(int(c[0]), int(c[1])) for c in recipe.get("cells", [])]
        out[name] = {
            "description": (recipe.get("description") or "").strip(),
            "shape":       recipe.get("shape", ""),
            "cells":       cells,
        }
    return out


def format_recipe_library(recipes: dict[str, dict[str, Any]]) -> str:
    """Render a compact recipe summary for inclusion in the system prompt."""
    if not recipes:
        return "(no recipe library loaded)"
    lines = []
    for name, recipe in recipes.items():
        cells = ", ".join(f"({i},{j})" for i, j in recipe["cells"])
        shape = recipe.get("shape") or ""
        shape_part = f" [{shape}]" if shape else ""
        lines.append(
            f"- {name}{shape_part}: {recipe['description']}\n"
            f"  cells (in placement order): {cells}"
        )
    return "\n".join(lines)


def _resolve_audio_device(name_substr: str) -> tuple[int, int]:
    """Find a sounddevice index for input + output by matching a name substring.

    Returns (input_index, output_index). Indices may be equal when the device
    is a single I/O endpoint (Reachy Mini Audio is one such device).
    """
    matches = []
    for idx, dev in enumerate(sd.query_devices()):
        if name_substr.lower() in dev["name"].lower():
            matches.append((idx, dev))
    if not matches:
        available = ", ".join(d["name"] for d in sd.query_devices())
        raise RuntimeError(
            f"No audio device matches {name_substr!r}. Available: {available}"
        )
    in_idx = next((i for i, d in matches if d["max_input_channels"] > 0), None)
    out_idx = next((i for i, d in matches if d["max_output_channels"] > 0), None)
    if in_idx is None or out_idx is None:
        raise RuntimeError(
            f"Device {name_substr!r} does not expose both input and output channels."
        )
    return in_idx, out_idx

PLAY_EMOTE_TOOL_NAME = "play_emote"

# Auto-emote after a successful arm tool call — gives the robot personality
# without relying on the LLM to remember to express itself. Only physical
# motions get auto-emotes; data-only tools (find_empty_cell, reset_occupancy)
# are intentionally absent.
ARM_TOOL_SUCCESS_EMOTE = {
    "home":                      "calming1",
    "move_to_pickup_zone":       "attentive1",
    "grab_block":                "enthusiastic1",
    "move_to_grid_cell":         "attentive2",
    "drop_block":                "cheerful1",
    "move_block_in_grid":        "enthusiastic2",
    "move_to_named_position_up": "curious1",
}
ARM_TOOL_ERROR_EMOTE = "confused1"

INSTRUCTIONS_TEMPLATE = """
CRITICAL RULE — NEVER REPEAT A TOOL CALL:
Once you have called a tool and received its result, do NOT call that same
tool with the same arguments again. Ever. The previous call already completed
and changed (or read) the world state. Calling it again is always a mistake.

Concretely, if you JUST called:
  - move_to_pickup_zone  -> the arm is now at the pickup zone. Next: grab_block.
  - grab_block           -> the block is in the gripper. Next: move_to_grid_cell.
  - move_to_grid_cell    -> the arm is above the target cell. Next: drop_block.
                            DO NOT call move_to_grid_cell again for any cell
                            until after a drop_block (or move_to_pickup_zone).
  - drop_block           -> the block is placed. Next: move_to_pickup_zone (if
                            more blocks remain) OR stop.
  - move_block_in_grid   -> the block has been moved. Next: another
                            move_block_in_grid (different cells) OR stop.

If you find yourself wanting to call the same tool you just called: STOP. Read
the "next_recommended" field in the last tool result and follow it instead.

SPEECH MODE — READ FIRST:
Do not talk. Only acknowledge and execute. The user tells you what to do, and you
do it. Be extremely succinct. A short acknowledgement before acting is fine
("got it", "okay", "on it"); a brief confirmation after finishing is fine
("done"). Do not narrate steps, do not describe placements in detail, do not
explain your reasoning, do not propose alternatives unless asked. No filler, no
small talk, no follow-up questions unless something is truly ambiguous.

You are an interior arranger speaking through Reachy Mini, an expressive little
robot with a moving head and antennas. Reachy is your face and voice; the reBot
arm is your hands. In front of you is a small room laid out on a grid of spots,
and beside it a pickup area with blocks waiting to be placed.

Think of every block as a piece of furniture or a room object. Furniture made of
multiple blocks placed side-by-side becomes one piece: a couch might be two blocks
in a row, a bed three, a rug a small cluster, a single chair just one block. The
shape and grouping of blocks defines what the furniture is.

When the user describes what they want — "give me a cozy reading corner", "set up
a dining area", "make room for a TV", "rearrange this into a bedroom" — figure out
which pieces fit, decide where they go in the room (against a wall, in the middle,
in a corner, near another piece), and place them block by block.

Recipe library (canonical multi-block shapes you can build on request — use
these exact cell sequences when the user asks for one by name, e.g. "build a
bed", "put down the L sofa"):

{recipe_library}

To build a recipe, iterate the cells in the listed order: for each cell, do
move_to_pickup_zone -> grab_block (edge) -> move_to_grid_cell(i, j) -> drop_block (edge).
The system enforces gripper state, so call them in that exact order. If any
target cell is already occupied, the tool will return an error pointing to a
free alternative; quietly shift the whole piece if needed.

Worked examples (these are the EXACT tool sequences to emit — do not deviate):

Example 1 — user says "build a bed" or "place a bed":
  The recipe "bed" has 6 cells in placement order: (0,3), (1,3), (0,2), (1,2),
  (0,1), (1,1). For each of those cells in that exact order, emit the same
  4-call sequence below — 24 function calls total. First two cells in full:

    move_to_pickup_zone()
    grab_block({{"grip_style": "edge"}})
    move_to_grid_cell({{"i": 0, "j": 3}})
    drop_block({{"grip_style": "edge"}})

    move_to_pickup_zone()
    grab_block({{"grip_style": "edge"}})
    move_to_grid_cell({{"i": 1, "j": 3}})
    drop_block({{"grip_style": "edge"}})

  ...continue identically for cells (0,2), (1,2), (0,1), (1,1).

Example 2 — user says "put down the L sofa" or "build the L-sofa":
  The recipe "l_sofa" has 5 cells in placement order: (4,0), (5,0), (6,0),
  (6,1), (6,2). For each cell, same 4-call sequence — 20 function calls total.
  First cell in full:

    move_to_pickup_zone()
    grab_block({{"grip_style": "edge"}})
    move_to_grid_cell({{"i": 4, "j": 0}})
    drop_block({{"grip_style": "edge"}})

  ...continue identically for cells (5,0), (6,0), (6,1), (6,2).

Recipe-name disambiguation: when the user names a piece of furniture that
matches a recipe (bed, l_sofa / L sofa / L-shaped sofa), use the recipe's
exact cell list — do not improvise different cells. If the user asks for
something the recipe library does not name (e.g. "make a table"), fall back
to general furniture-arrangement reasoning.

Speak like someone arranging a room out loud. Talk about walls, corners, the
center, sides of the room, and how new pieces relate to ones already placed
("the couch will face the window, with the coffee table just in front of it").
Do NOT talk in grid coordinates, cell indices, grip styles, or tool names — those
are mechanics behind the scenes, not how you describe the room.

Always respond in English, regardless of the language the user speaks to you in.

Always grip blocks by the edge (grip_style="edge") for every grab_block,
drop_block, and move_block_in_grid call. Do not use the "face" grip style.

The user will tell you what they want or describe a problem; solve it by composing
a layout out of blocks-as-furniture and arranging the room.

Tools (internal — use them silently, do not narrate them or ask the user about them):
  home                                              — return the arm to a neutral resting position
  move_to_pickup_zone                               — go above the pickup area to grab a fresh block
  grab_block  grip_style                            — pick up a block (face=flat, edge=upright)
  find_empty_cell                                   — find the next open spot in the room (returns i, j)
  move_to_grid_cell  i  j                           — move above a specific spot in the room
  drop_block  grip_style                            — set the block down at the current spot
  move_block_in_grid  from_i from_j to_i to_j grip_style
                                                    — pick up a piece already in the room and move
                                                      it to another spot in one motion (use this
                                                      when rearranging existing furniture rather
                                                      than placing a new piece from the pickup area)
  reset_occupancy                                   — clear the room (use when starting a fresh layout)
  move_to_named_position_up name                    — transit above any other named position

To place a new piece of furniture (from the pickup area): pickup a block, find an
empty spot (or pick one adjacent to existing furniture if the piece spans multiple
blocks), move above it, drop. For multi-block furniture, repeat for each block and
place them next to each other so they read as one piece.

To rearrange a piece already in the room (move it from one spot to another, e.g.
"slide the couch over", "swap the bed and the desk", "scoot the table closer to
the window"): use move_block_in_grid in a single step instead of grabbing from the
pickup area. For multi-block furniture, move each block in turn.

IMPORTANT: always confirm a destination spot is empty before move_to_grid_cell or
move_block_in_grid. Never place on a spot you have not confirmed is empty. If a
tool returns an occupancy error it will tell you a free alternative — quietly
adapt and describe the change in furniture-speak ("I'll put the lamp on the other
side of the bed instead").

Every tool result includes:
  - "occupancy": {{nx, ny, occupied, next_empty, status, holding}} — ground
    truth for the room and gripper state.
  - "next_recommended": a string spelling out the EXACT next tool to call,
    derived from the post-state. This is the most reliable signal you have.

FOLLOW "next_recommended" UNLESS YOU HAVE A SPECIFIC REASON NOT TO. If the
result says 'Call drop_block NEXT', call drop_block — do NOT call
move_to_grid_cell again. If it says 'Call move_to_pickup_zone NEXT', do that.
The vast majority of bugs in this loop come from re-issuing the same call
you just made; "next_recommended" exists to prevent that.

Also use the occupancy snapshot to choose spots that make spatial sense
(adjacent to existing pieces for multi-block furniture, away from clutter when
starting a new piece, etc.). When the snapshot is empty, you can place wherever
fits the user's intent.

Gripper state machine (READ THE "holding" FIELD BEFORE EVERY CALL — the system
enforces these and a wrong call will be rejected):

  If holding == false, your ONLY valid next arm calls are:
      move_to_pickup_zone, grab_block, move_to_grid_cell, move_to_named_position_up,
      move_block_in_grid, find_empty_cell, reset_occupancy, home
  If holding == true, your ONLY valid next arm calls are:
      drop_block, move_to_grid_cell, move_to_named_position_up,
      find_empty_cell, reset_occupancy, home
  When holding == true you must NEVER call grab_block or move_block_in_grid —
  call drop_block first.

After a successful grab_block, holding becomes true. After a successful
drop_block or move_block_in_grid, holding becomes false. If a tool result says
"error: a block is ALREADY in the gripper" or "error: the gripper is EMPTY",
that means you misread the holding flag — read the latest occupancy.holding
and pick the tool the error message tells you to use. Do NOT retry the same
blocked call.

When the user says they want to start over or change the room, call reset_occupancy
and say something natural like "okay, clearing the room".

Expressive tool (separate from the arm):
  play_emote  name               — play a Reachy Mini emote (head + antenna motion)
                                   when it fits the conversation: greet the user,
                                   show you're thinking, react to surprise, etc.
                                   You do NOT need to call this after arm motions —
                                   the system already plays a fitting emote after
                                   every arm tool call. Use play_emote for purely
                                   expressive moments (hello, agreement, confusion
                                   about a request, etc.).
                                   Available names: {emote_names}.

After finishing a request, give at most one short confirmation ("done", "all
set"). Do not describe what you placed unless the user asks. If a request is
truly ambiguous, ask one short clarifying question — otherwise just execute.
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


def _build_play_emote_tool(emote_names: list[str]) -> dict:
    """OpenAI Realtime tool definition for triggering a Reachy Mini emote."""
    return {
        "type": "function",
        "name": PLAY_EMOTE_TOOL_NAME,
        "description": (
            "Play a Reachy Mini emote (head + antenna motion, optionally with a "
            "short sound). Use to add personality during conversation — e.g. nod "
            "when greeting, look curious when thinking, celebrate after a pick "
            "and place."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": emote_names,
                    "description": "Name of the recorded emote to play.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    }


# ── occupancy grid ────────────────────────────────────────────────────────────

class OccupancyGrid:
    def __init__(self, nx: int, ny: int, state_file: Path | None = None) -> None:
        self.nx         = nx
        self.ny         = ny
        self.state_file = state_file
        self._cells: set[tuple[int, int]] = set()

    def is_occupied(self, i: int, j: int) -> bool:
        return (i, j) in self._cells

    def mark_occupied(self, i: int, j: int) -> None:
        self._cells.add((i, j))
        self._save()

    def mark_empty(self, i: int, j: int) -> None:
        self._cells.discard((i, j))
        self._save()

    def next_empty(self) -> tuple[int, int] | None:
        for j in range(self.ny):
            for i in range(self.nx):
                if (i, j) not in self._cells:
                    return (i, j)
        return None

    def reset(self) -> None:
        self._cells.clear()
        self._save()

    def load(self) -> None:
        if self.state_file and self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self._cells = {(int(c[0]), int(c[1])) for c in data.get("occupied", [])}
                print(f"[occupancy] loaded from {self.state_file.name}: {self.status()}")
            except Exception as exc:
                print(f"[occupancy] load error: {exc}")

    def _save(self) -> None:
        if self.state_file:
            try:
                data = {"occupied": sorted([list(c) for c in self._cells])}
                self.state_file.write_text(json.dumps(data, indent=2))
            except Exception:
                pass

    def status(self) -> str:
        n   = len(self._cells)
        tot = self.nx * self.ny
        if n == 0:
            return f"grid {self.nx}×{self.ny}: all {tot} cells empty"
        cells = sorted(self._cells, key=lambda c: (c[1], c[0]))
        return f"grid {self.nx}×{self.ny}: {n}/{tot} occupied — {cells}"


# ── Reachy emote controller ────────────────────────────────────────────────────

class ReachyEmoteController:
    """Owns a ReachyMini SDK client and plays emotes from the emotions library."""

    DATASET = "pollen-robotics/reachy-mini-emotions-library"

    def __init__(self) -> None:
        from reachy_mini import ReachyMini  # noqa: PLC0415
        from reachy_mini.motion.recorded_move import RecordedMoves  # noqa: PLC0415

        self._reachy  = ReachyMini(
            connection_mode="auto", media_backend="no_media", timeout=15.0,
        )
        self._library = RecordedMoves(self.DATASET)
        self._moves   = self._library.list_moves()
        print(f"Reachy emotes loaded ({len(self._moves)} available).")

    @property
    def available(self) -> list[str]:
        return list(self._moves)

    def play(self, name: str) -> str:
        if name not in self._moves:
            return f"error: unknown emote {name!r}"
        try:
            self._reachy.play_move(self._library.get(name), initial_goto_duration=0.4)
        except Exception as exc:
            return f"error: emote {name!r} failed: {exc}"
        return f"played emote {name!r}"

    def close(self) -> None:
        try:
            self._reachy.client.disconnect()
        except Exception:
            pass


# ── executors ──────────────────────────────────────────────────────────────────

class GraspExecutor(ABC):
    def __init__(self) -> None:
        self._occupancy: OccupancyGrid | None = None
        self._pending_cell: tuple[int, int] | None = None
        self._pending_move_src: tuple[int, int] | None = None
        self._holding: bool = False

    @abstractmethod
    def execute_tool(self, name: str, tool_input: dict) -> str: ...

    def _gripper_pre(self, name: str) -> str | None:
        """Block grabs when already holding, and drops when not holding.

        Error messages are directive: they spell out the next valid tool call so
        the model can recover without re-trying the same blocked call.
        """
        if name == "grab_block" and self._holding:
            return (
                "error: a block is ALREADY in the gripper (holding=true). Do NOT "
                "call grab_block. Your next call must be drop_block (to place it "
                "where you are now) or move_to_grid_cell (to move above another "
                "spot first, then drop_block)."
            )
        if name == "drop_block" and not self._holding:
            return (
                "error: the gripper is EMPTY (holding=false). Do NOT call "
                "drop_block. Your next call must be move_to_pickup_zone followed "
                "by grab_block to pick up a fresh block."
            )
        if name == "move_block_in_grid" and self._holding:
            return (
                "error: a block is ALREADY in the gripper (holding=true). Do NOT "
                "call move_block_in_grid. Your next call must be drop_block "
                "(to place the held block first)."
            )
        return None

    def _gripper_post(self, name: str, result: str) -> None:
        """Update gripper state after a successful motion that touched it."""
        if result.startswith("error") or result.startswith("IK failed"):
            return
        if name == "grab_block":
            self._holding = True
        elif name in ("drop_block", "move_block_in_grid"):
            self._holding = False

    def _occupancy_pre(self, name: str, tool_input: dict) -> str | None:
        """Intercept occupancy tools; also gate move_to_grid_cell on free cells.
        Returns a result string if the call is fully handled, None to continue dispatch."""
        occ = self._occupancy
        if name == "find_empty_cell":
            if occ is None:
                return "error: no grid loaded"
            cell = occ.next_empty()
            return f"i={cell[0]} j={cell[1]}" if cell else "error: all cells are occupied"
        if name == "reset_occupancy":
            if occ is not None:
                occ.reset()
            return "occupancy grid cleared — all cells empty"
        if name == "move_to_grid_cell" and occ is not None:
            try:
                i, j = int(tool_input["i"]), int(tool_input["j"])
            except (KeyError, ValueError):
                return None
            if occ.is_occupied(i, j):
                alt = occ.next_empty()
                hint = f" Try i={alt[0]} j={alt[1]}." if alt else " No empty cells remain."
                return f"error: cell ({i},{j}) is already occupied.{hint}"
            self._pending_cell = (i, j)
        if name == "move_block_in_grid" and occ is not None:
            try:
                to_i, to_j   = int(tool_input["to_i"]),   int(tool_input["to_j"])
                from_i, from_j = int(tool_input["from_i"]), int(tool_input["from_j"])
            except (KeyError, ValueError):
                return None
            if occ.is_occupied(to_i, to_j):
                alt = occ.next_empty()
                hint = f" Try i={alt[0]} j={alt[1]}." if alt else " No empty cells remain."
                return f"error: destination ({to_i},{to_j}) is already occupied.{hint}"
            self._pending_cell     = (to_i, to_j)
            self._pending_move_src = (from_i, from_j)
        return None

    def _occupancy_post_drop(self, result: str) -> None:
        """Mark the pending cell occupied once drop_block succeeds."""
        if self._occupancy and self._pending_cell and not result.startswith("error"):
            self._occupancy.mark_occupied(*self._pending_cell)
            self._pending_cell = None

    def _occupancy_post_move_block(self, result: str) -> None:
        """After move_block_in_grid: mark destination occupied, source empty."""
        if self._occupancy and self._pending_cell and self._pending_move_src \
                and not result.startswith("error"):
            self._occupancy.mark_occupied(*self._pending_cell)
            self._occupancy.mark_empty(*self._pending_move_src)
        self._pending_cell     = None
        self._pending_move_src = None

    def occupancy_snapshot(self) -> dict | None:
        """Structured snapshot of the room layout for inclusion in tool results."""
        occ = self._occupancy
        if occ is None:
            return {"holding": self._holding}
        return {
            "nx":         occ.nx,
            "ny":         occ.ny,
            "occupied":   sorted([list(c) for c in occ._cells]),
            "next_empty": list(occ.next_empty()) if occ.next_empty() else None,
            "status":     occ.status(),
            "holding":    self._holding,
        }

    def info(self) -> str:
        return ""

    def close(self) -> None:
        pass


class DryRunGraspExecutor(GraspExecutor):
    def __init__(self, nx: int = 4, ny: int = 3) -> None:
        super().__init__()
        self._occupancy = OccupancyGrid(nx, ny)

    def execute_tool(self, name: str, tool_input: dict) -> str:
        gripper_err = self._gripper_pre(name)
        if gripper_err is not None:
            return gripper_err
        result = self._occupancy_pre(name, tool_input)
        if result is not None:
            return result
        args_str = ", ".join(f"{k}={v!r}" for k, v in tool_input.items())
        result = f"dry-run: {name}({args_str})"
        if name == "drop_block":
            self._occupancy_post_drop(result)
        elif name == "move_block_in_grid":
            self._occupancy_post_move_block(result)
        self._gripper_post(name, result)
        return result

    def info(self) -> str:
        occ = self._occupancy.status() if self._occupancy else "no grid"
        return f"Backend: dry-run (no hardware) | {occ}"


class ReBotGraspExecutor(GraspExecutor):
    def __init__(
        self,
        positions_file: Path,
        grid_file: Path,
        arm_config: str | None,
        duration: float,
        state_file: Path | None = None,
    ) -> None:
        super().__init__()
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
            self._occupancy = OccupancyGrid(self.grid_nx, self.grid_ny, state_file)
            self._occupancy.load()

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
        gripper_err = self._gripper_pre(name)
        if gripper_err is not None:
            return gripper_err
        pre = self._occupancy_pre(name, tool_input)
        if pre is not None:
            return pre
        result = self._dispatch(
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
        if name == "drop_block":
            self._occupancy_post_drop(result)
        elif name == "move_block_in_grid":
            self._occupancy_post_move_block(result)
        self._gripper_post(name, result)
        return result

    def info(self) -> str:
        pos_names = [n for n in self._raw if n.lower() not in self._JOINT_LABELS]
        occ = self._occupancy.status() if self._occupancy else "no grid loaded"
        return f"positions: {', '.join(pos_names)}  |  {occ}"

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
        instructions: str,
        voice: str,
        verbose: bool,
        executor: GraspExecutor,
        openai_tools: list[dict],
        context_info: str,
        audio_input_device: int,
        audio_output_device: int,
        emote_controller: "ReachyEmoteController | None",
        event_log_path: Path | None = None,
    ) -> None:
        self.api_key      = api_key
        self.model        = model
        self.instructions = instructions
        self.voice        = voice
        self.verbose      = verbose
        self.executor     = executor
        self.openai_tools = openai_tools
        self.context_info = context_info
        self.audio_input_device  = audio_input_device
        self.audio_output_device = audio_output_device
        self.emote_controller    = emote_controller
        self.ws: websocket.WebSocket | None = None
        self.stop_event   = threading.Event()
        self.audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=100)
        self._dispatched_calls: set[str] = set()   # dedup tool calls across events
        self._pending_followup: bool = False       # ask model to follow up after current response.done
        self._last_call_key: str | None = None     # signature of most-recent tool call (name + sorted args)
        self._last_call_count: int = 0             # consecutive same-signature calls (success OR fail)
        self._last_recommended_next: str | None = None  # what the prior result told the model to do
        self._current_response_id: str | None = None  # active response.id (None if no response in flight)
        self._calls_this_response: int = 0         # how many function_calls dispatched in the current response
        # Outputs are buffered and flushed at response.done so the conversation
        # history is consistent before the model is asked to follow up.
        self._pending_outputs: list[dict[str, Any]] = []
        self._event_log_fp: Any | None = None
        if event_log_path is not None:
            self._event_log_fp = open(event_log_path, "a", buffering=1)
            print(f"Logging raw events to {event_log_path}")
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
            device=self.audio_input_device,
        )
        self._output_stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=1200, callback=self._output_callback,
            device=self.audio_output_device,
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
        if self.emote_controller is not None:
            self.emote_controller.close()
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
        if self._event_log_fp is not None:
            try:
                self._event_log_fp.close()
            except Exception:
                pass
            self._event_log_fp = None

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
                "instructions": self.instructions.strip() + "\n\n" + self.context_info,
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
                self._pending_followup = False
                self._pending_outputs.clear()

            elif etype == "response.created":
                # Mark a new response in flight so per-response logs are linkable.
                self._current_response_id = (
                    event.get("response", {}).get("id")
                )
                self._calls_this_response = 0
                print(f"[response.created]  id={self._current_response_id!r}")

            elif etype == "response.output_item.done":
                # Execute the tool now (low latency) but BUFFER the output —
                # it will be sent at response.done so the conversation history
                # is consistent before we request the follow-up response.
                item  = event.get("item", {})
                itype = item.get("type")
                if itype == "function_call":
                    self._calls_this_response += 1
                    rid = self._current_response_id
                    cid = item.get("call_id", "")
                    print(f"[tool-call]  response={rid!r} call_id={cid!r} "
                          f"name={item.get('name')!r} args={item.get('arguments')!r}")
                    if self._dispatch_tool_call(item):
                        self._pending_followup = True
                elif self.verbose:
                    print(f"[item]  type={itype!r}  name={item.get('name')!r}")

            elif etype == "response.done":
                rid = event.get("response", {}).get("id")
                output = event.get("response", {}).get("output", [])
                fc_count = sum(1 for it in output if it.get("type") == "function_call")
                print(f"[response.done]  id={rid!r} items={len(output)} "
                      f"function_calls={fc_count} dispatched_this_resp="
                      f"{self._calls_this_response}")

                # Flush buffered tool outputs, then request follow-up — once,
                # regardless of how many calls ran.
                if self._pending_outputs:
                    for output_event in self._pending_outputs:
                        self._send(output_event)
                    self._pending_outputs.clear()

                if self._pending_followup:
                    self._pending_followup = False
                    self._send({"type": "response.create"})

                self._current_response_id = None
                self._calls_this_response = 0

            elif etype == "response.function_call_arguments.done":
                rid = event.get("response_id")
                cid = event.get("call_id")
                print(f"[args-done]  response={rid!r} call_id={cid!r} "
                      f"name={event.get('name')!r} args={event.get('arguments')!r}")

            elif etype == "error":
                print(
                    f"[api-error] {json.dumps(event.get('error', event), indent=2)}",
                    file=sys.stderr,
                )

            elif self.verbose:
                print(f"[event] {json.dumps(event, indent=2)}")

    def _dispatch_tool_call(self, item: dict[str, Any]) -> bool:
        """Dispatch a function_call item.

        Returns False when the model emitted unparseable JSON args — a strong
        signal the response was interrupted mid-stream. Callers should skip
        the follow-up response.create in that case, otherwise the model just
        truncates again on the next response and the loop spins.
        """
        name    = item.get("name", "")
        call_id = item.get("call_id", "")
        if call_id in self._dispatched_calls:
            return True
        self._dispatched_calls.add(call_id)
        try:
            args = json.loads(item.get("arguments") or "{}")
            bad_json    = False
            parse_error: json.JSONDecodeError | None = None
            call_key    = f"{name}|{json.dumps(args, sort_keys=True)}"
        except json.JSONDecodeError as exc:
            args        = {}
            bad_json    = True
            parse_error = exc
            call_key    = f"{name}|<bad-args>"

        # Loop-breaker: abort consecutive same-signature calls. Bad-JSON
        # calls trip after just 2 in a row — the truncated bytes are
        # unrecoverable, so retrying produces the same garbage. Parseable
        # calls allow 3 before aborting on the 4th, since the model
        # sometimes needs a beat to react to a result.
        breaker_threshold = 1 if bad_json else 3
        if (
            self._last_call_key == call_key
            and self._last_call_count >= breaker_threshold
        ):
            directive = (
                self._last_recommended_next
                or "Read the 'occupancy' snapshot — especially the "
                   "'holding' field — and pick a DIFFERENT next tool."
            )
            result_str = (
                "error: REPEAT-CALL ABORTED. You have just called this "
                f"exact tool with these arguments {self._last_call_count} "
                "times in a row with no other progress. STOP repeating. "
                f"Required next action: {directive}"
            )
            print(f"[loop-breaker] {call_key} after "
                  f"{self._last_call_count} consecutive calls")
        elif bad_json:
            result_str = f"error: bad JSON args: {parse_error}"
        else:
            print(f"[tool→exec]  {name}({json.dumps(args, sort_keys=True)})")
            if name == PLAY_EMOTE_TOOL_NAME:
                if self.emote_controller is None:
                    result_str = "error: reachy emote controller not available"
                else:
                    result_str = self.emote_controller.play(args.get("name", ""))
            else:
                result_str = self._run_arm_tool_with_ik_fallback(name, args)

        success = not (
            result_str.startswith("error") or result_str.startswith("IK failed")
        )

        # Track consecutive same-signature calls (regardless of outcome).
        if self._last_call_key == call_key:
            self._last_call_count += 1
        else:
            self._last_call_key = call_key
            self._last_call_count = 1

        result: dict[str, Any] = {"success": success, "message": result_str}
        snapshot = self.executor.occupancy_snapshot()
        if snapshot is not None:
            result["occupancy"] = snapshot

        # Directive next-call hint — tells the model the EXACT tool to call
        # next based on the post-state. Empirically the model loops when the
        # prompt only describes "valid" tools; a concrete recommendation per
        # result is much more reliable.
        holding = snapshot.get("holding", False) if snapshot else False
        next_hint = self._recommended_next(name, success, holding)
        if next_hint is not None:
            result["next_recommended"] = next_hint
            self._last_recommended_next = next_hint
        print(f"[tool←result]  {json.dumps(result, sort_keys=True)}")

        # Auto-emote after arm tools: motion first, then personality.
        if self.emote_controller is not None and name != PLAY_EMOTE_TOOL_NAME:
            emote_name = (
                ARM_TOOL_SUCCESS_EMOTE.get(name) if success else ARM_TOOL_ERROR_EMOTE
            )
            if emote_name and emote_name in self.emote_controller.available:
                print(f"[auto-emote] {emote_name}")
                self.emote_controller.play(emote_name)
        # Buffer the output — the receive loop flushes all pending outputs at
        # response.done (before sending response.create), so the conversation
        # history is consistent when the model is asked to follow up. This
        # avoids racing the in-flight response and prevents the model from
        # re-emitting a function_call because it can't see the prior output.
        self._pending_outputs.append({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
            },
        })
        return not bad_json

    def _recommended_next(self, name: str, success: bool, holding: bool) -> str | None:
        """Return the directive next call the model should make.

        The recommendation depends ONLY on (just-executed tool, success, holding)
        — a small explicit state machine that mirrors the canonical pick-and-place
        flow. This is the single most effective tool against the model looping
        on a recently-successful call.
        """
        if not success:
            # On error the rejection / IK / loop-breaker message already tells
            # the model what to do; don't add a hint that might conflict.
            return None
        if name == PLAY_EMOTE_TOOL_NAME:
            return None
        if name == "move_to_pickup_zone":
            if holding:
                return ("Unexpected: gripper holding after move_to_pickup_zone. "
                        "Call drop_block to clear.")
            return ("Call grab_block({\"grip_style\": \"edge\"}) NEXT to pick "
                    "up the block from the pickup zone.")
        if name == "grab_block":
            return ("Call move_to_grid_cell({\"i\": <col>, \"j\": <row>}) NEXT "
                    "with the target cell from the recipe you are building. "
                    "After that, drop_block.")
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
                    "more cells to rearrange, call move_block_in_grid again "
                    "for the next pair; otherwise stop.")
        if name == "move_to_named_position_up":
            if holding:
                return ("Call drop_block NEXT to place the block here, or "
                        "move_to_grid_cell to relocate.")
            return ("Call grab_block NEXT if this is a pickup spot, otherwise "
                    "decide based on context.")
        if name == "find_empty_cell":
            return ("Call move_to_grid_cell with the (i, j) just returned, then "
                    "drop_block.")
        if name == "reset_occupancy":
            return ("Room cleared. Wait for the next user instruction unless "
                    "you were already mid-recipe.")
        if name == "home":
            return None
        return None

    def _run_arm_tool_with_ik_fallback(self, name: str, args: dict[str, Any]) -> str:
        """Execute an arm tool; if IK fails, go home and retry once.

        Skip the retry when the failure left a block in the gripper
        (move_block_in_grid destination IK fail) — retrying would attempt to
        re-grab a now-empty source. Also skip when the failed tool is 'home'
        itself.
        """
        result = self.executor.execute_tool(name, args)
        if (
            result.startswith("IK failed")
            and "block still held" not in result
            and name != "home"
        ):
            print(f"[ik-fallback] {name} failed: {result!r} — going home and retrying")
            home_result = self.executor.execute_tool("home", {})
            print(f"[ik-fallback] home -> {home_result!r}")
            if not (
                home_result.startswith("error") or home_result.startswith("IK failed")
            ):
                retry_result = self.executor.execute_tool(name, args)
                print(f"[ik-fallback] retry -> {retry_result!r}")
                return retry_result
        return result

    # ── ws helpers ─────────────────────────────────────────────────────────────

    _NOISY_EVENT_TYPES = frozenset({
        "response.output_audio.delta",
        "response.audio.delta",
        "response.function_call_arguments.delta",
        "response.output_audio_transcript.delta",
        "response.audio_transcript.delta",
        "input_audio_buffer.append",
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
    })

    def _log_event(self, direction: str, event: dict[str, Any]) -> None:
        """Append a raw event to the event log (if configured), excluding deltas."""
        if self._event_log_fp is None:
            return
        if event.get("type") in self._NOISY_EVENT_TYPES:
            return
        try:
            self._event_log_fp.write(json.dumps({
                "t":   time.time(),
                "dir": direction,
                "ev":  event,
            }) + "\n")
        except Exception:
            pass

    def _send(self, event: dict[str, Any]) -> None:
        if self.verbose and event.get("type") != "input_audio_buffer.append":
            print(f"[send] {json.dumps(event, indent=2)}")
        self._log_event("send", event)
        if self.ws is None:
            raise RuntimeError("WebSocket not connected")
        self.ws.send(json.dumps(event))

    def _recv(self) -> dict[str, Any]:
        if self.ws is None:
            raise RuntimeError("WebSocket not connected")
        event = json.loads(self.ws.recv())
        self._log_event("recv", event)
        if self.verbose and event.get("type") not in self._NOISY_EVENT_TYPES:
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
    parser.add_argument("--state-file",     default=DEFAULT_STATE_FILE,
                        help="JSON file shared with grid_gui.py for occupancy sync.")
    parser.add_argument("--recipes-file",   default=DEFAULT_RECIPES_FILE,
                        help=f"YAML furniture recipe library (default: "
                             f"{DEFAULT_RECIPES_FILE!r}).")
    parser.add_argument("--arm-config",     default=None,
                        help="Path to reBotArm arm.yaml (auto-detected when omitted).")
    parser.add_argument("--duration",       type=float, default=2.0,
                        help="Default motion duration in seconds.")
    parser.add_argument("--dry-run",        action="store_true",
                        help="No hardware — print tool results only.")
    parser.add_argument("--audio-device",   default=DEFAULT_AUDIO_DEVICE,
                        help=f"Substring of audio device name for mic + speaker "
                             f"(default: {DEFAULT_AUDIO_DEVICE!r}).")
    parser.add_argument("--no-emotes",      action="store_true",
                        help="Disable Reachy emote tool (skip SDK + emote library load).")
    parser.add_argument("--event-log",      default=None,
                        help="Write raw OpenAI Realtime events (sent + received, "
                             "minus audio deltas) to this JSONL file for forensic "
                             "debugging of tool-call streams.")
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

    state_path = Path(args.state_file)
    if not state_path.is_absolute():
        state_path = _REPO_ROOT / state_path

    recipes_path = Path(args.recipes_file)
    if not recipes_path.is_absolute():
        recipes_path = _REPO_ROOT / recipes_path
    recipes = load_recipes(recipes_path)
    if recipes:
        print(f"Loaded {len(recipes)} recipe(s) from {recipes_path.name}: "
              f"{list(recipes)}")
    else:
        print(f"[warn] no recipes loaded from {recipes_path}")

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
            state_file=state_path,
        )

    context_info = executor.info()

    emote_controller: ReachyEmoteController | None = None
    emote_names: list[str] = []
    if not args.no_emotes:
        emote_controller = ReachyEmoteController()
        emote_names = emote_controller.available
        openai_tools.append(_build_play_emote_tool(emote_names))

    instructions = INSTRUCTIONS_TEMPLATE.format(
        emote_names=", ".join(emote_names) if emote_names else "(emotes disabled)",
        recipe_library=format_recipe_library(recipes),
    )

    in_idx, out_idx = _resolve_audio_device(args.audio_device)
    in_name  = sd.query_devices(in_idx)["name"]
    out_name = sd.query_devices(out_idx)["name"]
    print(f"Audio I/O: input=[{in_idx}] {in_name!r}  output=[{out_idx}] {out_name!r}")

    event_log_path = Path(args.event_log) if args.event_log else None
    if event_log_path is not None and not event_log_path.is_absolute():
        event_log_path = _REPO_ROOT / event_log_path

    loop = RealtimeVoiceToolLoop(
        api_key=api_key,
        model=args.model,
        instructions=instructions,
        voice=args.voice,
        verbose=args.verbose,
        executor=executor,
        openai_tools=openai_tools,
        context_info=context_info,
        audio_input_device=in_idx,
        audio_output_device=out_idx,
        emote_controller=emote_controller,
        event_log_path=event_log_path,
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
