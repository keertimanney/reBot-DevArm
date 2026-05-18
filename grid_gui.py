#!/usr/bin/env python3
"""Interactive GUI for managing grid cell occupancy.

Displays the grid defined in grid_config.yaml as clickable cells.
Click a cell to toggle occupied/empty.  State is written to
grid_occupancy.json so run_voice_agent.py can read/write the same file.

The GUI polls the state file every 500 ms so it reflects changes made by
the voice agent in real time (bidirectional sync via a shared JSON file).

Usage:
    python grid_gui.py
    python grid_gui.py --grid-file grid_config.yaml
    python grid_gui.py --state-file grid_occupancy.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
from compute_grid import compute_grid_map

_DEFAULT_GRID_FILE  = "grid_config.yaml"
_DEFAULT_STATE_FILE = "grid_occupancy.json"

# ── colours ───────────────────────────────────────────────────────────────────
_C_EMPTY          = "#43A047"   # green — free cell
_C_EMPTY_HOVER    = "#66BB6A"
_C_OCCUPIED       = "#E53935"   # red — blocked cell
_C_OCCUPIED_HOVER = "#EF5350"
_C_BG             = "#263238"   # dark background
_C_PANEL          = "#37474F"
_C_TEXT           = "#ECEFF1"
_C_STATUS_EMPTY   = "#A5D6A7"
_C_STATUS_OCC     = "#EF9A9A"


class GridGUI:
    _POLL_MS = 500   # file-sync poll interval

    def __init__(
        self,
        root: tk.Tk,
        nx: int,
        ny: int,
        grid_map: dict[tuple[int, int], tuple[float, float]],
        z_height: float,
        state_file: Path,
    ) -> None:
        self.root       = root
        self.nx         = nx
        self.ny         = ny
        self.grid_map   = grid_map
        self.z_height   = z_height
        self.state_file = state_file

        self.occupied: set[tuple[int, int]] = set()
        self.buttons:  dict[tuple[int, int], tk.Button] = {}
        self._file_mtime: float = 0.0

        self._build_ui()
        self._load_state(from_poll=False)
        self._schedule_poll()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.title("Grid Occupancy Manager")
        self.root.configure(bg=_C_BG)
        self.root.resizable(False, False)

        # ── header ────────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=_C_BG)
        header.pack(fill="x", padx=16, pady=(12, 4))

        tk.Label(
            header,
            text="Grid Occupancy",
            font=("Helvetica", 16, "bold"),
            bg=_C_BG, fg=_C_TEXT,
        ).pack(side="left")

        tk.Label(
            header,
            text=f"{self.nx} cols × {self.ny} rows   z={self.z_height:.3f} m",
            font=("Helvetica", 10),
            bg=_C_BG, fg="#90A4AE",
        ).pack(side="right")

        # ── legend ────────────────────────────────────────────────────────────
        legend = tk.Frame(self.root, bg=_C_BG)
        legend.pack(fill="x", padx=16, pady=(0, 6))
        for colour, label in ((_C_EMPTY, "empty"), (_C_OCCUPIED, "occupied")):
            tk.Frame(legend, bg=colour, width=14, height=14).pack(side="left", padx=(0, 4))
            tk.Label(legend, text=label, font=("Helvetica", 9),
                     bg=_C_BG, fg="#90A4AE").pack(side="left", padx=(0, 14))

        # ── grid cells ────────────────────────────────────────────────────────
        grid_frame = tk.Frame(self.root, bg=_C_PANEL, bd=0)
        grid_frame.pack(padx=16, pady=6)

        cell_w = max(90, 320 // self.nx)
        cell_h = max(60, 200 // self.ny)

        # j=ny-1 at top (far row), j=0 at bottom (near row) — matches physical table
        for row_idx, j in enumerate(range(self.ny - 1, -1, -1)):
            # row label (j value)
            tk.Label(
                grid_frame,
                text=f"j={j}",
                font=("Courier", 8),
                bg=_C_PANEL, fg="#78909C",
                width=3,
            ).grid(row=row_idx, column=0, padx=(4, 2), pady=2)

            for i in range(self.nx):
                xy = self.grid_map.get((i, j))
                xy_str = f"{xy[0]:+.3f}\n{xy[1]:+.3f}" if xy else ""
                btn = tk.Button(
                    grid_frame,
                    text=f"({i},{j})\n{xy_str}",
                    width=cell_w // 8,
                    height=3,
                    font=("Courier", 8),
                    relief="flat",
                    bd=0,
                    cursor="hand2",
                    command=lambda ci=i, cj=j: self._toggle(ci, cj),
                )
                btn.grid(row=row_idx, column=i + 1, padx=3, pady=3)
                self.buttons[(i, j)] = btn

        # column labels (i values)
        tk.Label(grid_frame, text="", bg=_C_PANEL, width=3).grid(row=self.ny, column=0)
        for i in range(self.nx):
            tk.Label(
                grid_frame,
                text=f"i={i}",
                font=("Courier", 8),
                bg=_C_PANEL, fg="#78909C",
            ).grid(row=self.ny, column=i + 1, pady=(0, 4))

        # ── status bar ────────────────────────────────────────────────────────
        self.status_var = tk.StringVar()
        tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Helvetica", 10),
            bg=_C_BG, fg=_C_TEXT,
            pady=4,
        ).pack()

        # ── control buttons ───────────────────────────────────────────────────
        btn_frame = tk.Frame(self.root, bg=_C_BG)
        btn_frame.pack(pady=(4, 12))

        for label, cmd, colour in (
            ("Clear All",  self._clear_all,  "#43A047"),
            ("Fill All",   self._fill_all,   "#E53935"),
            ("Reload",     lambda: self._load_state(from_poll=False), "#1E88E5"),
        ):
            tk.Button(
                btn_frame,
                text=label,
                command=cmd,
                bg=colour,
                fg="white",
                activebackground=colour,
                font=("Helvetica", 10, "bold"),
                relief="flat",
                bd=0,
                padx=14,
                pady=6,
                cursor="hand2",
            ).pack(side="left", padx=6)

        # auto-save toggle
        self.auto_save = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.root,
            text="Auto-save",
            variable=self.auto_save,
            bg=_C_BG, fg="#90A4AE",
            selectcolor=_C_PANEL,
            activebackground=_C_BG,
            font=("Helvetica", 9),
        ).pack()

        # file path label
        tk.Label(
            self.root,
            text=f"state: {self.state_file.name}",
            font=("Helvetica", 8),
            bg=_C_BG, fg="#546E7A",
        ).pack(pady=(0, 8))

    # ── cell interaction ──────────────────────────────────────────────────────

    def _toggle(self, i: int, j: int) -> None:
        if (i, j) in self.occupied:
            self.occupied.discard((i, j))
        else:
            self.occupied.add((i, j))
        self._refresh_cell(i, j)
        self._update_status()
        if self.auto_save.get():
            self._save_state()

    def _refresh_cell(self, i: int, j: int) -> None:
        btn = self.buttons[(i, j)]
        if (i, j) in self.occupied:
            btn.config(bg=_C_OCCUPIED, fg="white", activebackground=_C_OCCUPIED_HOVER)
        else:
            btn.config(bg=_C_EMPTY,    fg="white", activebackground=_C_EMPTY_HOVER)

    def _refresh_all(self) -> None:
        for key in self.buttons:
            self._refresh_cell(*key)
        self._update_status()

    def _update_status(self) -> None:
        total = self.nx * self.ny
        n     = len(self.occupied)
        empty = total - n
        colour = _C_STATUS_OCC if n > 0 else _C_STATUS_EMPTY
        self.status_var.set(f"  {n} occupied  ·  {empty} empty  ·  {total} total  ")
        # find the status Label widget and update its colour
        for w in self.root.winfo_children():
            if isinstance(w, tk.Label) and w.cget("textvariable") == str(self.status_var):
                w.config(fg=colour)
                break

    # ── bulk operations ───────────────────────────────────────────────────────

    def _clear_all(self) -> None:
        self.occupied.clear()
        self._refresh_all()
        if self.auto_save.get():
            self._save_state()

    def _fill_all(self) -> None:
        for j in range(self.ny):
            for i in range(self.nx):
                self.occupied.add((i, j))
        self._refresh_all()
        if self.auto_save.get():
            self._save_state()

    # ── file I/O ──────────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        data = {"occupied": sorted([list(c) for c in self.occupied])}
        self.state_file.write_text(json.dumps(data, indent=2))
        self._file_mtime = self.state_file.stat().st_mtime
        self.root.title(f"Grid Occupancy Manager  —  saved")

    def _load_state(self, *, from_poll: bool = True) -> None:
        if not self.state_file.exists():
            return
        try:
            mtime = self.state_file.stat().st_mtime
            if from_poll and mtime <= self._file_mtime:
                return   # no change
            data = json.loads(self.state_file.read_text())
            self.occupied = {(int(c[0]), int(c[1])) for c in data.get("occupied", [])}
            self._file_mtime = mtime
            self._refresh_all()
            if not from_poll:
                self.root.title("Grid Occupancy Manager  —  loaded")
        except Exception:
            pass

    def _schedule_poll(self) -> None:
        self._load_state(from_poll=True)
        self.root.after(self._POLL_MS, self._schedule_poll)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive grid occupancy editor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--grid-file",  default=_DEFAULT_GRID_FILE)
    parser.add_argument("--state-file", default=_DEFAULT_STATE_FILE)
    args = parser.parse_args()

    grid_path  = Path(args.grid_file)
    state_path = Path(args.state_file)
    if not grid_path.is_absolute():
        grid_path = _REPO_ROOT / grid_path
    if not state_path.is_absolute():
        state_path = _REPO_ROOT / state_path

    if not grid_path.exists():
        sys.exit(f"Grid file not found: {grid_path}")

    cfg      = yaml.safe_load(grid_path.read_text())
    nx: int  = cfg["grid"]["nx"]
    ny: int  = cfg["grid"]["ny"]
    grid_map, z_height = compute_grid_map(grid_path)

    root = tk.Tk()
    GridGUI(root, nx, ny, grid_map, z_height, state_path)
    root.mainloop()


if __name__ == "__main__":
    main()
