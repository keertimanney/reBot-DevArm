"""CAD library — loads meshes and per-block metadata from a manifest."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class CADEntry:
    cad_id: str
    mesh_path: Path
    color_hint: Optional[str] = None       # for color-prompted grounding
    symmetry: str = "none"                  # "none" | "z2" | "z4" | "spherical"
    grasp_sites: list[dict] = None          # list of {name, pose_in_obj}
    _mesh_cache: object = None              # populated lazily


class CADLibrary:
    """Loads `manifest.yaml`, gives you trimesh meshes on demand.

    Manifest format:
        blocks:
          - cad_id: brick_2x4_red
            mesh: meshes/brick_2x4.obj
            color: red
            symmetry: z2
            grasp_sites:
              - name: top_center
                pose: [[1,0,0,0],[0,1,0,0],[0,0,1,0.012],[0,0,0,1]]
    """

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        with open(manifest_path) as f:
            raw = yaml.safe_load(f)

        self.entries: dict[str, CADEntry] = {}
        root = self.manifest_path.parent
        for b in raw.get("blocks", []):
            entry = CADEntry(
                cad_id=b["cad_id"],
                mesh_path=(root / b["mesh"]).resolve(),
                color_hint=b.get("color"),
                symmetry=b.get("symmetry", "none"),
                grasp_sites=b.get("grasp_sites", []),
            )
            self.entries[entry.cad_id] = entry

    def mesh(self, cad_id: str):
        """Return a trimesh.Trimesh, cached after first load."""
        try:
            entry = self.entries[cad_id]
        except KeyError:
            raise KeyError(f"Unknown cad_id: {cad_id!r}. Known: {list(self.entries)}")
        if entry._mesh_cache is None:
            import trimesh  # local import keeps this module light
            entry._mesh_cache = trimesh.load(entry.mesh_path, force="mesh")
        return entry._mesh_cache

    def ids(self) -> list[str]:
        return list(self.entries.keys())

    def color_hint(self, cad_id: str) -> Optional[str]:
        return self.entries[cad_id].color_hint
