# meshes/

Drop your CAD meshes here. Recommended formats:

- `.obj` — broadly supported, FoundationPose-friendly
- `.stl` — fine, but no material info
- `.ply` — fine

Conventions:

- Origin at the geometric center of the block, or at the natural grasp point.
  Whatever you choose, the grasp_sites in `manifest.yaml` are expressed in this
  local frame.
- Units: meters. FoundationPose expects meters. If you exported from CAD in mm,
  scale by 0.001 before saving.
- Watertight. FoundationPose handles non-watertight meshes but quality drops.
  Run `meshlab` or `trimesh.repair.fix_normals` if exports look wrong.
- Triangle count: 2k-20k is fine. Above 100k just slows FoundationPose down
  without helping accuracy.

After adding a mesh, register it in `../manifest.yaml`.
