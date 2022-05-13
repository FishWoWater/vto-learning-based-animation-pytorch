#!/usr/bin/env python
import sys 
import os
import glob

sys.path.append(".")

from rendering.renderer import GarmentRenderer

try:
    index = sys.argv.index("--cloth_path")
    cloth_path = sys.argv[index + 1]
    index = sys.argv.index("--body_path")
    body_path = sys.argv[index + 1]
    index = sys.argv.index("--save_path")
    save_path = sys.argv[index + 1]
except ValueError:
    print("Usage: blender --background rendering/scene.blend --python rendering/render.py --cloth_path <path_to_meshes> --body_path <> --savepath <>")


renderer = GarmentRenderer(
    cloth_paths=[],
    body_paths=[],
    cloth_material="ClothMaterialPurple",
    body_material="MannequinMaterial"
)

renderer.render_given_path(cloth_path, body_path, save_path)

