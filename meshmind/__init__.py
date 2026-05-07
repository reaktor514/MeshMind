"""MeshMind — own neural network for 2D/3D modelling.

Public API:
    >>> from meshmind import MeshMindPipeline, MeshMindConfig
    >>> pipe = MeshMindPipeline.from_pretrained()
    >>> img = pipe.text_to_image("a red cyber-dragon, studio render")
    >>> mesh = pipe.text_to_mesh("a red cyber-dragon, studio render")
    >>> mesh.export("dragon.glb")

The package is split into:
    meshmind.models    — neural network modules (diffusion, ShapE, custom refiner).
    meshmind.utils     — IO helpers (mesh export, image preprocessing, seeding).
    meshmind.pipeline  — high-level orchestration of the full text/image -> 2D/3D flow.
"""

from meshmind.config import MeshMindConfig
from meshmind.pipeline import MeshMindPipeline

__all__ = ["MeshMindConfig", "MeshMindPipeline"]
__version__ = "0.1.0"
