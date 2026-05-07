"""Neural network modules used by MeshMind.

Heavy backbones (Stable Diffusion, ShapE) are loaded lazily on first use so
``import meshmind`` is cheap and side-effect free.
"""

from meshmind.models.refiner import MeshMindRefiner

__all__ = ["MeshMindRefiner"]
