"""Mesh export utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def shape_e_output_to_trimesh(result: Any):  # noqa: ANN201
    """Convert a :class:`diffusers.ShapEPipelineOutput` mesh to ``trimesh.Trimesh``.

    ShapE returns a list of ``MeshDecoderOutput`` objects with ``verts`` /
    ``faces`` tensors.  We take the first mesh in the batch.
    """
    import torch
    import trimesh

    if not hasattr(result, "images"):
        raise TypeError(f"Unexpected ShapE result type: {type(result)!r}")

    meshes = result.images
    if not meshes:
        raise RuntimeError("ShapE pipeline produced no meshes")
    mesh = meshes[0]

    verts = mesh.verts
    faces = mesh.faces
    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)

    vertex_colors = None
    rgb = getattr(mesh, "vertex_channels", None)
    if rgb is not None and {"R", "G", "B"} <= set(rgb):
        r = _to_numpy(rgb["R"])
        g = _to_numpy(rgb["G"])
        b = _to_numpy(rgb["B"])
        vertex_colors = np.clip(np.stack([r, g, b], axis=-1) * 255.0, 0, 255).astype(np.uint8)

    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vertex_colors, process=False)


def _to_numpy(x: Any) -> np.ndarray:
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_mesh(mesh, path: str | Path) -> Path:  # noqa: ANN001
    """Export a :class:`trimesh.Trimesh` to ``.obj``, ``.glb``, ``.ply`` or ``.stl``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower().lstrip(".")
    if suffix not in {"obj", "glb", "gltf", "ply", "stl"}:
        raise ValueError(f"Unsupported mesh format: .{suffix}")
    mesh.export(p)
    return p
