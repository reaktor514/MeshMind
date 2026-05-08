"""Mesh export and post-processing utilities."""

from __future__ import annotations

from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


@dataclass
class CleanupOptions:
    """Knobs for :func:`clean_mesh`.

    Defaults are tuned for ShapE outputs: a single subject blob plus a few
    small floating clusters and noisy normals.
    """

    drop_floaters: bool = True
    floater_min_face_fraction: float = 0.01
    smooth_iters: int = 3
    smooth_lamb: float = 0.5
    fix_normals: bool = True
    fill_holes: bool = False
    target_faces: int | None = None


def largest_connected_component(mesh):  # noqa: ANN001, ANN201
    """Return the biggest connected component of ``mesh`` by face count."""
    parts = mesh.split(only_watertight=False)
    if len(parts) == 0:
        return mesh
    return max(parts, key=lambda m: len(m.faces))


def drop_small_components(mesh, min_face_fraction: float = 0.01):  # noqa: ANN001, ANN201
    """Drop connected components whose face count is below
    ``min_face_fraction`` of the total.  Falls back to keeping the largest
    component if everything would otherwise be filtered out.
    """
    import trimesh

    parts = list(mesh.split(only_watertight=False))
    if len(parts) <= 1:
        return mesh
    total = sum(len(p.faces) for p in parts)
    if total == 0:
        return mesh
    threshold = max(1, int(total * float(min_face_fraction)))
    kept = [p for p in parts if len(p.faces) >= threshold]
    if not kept:
        return largest_connected_component(mesh)
    if len(kept) == 1:
        return kept[0]
    return trimesh.util.concatenate(kept)


def smooth_mesh(mesh, iterations: int = 3, lamb: float = 0.5):  # noqa: ANN001, ANN201
    """Apply Laplacian smoothing and return the (in-place mutated) mesh."""
    import trimesh

    if iterations <= 0:
        return mesh
    trimesh.smoothing.filter_laplacian(mesh, lamb=float(lamb), iterations=int(iterations))
    return mesh


def fix_mesh_normals(mesh):  # noqa: ANN001, ANN201
    """Reorient faces consistently and recompute normals."""
    mesh.fix_normals()
    return mesh


def decimate_mesh(mesh, target_faces: int):  # noqa: ANN001, ANN201
    """Reduce face count toward ``target_faces`` using trimesh quadric
    decimation.  Returns the original mesh untouched if no decimation backend
    (``open3d`` / ``fast-simplification``) is available.
    """
    if target_faces is None or target_faces <= 0:
        return mesh
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(int(target_faces))
    except Exception:
        return mesh


def clean_mesh(mesh, options: CleanupOptions | None = None):  # noqa: ANN001, ANN201
    """Run a configurable cleanup pipeline on a :class:`trimesh.Trimesh`.

    Order matters: drop floaters first (so smoothing doesn't waste effort on
    them), then smooth, fix normals, optionally fill holes, and finally
    decimate to a target poly budget.
    """
    options = options or CleanupOptions()

    if options.drop_floaters:
        mesh = drop_small_components(mesh, min_face_fraction=options.floater_min_face_fraction)

    if options.smooth_iters > 0:
        mesh = smooth_mesh(mesh, iterations=options.smooth_iters, lamb=options.smooth_lamb)

    if options.fix_normals:
        mesh = fix_mesh_normals(mesh)

    if options.fill_holes:
        mesh.fill_holes()

    if options.target_faces:
        mesh = decimate_mesh(mesh, options.target_faces)

    return mesh


def mesh_summary(mesh) -> dict:  # noqa: ANN001
    """Return a small dict with counts and extents for logging."""
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "extents": [float(x) for x in mesh.extents],
    }
