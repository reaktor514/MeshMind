"""Unit tests for the mesh post-processing helpers."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def trimesh_module():
    return pytest.importorskip("trimesh")


def _box(extent: float = 1.0):
    import trimesh

    return trimesh.creation.box(extents=(extent, extent, extent))


def _hi_res_blob():
    """Higher-poly mesh so it dominates a face-count threshold against simple boxes."""
    import trimesh

    return trimesh.creation.icosphere(subdivisions=3)


def test_largest_connected_component_keeps_biggest(trimesh_module) -> None:
    import trimesh

    from meshmind.utils.mesh_io import largest_connected_component

    big = _hi_res_blob()
    small = _box(0.2)
    small.apply_translation([5.0, 0.0, 0.0])
    combined = trimesh.util.concatenate([big, small])

    out = largest_connected_component(combined)

    assert len(out.faces) == len(big.faces)


def test_drop_small_components_removes_only_floaters(trimesh_module) -> None:
    import trimesh

    from meshmind.utils.mesh_io import drop_small_components

    big = _hi_res_blob()
    floater_a = _box(0.05)
    floater_a.apply_translation([5.0, 0.0, 0.0])
    floater_b = _box(0.05)
    floater_b.apply_translation([0.0, 5.0, 0.0])
    combined = trimesh.util.concatenate([big, floater_a, floater_b])

    out = drop_small_components(combined, min_face_fraction=0.1)

    assert len(out.faces) == len(big.faces)


def test_drop_small_components_falls_back_to_largest(trimesh_module) -> None:
    import trimesh

    from meshmind.utils.mesh_io import drop_small_components

    big = _hi_res_blob()
    medium = _box(0.5)
    medium.apply_translation([5.0, 0.0, 0.0])
    combined = trimesh.util.concatenate([big, medium])

    # threshold=0.99 of total faces — every component fails it; should fall
    # back to the largest island instead of returning an empty mesh.
    out = drop_small_components(combined, min_face_fraction=0.99)

    assert len(out.faces) == len(big.faces)


def test_smooth_mesh_zero_iters_is_noop(trimesh_module) -> None:
    from meshmind.utils.mesh_io import smooth_mesh

    box = _box(1.0)
    before = box.vertices.copy()
    smooth_mesh(box, iterations=0)
    np.testing.assert_array_equal(before, box.vertices)


def test_smooth_mesh_moves_vertices(trimesh_module) -> None:
    from meshmind.utils.mesh_io import smooth_mesh

    box = _box(1.0)
    before = box.vertices.copy()
    smooth_mesh(box, iterations=2, lamb=0.5)
    # Laplacian smoothing must actually deform a sharp box.
    assert np.linalg.norm(box.vertices - before) > 1e-6


def test_fix_mesh_normals_runs(trimesh_module) -> None:
    from meshmind.utils.mesh_io import fix_mesh_normals

    box = _box(1.0)
    out = fix_mesh_normals(box)
    assert out is box
    assert out.face_normals.shape[0] == len(box.faces)


def test_decimate_no_op_when_below_target(trimesh_module) -> None:
    from meshmind.utils.mesh_io import decimate_mesh

    box = _box(1.0)
    out = decimate_mesh(box, target_faces=10_000)
    assert len(out.faces) == len(box.faces)


def test_clean_mesh_pipeline_runs(trimesh_module) -> None:
    import trimesh

    from meshmind.utils.mesh_io import CleanupOptions, clean_mesh

    big = _hi_res_blob()
    floater = _box(0.05)
    floater.apply_translation([5.0, 0.0, 0.0])
    combined = trimesh.util.concatenate([big, floater])

    out = clean_mesh(
        combined,
        CleanupOptions(
            drop_floaters=True,
            floater_min_face_fraction=0.1,
            smooth_iters=1,
            smooth_lamb=0.3,
            fix_normals=True,
            fill_holes=False,
            target_faces=None,
        ),
    )

    # Floater dropped, normals computed.
    assert len(out.faces) == len(big.faces)
    assert out.face_normals.shape[0] == len(out.faces)


def test_clean_mesh_default_options_safe(trimesh_module) -> None:
    from meshmind.utils.mesh_io import clean_mesh

    box = _box(1.0)
    out = clean_mesh(box)
    assert len(out.faces) == len(box.faces)


def test_mesh_summary_keys(trimesh_module) -> None:
    from meshmind.utils.mesh_io import mesh_summary

    box = _box(1.0)
    summary = mesh_summary(box)
    assert summary["faces"] == len(box.faces)
    assert summary["vertices"] == len(box.vertices)
    assert isinstance(summary["watertight"], bool)
    assert len(summary["extents"]) == 3
