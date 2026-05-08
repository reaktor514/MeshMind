"""Tests for the TripoSR backbone wrapper.

These tests intentionally avoid downloading the actual TripoSR weights — the
asserts cover the *contract* (lazy construction, arg validation, public
surface). End-to-end inference is exercised separately during integration
testing of the live Gradio app.
"""

from __future__ import annotations

import importlib

import pytest


def test_triposr_wrapper_is_lazy() -> None:
    """Constructing the wrapper must NOT trigger weight downloads."""
    from meshmind import MeshMindPipeline

    pipe = MeshMindPipeline()
    assert pipe._triposr is None  # noqa: SLF001

    triposr = pipe.triposr
    assert triposr is not None
    assert pipe._triposr is triposr  # noqa: SLF001
    # Same access returns the same lazy instance.
    assert pipe.triposr is triposr
    # Backbone is held but the model itself is still unloaded.
    assert triposr._model is None  # noqa: SLF001


def test_triposr_config_defaults() -> None:
    from meshmind import MeshMindConfig

    cfg = MeshMindConfig()
    assert cfg.triposr_model_id == "stabilityai/TripoSR"
    assert cfg.triposr_mc_resolution >= 64
    assert cfg.triposr_chunk_size > 0


def test_triposr_pipeline_methods_exist() -> None:
    from meshmind import MeshMindPipeline

    pipe = MeshMindPipeline()
    assert callable(pipe.image_to_mesh_triposr)
    assert callable(pipe.smart_text_to_mesh_triposr)


def test_triposr_smart_validates_prompt() -> None:
    from meshmind import MeshMindPipeline

    pipe = MeshMindPipeline()
    with pytest.raises(ValueError, match="non-empty prompt"):
        pipe.smart_text_to_mesh_triposr(prompt="   ")


def test_vendored_tsr_imports() -> None:
    """The vendored ``meshmind._tsr`` package must import cleanly without
    requiring the optional ``torchmcubes`` extension."""
    tsr_system = importlib.import_module("meshmind._tsr.system")
    assert hasattr(tsr_system, "TSR")

    iso = importlib.import_module("meshmind._tsr.models.isosurface")
    assert hasattr(iso, "MarchingCubeHelper")

    utils = importlib.import_module("meshmind._tsr.utils")
    # find_class must rewrite the upstream ``tsr.<...>`` prefix.
    cls = utils.find_class("tsr.models.network_utils.TriplaneUpsampleNetwork")
    assert cls.__module__ == "meshmind._tsr.models.network_utils"


def test_marching_cube_helper_extracts_sphere() -> None:
    """Pure-Python skimage marching cubes must work on a synthetic sphere."""
    import numpy as np
    import torch

    from meshmind._tsr.models.isosurface import MarchingCubeHelper

    res = 32
    helper = MarchingCubeHelper(res)
    coords = np.linspace(-1.0, 1.0, res, dtype=np.float32)
    xs, ys, zs = np.meshgrid(coords, coords, coords, indexing="ij")
    distance = np.sqrt(xs**2 + ys**2 + zs**2)
    # ``forward`` expects ``-level`` so we negate to match the contract.
    level = torch.from_numpy((distance - 0.6).astype(np.float32))
    verts, faces = helper(-level.flatten())
    assert verts.shape[0] > 0
    assert faces.shape[0] > 0
    # Vertices should sit roughly on the unit cube parameterisation [0, 1].
    assert verts.min().item() >= -1e-3
    assert verts.max().item() <= 1.0 + 1e-3
