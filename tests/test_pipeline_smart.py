"""Tests for ``MeshMindPipeline.smart_text_to_mesh`` argument validation and
cleanup-options plumbing.  Backbone calls are stubbed so this stays a unit
test (no SD-Turbo / ShapE downloads required).
"""

from __future__ import annotations

import pytest


def test_smart_text_to_mesh_requires_prompt() -> None:
    from meshmind import MeshMindPipeline

    pipe = MeshMindPipeline()
    with pytest.raises(ValueError):
        pipe.smart_text_to_mesh("")
    with pytest.raises(ValueError):
        pipe.smart_text_to_mesh("   ")


def test_generate_mesh_requires_inputs() -> None:
    from meshmind import MeshMindPipeline

    pipe = MeshMindPipeline()
    with pytest.raises(ValueError):
        pipe.generate_mesh(prompt=None, reference=None)


def test_build_cleanup_options_uses_config() -> None:
    from meshmind import MeshMindConfig, MeshMindPipeline

    cfg = MeshMindConfig()
    cfg.cleanup_smooth_iters = 7
    cfg.cleanup_smooth_lamb = 0.42
    cfg.cleanup_fill_holes = True
    cfg.cleanup_target_faces = 1234

    pipe = MeshMindPipeline(cfg)
    opts = pipe._build_cleanup_options()

    assert opts.smooth_iters == 7
    assert opts.smooth_lamb == 0.42
    assert opts.fill_holes is True
    assert opts.target_faces == 1234


def test_smart_text_to_mesh_routes_through_t2i_then_i2m(monkeypatch) -> None:  # noqa: ANN001
    """Stub backbones and verify the smart pipeline calls SD-Turbo then
    ShapE-img2img with the same prompt and the i2m route, not the t2m route.
    """
    import trimesh
    from PIL import Image

    from meshmind import MeshMindPipeline
    from meshmind.utils import mesh_io

    pipe = MeshMindPipeline()
    calls: dict[str, dict] = {}

    fake_image = Image.new("RGB", (32, 32), color=(123, 45, 67))
    fake_mesh = trimesh.creation.box(extents=(1, 1, 1))

    class _StubT2I:
        def generate(self, **kwargs):  # noqa: ANN001, ANN201, ANN003
            calls["t2i"] = kwargs
            return [fake_image]

    class _StubI2M:
        def generate(self, **kwargs):  # noqa: ANN001, ANN201, ANN003
            calls["i2m"] = kwargs
            return fake_mesh

    class _StubT2M:
        def generate(self, **kwargs):  # noqa: ANN001, ANN201, ANN003
            calls["t2m"] = kwargs
            return fake_mesh

    pipe._t2i = _StubT2I()  # type: ignore[assignment]
    pipe._i2m = _StubI2M()  # type: ignore[assignment]
    pipe._t2m = _StubT2M()  # type: ignore[assignment]

    monkeypatch.setattr(mesh_io, "clean_mesh", lambda m, _opts=None: m)

    out_mesh, out_image = pipe.smart_text_to_mesh(
        "a scary monster",
        negative_prompt="blurry",
        seed=7,
        steps=20,
        guidance=12.0,
        resolution=96,
        cleanup=True,
        return_image=True,
    )

    assert "t2i" in calls and calls["t2i"]["prompt"] == "a scary monster"
    assert calls["t2i"]["negative_prompt"] == "blurry"
    assert calls["t2i"]["seed"] == 7
    assert "i2m" in calls and calls["i2m"]["reference"] is fake_image
    assert calls["i2m"]["seed"] == 7
    assert calls["i2m"]["steps"] == 20
    assert calls["i2m"]["guidance"] == 12.0
    assert calls["i2m"]["resolution"] == 96
    assert "t2m" not in calls  # smart path skips text-conditioned ShapE
    assert out_mesh is fake_mesh
    assert out_image is fake_image
