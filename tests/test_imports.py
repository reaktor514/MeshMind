"""Smoke tests: ``import meshmind`` should not crash and submodules should be reachable."""

from __future__ import annotations


def test_import_package() -> None:
    import meshmind

    assert hasattr(meshmind, "MeshMindPipeline")
    assert hasattr(meshmind, "MeshMindConfig")
    assert meshmind.__version__


def test_import_submodules() -> None:
    from meshmind import MeshMindConfig, MeshMindPipeline
    from meshmind.models.image_to_image import ImageToImage
    from meshmind.models.image_to_mesh import ImageToMesh
    from meshmind.models.refiner import MeshMindRefiner
    from meshmind.models.text_to_image import TextToImage
    from meshmind.models.text_to_mesh import TextToMesh
    from meshmind.utils.mesh_io import save_mesh, shape_e_output_to_trimesh
    from meshmind.utils.seeding import seed_everything

    assert MeshMindConfig().image_size == 512
    pipe = MeshMindPipeline()
    assert pipe.config.t2i_model_id
    assert callable(seed_everything)
    assert callable(save_mesh)
    assert callable(shape_e_output_to_trimesh)
    assert TextToImage is not None
    assert ImageToImage is not None
    assert TextToMesh is not None
    assert ImageToMesh is not None
    assert MeshMindRefiner is not None


def test_refiner_forward_image_latents() -> None:
    import torch

    from meshmind.models.refiner import MeshMindRefiner

    refiner = MeshMindRefiner(in_channels=4, dim=64, depth=2, heads=4, ctx_dim=128)
    latents = torch.randn(1, 4, 8, 8)
    context = torch.randn(1, 16, 128)

    out = refiner(latents, context, strength=1.0)
    assert out.shape == latents.shape

    # strength=0 must be a true no-op (identity passthrough).
    same = refiner(latents, context, strength=0.0)
    assert torch.equal(same, latents)


def test_refiner_forward_token_latents() -> None:
    import torch

    from meshmind.models.refiner import MeshMindRefiner

    refiner = MeshMindRefiner(in_channels=8, dim=32, depth=2, heads=4, ctx_dim=64)
    tokens = torch.randn(2, 12, 8)
    out = refiner(tokens, context=None, strength=0.5)
    assert out.shape == tokens.shape


def test_refiner_zero_init_delta() -> None:
    import torch

    from meshmind.models.refiner import MeshMindRefiner

    # With zero-initialised proj_out the residual delta starts at zero, so a
    # freshly-constructed refiner is an exact identity even at strength=1.
    refiner = MeshMindRefiner(in_channels=4, dim=64, depth=2, heads=4, ctx_dim=128)
    latents = torch.randn(1, 4, 8, 8)
    out = refiner(latents, context=None, strength=1.0)
    assert torch.allclose(out, latents, atol=1e-6)
