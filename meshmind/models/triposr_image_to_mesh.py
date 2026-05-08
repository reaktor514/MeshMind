"""Image-to-mesh module backed by TripoSR (vendored under ``meshmind._tsr``).

TripoSR is a feed-forward (non-diffusion) Large Reconstruction Model for image
to 3D. Compared to ShapE-img2img it produces noticeably cleaner geometry on
complex/cinematic prompts because it does a direct regression instead of
running a noisy latent diffusion pass.

The class is intentionally lazy: constructing the wrapper does *not* download
weights, only the first ``generate(...)`` call does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import trimesh
    from PIL.Image import Image

    from meshmind.config import MeshMindConfig


class TripoSRImageToMesh:
    """Lazy wrapper around the vendored TripoSR ``TSR`` model."""

    def __init__(self, config: MeshMindConfig) -> None:
        self.config = config
        self._model = None
        self._device: str | None = None

    # ------------------------------------------------------------------ load
    def _load(self):  # noqa: ANN202
        if self._model is not None:
            return self._model

        from meshmind._tsr.system import TSR
        from meshmind.utils.perf import configure_threads

        configure_threads()
        model = TSR.from_pretrained(
            self.config.triposr_model_id,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        # ``set_chunk_size`` keeps memory usage in check on CPU.
        model.renderer.set_chunk_size(self.config.triposr_chunk_size)
        device = self.config.resolve_device()
        model.to(device)
        model.eval()
        self._model = model
        self._device = device
        return model

    # -------------------------------------------------------------- generate
    def generate(
        self,
        reference: Image,
        *,
        mc_resolution: int | None = None,
        foreground_ratio: float = 0.85,
        remove_bg: bool = False,
    ) -> trimesh.Trimesh:
        """Run TripoSR on a single reference image and return a ``trimesh.Trimesh``.

        ``remove_bg`` is opt-in because ``rembg`` is a heavy dep. If the input
        already has a clean alpha background, leave it disabled.
        """
        import numpy as np
        import torch
        from PIL import Image as PILImage

        model = self._load()
        device = self._device or self.config.resolve_device()

        prepared = reference.convert("RGB")
        if remove_bg:
            from meshmind._tsr.utils import (
                remove_background as _remove_background,
            )
            from meshmind._tsr.utils import (
                resize_foreground as _resize_foreground,
            )
            try:
                rgba = _remove_background(reference.convert("RGBA"))
                rgba = _resize_foreground(rgba, foreground_ratio)
                arr = np.array(rgba).astype(np.float32) / 255.0
                rgb = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
                prepared = PILImage.fromarray((rgb * 255.0).astype(np.uint8))
            except Exception:  # noqa: BLE001
                # ``rembg`` not installed or model download failed — fall back
                # to the raw reference. TripoSR will still work, just less crisp.
                prepared = reference.convert("RGB")

        with torch.no_grad():
            scene_codes = model([prepared], device=device)
            meshes = model.extract_mesh(
                scene_codes,
                has_vertex_color=True,
                resolution=int(mc_resolution or self.config.triposr_mc_resolution),
            )
        if not meshes:
            raise RuntimeError("TripoSR returned no meshes")
        return meshes[0]
