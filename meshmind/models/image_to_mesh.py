"""Image-to-mesh module backed by ShapE img2img."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import trimesh
    from PIL.Image import Image

    from meshmind.config import MeshMindConfig


class ImageToMesh:
    def __init__(self, config: MeshMindConfig) -> None:
        self.config = config
        self._pipe = None

    def _load(self):  # noqa: ANN202
        if self._pipe is not None:
            return self._pipe
        from diffusers import ShapEImg2ImgPipeline

        from meshmind.utils.perf import configure_threads, optimize_diffusers_pipe

        configure_threads()
        pipe = ShapEImg2ImgPipeline.from_pretrained(
            self.config.i2m_model_id,
            torch_dtype=self.config.resolve_dtype(),
            cache_dir=self.config.cache_dir,
        )
        pipe.set_progress_bar_config(disable=True)
        pipe = pipe.to(self.config.resolve_device())
        pipe = optimize_diffusers_pipe(pipe)
        self._pipe = pipe
        return pipe

    def generate(
        self,
        reference: Image,
        *,
        seed: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        resolution: int | None = None,
    ) -> trimesh.Trimesh:
        import torch

        from meshmind.utils.mesh_io import shape_e_output_to_trimesh

        pipe = self._load()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.config.resolve_device()).manual_seed(int(seed))

        prepared = reference.convert("RGB").resize((256, 256))
        result = pipe(
            image=prepared,
            num_inference_steps=steps or self.config.mesh_steps,
            guidance_scale=guidance if guidance is not None else max(self.config.mesh_guidance, 3.0),
            frame_size=resolution or self.config.mesh_resolution,
            output_type="mesh",
            generator=generator,
        )
        return shape_e_output_to_trimesh(result)
