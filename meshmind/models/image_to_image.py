"""Image-to-image module wrapping Stable Diffusion img2img.

Used for "reference image" 2D generation: take a user-supplied image and
re-paint it conditioned on a text prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

    from meshmind.config import MeshMindConfig


class ImageToImage:
    def __init__(self, config: MeshMindConfig) -> None:
        self.config = config
        self._pipe = None

    def _load(self):  # noqa: ANN202
        if self._pipe is not None:
            return self._pipe
        from diffusers import AutoPipelineForImage2Image

        pipe = AutoPipelineForImage2Image.from_pretrained(
            self.config.i2i_model_id,
            torch_dtype=self.config.resolve_dtype(),
            cache_dir=self.config.cache_dir,
        )
        pipe.set_progress_bar_config(disable=True)
        pipe = pipe.to(self.config.resolve_device())
        self._pipe = pipe
        return pipe

    def generate(
        self,
        prompt: str,
        reference: Image,
        *,
        negative_prompt: str | None = None,
        strength: float = 0.6,
        num_images: int = 1,
        seed: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        size: int | None = None,
    ) -> list[Image]:
        import torch

        pipe = self._load()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.config.resolve_device()).manual_seed(int(seed))

        side = size or self.config.image_size
        prepared = reference.convert("RGB").resize((side, side))
        result = pipe(
            prompt=prompt,
            image=prepared,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=steps or max(self.config.image_steps, 8),
            guidance_scale=guidance if guidance is not None else max(self.config.image_guidance, 1.5),
            strength=strength,
            generator=generator,
        )
        return list(result.images)
