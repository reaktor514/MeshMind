"""Text-to-image module wrapping a Stable Diffusion (Turbo) pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

    from meshmind.config import MeshMindConfig


class TextToImage:
    """Lazy wrapper around ``diffusers.AutoPipelineForText2Image``.

    The wrapper is intentionally thin so we keep our own contract (always
    return ``PIL.Image.Image`` lists, accept our config) while delegating the
    actual UNet/VAE math to the pretrained backbone.
    """

    def __init__(self, config: MeshMindConfig) -> None:
        self.config = config
        self._pipe = None

    def _load(self):  # noqa: ANN202
        if self._pipe is not None:
            return self._pipe
        from diffusers import AutoPipelineForText2Image

        pipe = AutoPipelineForText2Image.from_pretrained(
            self.config.t2i_model_id,
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
        *,
        negative_prompt: str | None = None,
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
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=steps or self.config.image_steps,
            guidance_scale=guidance if guidance is not None else self.config.image_guidance,
            height=side,
            width=side,
            generator=generator,
        )
        return list(result.images)
