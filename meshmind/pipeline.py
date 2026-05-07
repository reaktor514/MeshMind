"""High-level orchestration of MeshMind."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from meshmind.config import MeshMindConfig

if TYPE_CHECKING:
    import trimesh
    from PIL.Image import Image

    from meshmind.models.image_to_image import ImageToImage
    from meshmind.models.image_to_mesh import ImageToMesh
    from meshmind.models.refiner import MeshMindRefiner
    from meshmind.models.text_to_image import TextToImage
    from meshmind.models.text_to_mesh import TextToMesh


@dataclass
class GenerationResult:
    """Container returned by high-level pipeline methods."""

    images: list[Image]
    mesh: trimesh.Trimesh | None = None
    saved_paths: list[Path] | None = None


class MeshMindPipeline:
    """Top-level MeshMind orchestration.

    The pipeline composes four pretrained backbones (text->image, image->image,
    text->mesh, image->mesh) with our own :class:`MeshMindRefiner` to produce
    2D images and 3D meshes from text prompts and/or reference images.

    Submodules are loaded lazily, so constructing the pipeline is cheap.
    """

    def __init__(self, config: MeshMindConfig | None = None) -> None:
        self.config = config or MeshMindConfig()
        self._t2i: TextToImage | None = None
        self._i2i: ImageToImage | None = None
        self._t2m: TextToMesh | None = None
        self._i2m: ImageToMesh | None = None
        self._refiner: MeshMindRefiner | None = None

    @classmethod
    def from_pretrained(cls, **overrides) -> MeshMindPipeline:  # noqa: ANN003
        return cls(MeshMindConfig(**overrides))

    @property
    def t2i(self) -> TextToImage:
        if self._t2i is None:
            from meshmind.models.text_to_image import TextToImage

            self._t2i = TextToImage(self.config)
        return self._t2i

    @property
    def i2i(self) -> ImageToImage:
        if self._i2i is None:
            from meshmind.models.image_to_image import ImageToImage

            self._i2i = ImageToImage(self.config)
        return self._i2i

    @property
    def t2m(self) -> TextToMesh:
        if self._t2m is None:
            from meshmind.models.text_to_mesh import TextToMesh

            self._t2m = TextToMesh(self.config)
        return self._t2m

    @property
    def i2m(self) -> ImageToMesh:
        if self._i2m is None:
            from meshmind.models.image_to_mesh import ImageToMesh

            self._i2m = ImageToMesh(self.config)
        return self._i2m

    @property
    def refiner(self) -> MeshMindRefiner:
        if self._refiner is None:
            from meshmind.models.refiner import MeshMindRefiner

            self._refiner = MeshMindRefiner(
                in_channels=4,
                dim=self.config.refiner_dim,
                depth=self.config.refiner_depth,
                heads=self.config.refiner_heads,
            )
        return self._refiner

    # ------------------------------------------------------------------
    # 2D
    # ------------------------------------------------------------------
    def text_to_image(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        reference: Image | None = None,
        num_images: int = 1,
        seed: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
    ) -> list[Image]:
        """Generate 2D images from text and (optionally) a reference image."""
        if reference is not None:
            return self.i2i.generate(
                prompt=prompt,
                reference=reference,
                negative_prompt=negative_prompt,
                num_images=num_images,
                seed=seed,
                steps=steps,
                guidance=guidance,
            )
        return self.t2i.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            seed=seed,
            steps=steps,
            guidance=guidance,
        )

    # ------------------------------------------------------------------
    # 3D
    # ------------------------------------------------------------------
    def text_to_mesh(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        resolution: int | None = None,
    ) -> trimesh.Trimesh:
        return self.t2m.generate(
            prompt=prompt,
            seed=seed,
            steps=steps,
            guidance=guidance,
            resolution=resolution,
        )

    def image_to_mesh(
        self,
        reference: Image,
        *,
        seed: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        resolution: int | None = None,
    ) -> trimesh.Trimesh:
        return self.i2m.generate(
            reference=reference,
            seed=seed,
            steps=steps,
            guidance=guidance,
            resolution=resolution,
        )

    def generate_mesh(
        self,
        prompt: str | None = None,
        reference: Image | None = None,
        *,
        seed: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        resolution: int | None = None,
    ) -> trimesh.Trimesh:
        """Convenience: pick text- or image-conditioned 3D based on inputs."""
        if reference is None and not prompt:
            raise ValueError("generate_mesh requires either a prompt or a reference image")
        if reference is not None:
            return self.image_to_mesh(
                reference=reference,
                seed=seed,
                steps=steps,
                guidance=guidance,
                resolution=resolution,
            )
        assert prompt is not None
        return self.text_to_mesh(
            prompt=prompt,
            seed=seed,
            steps=steps,
            guidance=guidance,
            resolution=resolution,
        )
