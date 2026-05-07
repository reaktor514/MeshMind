"""Runtime configuration for MeshMind.

All defaults can be overridden via environment variables prefixed with ``MESHMIND_``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(f"MESHMIND_{name}", default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(f"MESHMIND_{name}")
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(f"MESHMIND_{name}")
    return float(raw) if raw else default


@dataclass
class MeshMindConfig:
    """Configuration container shared across all MeshMind submodules.

    Each weight identifier resolves to a public Hugging Face Hub repo so the
    pipeline works out of the box once weights are downloaded.
    """

    # 2D image generation backbones
    t2i_model_id: str = field(default_factory=lambda: _env("T2I_MODEL", "stabilityai/sd-turbo"))
    i2i_model_id: str = field(default_factory=lambda: _env("I2I_MODEL", "stabilityai/sd-turbo"))

    # 3D mesh generation backbones (ShapE family — outputs latent meshes directly)
    t2m_model_id: str = field(default_factory=lambda: _env("T2M_MODEL", "openai/shap-e"))
    i2m_model_id: str = field(default_factory=lambda: _env("I2M_MODEL", "openai/shap-e-img2img"))

    # Inference defaults
    image_size: int = field(default_factory=lambda: _env_int("IMAGE_SIZE", 512))
    image_steps: int = field(default_factory=lambda: _env_int("IMAGE_STEPS", 4))
    image_guidance: float = field(default_factory=lambda: _env_float("IMAGE_GUIDANCE", 0.0))

    mesh_steps: int = field(default_factory=lambda: _env_int("MESH_STEPS", 64))
    mesh_guidance: float = field(default_factory=lambda: _env_float("MESH_GUIDANCE", 15.0))
    mesh_resolution: int = field(default_factory=lambda: _env_int("MESH_RESOLUTION", 128))

    # Custom refiner (our own from-scratch architecture)
    refiner_dim: int = field(default_factory=lambda: _env_int("REFINER_DIM", 256))
    refiner_depth: int = field(default_factory=lambda: _env_int("REFINER_DEPTH", 4))
    refiner_heads: int = field(default_factory=lambda: _env_int("REFINER_HEADS", 4))
    refiner_strength: float = field(default_factory=lambda: _env_float("REFINER_STRENGTH", 0.0))

    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))
    dtype: str = field(default_factory=lambda: _env("DTYPE", "auto"))
    cache_dir: str | None = field(default_factory=lambda: os.environ.get("MESHMIND_CACHE_DIR"))

    def resolve_device(self) -> str:
        """Resolve ``device='auto'`` to ``cuda`` if available, otherwise ``cpu``."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def resolve_dtype(self):  # noqa: ANN201 — torch dtype is the natural return type
        """Resolve ``dtype='auto'`` to ``float16`` on CUDA, ``float32`` on CPU."""
        import torch

        if self.dtype == "float32":
            return torch.float32
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            return torch.bfloat16
        device = self.resolve_device()
        return torch.float16 if device == "cuda" else torch.float32
