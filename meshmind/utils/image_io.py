"""PIL image helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image


def load_image(path: str | Path) -> Image:
    from PIL import Image as _Image

    return _Image.open(path).convert("RGB")


def save_image(image: Image, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    image.save(p)
    return p
