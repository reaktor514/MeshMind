"""Gradio web UI for MeshMind.

Run with::

    python app.py

The UI exposes two tabs:

* **2D Generation** — text + optional reference image -> PNG.
* **3D Generation** — text + optional reference image -> ``.obj`` / ``.glb``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr

from meshmind import MeshMindConfig, MeshMindPipeline

if TYPE_CHECKING:
    from PIL.Image import Image


_PIPE: MeshMindPipeline | None = None


def _pipeline() -> MeshMindPipeline:
    global _PIPE
    if _PIPE is None:
        _PIPE = MeshMindPipeline(MeshMindConfig())
    return _PIPE


def _safe_int(value, default=None):  # noqa: ANN001
    if value in (None, "", 0):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default=None):  # noqa: ANN001
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def generate_image(
    prompt: str,
    negative_prompt: str,
    reference: Image | None,
    seed,  # noqa: ANN001
    steps,  # noqa: ANN001
    guidance,  # noqa: ANN001
    num_images: int,
):  # noqa: ANN201
    if not prompt or not prompt.strip():
        raise gr.Error("Введи текстовый промпт")
    pipe = _pipeline()
    images = pipe.text_to_image(
        prompt=prompt.strip(),
        negative_prompt=negative_prompt.strip() or None,
        reference=reference,
        num_images=int(num_images),
        seed=_safe_int(seed),
        steps=_safe_int(steps),
        guidance=_safe_float(guidance),
    )
    return images


def generate_mesh(
    prompt: str,
    reference: Image | None,
    seed,  # noqa: ANN001
    steps,  # noqa: ANN001
    guidance,  # noqa: ANN001
    resolution,  # noqa: ANN001
    fmt: str,
):  # noqa: ANN201
    if reference is None and (not prompt or not prompt.strip()):
        raise gr.Error("Введи текстовый промпт или загрузи референс-картинку")

    pipe = _pipeline()
    mesh = pipe.generate_mesh(
        prompt=prompt.strip() if prompt else None,
        reference=reference,
        seed=_safe_int(seed),
        steps=_safe_int(steps),
        guidance=_safe_float(guidance),
        resolution=_safe_int(resolution),
    )

    suffix = "." + fmt.lower()
    out = Path(tempfile.mkdtemp(prefix="meshmind_")) / f"meshmind_output{suffix}"
    mesh.export(out)
    return str(out), str(out)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="MeshMind — 2D/3D Generative Pipeline", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# MeshMind\n"
            "Своя нейросеть для **2D / 3D** моделирования. "
            "Генерируй по тексту или по картинке-референсу."
        )

        with gr.Tab("2D генерация"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_prompt = gr.Textbox(
                        label="Промпт",
                        placeholder="a red cyber-dragon, studio render, highly detailed",
                        lines=2,
                    )
                    img_negative = gr.Textbox(
                        label="Negative prompt",
                        placeholder="low quality, blurry, watermark",
                        lines=1,
                    )
                    img_reference = gr.Image(
                        label="Референс (опционально)",
                        type="pil",
                        sources=["upload", "clipboard"],
                    )
                    with gr.Row():
                        img_seed = gr.Number(label="Seed", value=None, precision=0)
                        img_steps = gr.Number(label="Steps", value=None, precision=0)
                        img_guidance = gr.Number(label="Guidance", value=None)
                        img_count = gr.Slider(
                            label="Картинок", minimum=1, maximum=4, step=1, value=1
                        )
                    img_btn = gr.Button("Сгенерировать 2D", variant="primary")
                with gr.Column(scale=1):
                    img_gallery = gr.Gallery(label="Результаты", columns=2, height=512)

            img_btn.click(
                fn=generate_image,
                inputs=[
                    img_prompt,
                    img_negative,
                    img_reference,
                    img_seed,
                    img_steps,
                    img_guidance,
                    img_count,
                ],
                outputs=[img_gallery],
            )

        with gr.Tab("3D генерация"):
            with gr.Row():
                with gr.Column(scale=1):
                    mesh_prompt = gr.Textbox(
                        label="Промпт",
                        placeholder="a stylized low-poly fox",
                        lines=2,
                    )
                    mesh_reference = gr.Image(
                        label="Референс (опционально)",
                        type="pil",
                        sources=["upload", "clipboard"],
                    )
                    with gr.Row():
                        mesh_seed = gr.Number(label="Seed", value=None, precision=0)
                        mesh_steps = gr.Number(label="Steps", value=64, precision=0)
                        mesh_guidance = gr.Number(label="Guidance", value=15.0)
                        mesh_resolution = gr.Slider(
                            label="Resolution",
                            minimum=64,
                            maximum=256,
                            step=32,
                            value=128,
                        )
                    mesh_format = gr.Radio(
                        label="Формат",
                        choices=["glb", "obj", "ply", "stl"],
                        value="glb",
                    )
                    mesh_btn = gr.Button("Сгенерировать 3D", variant="primary")
                with gr.Column(scale=1):
                    mesh_viewer = gr.Model3D(label="Превью меша", clear_color=[0.07, 0.07, 0.09, 1])
                    mesh_file = gr.File(label="Скачать")

            mesh_btn.click(
                fn=generate_mesh,
                inputs=[
                    mesh_prompt,
                    mesh_reference,
                    mesh_seed,
                    mesh_steps,
                    mesh_guidance,
                    mesh_resolution,
                    mesh_format,
                ],
                outputs=[mesh_viewer, mesh_file],
            )

        gr.Markdown(
            "_2D backbone: Stable Diffusion Turbo · "
            "3D backbone: ShapE · "
            "Custom: MeshMindRefiner (transformer latent refiner)._"
        )

    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0")
