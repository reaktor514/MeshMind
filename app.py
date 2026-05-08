"""Gradio web UI for MeshMind.

Run with::

    python app.py

The UI exposes three tabs:

* **2D Generation** — text + optional reference image -> PNG.
* **3D Generation** — text + optional reference image -> ``.obj`` / ``.glb``.
* **Smart 3D** — text -> 2D image -> 3D mesh (better for complex prompts).
"""

from __future__ import annotations

import os
import tempfile
import threading
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


def _warmup_pipeline_async() -> None:
    """Pre-load the SD-Turbo backbone in a background thread.

    Without this, the very first click on the Gradio site pays the model load
    cost (~5-10 s on warm cache, ~60 s cold).  We start that work as soon as
    the server boots so the first request lands on a hot pipeline.

    Disable with ``MESHMIND_NO_WARMUP=1`` (e.g. for unit tests / CI).
    """
    if os.environ.get("MESHMIND_NO_WARMUP", "").lower() in {"1", "true", "yes", "on"}:
        return

    def _run() -> None:
        try:
            pipe = _pipeline()
            pipe.t2i._load()  # pyright: ignore[reportPrivateUsage]
        except Exception:
            # Warmup failures must not crash the app — the on-click path will
            # surface any real loading error to the user.
            pass

    threading.Thread(target=_run, name="meshmind-warmup", daemon=True).start()


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


def _export_mesh(mesh, fmt: str) -> str:  # noqa: ANN001
    suffix = "." + fmt.lower()
    out = Path(tempfile.mkdtemp(prefix="meshmind_")) / f"meshmind_output{suffix}"
    mesh.export(out)
    return str(out)


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
    cleanup: bool,
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
        cleanup=bool(cleanup),
    )
    out = _export_mesh(mesh, fmt)
    return out, out


def smart_generate_mesh(
    prompt: str,
    negative_prompt: str,
    seed,  # noqa: ANN001
    image_steps,  # noqa: ANN001
    mesh_steps,  # noqa: ANN001
    mesh_guidance,  # noqa: ANN001
    resolution,  # noqa: ANN001
    fmt: str,
    cleanup: bool,
):  # noqa: ANN201
    if not prompt or not prompt.strip():
        raise gr.Error("Введи текстовый промпт")
    pipe = _pipeline()
    seed_val = _safe_int(seed)
    mesh, image = pipe.smart_text_to_mesh(
        prompt=prompt.strip(),
        negative_prompt=(negative_prompt or "").strip() or None,
        image_seed=seed_val,
        image_steps=_safe_int(image_steps),
        seed=seed_val,
        steps=_safe_int(mesh_steps),
        guidance=_safe_float(mesh_guidance),
        resolution=_safe_int(resolution),
        cleanup=bool(cleanup),
        return_image=True,
    )
    out = _export_mesh(mesh, fmt)
    return image, out, out


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
                    mesh_cleanup = gr.Checkbox(
                        label="Post-processing (выкинуть плавающие куски, сгладить, чинить нормали)",
                        value=True,
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
                    mesh_cleanup,
                ],
                outputs=[mesh_viewer, mesh_file],
            )

        with gr.Tab("Smart 3D (text → image → mesh)"):
            gr.Markdown(
                "ShapE плохо тянет сложные / OOD-промпты (фэнтези, монстры). "
                "Этот режим сначала генерирует 2D-картинку через SD-Turbo, "
                "а потом скармливает её ShapE-img2img — для всего, что сложнее «яблока», "
                "заметно чище."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    smart_prompt = gr.Textbox(
                        label="Промпт",
                        placeholder="a terrifying horror monster, sharp teeth, glowing eyes",
                        lines=2,
                    )
                    smart_negative = gr.Textbox(
                        label="Negative prompt",
                        placeholder="low quality, blurry, watermark, text",
                        lines=1,
                    )
                    with gr.Row():
                        smart_seed = gr.Number(label="Seed", value=None, precision=0)
                        smart_image_steps = gr.Number(label="Image steps", value=4, precision=0)
                        smart_mesh_steps = gr.Number(label="Mesh steps", value=64, precision=0)
                        smart_mesh_guidance = gr.Number(label="Mesh guidance", value=15.0)
                    smart_resolution = gr.Slider(
                        label="Resolution",
                        minimum=64,
                        maximum=256,
                        step=32,
                        value=128,
                    )
                    smart_format = gr.Radio(
                        label="Формат",
                        choices=["glb", "obj", "ply", "stl"],
                        value="glb",
                    )
                    smart_cleanup = gr.Checkbox(
                        label="Post-processing меша",
                        value=True,
                    )
                    smart_btn = gr.Button("Сгенерировать Smart 3D", variant="primary")
                with gr.Column(scale=1):
                    smart_image = gr.Image(label="Промежуточная 2D-картинка", type="pil")
                    smart_viewer = gr.Model3D(
                        label="Превью меша", clear_color=[0.07, 0.07, 0.09, 1]
                    )
                    smart_file = gr.File(label="Скачать")

            smart_btn.click(
                fn=smart_generate_mesh,
                inputs=[
                    smart_prompt,
                    smart_negative,
                    smart_seed,
                    smart_image_steps,
                    smart_mesh_steps,
                    smart_mesh_guidance,
                    smart_resolution,
                    smart_format,
                    smart_cleanup,
                ],
                outputs=[smart_image, smart_viewer, smart_file],
            )

        gr.Markdown(
            "_2D backbone: Stable Diffusion Turbo · "
            "3D backbone: ShapE · "
            "Custom: MeshMindRefiner (transformer latent refiner) + cleanup pipeline._"
        )

    return demo


if __name__ == "__main__":
    _warmup_pipeline_async()
    build_app().launch(server_name="0.0.0.0")
