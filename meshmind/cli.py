"""CLI entry-points (``meshmind-app``)."""

from __future__ import annotations


def launch_app() -> None:
    """Launch the MeshMind Gradio web UI."""
    from app import build_app

    build_app().launch()
