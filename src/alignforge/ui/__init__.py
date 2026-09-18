"""Gradio UI — a client over the API. Never imports train/eval/models."""

from alignforge.ui.app import create_ui

__all__ = ["create_ui"]
