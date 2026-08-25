# src/alignforge/__init__.py
"""AlignForge — LLM post-training pipeline (QLoRA SFT → DPO → eval → serve)."""

# Single source of truth for the version; read by hatchling at build time.
# Kept import-free so `import alignforge` stays instant.
__version__ = "0.1.0"

__all__ = ["__version__"]
