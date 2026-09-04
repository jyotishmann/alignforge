"""Data pipeline: load, validate, filter, dedup, decontaminate, format, write."""

from alignforge.data.build import build_dataset

__all__ = ["build_dataset"]
