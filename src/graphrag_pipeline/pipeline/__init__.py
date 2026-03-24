"""Pipeline primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import PipelineStep

if TYPE_CHECKING:
    from .runner import PipelineRunner

__all__ = ["PipelineRunner", "PipelineStep"]


def __getattr__(name: str):
    if name == "PipelineRunner":
        from .runner import PipelineRunner

        return PipelineRunner
    raise AttributeError(name)
