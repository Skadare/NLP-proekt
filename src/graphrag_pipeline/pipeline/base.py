"""Base step contracts for the pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

from graphrag_pipeline.context import PipelineContext


class PipelineStep(ABC):
    name: str

    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError
