"""Pipeline runner for ordered execution of steps."""

from __future__ import annotations

from dataclasses import dataclass, field

from graphrag_pipeline.context import PipelineContext
from graphrag_pipeline.steps.answering.step import AnsweringStep
from graphrag_pipeline.steps.standardization.step import StandardizationStep
from graphrag_pipeline.steps.subgraph_retrieval.step import SubgraphRetrievalStep

from .base import PipelineStep


@dataclass
class PipelineRunner:
    steps: list[PipelineStep] = field(default_factory=list)

    def run(self, context: PipelineContext) -> PipelineContext:
        current = context
        for step in self.steps:
            current = step.run(current)
        return current

    @classmethod
    def default(cls) -> "PipelineRunner":
        return cls(
            steps=[
                StandardizationStep(),
                SubgraphRetrievalStep(),
                AnsweringStep(),
            ]
        )
