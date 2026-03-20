"""Pipeline step placeholder for answer generation."""

from graphrag_pipeline.context import PipelineContext
from graphrag_pipeline.pipeline.base import PipelineStep


class AnsweringStep(PipelineStep):
    name = "answering"

    def run(self, context: PipelineContext) -> PipelineContext:
        context.metadata.setdefault("steps", []).append(self.name)
        return context
