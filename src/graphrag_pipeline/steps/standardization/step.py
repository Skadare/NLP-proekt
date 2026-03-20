"""Pipeline step placeholder for question standardization."""

from graphrag_pipeline.context import PipelineContext
from graphrag_pipeline.pipeline.base import PipelineStep


class StandardizationStep(PipelineStep):
    name = "standardization"

    def run(self, context: PipelineContext) -> PipelineContext:
        context.metadata.setdefault("steps", []).append(self.name)
        return context
