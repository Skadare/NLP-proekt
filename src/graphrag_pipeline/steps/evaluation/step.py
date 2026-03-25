"""Pipeline step for evaluation orchestration."""

from graphrag_pipeline.context import PipelineContext
from graphrag_pipeline.pipeline.base import PipelineStep


class EvaluationStep(PipelineStep):
    name = "evaluation"

    def run(self, context: PipelineContext) -> PipelineContext:
        context.metadata.setdefault("steps", []).append(self.name)
        context.metadata.setdefault("evaluation", {})
        return context
