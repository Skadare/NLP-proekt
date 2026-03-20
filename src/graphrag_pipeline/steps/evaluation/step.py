"""Pipeline step placeholder for evaluation."""

from graphrag_pipeline.context import PipelineContext
from graphrag_pipeline.pipeline.base import PipelineStep


class EvaluationStep(PipelineStep):
    name = "evaluation"

    def run(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError("Evaluation step is not implemented yet.")
