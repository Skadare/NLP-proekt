"""Pipeline step placeholder for KG generation."""

from graphrag_pipeline.context import PipelineContext
from graphrag_pipeline.pipeline.base import PipelineStep


class KGGenStep(PipelineStep):
    name = "kg_gen"

    def run(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError("KG generation step is not implemented yet.")
