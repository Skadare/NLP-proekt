"""Pipeline step placeholder for subgraph retrieval."""

from graphrag_pipeline.context import PipelineContext
from graphrag_pipeline.pipeline.base import PipelineStep


class SubgraphRetrievalStep(PipelineStep):
    name = "subgraph_retrieval"

    def run(self, context: PipelineContext) -> PipelineContext:
        context.metadata.setdefault("steps", []).append(self.name)
        return context
