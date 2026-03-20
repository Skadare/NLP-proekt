"""Filter helpers placeholder for future pipeline composition."""

from graphrag_pipeline.context import PipelineContext


def passthrough(context: PipelineContext) -> PipelineContext:
    return context
