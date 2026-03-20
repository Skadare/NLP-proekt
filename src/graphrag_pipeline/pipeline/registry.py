"""Step registry placeholder."""

from graphrag_pipeline.pipeline.base import PipelineStep


STEP_REGISTRY: dict[str, type[PipelineStep]] = {}
