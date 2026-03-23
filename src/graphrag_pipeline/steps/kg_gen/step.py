"""Pipeline step placeholder for KG generation."""

from graphrag_pipeline.context import PipelineContext
from graphrag_pipeline.pipeline.base import PipelineStep

from .command import run_command


class KGGenStep(PipelineStep):
    name = "kg_gen"

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.input_path is None:
            raise ValueError("PipelineContext.input_path is required for kg_gen step.")

        summary = run_command(
            context.input_path,
            kg_root=context.kg_dir or "data/kg",
            provider="openai",
            model="gpt-4o-mini",
        )
        context.kg_dir = str(summary["output_dir"])
        context.metadata.setdefault("kg_build", summary)
        artifacts = summary.get("artifacts")
        if isinstance(artifacts, dict):
            context.artifacts.update(artifacts)
        return context
