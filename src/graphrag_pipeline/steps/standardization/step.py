"""Pipeline step placeholder for question standardization."""

from graphrag_pipeline.context import PipelineContext
from graphrag_pipeline.pipeline.base import PipelineStep

from .aliases import load_alias_records, replace_aliases
from .normalize import normalize_question


class StandardizationStep(PipelineStep):
    name = "standardization"

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini") -> None:
        self.provider = provider
        self.model = model

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.raw_question is None or not context.raw_question.strip():
            raise ValueError("PipelineContext.raw_question is required for standardization step.")

        try:
            llm_normalized = normalize_question(
                context.raw_question,
                provider=self.provider,
                model=self.model,
            )
        except RuntimeError as exc:
            llm_normalized = context.raw_question
            context.metadata["standardization_warning"] = str(exc)

        alias_records = load_alias_records(context.kg_dir) if context.kg_dir is not None else []
        final_normalized, linked_entities = replace_aliases(llm_normalized, alias_records)

        context.normalized_question = final_normalized
        context.linked_entities = linked_entities
        context.metadata["llm_normalized_question"] = llm_normalized
        context.metadata["standardization_provider"] = self.provider
        context.metadata["standardization_model"] = self.model
        context.metadata.setdefault("steps", []).append(self.name)
        return context
