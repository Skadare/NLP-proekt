"""Pipeline step for answer generation."""

from graphrag_pipeline.context import PipelineContext
from graphrag_pipeline.pipeline.base import PipelineStep
from graphrag_pipeline.steps.answering.answer import generate_answer
from graphrag_pipeline.steps.answering.reasoning import generate_reasoning


class AnsweringStep(PipelineStep):
    name = "answering"

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini") -> None:
        self.provider = provider
        self.model = model

    def run(self, context: PipelineContext) -> PipelineContext:
        question = context.raw_question or context.normalized_question
        if question is None:
            raise ValueError("Question is required for answer generation.")

        debug = bool(context.metadata.get("debug"))

        context.answer_result = generate_answer(
            question,
            context.subgraph,
            context.provenance,
            provider=self.provider,
            model=self.model,
            debug=debug,
        )
        context.answer_result.reasoning = generate_reasoning(
            question,
            context.subgraph,
            context.answer_result,
            context.provenance,
            provider=self.provider,
            model=self.model,
        )
        context.metadata.setdefault("steps", []).append(self.name)
        return context
