"""Formatting utilities for readable CLI responses."""

from __future__ import annotations

from .context import PipelineContext
from .types import StructuredResponse


def build_structured_response(context: PipelineContext) -> StructuredResponse:
    return StructuredResponse(
        sections=[
            {
                "name": "question",
                "raw_question": context.raw_question,
                "normalized_question": context.normalized_question,
            },
            {
                "name": "linked_entities",
                "items": [item.model_dump() for item in context.linked_entities],
            },
            {
                "name": "subgraph",
                "items": [fact.model_dump() for fact in context.subgraph.facts],
            },
            {
                "name": "answer",
                "answer": context.answer_result.answer,
                "reasoning": context.answer_result.reasoning,
            },
        ]
    )
