"""Reasoning generation for the answering step."""

from __future__ import annotations

from typing import cast

from graphrag_pipeline.llm.prompts import get_prompt
from graphrag_pipeline.steps.answering.formatter import format_evidence
from graphrag_pipeline.types import AnswerResult, ProvenanceRecord, SubgraphResult


def generate_reasoning(
    question: str,
    subgraph: SubgraphResult,
    answer: AnswerResult,
    provenance: list[ProvenanceRecord],
    *,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
) -> str:
    """Generate reasoning grounded in retrieved evidence.

    This implementation returns a deterministic grounded response until
    the LLM client is wired in.
    """
    if not question.strip():
        raise ValueError("Question is empty; cannot generate reasoning.")

    if answer.answer is None or not answer.answer.strip():
        return "No answer available to justify with evidence."

    if not subgraph.facts:
        return "No evidence facts were available to justify the answer."

    prompt = get_prompt("reasoning")
    formatted = format_evidence(subgraph, provenance)
    evidence_text = cast(str, formatted["evidence_text"])
    evidence_ids = cast(list[str], formatted["evidence_ids"])

    if not evidence_text:
        return "No evidence facts were available to justify the answer."

    _ = provider
    _ = model
    _ = prompt

    citations = " ".join(f"[{evidence_id}]" for evidence_id in evidence_ids)
    top_fact = subgraph.facts[0]
    return (
        f"The evidence shows {top_fact.head} {top_fact.relation} {top_fact.tail}. "
        f"This supports the answer: {answer.answer} {citations}"
    )
