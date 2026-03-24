"""Answer generation for the answering step."""

from __future__ import annotations

from typing import cast

from graphrag_pipeline.llm.prompts import get_prompt
from graphrag_pipeline.steps.answering.formatter import format_evidence
from graphrag_pipeline.types import AnswerResult, ProvenanceRecord, SubgraphResult


def _abstained_result() -> AnswerResult:
    return AnswerResult(
        answer="Insufficient evidence in retrieved subgraph.",
        reasoning=None,
        evidence_ids=[],
    )


def generate_answer(
    question: str,
    subgraph: SubgraphResult,
    provenance: list[ProvenanceRecord],
    *,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
) -> AnswerResult:
    """Generate an answer grounded in retrieved evidence.

    This implementation returns a deterministic grounded response until
    the LLM client is wired in.
    """
    if not question.strip():
        raise ValueError("Question is empty; cannot generate answer.")

    if not subgraph.facts:
        return _abstained_result()

    prompt = get_prompt("answer")
    formatted = format_evidence(subgraph, provenance)
    evidence_text = cast(str, formatted["evidence_text"])
    evidence_ids = cast(list[str], formatted["evidence_ids"])

    if not evidence_text:
        return _abstained_result()

    _ = provider
    _ = model
    _ = prompt

    top_fact = subgraph.facts[0]
    top_id = evidence_ids[0] if evidence_ids else "E1"
    answer = f"{top_fact.head} {top_fact.relation} {top_fact.tail}. [{top_id}]"
    return AnswerResult(answer=answer, reasoning=None, evidence_ids=evidence_ids)
