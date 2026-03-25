"""Reasoning generation for the answering step."""

from __future__ import annotations

from typing import cast

from graphrag_pipeline.llm.client import LLMClient
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

    if not evidence_text:
        return "No evidence facts were available to justify the answer."

    user_prompt = (
        "Explain briefly why the answer is supported by the evidence.\n"
        "Cite evidence ids inline like [E1].\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer.answer}\n\n"
        f"Evidence:\n{evidence_text}\n"
    )

    client = LLMClient()
    try:
        return client.complete(
            provider=provider,
            model=model,
            system_prompt=prompt,
            user_prompt=user_prompt,
        )
    except Exception:
        return "Unable to generate reasoning due to LLM error."
