"""Answer generation for the answering step."""

from __future__ import annotations

from typing import cast

from graphrag_pipeline.llm.client import LLMClient
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

    user_prompt = (
        "Answer the question using only the provided evidence.\n"
        "If the evidence is insufficient, say so briefly.\n"
        "Cite evidence ids inline like [E1].\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence:\n{evidence_text}\n"
    )

    client = LLMClient()
    try:
        answer = client.complete(
            provider=provider,
            model=model,
            system_prompt=prompt,
            user_prompt=user_prompt,
        )
        print(f"[generate_answer] raw answer: {answer!r}")
    except Exception as e:
        print(f"[generate_answer] LLM error: {e}")
        return _abstained_result()

    if not answer or not answer.strip():
        print("[generate_answer] empty answer returned by LLM")
        return _abstained_result()

    return AnswerResult(answer=answer.strip(), reasoning=None, evidence_ids=evidence_ids)