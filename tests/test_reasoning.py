import pytest

from graphrag_pipeline.steps.answering.reasoning import generate_reasoning
from graphrag_pipeline.types import AnswerResult, ProvenanceRecord, RetrievedFact, SubgraphResult


def test_generate_reasoning_is_grounded() -> None:
    subgraph = SubgraphResult(
        facts=[
            RetrievedFact(
                triple_id="trp_1",
                score=0.9,
                head="Azure",
                relation="offers",
                tail="AKS",
                provenance_id="prov_1",
            )
        ]
    )
    provenance = [
        ProvenanceRecord(
            provenance_id="prov_1",
            doc_id="doc_1",
            passage_id="p1",
            snippet="Azure offers AKS.",
        )
    ]
    answer = AnswerResult(answer="Azure offers AKS.", evidence_ids=["E1"])

    reasoning = generate_reasoning(
        "What does Azure offer?",
        subgraph,
        answer,
        provenance,
    )

    assert "Azure offers AKS" in reasoning
    assert "[E1]" in reasoning


def test_generate_reasoning_rejects_blank_question() -> None:
    with pytest.raises(ValueError):
        generate_reasoning(
            " ",
            SubgraphResult(),
            AnswerResult(answer="ok"),
            provenance=[],
        )
