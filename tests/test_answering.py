import pytest

from graphrag_pipeline.context import PipelineContext
from graphrag_pipeline.steps.answering.answer import generate_answer
from graphrag_pipeline.steps.answering.step import AnsweringStep
from graphrag_pipeline.types import AnswerResult, RetrievedFact, SubgraphResult


def test_answering_step_requires_question() -> None:
    context = PipelineContext()
    with pytest.raises(ValueError):
        AnsweringStep().run(context)


def test_generate_answer_abstains_on_empty_subgraph() -> None:
    result = generate_answer(
        "What is GraphRAG?",
        SubgraphResult(),
        provenance=[],
    )
    assert result.answer == "Insufficient evidence in retrieved subgraph."
    assert result.reasoning is None
    assert result.evidence_ids == []


def test_generate_answer_with_single_fact_is_grounded() -> None:
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
    result = generate_answer("What does Azure offer?", subgraph, provenance=[])
    assert result.answer is not None and result.answer.strip()
    assert "Azure offers AKS" in result.answer
    assert "[E1]" in result.answer
    assert result.evidence_ids == ["E1"]


def test_answering_step_prefers_raw_question(monkeypatch) -> None:
    captured = {}

    def fake_generate_answer(question, subgraph, provenance, *, provider, model):
        captured["question"] = question
        return AnswerResult(answer="ok", reasoning="ok", evidence_ids=[])

    monkeypatch.setattr(
        "graphrag_pipeline.steps.answering.step.generate_answer",
        fake_generate_answer,
    )

    context = PipelineContext(
        raw_question="raw question",
        normalized_question="normalized question",
    )
    AnsweringStep().run(context)
    assert captured["question"] == "raw question"


def test_generate_answer_rejects_blank_question() -> None:
    with pytest.raises(ValueError):
        generate_answer("  ", SubgraphResult(), provenance=[])


def test_answering_step_appends_metadata_steps(monkeypatch) -> None:
    def fake_generate_answer(question, subgraph, provenance, *, provider, model):
        return AnswerResult(answer="ok", reasoning="ok", evidence_ids=[])

    monkeypatch.setattr(
        "graphrag_pipeline.steps.answering.step.generate_answer",
        fake_generate_answer,
    )

    context = PipelineContext(raw_question="What is GraphRAG?")
    AnsweringStep().run(context)
    assert context.metadata["steps"][-1] == "answering"
