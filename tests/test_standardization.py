from pathlib import Path

from graphrag_pipeline.context import PipelineContext
from graphrag_pipeline.types import ConversationMessage
from graphrag_pipeline.steps.standardization.aliases import load_alias_records, replace_aliases
from graphrag_pipeline.steps.standardization.step import StandardizationStep


def test_replace_aliases_prefers_longest_match() -> None:
    records = [
        {"alias": "US", "entity_id": "ent_us", "canonical_name": "United States"},
        {
            "alias": "U.S.",
            "entity_id": "ent_us",
            "canonical_name": "United States",
        },
        {
            "alias": "United States",
            "entity_id": "ent_us",
            "canonical_name": "United States",
        },
    ]

    replaced, linked = replace_aliases("What is the capital of U.S.?", records)
    assert replaced == "What is the capital of United States?"
    assert len(linked) == 1
    assert linked[0].canonical_name == "United States"


def test_load_alias_records_reads_jsonl(tmp_path: Path) -> None:
    kg_dir = tmp_path / "kg"
    kg_dir.mkdir(parents=True)
    (kg_dir / "aliases.jsonl").write_text(
        """{"alias":"AKS","entity_id":"ent_1","canonical_name":"Azure Kubernetes Service"}\n""",
        encoding="utf-8",
    )

    records = load_alias_records(str(kg_dir))
    assert len(records) == 1
    assert records[0]["alias"] == "AKS"


def test_standardization_step_llm_then_alias(monkeypatch, tmp_path: Path) -> None:
    kg_dir = tmp_path / "kg"
    kg_dir.mkdir(parents=True)
    (kg_dir / "aliases.jsonl").write_text(
        """{"alias":"US","entity_id":"ent_us","canonical_name":"United States"}\n""",
        encoding="utf-8",
    )

    def fake_normalize(question: str, *, provider: str, model: str) -> str:
        assert question == "capital of us?"
        return "What is the capital of US?"

    monkeypatch.setattr(
        "graphrag_pipeline.steps.standardization.step.normalize_question", fake_normalize
    )

    context = PipelineContext(raw_question="capital of us?", kg_dir=str(kg_dir))
    result = StandardizationStep().run(context)

    assert result.metadata["llm_normalized_question"] == "What is the capital of US?"
    assert result.normalized_question == "What is the capital of United States?"
    assert len(result.linked_entities) == 1


def test_standardization_uses_full_conversation_when_available(monkeypatch) -> None:
    captured = {"question": ""}

    def fake_normalize(question: str, *, provider: str, model: str) -> str:
        captured["question"] = question
        return "Where do they play this week?"

    monkeypatch.setattr(
        "graphrag_pipeline.steps.standardization.step.normalize_question", fake_normalize
    )

    context = PipelineContext(
        raw_question="Where do they play this week?",
        conversation_messages=[
            ConversationMessage(speaker="user", text="Tell me about the Arizona Cardinals."),
            ConversationMessage(speaker="assistant", text="They are an NFL team."),
            ConversationMessage(speaker="user", text="Where do they play this week?"),
        ],
    )

    StandardizationStep().run(context)

    assert captured["question"] == (
        "|user|: Tell me about the Arizona Cardinals.\n"
        "|assistant|: They are an NFL team.\n"
        "|user|: Where do they play this week?"
    )
