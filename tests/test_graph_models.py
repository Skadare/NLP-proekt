from graphrag_pipeline.types import Entity


def test_entity_model_aliases_default_empty() -> None:
    entity = Entity(entity_id="e1", canonical_name="Azure")
    assert entity.aliases == []
