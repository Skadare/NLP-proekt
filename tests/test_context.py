from graphrag_pipeline.context import PipelineContext


def test_context_defaults() -> None:
    context = PipelineContext()
    assert context.raw_question is None
    assert context.entities == []
    assert context.conversation_messages == []
