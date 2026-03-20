from graphrag_pipeline.context import PipelineContext
from graphrag_pipeline.pipeline.runner import PipelineRunner


def test_default_runner_preserves_question() -> None:
    context = PipelineContext(raw_question="What is GraphRAG?")
    result = PipelineRunner.default().run(context)
    assert result.raw_question == "What is GraphRAG?"
