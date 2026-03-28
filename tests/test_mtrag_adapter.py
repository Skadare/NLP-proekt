from graphrag_pipeline.steps.evaluation.mtrag_adapter import (
    build_generation_record,
    build_retrieval_record,
)


def test_build_retrieval_record_sets_collection_fallback() -> None:
    task = {
        "task_id": "t1",
        "collection": "mt-rag-fiqa-beir-elser-512-100-20240501",
    }

    record = build_retrieval_record(task, contexts=[])

    assert record["Collection"] == "mt-rag-fiqa-beir-elser-512-100-20240501"
    assert record["contexts"] == []


def test_build_generation_record_adds_prediction_and_answerability_fallback() -> None:
    task = {
        "task_id": "t2",
        "Collection": "mt-rag-clapnq-elser-512-100-20240503",
        "Answerability": ["UNANSWERABLE"],
    }

    record = build_generation_record(task, contexts=[], answer_text="Insufficient evidence.")

    assert record["predictions"] == [{"text": "Insufficient evidence."}]
    assert record["answerability"] == ["UNANSWERABLE"]
