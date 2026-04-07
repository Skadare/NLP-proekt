import json
from pathlib import Path

from graphrag_pipeline.steps.subgraph_retrieval.step import SubgraphRetrievalStep
from graphrag_pipeline.steps.evaluation.runner import (
    _build_runner,
    _clean_retrieval_text,
    _merge_contexts,
    _filter_tasks_by_ids,
    _load_mtrag_retrieval_tasks,
    _resolve_kg_dir_for_collection,
    _sample_tasks,
)


def _task(task_id: str, collection: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "Collection": collection,
        "input": [{"speaker": "user", "text": "q"}],
    }


def test_stratified_sampling_balances_collections() -> None:
    tasks = [
        _task("a1", "col_a"),
        _task("a2", "col_a"),
        _task("a3", "col_a"),
        _task("b1", "col_b"),
        _task("b2", "col_b"),
        _task("b3", "col_b"),
    ]

    sampled = _sample_tasks(tasks, max_tasks=4, sample_mode="stratified", seed=7)

    assert len(sampled) == 4
    collection_counts = {"col_a": 0, "col_b": 0}
    for task in sampled:
        collection = task.get("Collection")
        if isinstance(collection, str):
            collection_counts[collection] += 1

    assert collection_counts["col_a"] == 2
    assert collection_counts["col_b"] == 2


def test_filter_tasks_by_ids_preserves_requested_order() -> None:
    tasks = [_task("t1", "col_a"), _task("t2", "col_a"), _task("t3", "col_b")]

    filtered = _filter_tasks_by_ids(tasks, ["t3", "t1"])

    assert [item["task_id"] for item in filtered] == ["t3", "t1"]


def test_stratified_sampling_spreads_conversations() -> None:
    tasks = [
        {
            "task_id": "a::1",
            "conversation_id": "conv_a",
            "Collection": "col_a",
            "input": [{"speaker": "user", "text": "q"}],
        },
        {
            "task_id": "a::2",
            "conversation_id": "conv_a",
            "Collection": "col_a",
            "input": [{"speaker": "user", "text": "q"}],
        },
        {
            "task_id": "b::1",
            "conversation_id": "conv_b",
            "Collection": "col_a",
            "input": [{"speaker": "user", "text": "q"}],
        },
    ]

    sampled = _sample_tasks(tasks, max_tasks=2, sample_mode="stratified", seed=7)

    assert len(sampled) == 2
    conversation_ids = {item["conversation_id"] for item in sampled}
    assert conversation_ids == {"conv_a", "conv_b"}


def test_resolve_kg_dir_uses_collection_subdir(tmp_path: Path) -> None:
    root = tmp_path / "kg"
    clapnq_dir = root / "clapnq"
    clapnq_dir.mkdir(parents=True)
    (clapnq_dir / "entities.jsonl").write_text("", encoding="utf-8")

    resolved = _resolve_kg_dir_for_collection(
        str(root),
        "mt-rag-clapnq-elser-512-100-20240503",
    )

    assert resolved == str(clapnq_dir)


def test_merge_contexts_prefers_higher_score_and_respects_top_k() -> None:
    merged = _merge_contexts(
        [
            {"document_id": "d1", "score": 1.0, "text": "a"},
            {"document_id": "d2", "score": 0.3, "text": "b"},
        ],
        [
            {"document_id": "d1", "score": 2.0, "text": "better"},
            {"document_id": "d3", "score": 0.5, "text": "c"},
        ],
        top_k=2,
    )

    assert [item["document_id"] for item in merged] == ["d1", "d3"]
    first_score = merged[0]["score"]
    second_score = merged[1]["score"]
    assert isinstance(first_score, (int, float))
    assert isinstance(second_score, (int, float))
    assert first_score > second_score


def test_build_runner_can_disable_two_hop_retrieval() -> None:
    runner = _build_runner(
        include_answering=False,
        provider="openai",
        model="gpt-4o-mini",
        include_two_hop=False,
    )

    retrieval_step = runner.steps[1]
    assert isinstance(retrieval_step, SubgraphRetrievalStep)
    assert retrieval_step.include_two_hop is False


def test_clean_retrieval_text_removes_user_prefixes() -> None:
    raw = "|user|: First question\n|user|: Follow up"

    cleaned = _clean_retrieval_text(raw)

    assert cleaned == "First question\nFollow up"


def test_load_mtrag_retrieval_tasks_builds_records(tmp_path: Path) -> None:
    retrieval_root = tmp_path / "mtrag-human" / "retrieval_tasks"
    for stem in ["clapnq", "cloud", "fiqa", "govt"]:
        stem_dir = retrieval_root / stem
        stem_dir.mkdir(parents=True)
        payload = {"_id": f"{stem}<::>2", "text": "|user|: Sample question"}
        (stem_dir / f"{stem}_rewrite.jsonl").write_text(
            json.dumps(payload) + "\n",
            encoding="utf-8",
        )

    tasks = _load_mtrag_retrieval_tasks(mtrag_root=tmp_path, mode="rewrite")

    assert len(tasks) == 4
    task_ids = {task["task_id"] for task in tasks}
    assert task_ids == {
        "clapnq<::>2",
        "cloud<::>2",
        "fiqa<::>2",
        "govt<::>2",
    }
    for task in tasks:
        assert isinstance(task.get("Collection"), str)
        input_messages = task.get("input")
        assert isinstance(input_messages, list)
        assert input_messages[0]["text"] == "Sample question"
