from pathlib import Path
import json

from graphrag_pipeline.steps.kg_gen.command import run_command
from graphrag_pipeline.steps.kg_gen.extractor import KGExtractionArtifacts
from graphrag_pipeline.steps.kg_gen.mtrag_command import run_mtrag_command
from graphrag_pipeline.types import Entity, ProvenanceRecord, Relation, Triple


def test_kg_build_creates_artifact_files(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_text("Azure offers AKS.", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    fake_result = KGExtractionArtifacts(
        entities=[Entity(entity_id="ent_1", canonical_name="Azure", aliases=["Azure"])],
        relations=[Relation(relation_id="rel_1", canonical_name="offers", aliases=["offers"])],
        triples=[
            Triple(
                triple_id="trp_1",
                head_id="ent_1",
                relation_id="rel_1",
                tail_id="ent_2",
                provenance_id="prov_1",
            )
        ],
        provenance=[
            ProvenanceRecord(
                provenance_id="prov_1",
                source_path=str(input_path),
                doc_id=str(input_path),
                passage_id="chunk_1",
                extractor="kg-gen",
            )
        ],
        aliases=[
            {
                "alias": "Azure",
                "entity_id": "ent_1",
                "canonical_name": "Azure",
            }
        ],
        metadata={"model": "openai/gpt-4o-mini"},
    )

    def fake_extract(*args, **kwargs):  # type: ignore[no-untyped-def]
        return fake_result

    monkeypatch.setattr(
        "graphrag_pipeline.steps.kg_gen.command.extract_graph_from_text", fake_extract
    )

    summary = run_command(
        input_path=str(input_path),
        kg_root=str(tmp_path / "kg_root"),
        kg_name="sample",
    )

    output_dir_value = summary["output_dir"]
    assert isinstance(output_dir_value, str)
    output_dir = Path(output_dir_value)
    assert output_dir.exists()
    assert (output_dir / "entities.jsonl").exists()
    assert (output_dir / "relations.jsonl").exists()
    assert (output_dir / "triples.jsonl").exists()
    assert (output_dir / "aliases.jsonl").exists()
    assert (output_dir / "provenance.jsonl").exists()
    assert (output_dir / "metadata.json").exists()


def test_kg_build_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.txt"
    try:
        run_command(input_path=str(missing_path), kg_root=str(tmp_path))
    except FileNotFoundError:
        assert True
        return

    assert False, "Expected FileNotFoundError for missing input file."


def test_kg_build_loads_api_key_from_dotenv(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_text("Azure offers AKS.", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=dotenv-key\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    fake_result = KGExtractionArtifacts(
        entities=[],
        relations=[],
        triples=[],
        provenance=[],
        aliases=[],
        metadata={"model": "openai/gpt-4o-mini"},
    )

    def fake_extract(*args, **kwargs):  # type: ignore[no-untyped-def]
        return fake_result

    monkeypatch.setattr(
        "graphrag_pipeline.steps.kg_gen.command.extract_graph_from_text", fake_extract
    )

    summary = run_command(
        input_path=str(input_path),
        kg_root=str(tmp_path / "kg_root"),
        kg_name="from_dotenv",
    )

    output_dir_value = summary["output_dir"]
    assert isinstance(output_dir_value, str)
    assert Path(output_dir_value).exists()


def test_mtrag_build_resume_skips_completed_collection(tmp_path: Path, monkeypatch) -> None:
    input_file = tmp_path / "tasks.jsonl"
    input_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": "t1",
                        "Collection": "mt-rag-clapnq-elser-512-100-20240503",
                        "contexts": [{"document_id": "doc_1", "text": "alpha text"}],
                    }
                ),
                json.dumps(
                    {
                        "task_id": "t2",
                        "Collection": "mt-rag-govt-elser-512-100-20240611",
                        "contexts": [{"document_id": "doc_2", "text": "beta text"}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    clapnq_dir = tmp_path / "out" / "clapnq"
    clapnq_dir.mkdir(parents=True)
    completed_summary = {
        "output_dir": str(clapnq_dir),
        "passages_processed": 1,
        "collections": {"mt-rag-clapnq-elser-512-100-20240503": 1},
    }
    (clapnq_dir / "checkpoint.json").write_text(
        json.dumps({"status": "completed", "summary": completed_summary}),
        encoding="utf-8",
    )
    (tmp_path / "out" / "checkpoint.json").write_text(
        json.dumps({"completed_collections": ["clapnq"]}),
        encoding="utf-8",
    )

    fake_result = KGExtractionArtifacts(
        entities=[],
        relations=[],
        triples=[],
        provenance=[],
        aliases=[],
        metadata={"model": "openai/gpt-4o-mini"},
    )
    calls: list[str] = []

    def fake_extract(text: str, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(text)
        return fake_result

    monkeypatch.setattr(
        "graphrag_pipeline.steps.kg_gen.mtrag_command.extract_graph_from_text",
        fake_extract,
    )

    summary = run_mtrag_command(
        mtrag_root=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        input_file=str(input_file),
        source_mode="task-contexts",
        split_by_collection=True,
        collections=["clapnq", "govt"],
        max_passages_per_collection=1,
        resume=True,
    )

    assert len(calls) == 1
    assert calls[0] == "beta text"
    collections = summary["collections"]
    assert isinstance(collections, dict)
    assert "clapnq" in collections
    assert "govt" in collections


def test_mtrag_build_passage_corpus_requires_explicit_full_corpus_opt_in(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    try:
        run_mtrag_command(
            mtrag_root=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            source_mode="passage-corpus",
            max_passages=0,
            allow_full_corpus=False,
            collections=["clapnq"],
        )
    except ValueError as exc:
        assert "--allow-full-corpus" in str(exc)
        return

    assert False, "Expected ValueError when passage-corpus is uncapped without allow_full_corpus."


def test_mtrag_build_passage_corpus_allows_uncapped_with_explicit_opt_in(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    fake_result = KGExtractionArtifacts(
        entities=[],
        relations=[],
        triples=[],
        provenance=[],
        aliases=[],
        metadata={"model": "openai/gpt-4o-mini"},
    )

    def fake_extract(*args, **kwargs):  # type: ignore[no-untyped-def]
        return fake_result

    monkeypatch.setattr(
        "graphrag_pipeline.steps.kg_gen.mtrag_command.extract_graph_from_text",
        fake_extract,
    )

    monkeypatch.setattr(
        "graphrag_pipeline.steps.kg_gen.mtrag_command._iter_unique_passages_from_corpus",
        lambda **kwargs: [
            {
                "document_id": "doc_1",
                "text": "alpha text",
                "collection": "mt-rag-clapnq-elser-512-100-20240503",
                "source": "",
                "url": "",
            }
        ],
    )

    summary = run_mtrag_command(
        mtrag_root=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        source_mode="passage-corpus",
        max_passages=0,
        allow_full_corpus=True,
        collections=["clapnq"],
    )

    processed = summary.get("passages_processed")
    assert processed == 1
