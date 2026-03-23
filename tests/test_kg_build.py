from pathlib import Path

from graphrag_pipeline.steps.kg_gen.command import run_command
from graphrag_pipeline.steps.kg_gen.extractor import KGExtractionArtifacts
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
