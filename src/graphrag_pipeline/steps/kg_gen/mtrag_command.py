"""CLI-facing entrypoint for MT-RAG aligned KG generation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import zipfile

from graphrag_pipeline.types import Entity, ProvenanceRecord, Relation, Triple
from graphrag_pipeline.utils.env import load_dotenv, required_key_for_provider

from .extractor import extract_graph_from_text
from .serializer import save_artifacts


COLLECTION_NAME_BY_STEM: dict[str, str] = {
    "clapnq": "mt-rag-clapnq-elser-512-100-20240503",
    "govt": "mt-rag-govt-elser-512-100-20240611",
    "fiqa": "mt-rag-fiqa-beir-elser-512-100-20240501",
    "cloud": "mt-rag-ibmcloud-elser-512-100-20240502",
}

STEM_BY_COLLECTION_NAME = {value: key for key, value in COLLECTION_NAME_BY_STEM.items()}


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _default_input_file(mtrag_root: str) -> Path:
    return Path(mtrag_root) / "mtrag-human" / "generation_tasks" / "RAG.jsonl"


def _default_corpus_dir(mtrag_root: str) -> Path:
    return Path(mtrag_root) / "corpora" / "passage_level"


def _extract_collection(task: dict[str, object]) -> str:
    collection = task.get("Collection")
    if isinstance(collection, str) and collection:
        return collection
    collection = task.get("collection")
    if isinstance(collection, str) and collection:
        return collection
    return "unknown"


def _iter_unique_passages_from_tasks(
    tasks: list[dict[str, object]],
    *,
    max_passages: int,
    allowed_collections: set[str] | None = None,
) -> list[dict[str, str]]:
    seen: set[str] = set()
    passages: list[dict[str, str]] = []

    for task in tasks:
        contexts = task.get("contexts", [])
        if not isinstance(contexts, list):
            continue

        collection = _extract_collection(task)
        if allowed_collections is not None and collection not in allowed_collections:
            continue
        for context in contexts:
            if not isinstance(context, dict):
                continue
            document_id = context.get("document_id")
            text = context.get("text")
            if not isinstance(document_id, str) or not document_id:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            if document_id in seen:
                continue

            title = context.get("title")
            source = context.get("source")
            url = context.get("url")
            passages.append(
                {
                    "document_id": document_id,
                    "text": text,
                    "title": title if isinstance(title, str) else "",
                    "source": source if isinstance(source, str) else "",
                    "url": url if isinstance(url, str) else "",
                    "collection": collection,
                }
            )
            seen.add(document_id)
            if max_passages > 0 and len(passages) >= max_passages:
                return passages

    return passages


def _collection_names_from_stems(stems: list[str]) -> set[str]:
    return {COLLECTION_NAME_BY_STEM[stem] for stem in stems}


def _resolve_collection_stems(collections: list[str] | None) -> list[str]:
    if not collections:
        return list(COLLECTION_NAME_BY_STEM.keys())

    stems: list[str] = []
    for item in collections:
        value = item.strip()
        if not value:
            continue
        lower_value = value.lower()
        if lower_value in COLLECTION_NAME_BY_STEM:
            stems.append(lower_value)
            continue
        mapped = STEM_BY_COLLECTION_NAME.get(value)
        if mapped is not None:
            stems.append(mapped)
            continue
        raise ValueError(
            f"Unsupported collection '{item}'. Use one of: "
            "clapnq, govt, fiqa, cloud, or full MT-RAG Collection names."
        )

    unique_stems: list[str] = []
    seen: set[str] = set()
    for stem in stems:
        if stem in seen:
            continue
        seen.add(stem)
        unique_stems.append(stem)
    return unique_stems


def _iter_unique_passages_from_corpus(
    *,
    corpus_dir: Path,
    collection_stems: list[str],
    max_passages: int,
) -> list[dict[str, str]]:
    passages: list[dict[str, str]] = []
    seen: set[str] = set()

    for stem in collection_stems:
        zip_path = corpus_dir / f"{stem}.jsonl.zip"
        if not zip_path.exists() or not zip_path.is_file():
            raise FileNotFoundError(f"Expected MT-RAG corpus zip missing: {zip_path}")

        with zipfile.ZipFile(zip_path) as archive:
            members = archive.namelist()
            if not members:
                continue
            with archive.open(members[0]) as handle:
                for raw_line in handle:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue

                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        continue

                    document_id = payload.get("id") or payload.get("_id")
                    text = payload.get("text")
                    if not isinstance(document_id, str) or not document_id:
                        continue
                    if not isinstance(text, str) or not text.strip():
                        continue
                    if document_id in seen:
                        continue

                    title = payload.get("title")
                    url = payload.get("url")
                    passages.append(
                        {
                            "document_id": document_id,
                            "text": text,
                            "title": title if isinstance(title, str) else "",
                            "source": "",
                            "url": url if isinstance(url, str) else "",
                            "collection": COLLECTION_NAME_BY_STEM[stem],
                        }
                    )
                    seen.add(document_id)
                    if max_passages > 0 and len(passages) >= max_passages:
                        return passages

    return passages


def run_mtrag_command(
    *,
    mtrag_root: str,
    output_dir: str,
    input_file: str | None = None,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    chunk_size: int = 5000,
    cluster: bool = False,
    max_tasks: int = 0,
    max_passages: int = 0,
    source_mode: str = "task-contexts",
    collections: list[str] | None = None,
    split_by_collection: bool = False,
    max_passages_per_collection: int = 0,
) -> dict[str, object]:
    """Build an MT-RAG-aligned KG using benchmark passage ids as provenance."""

    load_dotenv()

    key_name = required_key_for_provider(provider)
    if key_name is not None and not os.getenv(key_name):
        raise RuntimeError(
            f"Missing required API key for provider '{provider}'. "
            f"Set environment variable {key_name} (or add it to .env), "
            "or choose another provider."
        )

    tasks: list[dict[str, object]] = []
    input_path: Path | None = None
    corpus_dir: Path | None = None
    collection_stems = _resolve_collection_stems(collections)

    if split_by_collection:
        per_collection_limit = (
            max_passages_per_collection if max_passages_per_collection > 0 else max_passages
        )
        if source_mode == "passage-corpus" and per_collection_limit <= 0:
            raise ValueError(
                "split-by-collection with passage-corpus requires --max-passages "
                "or --max-passages-per-collection."
            )

        output_root = Path(output_dir)
        collection_summaries: dict[str, dict[str, object]] = {}
        total_passages = 0
        for stem in collection_stems:
            summary = run_mtrag_command(
                mtrag_root=mtrag_root,
                output_dir=str(output_root / stem),
                input_file=input_file,
                provider=provider,
                model=model,
                chunk_size=chunk_size,
                cluster=cluster,
                max_tasks=max_tasks,
                max_passages=per_collection_limit,
                source_mode=source_mode,
                collections=[stem],
                split_by_collection=False,
                max_passages_per_collection=0,
            )
            collection_summaries[stem] = summary

            processed = summary.get("passages_processed")
            if isinstance(processed, int):
                total_passages += processed

        return {
            "output_dir": output_dir,
            "source_mode": source_mode,
            "split_by_collection": True,
            "collections": collection_summaries,
            "total_passages_processed": total_passages,
        }

    allowed_collections = _collection_names_from_stems(collection_stems)

    if source_mode == "task-contexts":
        input_path = Path(input_file) if input_file else _default_input_file(mtrag_root)
        if not input_path.exists() or not input_path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        tasks = _load_jsonl(input_path)
        if max_tasks > 0:
            tasks = tasks[:max_tasks]
        passages = _iter_unique_passages_from_tasks(
            tasks,
            max_passages=max_passages,
            allowed_collections=allowed_collections,
        )
    elif source_mode == "passage-corpus":
        if max_passages <= 0:
            raise ValueError(
                "--max-passages is required for passage-corpus mode to avoid accidental full-corpus runs."
            )
        corpus_dir = _default_corpus_dir(mtrag_root)
        passages = _iter_unique_passages_from_corpus(
            corpus_dir=corpus_dir,
            collection_stems=collection_stems,
            max_passages=max_passages,
        )
    else:
        raise ValueError("source_mode must be either 'task-contexts' or 'passage-corpus'.")

    if not passages:
        raise ValueError(
            "No passages with document_id/text were found for the selected source mode."
        )

    entities_by_id: dict[str, Entity] = {}
    relations_by_id: dict[str, Relation] = {}
    triples_by_id: dict[str, Triple] = {}
    provenance_by_id: dict[str, ProvenanceRecord] = {}
    alias_pairs: set[tuple[str, str, str]] = set()
    aliases: list[dict[str, str]] = []
    collection_counts: dict[str, int] = {}

    for passage in passages:
        document_id = passage["document_id"]
        text = passage["text"]
        collection = passage["collection"]
        collection_counts[collection] = collection_counts.get(collection, 0) + 1

        try:
            artifacts = extract_graph_from_text(
                text,
                source_path=document_id,
                provider=provider,
                model=model,
                chunk_size=chunk_size,
                cluster=cluster,
            )
        except Exception as exc:  # pragma: no cover - defensive runtime mapping
            message = str(exc)
            if "api_key client option must be set" in message:
                raise RuntimeError(
                    f"Provider '{provider}' requires an API key. "
                    "Set the matching environment variable before running kg-build-mtrag."
                ) from exc
            raise

        local_provenance: dict[str, ProvenanceRecord] = {}
        for record in artifacts.provenance:
            rewritten = ProvenanceRecord(
                provenance_id=f"{document_id}::{record.provenance_id}",
                source_path=passage["source"] or passage["url"] or document_id,
                doc_id=document_id,
                passage_id=document_id,
                snippet=text,
                extractor=record.extractor,
                confidence=record.confidence,
            )
            provenance_by_id[rewritten.provenance_id] = rewritten
            local_provenance[record.provenance_id] = rewritten

        for entity in artifacts.entities:
            entities_by_id[entity.entity_id] = entity
        for relation in artifacts.relations:
            relations_by_id[relation.relation_id] = relation
        for triple in artifacts.triples:
            rewritten_provenance_id = None
            if triple.provenance_id is not None:
                mapped = local_provenance.get(triple.provenance_id)
                if mapped is not None:
                    rewritten_provenance_id = mapped.provenance_id

            triples_by_id[triple.triple_id] = Triple(
                triple_id=triple.triple_id,
                head_id=triple.head_id,
                relation_id=triple.relation_id,
                tail_id=triple.tail_id,
                provenance_id=rewritten_provenance_id,
                confidence=triple.confidence,
            )

        for alias in artifacts.aliases:
            alias_value = alias.get("alias")
            entity_id = alias.get("entity_id")
            canonical_name = alias.get("canonical_name")
            if not isinstance(alias_value, str):
                continue
            if not isinstance(entity_id, str):
                continue
            if not isinstance(canonical_name, str):
                continue
            alias_key = (alias_value, entity_id, canonical_name)
            if alias_key in alias_pairs:
                continue
            alias_pairs.add(alias_key)
            aliases.append(
                {
                    "alias": alias_value,
                    "entity_id": entity_id,
                    "canonical_name": canonical_name,
                }
            )

    from .extractor import KGExtractionArtifacts

    merged_artifacts = KGExtractionArtifacts(
        entities=list(entities_by_id.values()),
        relations=list(relations_by_id.values()),
        triples=list(triples_by_id.values()),
        provenance=list(provenance_by_id.values()),
        aliases=aliases,
        metadata={
            "dataset": "mt-rag-benchmark",
            "source_mode": source_mode,
            "input_file": str(input_path) if input_path is not None else None,
            "corpus_dir": str(corpus_dir) if corpus_dir is not None else None,
            "collections_requested": collection_stems,
            "provider": provider,
            "model": model,
            "chunk_size": chunk_size,
            "cluster": cluster,
            "tasks_processed": len(tasks),
            "passages_processed": len(passages),
            "collections": collection_counts,
        },
    )

    file_paths = save_artifacts(output_dir, merged_artifacts)
    return {
        "output_dir": output_dir,
        "entities": len(merged_artifacts.entities),
        "relations": len(merged_artifacts.relations),
        "triples": len(merged_artifacts.triples),
        "provenance": len(merged_artifacts.provenance),
        "passages_processed": len(passages),
        "collections": collection_counts,
        "artifacts": file_paths,
    }
