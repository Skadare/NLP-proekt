"""KG artifact serializer."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from .extractor import KGExtractionArtifacts


def _to_jsonable(record: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, BaseModel):
        return record.model_dump()
    return record


def _write_jsonl(path: Path, records: Sequence[BaseModel | dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_to_jsonable(record), ensure_ascii=False))
            handle.write("\n")


def save_artifacts(output_dir: str, artifacts: KGExtractionArtifacts) -> dict[str, str]:
    """Persist extracted KG artifacts under the target directory."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    entities_path = root / "entities.jsonl"
    relations_path = root / "relations.jsonl"
    triples_path = root / "triples.jsonl"
    aliases_path = root / "aliases.jsonl"
    provenance_path = root / "provenance.jsonl"
    metadata_path = root / "metadata.json"

    _write_jsonl(entities_path, artifacts.entities)
    _write_jsonl(relations_path, artifacts.relations)
    _write_jsonl(triples_path, artifacts.triples)
    _write_jsonl(aliases_path, artifacts.aliases)
    _write_jsonl(provenance_path, artifacts.provenance)

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(artifacts.metadata, handle, ensure_ascii=False, indent=2)

    return {
        "output_dir": str(root),
        "entities": str(entities_path),
        "relations": str(relations_path),
        "triples": str(triples_path),
        "aliases": str(aliases_path),
        "provenance": str(provenance_path),
        "metadata": str(metadata_path),
    }
