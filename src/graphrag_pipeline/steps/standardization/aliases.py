"""Alias loading and replacement utilities."""

from __future__ import annotations

import json
from pathlib import Path
import re

from graphrag_pipeline.types import LinkedEntity


def load_alias_records(kg_dir: str) -> list[dict[str, str]]:
    """Load alias records from a built KG directory."""
    aliases_path = Path(kg_dir) / "aliases.jsonl"
    if not aliases_path.exists() or not aliases_path.is_file():
        return []

    records: list[dict[str, str]] = []
    for raw_line in aliases_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            continue
        alias = payload.get("alias")
        entity_id = payload.get("entity_id")
        canonical_name = payload.get("canonical_name")
        if (
            isinstance(alias, str)
            and isinstance(entity_id, str)
            and isinstance(canonical_name, str)
        ):
            records.append(
                {
                    "alias": alias,
                    "entity_id": entity_id,
                    "canonical_name": canonical_name,
                }
            )
    return records


def replace_aliases(
    question: str,
    alias_records: list[dict[str, str]],
) -> tuple[str, list[LinkedEntity]]:
    """Replace entity aliases with canonical names in a question string."""
    if not question.strip() or not alias_records:
        return question, []

    replaced_question = question
    linked: dict[tuple[str, str], LinkedEntity] = {}

    unique_records: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    for record in alias_records:
        alias = record["alias"]
        canonical_name = record["canonical_name"]
        entity_id = record["entity_id"]
        key = (alias.lower(), canonical_name.lower(), entity_id)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        unique_records.append(record)

    unique_records.sort(key=lambda item: len(item["alias"]), reverse=True)

    for record in unique_records:
        alias = record["alias"]
        canonical_name = record["canonical_name"]
        entity_id = record["entity_id"]

        pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", flags=re.IGNORECASE)
        if pattern.search(replaced_question) is None:
            continue

        replaced_question = pattern.sub(canonical_name, replaced_question)
        mention_key = (alias.lower(), entity_id)
        linked[mention_key] = LinkedEntity(
            mention=alias,
            entity_id=entity_id,
            canonical_name=canonical_name,
            score=1.0,
        )

    return replaced_question, list(linked.values())
