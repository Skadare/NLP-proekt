"""Candidate graph builder."""

from __future__ import annotations

import re

import networkx as nx

from graphrag_pipeline.types import Entity, LinkedEntity


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "many",
    "of",
    "on",
    "or",
    "the",
    "to",
    "use",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


def _normalize_match_text(text: str) -> str:
    cleaned = text.lower()
    cleaned = cleaned.replace("\u2019", "'")
    cleaned = cleaned.replace("'", "")
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _important_tokens(text: str) -> set[str]:
    normalized = _normalize_match_text(text)
    if not normalized:
        return set()

    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    expanded: set[str] = set()
    for token in tokens:
        if len(token) <= 2 or token in _STOPWORDS:
            continue
        expanded.add(token)
        if token.endswith("s") and len(token) > 3:
            expanded.add(token[:-1])
    return expanded


def _match_phrase(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False

    pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", flags=re.IGNORECASE)
    if pattern.search(text) is not None:
        return True

    normalized_text = _normalize_match_text(text)
    normalized_phrase = _normalize_match_text(phrase)
    if not normalized_text or not normalized_phrase:
        return False

    normalized_pattern = re.compile(
        rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)",
        flags=re.IGNORECASE,
    )
    return normalized_pattern.search(normalized_text) is not None


def _anchor_ids_from_text(question: str, entities: list[Entity]) -> set[str]:
    anchor_ids: set[str] = set()
    for entity in entities:
        if _match_phrase(question, entity.canonical_name):
            anchor_ids.add(entity.entity_id)
            continue
        for alias in entity.aliases:
            if _match_phrase(question, alias):
                anchor_ids.add(entity.entity_id)
                break
    return anchor_ids


def _anchor_ids_from_token_overlap(question: str, entities: list[Entity]) -> set[str]:
    question_tokens = _important_tokens(question)
    if not question_tokens:
        return set()

    anchor_ids: set[str] = set()
    for entity in entities:
        names = [entity.canonical_name, *entity.aliases]
        for name in names:
            overlap = question_tokens.intersection(_important_tokens(name))
            if overlap:
                anchor_ids.add(entity.entity_id)
                break

    return anchor_ids


def build_candidates(
    question: str,
    linked_entities: list[LinkedEntity],
    entities: list[Entity],
    graph: nx.MultiDiGraph,
    *,
    include_two_hop: bool = True,
) -> tuple[list[dict[str, object]], set[str]]:
    if not question.strip():
        raise ValueError("Question is empty; cannot build subgraph candidates.")

    anchor_ids = {linked.entity_id for linked in linked_entities}
    if not anchor_ids:
        anchor_ids = _anchor_ids_from_text(question, entities)
    if not anchor_ids:
        anchor_ids = _anchor_ids_from_token_overlap(question, entities)

    if not anchor_ids:
        return [], set()

    def collect_candidates(active_anchor_ids: set[str]) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        seen_triples: set[str] = set()

        def add_edge(u: str, v: str, key: str, data: dict[str, object], hop: int) -> None:
            triple_id = str(data.get("triple_id", key))
            if triple_id in seen_triples:
                return
            seen_triples.add(triple_id)

            head_name = ""
            tail_name = ""
            if graph.has_node(u):
                head_name = str(graph.nodes[u].get("canonical_name") or "")
            if graph.has_node(v):
                tail_name = str(graph.nodes[v].get("canonical_name") or "")

            candidates.append(
                {
                    "head_id": u,
                    "tail_id": v,
                    "triple_id": triple_id,
                    "relation_id": data.get("relation_id"),
                    "relation_name": data.get("relation_name"),
                    "provenance_id": data.get("provenance_id"),
                    "confidence": data.get("confidence"),
                    "hop": hop,
                    "head_name": head_name,
                    "tail_name": tail_name,
                }
            )

        for anchor_id in active_anchor_ids:
            if anchor_id not in graph:
                continue
            for u, v, key, data in graph.out_edges(anchor_id, keys=True, data=True):
                add_edge(u, v, key, data, hop=1)
            for u, v, key, data in graph.in_edges(anchor_id, keys=True, data=True):
                add_edge(u, v, key, data, hop=1)

        if include_two_hop:
            one_hop_nodes = {c["head_id"] for c in candidates} | {c["tail_id"] for c in candidates}
            for node_id in one_hop_nodes:
                if node_id not in graph or node_id in active_anchor_ids:
                    continue
                for u, v, key, data in graph.out_edges(node_id, keys=True, data=True):
                    add_edge(u, v, key, data, hop=2)
                for u, v, key, data in graph.in_edges(node_id, keys=True, data=True):
                    add_edge(u, v, key, data, hop=2)

        return candidates

    candidates = collect_candidates(anchor_ids)

    if not candidates:
        overlap_anchor_ids = _anchor_ids_from_token_overlap(question, entities)
        if overlap_anchor_ids:
            anchor_ids = anchor_ids.union(overlap_anchor_ids)
            candidates = collect_candidates(anchor_ids)

    return candidates, anchor_ids
