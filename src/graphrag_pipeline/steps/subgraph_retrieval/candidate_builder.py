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

    scored_entities: list[tuple[str, tuple[int, float, int]]] = []
    for entity in entities:
        names = [entity.canonical_name, *entity.aliases]
        best_score: tuple[int, float, int] | None = None
        for name in names:
            name_tokens = _important_tokens(name)
            overlap = question_tokens.intersection(name_tokens)
            if not overlap:
                continue

            # Long benchmark questions become noisy when we anchor on generic
            # single-word entities like country names. Keep these only for short
            # queries or richer multi-token entity names.
            if len(overlap) == 1 and len(question_tokens) >= 4 and len(name_tokens) == 1:
                continue

            score = (
                len(overlap),
                len(overlap) / max(len(name_tokens), 1),
                max(len(token) for token in overlap),
            )
            if best_score is None or score > best_score:
                best_score = score

        if best_score is not None:
            scored_entities.append((entity.entity_id, best_score))

    if not scored_entities:
        return set()

    best_overlap = max(score[0] for _, score in scored_entities)
    strongest = [item for item in scored_entities if item[1][0] == best_overlap]
    best_coverage = max(score[1] for _, score in strongest)
    coverage_margin = 0.0 if best_overlap == 1 else 0.25

    return {
        entity_id for entity_id, score in strongest if score[1] >= best_coverage - coverage_margin
    }


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

        def iter_out_edges(node_id: str):
            for neighbor_id, edge_map in graph.succ[node_id].items():
                for key, data in edge_map.items():
                    yield node_id, neighbor_id, key, data

        def iter_in_edges(node_id: str):
            for neighbor_id, edge_map in graph.pred[node_id].items():
                for key, data in edge_map.items():
                    yield neighbor_id, node_id, key, data

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
            for u, v, key, data in iter_out_edges(anchor_id):
                add_edge(u, v, key, data, hop=1)
            for u, v, key, data in iter_in_edges(anchor_id):
                add_edge(u, v, key, data, hop=1)

        if include_two_hop:
            one_hop_nodes = {str(c["head_id"]) for c in candidates} | {
                str(c["tail_id"]) for c in candidates
            }
            for node_id in one_hop_nodes:
                if node_id not in graph or node_id in active_anchor_ids:
                    continue
                for u, v, key, data in iter_out_edges(node_id):
                    add_edge(u, v, key, data, hop=2)
                for u, v, key, data in iter_in_edges(node_id):
                    add_edge(u, v, key, data, hop=2)

        return candidates

    candidates = collect_candidates(anchor_ids)

    if not candidates:
        remaining_entities = [entity for entity in entities if entity.entity_id not in anchor_ids]
        overlap_anchor_ids = _anchor_ids_from_token_overlap(question, remaining_entities)
        if overlap_anchor_ids:
            anchor_ids = anchor_ids.union(overlap_anchor_ids)
            candidates = collect_candidates(anchor_ids)

    return candidates, anchor_ids
