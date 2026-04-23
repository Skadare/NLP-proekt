"""Subgraph retrieval utilities."""

from __future__ import annotations

import json
from pathlib import Path
import re

import networkx as nx

from graphrag_pipeline.types import (
    Entity,
    LinkedEntity,
    ProvenanceRecord,
    Relation,
    RetrievedFact,
    SubgraphResult,
    Triple,
)

from .candidate_builder import build_candidates
from .scorer import score_candidates


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required KG artifact is missing: {path}")
    records: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.strip().lower().split())


def _important_tokens(text: str) -> set[str]:
    normalized = _normalize_text(text)
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


def _dedupe_scored_candidates(
    scored: list[tuple[dict[str, object], float]],
) -> list[tuple[dict[str, object], float]]:
    deduped: list[tuple[dict[str, object], float]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for edge_data, score in scored:
        head_name = _normalize_text(str(edge_data.get("head_name") or ""))
        relation_name = _normalize_text(str(edge_data.get("relation_name") or ""))
        tail_name = _normalize_text(str(edge_data.get("tail_name") or ""))
        key = (head_name, relation_name, tail_name)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append((edge_data, score))
    return deduped


def load_kg_artifacts(
    kg_dir: str,
) -> tuple[list[Entity], list[Relation], list[Triple], list[ProvenanceRecord]]:
    root = Path(kg_dir)
    entities = [Entity.model_validate(payload) for payload in _load_jsonl(root / "entities.jsonl")]
    relations = [
        Relation.model_validate(payload) for payload in _load_jsonl(root / "relations.jsonl")
    ]
    triples = [Triple.model_validate(payload) for payload in _load_jsonl(root / "triples.jsonl")]
    provenance = [
        ProvenanceRecord.model_validate(payload)
        for payload in _load_jsonl(root / "provenance.jsonl")
    ]
    return entities, relations, triples, provenance


def build_kg_graph(
    entities: list[Entity],
    relations: list[Relation],
    triples: list[Triple],
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    relation_map = {rel.relation_id: rel.canonical_name for rel in relations}

    for entity in entities:
        graph.add_node(
            entity.entity_id,
            canonical_name=entity.canonical_name,
            aliases=entity.aliases,
            entity_type=entity.entity_type,
        )

    for triple in triples:
        graph.add_edge(
            triple.head_id,
            triple.tail_id,
            key=triple.triple_id,
            triple_id=triple.triple_id,
            relation_id=triple.relation_id,
            relation_name=relation_map.get(triple.relation_id, triple.relation_id),
            provenance_id=triple.provenance_id,
            confidence=triple.confidence,
        )

    return graph


def _build_subgraph(
    scored: list[tuple[dict[str, object], float]],
    entities: list[Entity],
    relations: list[Relation],
    *,
    top_k: int,
) -> SubgraphResult:
    if top_k <= 0:
        return SubgraphResult()

    entity_map = {entity.entity_id: entity.canonical_name for entity in entities}
    relation_map = {relation.relation_id: relation.canonical_name for relation in relations}

    facts: list[RetrievedFact] = []
    triple_ids: list[str] = []
    node_ids: list[str] = []
    seen_nodes: set[str] = set()

    for edge_data, score in scored[:top_k]:
        triple_id = str(edge_data["triple_id"])
        head_id = str(edge_data["head_id"])
        tail_id = str(edge_data["tail_id"])
        relation_id = str(edge_data["relation_id"])
        provenance_id = edge_data.get("provenance_id")

        facts.append(
            RetrievedFact(
                triple_id=triple_id,
                score=score,
                head=entity_map.get(head_id, head_id),
                relation=relation_map.get(relation_id, relation_id),
                tail=entity_map.get(tail_id, tail_id),
                provenance_id=str(provenance_id) if provenance_id is not None else None,
            )
        )
        triple_ids.append(triple_id)
        for node_id in (head_id, tail_id):
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                node_ids.append(node_id)

    return SubgraphResult(node_ids=node_ids, triple_ids=triple_ids, facts=facts)


def _snippet_preview(text: str | None, *, limit: int = 160) -> str:
    if not text:
        return ""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _score_passage(question: str, record: ProvenanceRecord) -> float:
    query_tokens = _important_tokens(question)
    if not query_tokens:
        return 0.0

    title = record.doc_id or ""
    snippet = record.snippet or ""
    haystack = f"{title} {snippet}".strip()
    passage_tokens = _important_tokens(haystack)
    if not passage_tokens:
        return 0.0

    overlap = query_tokens.intersection(passage_tokens)
    if not overlap:
        return 0.0

    score = float(len(overlap))
    score += len(overlap) / max(len(query_tokens), 1)

    normalized_question = _normalize_text(question)
    normalized_title = _normalize_text(title)
    normalized_snippet = _normalize_text(snippet)
    if normalized_title and normalized_title in normalized_question:
        score += 1.5
    for token in overlap:
        if token in normalized_title:
            score += 0.4
        if token in normalized_snippet:
            score += 0.1

    return score


def _build_passage_subgraph(
    question: str,
    provenance: list[ProvenanceRecord],
    *,
    top_k: int,
) -> SubgraphResult:
    scored_records: list[tuple[ProvenanceRecord, float]] = []
    for record in provenance:
        if not record.provenance_id or not record.passage_id:
            continue
        score = _score_passage(question, record)
        if score <= 0.0:
            continue
        scored_records.append((record, score))

    scored_records.sort(
        key=lambda item: (-item[1], item[0].passage_id or item[0].provenance_id or "")
    )
    if top_k > 0:
        scored_records = scored_records[:top_k]

    facts: list[RetrievedFact] = []
    triple_ids: list[str] = []
    node_ids: list[str] = []
    for index, (record, score) in enumerate(scored_records, start=1):
        passage_id = record.passage_id or record.provenance_id or f"passage_{index}"
        title = record.doc_id or passage_id
        snippet = _snippet_preview(record.snippet)
        fact = RetrievedFact(
            triple_id=f"passage::{passage_id}",
            score=score,
            head=title,
            relation="mentions",
            tail=snippet or title,
            provenance_id=record.provenance_id,
        )
        facts.append(fact)
        triple_ids.append(fact.triple_id)
        node_ids.append(title)

    return SubgraphResult(node_ids=node_ids, triple_ids=triple_ids, facts=facts)


def _merge_subgraphs(*subgraphs: SubgraphResult, top_k: int) -> SubgraphResult:
    fact_scores: dict[tuple[str, str | None], RetrievedFact] = {}
    for subgraph in subgraphs:
        for fact in subgraph.facts:
            key = (fact.triple_id, fact.provenance_id)
            existing = fact_scores.get(key)
            if existing is None or fact.score > existing.score:
                fact_scores[key] = fact

    ranked_facts = sorted(fact_scores.values(), key=lambda fact: (-fact.score, fact.triple_id))
    if top_k > 0:
        ranked_facts = ranked_facts[:top_k]

    node_ids: list[str] = []
    seen_nodes: set[str] = set()
    triple_ids: list[str] = []
    for fact in ranked_facts:
        triple_ids.append(fact.triple_id)
        for node in (fact.head, fact.tail):
            if node in seen_nodes:
                continue
            seen_nodes.add(node)
            node_ids.append(node)

    return SubgraphResult(node_ids=node_ids, triple_ids=triple_ids, facts=ranked_facts)


def retrieve_subgraph(
    question: str,
    linked_entities: list[LinkedEntity],
    entities: list[Entity],
    relations: list[Relation],
    triples: list[Triple],
    provenance: list[ProvenanceRecord],
    graph: nx.MultiDiGraph,
    *,
    top_k: int = 10,
    include_two_hop: bool = True,
) -> SubgraphResult:
    if not question.strip():
        raise ValueError("Question is empty; cannot retrieve subgraph.")
    _ = triples

    candidates, anchor_ids = build_candidates(
        question, linked_entities, entities, graph, include_two_hop=include_two_hop
    )
    graph_subgraph = SubgraphResult()
    if candidates and anchor_ids:
        scored = score_candidates(question, candidates, anchor_ids)
        if scored:
            scored = _dedupe_scored_candidates(scored)
            if scored:
                graph_subgraph = _build_subgraph(scored, entities, relations, top_k=top_k)

    passage_subgraph = _build_passage_subgraph(question, provenance, top_k=top_k)

    if not graph_subgraph.facts:
        return passage_subgraph
    if not passage_subgraph.facts:
        return graph_subgraph

    return _merge_subgraphs(graph_subgraph, passage_subgraph, top_k=top_k)
