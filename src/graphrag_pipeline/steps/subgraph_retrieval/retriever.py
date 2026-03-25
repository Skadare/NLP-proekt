"""Subgraph retrieval utilities."""

from __future__ import annotations

import json
from pathlib import Path

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
    entities = [Entity(**payload) for payload in _load_jsonl(root / "entities.jsonl")]
    relations = [Relation(**payload) for payload in _load_jsonl(root / "relations.jsonl")]
    triples = [Triple(**payload) for payload in _load_jsonl(root / "triples.jsonl")]
    provenance = [ProvenanceRecord(**payload) for payload in _load_jsonl(root / "provenance.jsonl")]
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
    _ = provenance
    _ = triples

    candidates, anchor_ids = build_candidates(
        question, linked_entities, entities, graph, include_two_hop=include_two_hop
    )
    if not candidates or not anchor_ids:
        return SubgraphResult()

    scored = score_candidates(question, candidates, anchor_ids)
    if not scored:
        return SubgraphResult()

    scored = _dedupe_scored_candidates(scored)
    if not scored:
        return SubgraphResult()

    return _build_subgraph(scored, entities, relations, top_k=top_k)
