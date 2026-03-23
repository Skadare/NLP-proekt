"""kg-gen integration for text-to-KG extraction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha1
import importlib.util
import sys
import types

from graphrag_pipeline.types import Entity, ProvenanceRecord, Relation, Triple


@dataclass
class KGExtractionArtifacts:
    entities: list[Entity]
    relations: list[Relation]
    triples: list[Triple]
    provenance: list[ProvenanceRecord]
    aliases: list[dict[str, str]]
    metadata: dict[str, object]


def _ensure_sentence_transformers_available() -> None:
    """Provide a lightweight stub when sentence-transformers is unavailable.

    kg-gen imports `SentenceTransformer` at module import time, even when only
    extraction APIs are used. We provide a local fallback stub so extraction can
    run without installing heavy retrieval dependencies.
    """

    try:
        if importlib.util.find_spec("sentence_transformers") is not None:
            return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("sentence_transformers")

    class SentenceTransformer:  # pragma: no cover - defensive runtime guard
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install it only if you plan to use kg-gen retrieval helpers."
            )

        def encode(self, *args: object, **kwargs: object) -> list[float]:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install it only if you plan to use kg-gen retrieval helpers."
            )

    setattr(module, "SentenceTransformer", SentenceTransformer)
    sys.modules["sentence_transformers"] = module


def _ensure_sklearn_pairwise_available() -> None:
    """Provide a lightweight sklearn cosine_similarity stub if sklearn is absent."""

    try:
        if importlib.util.find_spec("sklearn.metrics.pairwise") is not None:
            return
    except ModuleNotFoundError:
        pass

    sklearn_module = types.ModuleType("sklearn")
    metrics_module = types.ModuleType("sklearn.metrics")
    pairwise_module = types.ModuleType("sklearn.metrics.pairwise")

    def cosine_similarity(x: object, y: object) -> object:
        import numpy as np

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(1, -1)

        x_norm = np.linalg.norm(x_arr, axis=1, keepdims=True)
        y_norm = np.linalg.norm(y_arr, axis=1, keepdims=True)
        denom = x_norm @ y_norm.T
        denom[denom == 0.0] = 1e-12
        return (x_arr @ y_arr.T) / denom

    setattr(pairwise_module, "cosine_similarity", cosine_similarity)
    setattr(metrics_module, "pairwise", pairwise_module)
    setattr(sklearn_module, "metrics", metrics_module)

    sys.modules.setdefault("sklearn", sklearn_module)
    sys.modules.setdefault("sklearn.metrics", metrics_module)
    sys.modules.setdefault("sklearn.metrics.pairwise", pairwise_module)


def _stable_id(prefix: str, value: str) -> str:
    digest = sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _to_model_name(provider: str, model: str) -> str:
    if "/" in model:
        return model
    return f"{provider}/{model}"


def _as_attr_or_key(obj: object, name: str) -> object | None:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def _iter_triples(raw_relations: object | None) -> list[tuple[str, str, str]]:
    if raw_relations is None:
        return []
    if not isinstance(raw_relations, Iterable):
        return []

    triples: list[tuple[str, str, str]] = []
    for item in raw_relations:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            head, relation, tail = item
            triples.append((str(head), str(relation), str(tail)))
    return triples


def extract_graph_from_text(
    text: str,
    *,
    source_path: str,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    chunk_size: int = 5000,
    cluster: bool = True,
) -> KGExtractionArtifacts:
    """Extract a KG from text using kg-gen and convert it to internal models."""
    if not text.strip():
        raise ValueError("Input text is empty; cannot build KG.")

    _ensure_sentence_transformers_available()
    _ensure_sklearn_pairwise_available()

    try:
        from kg_gen import KGGen
    except ImportError as exc:
        raise RuntimeError(
            "kg-gen import failed. Ensure `kg-gen` is installed in this environment "
            "(run `pip install -e '.[dev]'`)."
        ) from exc

    model_name = _to_model_name(provider=provider, model=model)
    generator = KGGen(model=model_name, temperature=0.0)
    raw_graph = generator.generate(input_data=text, chunk_size=chunk_size, cluster=cluster)

    raw_entities = _as_attr_or_key(raw_graph, "entities") or []
    raw_edges = _as_attr_or_key(raw_graph, "edges") or []
    raw_relations = _as_attr_or_key(raw_graph, "relations") or []
    raw_entity_clusters = _as_attr_or_key(raw_graph, "entity_clusters") or {}
    raw_edge_clusters = _as_attr_or_key(raw_graph, "edge_clusters") or {}

    entity_alias_to_canonical: dict[str, str] = {}
    if not isinstance(raw_entity_clusters, dict):
        raw_entity_clusters = {}
    for canonical, alias_values in raw_entity_clusters.items():
        canonical_name = str(canonical)
        entity_alias_to_canonical[canonical_name] = canonical_name
        if isinstance(alias_values, Iterable):
            for alias in alias_values:
                entity_alias_to_canonical[str(alias)] = canonical_name

    if not isinstance(raw_entities, Iterable):
        raw_entities = []
    for entity in raw_entities:
        name = str(entity)
        entity_alias_to_canonical.setdefault(name, name)

    relation_alias_to_canonical: dict[str, str] = {}
    if not isinstance(raw_edge_clusters, dict):
        raw_edge_clusters = {}
    for canonical, alias_values in raw_edge_clusters.items():
        canonical_name = str(canonical)
        relation_alias_to_canonical[canonical_name] = canonical_name
        if isinstance(alias_values, Iterable):
            for alias in alias_values:
                relation_alias_to_canonical[str(alias)] = canonical_name

    if not isinstance(raw_edges, Iterable):
        raw_edges = []
    for edge in raw_edges:
        name = str(edge)
        relation_alias_to_canonical.setdefault(name, name)

    entities_by_name: dict[str, Entity] = {}
    aliases: list[dict[str, str]] = []
    for alias, canonical_name in entity_alias_to_canonical.items():
        entity = entities_by_name.get(canonical_name)
        if entity is None:
            entity = Entity(
                entity_id=_stable_id("ent", canonical_name.lower()),
                canonical_name=canonical_name,
                aliases=[],
            )
            entities_by_name[canonical_name] = entity
        if alias not in entity.aliases:
            entity.aliases.append(alias)
        aliases.append(
            {
                "alias": alias,
                "entity_id": entity.entity_id,
                "canonical_name": canonical_name,
            }
        )

    relations_by_name: dict[str, Relation] = {}
    for alias, canonical_name in relation_alias_to_canonical.items():
        relation = relations_by_name.get(canonical_name)
        if relation is None:
            relation = Relation(
                relation_id=_stable_id("rel", canonical_name.lower()),
                canonical_name=canonical_name,
                aliases=[],
            )
            relations_by_name[canonical_name] = relation
        if alias not in relation.aliases:
            relation.aliases.append(alias)

    triples: list[Triple] = []
    provenance: list[ProvenanceRecord] = []
    for index, (raw_head, raw_relation, raw_tail) in enumerate(
        _iter_triples(raw_relations), start=1
    ):
        head_canonical = entity_alias_to_canonical.get(raw_head, raw_head)
        tail_canonical = entity_alias_to_canonical.get(raw_tail, raw_tail)
        relation_canonical = relation_alias_to_canonical.get(raw_relation, raw_relation)

        if head_canonical not in entities_by_name:
            entities_by_name[head_canonical] = Entity(
                entity_id=_stable_id("ent", head_canonical.lower()),
                canonical_name=head_canonical,
                aliases=[head_canonical],
            )
        if tail_canonical not in entities_by_name:
            entities_by_name[tail_canonical] = Entity(
                entity_id=_stable_id("ent", tail_canonical.lower()),
                canonical_name=tail_canonical,
                aliases=[tail_canonical],
            )
        if relation_canonical not in relations_by_name:
            relations_by_name[relation_canonical] = Relation(
                relation_id=_stable_id("rel", relation_canonical.lower()),
                canonical_name=relation_canonical,
                aliases=[relation_canonical],
            )

        provenance_id = f"prov_{index:06d}"
        provenance.append(
            ProvenanceRecord(
                provenance_id=provenance_id,
                source_path=source_path,
                doc_id=source_path,
                passage_id=f"chunk_{index:06d}",
                snippet=None,
                extractor="kg-gen",
            )
        )

        head_id = entities_by_name[head_canonical].entity_id
        tail_id = entities_by_name[tail_canonical].entity_id
        relation_id = relations_by_name[relation_canonical].relation_id
        triple_key = f"{head_id}|{relation_id}|{tail_id}|{provenance_id}"
        triples.append(
            Triple(
                triple_id=_stable_id("trp", triple_key),
                head_id=head_id,
                relation_id=relation_id,
                tail_id=tail_id,
                provenance_id=provenance_id,
            )
        )

    return KGExtractionArtifacts(
        entities=list(entities_by_name.values()),
        relations=list(relations_by_name.values()),
        triples=triples,
        provenance=provenance,
        aliases=aliases,
        metadata={
            "provider": provider,
            "model": model_name,
            "chunk_size": chunk_size,
            "cluster": cluster,
            "source_path": source_path,
        },
    )
