"""MTRAG adapter helpers for evaluation formats."""

from __future__ import annotations

from dataclasses import dataclass

from graphrag_pipeline.types import ProvenanceRecord, SubgraphResult, Triple


@dataclass
class PassageEvidence:
    passage_id: str
    score: float
    text: str | None = None
    title: str | None = None
    url: str | None = None


def _build_provenance_map(
    provenance: list[ProvenanceRecord],
) -> dict[str, ProvenanceRecord]:
    return {record.provenance_id: record for record in provenance}


def _build_triple_map(triples: list[Triple]) -> dict[str, str]:
    return {
        triple.triple_id: triple.provenance_id
        for triple in triples
        if triple.provenance_id is not None
    }


def map_subgraph_to_contexts(
    subgraph: SubgraphResult,
    provenance: list[ProvenanceRecord],
    triples: list[Triple],
    *,
    top_k: int = 10,
) -> list[dict[str, object]]:
    """Map KG subgraph evidence to MTRAG contexts.

    Assumption (course-project level):
    - Each triple links to a ProvenanceRecord via provenance_id.
    - ProvenanceRecord provides passage_id or doc_id.
    - ProvenanceRecord.snippet holds passage text (best-effort for generation eval).
    """

    provenance_map = _build_provenance_map(provenance)
    triple_map = _build_triple_map(triples)

    aggregated: dict[str, PassageEvidence] = {}

    for fact in subgraph.facts:
        provenance_id = fact.provenance_id or triple_map.get(fact.triple_id)
        if provenance_id is None:
            continue
        record = provenance_map.get(provenance_id)
        if record is None:
            continue

        passage_id = record.passage_id or record.doc_id or record.source_path
        if passage_id is None:
            continue

        entry = aggregated.get(passage_id)
        if entry is None:
            entry = PassageEvidence(
                passage_id=passage_id,
                score=0.0,
                text=record.snippet,
                title=None,
                url=record.source_path,
            )
            aggregated[passage_id] = entry

        entry.score += float(fact.score)
        if entry.text is None and record.snippet:
            entry.text = record.snippet

    ranked = sorted(aggregated.values(), key=lambda item: item.score, reverse=True)
    if top_k > 0:
        ranked = ranked[:top_k]

    return [
        {
            "document_id": item.passage_id,
            "score": item.score,
            "text": item.text or "",
            "title": item.title or "",
            "url": item.url or "",
        }
        for item in ranked
    ]


def build_retrieval_record(
    task: dict[str, object], contexts: list[dict[str, object]]
) -> dict[str, object]:
    record = dict(task)
    record["contexts"] = contexts
    return record


def build_generation_record(
    task: dict[str, object],
    contexts: list[dict[str, object]],
    answer_text: str,
) -> dict[str, object]:
    record = dict(task)
    record["contexts"] = contexts
    record["predictions"] = [{"text": answer_text}]
    return record
