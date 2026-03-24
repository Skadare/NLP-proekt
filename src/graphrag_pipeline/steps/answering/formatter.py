"""Evidence formatting for answer generation."""

from __future__ import annotations

from graphrag_pipeline.types import ProvenanceRecord, SubgraphResult


def _build_provenance_map(
    provenance: list[ProvenanceRecord],
) -> dict[str, ProvenanceRecord]:
    return {record.provenance_id: record for record in provenance}


def format_evidence(
    subgraph: SubgraphResult,
    provenance: list[ProvenanceRecord],
) -> dict[str, object]:
    """Format evidence lines and ids for prompts."""
    if not subgraph.facts:
        return {
            "evidence_text": "",
            "evidence_ids": [],
            "provenance_map": {},
        }

    provenance_map = _build_provenance_map(provenance)
    evidence_lines: list[str] = []
    evidence_ids: list[str] = []

    for index, fact in enumerate(subgraph.facts, start=1):
        evidence_id = f"E{index}"
        evidence_ids.append(evidence_id)
        prov = provenance_map.get(fact.provenance_id or "")
        snippet = prov.snippet if prov is not None else None
        doc_id = prov.doc_id if prov is not None else None
        passage_id = prov.passage_id if prov is not None else None

        line_parts = [
            f"[{evidence_id}]",
            f"triple_id={fact.triple_id}",
            f"head={fact.head}",
            f"relation={fact.relation}",
            f"tail={fact.tail}",
            f"score={fact.score:.4f}",
        ]
        if fact.provenance_id:
            line_parts.append(f"provenance_id={fact.provenance_id}")
        if doc_id:
            line_parts.append(f"doc_id={doc_id}")
        if passage_id:
            line_parts.append(f"passage_id={passage_id}")
        if snippet:
            line_parts.append(f'snippet="{snippet}"')

        evidence_lines.append(" ".join(line_parts))

    return {
        "evidence_text": "\n".join(evidence_lines),
        "evidence_ids": evidence_ids,
        "provenance_map": provenance_map,
    }
