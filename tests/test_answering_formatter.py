from graphrag_pipeline.steps.answering.formatter import format_evidence
from graphrag_pipeline.types import ProvenanceRecord, RetrievedFact, SubgraphResult


def test_format_evidence_includes_provenance_fields() -> None:
    subgraph = SubgraphResult(
        facts=[
            RetrievedFact(
                triple_id="trp_1",
                score=0.75,
                head="Azure",
                relation="offers",
                tail="AKS",
                provenance_id="prov_1",
            )
        ]
    )
    provenance = [
        ProvenanceRecord(
            provenance_id="prov_1",
            doc_id="doc_1",
            passage_id="p1",
            snippet="Azure offers AKS.",
        )
    ]

    formatted = format_evidence(subgraph, provenance)
    evidence_text = formatted["evidence_text"]

    assert formatted["evidence_ids"] == ["E1"]
    assert "[E1]" in evidence_text
    assert "triple_id=trp_1" in evidence_text
    assert "provenance_id=prov_1" in evidence_text
    assert "doc_id=doc_1" in evidence_text
    assert "passage_id=p1" in evidence_text
    assert 'snippet="Azure offers AKS."' in evidence_text


def test_format_evidence_empty_subgraph() -> None:
    formatted = format_evidence(SubgraphResult(), provenance=[])
    assert formatted["evidence_text"] == ""
    assert formatted["evidence_ids"] == []
    assert formatted["provenance_map"] == {}
