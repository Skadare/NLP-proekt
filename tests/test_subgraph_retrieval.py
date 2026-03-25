from pathlib import Path

import networkx as nx
import pytest

from graphrag_pipeline.context import PipelineContext
from graphrag_pipeline.steps.subgraph_retrieval.retriever import (
    build_kg_graph,
    load_kg_artifacts,
    retrieve_subgraph,
)
from graphrag_pipeline.steps.subgraph_retrieval.step import SubgraphRetrievalStep
from graphrag_pipeline.types import Entity, LinkedEntity, Relation, Triple


def _simple_graph_fixture() -> tuple[list[Entity], list[Relation], list[Triple], nx.MultiDiGraph]:
    entities = [
        Entity(entity_id="ent_1", canonical_name="Azure"),
        Entity(entity_id="ent_2", canonical_name="AKS"),
        Entity(entity_id="ent_3", canonical_name="AWS"),
    ]
    relations = [Relation(relation_id="rel_1", canonical_name="offers")]
    triples = [
        Triple(triple_id="trp_1", head_id="ent_1", relation_id="rel_1", tail_id="ent_2"),
        Triple(triple_id="trp_2", head_id="ent_3", relation_id="rel_1", tail_id="ent_2"),
    ]
    graph = build_kg_graph(entities, relations, triples)
    return entities, relations, triples, graph


def test_graph_construction_from_kg_artifacts(tmp_path: Path) -> None:
    kg_dir = tmp_path / "kg"
    kg_dir.mkdir(parents=True)
    (kg_dir / "entities.jsonl").write_text(
        """{"entity_id":"ent_1","canonical_name":"Azure","aliases":[],"entity_type":"org"}\n""",
        encoding="utf-8",
    )
    (kg_dir / "relations.jsonl").write_text(
        """{"relation_id":"rel_1","canonical_name":"offers","aliases":[]}\n""",
        encoding="utf-8",
    )
    (kg_dir / "triples.jsonl").write_text(
        """{"triple_id":"trp_1","head_id":"ent_1","relation_id":"rel_1","tail_id":"ent_1","provenance_id":"prov_1","confidence":0.8}\n""",
        encoding="utf-8",
    )
    (kg_dir / "provenance.jsonl").write_text(
        """{"provenance_id":"prov_1","snippet":"Azure offers Azure."}\n""",
        encoding="utf-8",
    )

    entities, relations, triples, _ = load_kg_artifacts(str(kg_dir))
    graph = build_kg_graph(entities, relations, triples)

    assert graph.has_node("ent_1")
    assert graph.nodes["ent_1"]["canonical_name"] == "Azure"
    edges = list(graph.edges(keys=True, data=True))
    assert edges[0][2] == "trp_1"
    assert edges[0][3]["relation_name"] == "offers"
    assert edges[0][3]["provenance_id"] == "prov_1"


def test_retrieve_subgraph_anchors_on_linked_entities() -> None:
    entities, relations, triples, graph = _simple_graph_fixture()

    subgraph = retrieve_subgraph(
        "What does Azure offer?",
        linked_entities=[LinkedEntity(mention="Azure", entity_id="ent_1", canonical_name="Azure")],
        entities=entities,
        relations=relations,
        triples=triples,
        provenance=[],
        graph=graph,
        top_k=5,
        include_two_hop=False,
    )

    assert subgraph.triple_ids == ["trp_1"]
    assert subgraph.node_ids == ["ent_1", "ent_2"]
    assert subgraph.facts[0].head == "Azure"
    assert subgraph.facts[0].relation == "offers"
    assert subgraph.facts[0].tail == "AKS"


def test_retrieve_subgraph_falls_back_to_alias_match() -> None:
    entities = [
        Entity(entity_id="ent_1", canonical_name="Azure Kubernetes Service", aliases=["AKS"])
    ]
    relations = [Relation(relation_id="rel_1", canonical_name="supports")]
    triples = [Triple(triple_id="trp_1", head_id="ent_1", relation_id="rel_1", tail_id="ent_1")]
    graph = build_kg_graph(entities, relations, triples)

    subgraph = retrieve_subgraph(
        "What does AKS support?",
        linked_entities=[],
        entities=entities,
        relations=relations,
        triples=triples,
        provenance=[],
        graph=graph,
        include_two_hop=False,
    )

    assert subgraph.triple_ids == ["trp_1"]
    assert subgraph.facts[0].head == "Azure Kubernetes Service"


def test_retrieve_subgraph_returns_empty_when_no_match() -> None:
    entities = [Entity(entity_id="ent_1", canonical_name="Azure")]
    relations = [Relation(relation_id="rel_1", canonical_name="offers")]
    triples = [Triple(triple_id="trp_1", head_id="ent_1", relation_id="rel_1", tail_id="ent_1")]
    graph = build_kg_graph(entities, relations, triples)

    subgraph = retrieve_subgraph(
        "What does GCP offer?",
        linked_entities=[],
        entities=entities,
        relations=relations,
        triples=triples,
        provenance=[],
        graph=graph,
    )

    assert subgraph.facts == []
    assert subgraph.triple_ids == []
    assert subgraph.node_ids == []


def test_retrieve_subgraph_respects_top_k_and_ordering() -> None:
    entities = [
        Entity(entity_id="ent_1", canonical_name="Azure"),
        Entity(entity_id="ent_2", canonical_name="AKS"),
        Entity(entity_id="ent_3", canonical_name="Functions"),
        Entity(entity_id="ent_4", canonical_name="App Service"),
    ]
    relations = [Relation(relation_id="rel_1", canonical_name="offers")]
    triples = [
        Triple(triple_id="trp_2", head_id="ent_1", relation_id="rel_1", tail_id="ent_3"),
        Triple(triple_id="trp_1", head_id="ent_1", relation_id="rel_1", tail_id="ent_2"),
        Triple(triple_id="trp_3", head_id="ent_1", relation_id="rel_1", tail_id="ent_4"),
    ]
    graph = build_kg_graph(entities, relations, triples)

    subgraph = retrieve_subgraph(
        "What does Azure offer?",
        linked_entities=[LinkedEntity(mention="Azure", entity_id="ent_1", canonical_name="Azure")],
        entities=entities,
        relations=relations,
        triples=triples,
        provenance=[],
        graph=graph,
        top_k=2,
        include_two_hop=False,
    )

    assert subgraph.triple_ids == ["trp_1", "trp_2"]
    assert len(subgraph.facts) == 2


def test_retrieve_subgraph_relation_overlap_boosts_score() -> None:
    entities = [
        Entity(entity_id="ent_1", canonical_name="Azure"),
        Entity(entity_id="ent_2", canonical_name="AKS"),
    ]
    relations = [
        Relation(relation_id="rel_1", canonical_name="offers"),
        Relation(relation_id="rel_2", canonical_name="includes"),
    ]
    triples = [
        Triple(triple_id="trp_1", head_id="ent_1", relation_id="rel_1", tail_id="ent_2"),
        Triple(triple_id="trp_2", head_id="ent_1", relation_id="rel_2", tail_id="ent_2"),
    ]
    graph = build_kg_graph(entities, relations, triples)

    subgraph = retrieve_subgraph(
        "What does Azure offer?",
        linked_entities=[LinkedEntity(mention="Azure", entity_id="ent_1", canonical_name="Azure")],
        entities=entities,
        relations=relations,
        triples=triples,
        provenance=[],
        graph=graph,
        top_k=1,
        include_two_hop=False,
    )

    assert subgraph.triple_ids == ["trp_1"]


def test_self_referential_alias_is_downranked() -> None:
    entities = [
        Entity(entity_id="ent_1", canonical_name="Kubernetes"),
        Entity(entity_id="ent_2", canonical_name="a managed Kubernetes service"),
    ]
    relations = [
        Relation(relation_id="rel_1", canonical_name="is also called"),
        Relation(relation_id="rel_2", canonical_name="is"),
    ]
    triples = [
        Triple(triple_id="trp_alias", head_id="ent_1", relation_id="rel_1", tail_id="ent_1"),
        Triple(triple_id="trp_def", head_id="ent_1", relation_id="rel_2", tail_id="ent_2"),
    ]
    graph = build_kg_graph(entities, relations, triples)

    subgraph = retrieve_subgraph(
        "What is Kubernetes?",
        linked_entities=[
            LinkedEntity(mention="Kubernetes", entity_id="ent_1", canonical_name="Kubernetes")
        ],
        entities=entities,
        relations=relations,
        triples=triples,
        provenance=[],
        graph=graph,
        top_k=1,
        include_two_hop=False,
    )

    assert subgraph.triple_ids == ["trp_def"]


def test_definition_question_prefers_is_relation() -> None:
    entities = [
        Entity(entity_id="ent_1", canonical_name="AKS"),
        Entity(entity_id="ent_2", canonical_name="a managed Kubernetes service"),
        Entity(entity_id="ent_3", canonical_name="Kubernetes"),
    ]
    relations = [
        Relation(relation_id="rel_1", canonical_name="is"),
        Relation(relation_id="rel_2", canonical_name="offers"),
    ]
    triples = [
        Triple(triple_id="trp_is", head_id="ent_1", relation_id="rel_1", tail_id="ent_2"),
        Triple(triple_id="trp_offer", head_id="ent_1", relation_id="rel_2", tail_id="ent_3"),
    ]
    graph = build_kg_graph(entities, relations, triples)

    subgraph = retrieve_subgraph(
        "What is AKS?",
        linked_entities=[LinkedEntity(mention="AKS", entity_id="ent_1", canonical_name="AKS")],
        entities=entities,
        relations=relations,
        triples=triples,
        provenance=[],
        graph=graph,
        top_k=1,
        include_two_hop=False,
    )

    assert subgraph.triple_ids == ["trp_is"]


def test_deduplication_removes_duplicate_facts() -> None:
    entities = [
        Entity(entity_id="ent_1", canonical_name="Azure"),
        Entity(entity_id="ent_2", canonical_name="AKS"),
        Entity(entity_id="ent_3", canonical_name="Functions"),
    ]
    relations = [
        Relation(relation_id="rel_1", canonical_name="provides"),
        Relation(relation_id="rel_2", canonical_name="offers"),
    ]
    triples = [
        Triple(triple_id="trp_dup_1", head_id="ent_1", relation_id="rel_1", tail_id="ent_2"),
        Triple(triple_id="trp_dup_2", head_id="ent_1", relation_id="rel_1", tail_id="ent_2"),
        Triple(triple_id="trp_other", head_id="ent_1", relation_id="rel_2", tail_id="ent_3"),
    ]
    graph = build_kg_graph(entities, relations, triples)

    subgraph = retrieve_subgraph(
        "What does Azure provide?",
        linked_entities=[LinkedEntity(mention="Azure", entity_id="ent_1", canonical_name="Azure")],
        entities=entities,
        relations=relations,
        triples=triples,
        provenance=[],
        graph=graph,
        top_k=3,
        include_two_hop=False,
    )

    duplicates = {"trp_dup_1", "trp_dup_2"}
    assert len([triple_id for triple_id in subgraph.triple_ids if triple_id in duplicates]) == 1
    assert "trp_other" in subgraph.triple_ids


def test_useful_facts_are_preserved() -> None:
    entities = [
        Entity(entity_id="ent_1", canonical_name="Kubernetes"),
        Entity(entity_id="ent_2", canonical_name="teams deploy and scale Kubernetes clusters"),
    ]
    relations = [
        Relation(relation_id="rel_1", canonical_name="helps"),
        Relation(relation_id="rel_2", canonical_name="is also known as"),
    ]
    triples = [
        Triple(triple_id="trp_helpful", head_id="ent_1", relation_id="rel_1", tail_id="ent_2"),
        Triple(triple_id="trp_alias", head_id="ent_1", relation_id="rel_2", tail_id="ent_1"),
    ]
    graph = build_kg_graph(entities, relations, triples)

    subgraph = retrieve_subgraph(
        "What is Kubernetes?",
        linked_entities=[
            LinkedEntity(mention="Kubernetes", entity_id="ent_1", canonical_name="Kubernetes")
        ],
        entities=entities,
        relations=relations,
        triples=triples,
        provenance=[],
        graph=graph,
        top_k=2,
        include_two_hop=False,
    )

    assert "trp_helpful" in subgraph.triple_ids


def test_subgraph_step_uses_existing_context_graph(monkeypatch) -> None:
    entities, relations, triples, graph = _simple_graph_fixture()
    context = PipelineContext(
        raw_question="What does Azure offer?",
        normalized_question="What does Azure offer?",
        entities=entities,
        relations=relations,
        triples=triples,
        provenance=[],
        graph=graph,
    )

    def fail_build(*args, **kwargs):
        raise AssertionError("build_kg_graph should not be called")

    monkeypatch.setattr(
        "graphrag_pipeline.steps.subgraph_retrieval.step.build_kg_graph",
        fail_build,
    )

    result = SubgraphRetrievalStep(top_k=5, include_two_hop=False).run(context)
    assert result.graph is graph


def test_retrieve_subgraph_two_hop_expansion_adds_edges() -> None:
    entities = [
        Entity(entity_id="ent_1", canonical_name="Azure"),
        Entity(entity_id="ent_2", canonical_name="AKS"),
        Entity(entity_id="ent_3", canonical_name="Kubernetes"),
    ]
    relations = [Relation(relation_id="rel_1", canonical_name="offers")]
    triples = [
        Triple(triple_id="trp_1", head_id="ent_1", relation_id="rel_1", tail_id="ent_2"),
        Triple(triple_id="trp_2", head_id="ent_2", relation_id="rel_1", tail_id="ent_3"),
    ]
    graph = build_kg_graph(entities, relations, triples)

    subgraph = retrieve_subgraph(
        "What does Azure offer?",
        linked_entities=[LinkedEntity(mention="Azure", entity_id="ent_1", canonical_name="Azure")],
        entities=entities,
        relations=relations,
        triples=triples,
        provenance=[],
        graph=graph,
        include_two_hop=True,
        top_k=5,
    )

    assert "trp_1" in subgraph.triple_ids
    assert "trp_2" in subgraph.triple_ids


def test_subgraph_step_uses_normalized_question_first() -> None:
    entities, relations, triples, graph = _simple_graph_fixture()
    context = PipelineContext(
        raw_question="raw question",
        normalized_question="What does Azure offer?",
        entities=entities,
        relations=relations,
        triples=triples,
        provenance=[],
        graph=graph,
    )
    result = SubgraphRetrievalStep(top_k=1, include_two_hop=False).run(context)
    assert result.subgraph.triple_ids == ["trp_1"]


def test_subgraph_step_requires_question() -> None:
    context = PipelineContext()
    with pytest.raises(ValueError):
        SubgraphRetrievalStep().run(context)
