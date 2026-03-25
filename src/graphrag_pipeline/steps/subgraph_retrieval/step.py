"""Pipeline step for subgraph retrieval."""

from graphrag_pipeline.context import PipelineContext
from graphrag_pipeline.pipeline.base import PipelineStep

from .retriever import build_kg_graph, load_kg_artifacts, retrieve_subgraph


class SubgraphRetrievalStep(PipelineStep):
    name = "subgraph_retrieval"

    def __init__(self, top_k: int = 10, include_two_hop: bool = True) -> None:
        self.top_k = top_k
        self.include_two_hop = include_two_hop

    def run(self, context: PipelineContext) -> PipelineContext:
        question = context.normalized_question or context.raw_question
        if question is None or not question.strip():
            raise ValueError("Question is required for subgraph retrieval.")

        if (
            not context.entities or not context.relations or not context.triples
        ) and context.graph is None:
            if context.kg_dir is None:
                raise ValueError("PipelineContext.kg_dir is required for subgraph retrieval.")
            entities, relations, triples, provenance = load_kg_artifacts(context.kg_dir)
            context.entities = entities
            context.relations = relations
            context.triples = triples
            context.provenance = provenance

        if context.graph is None:
            context.graph = build_kg_graph(
                context.entities,
                context.relations,
                context.triples,
            )

        context.subgraph = retrieve_subgraph(
            question,
            context.linked_entities,
            context.entities,
            context.relations,
            context.triples,
            context.provenance,
            context.graph,
            top_k=self.top_k,
            include_two_hop=self.include_two_hop,
        )
        context.metadata.setdefault("steps", []).append(self.name)
        if context.metadata.get("debug"):
            print(
                "[retrieval]",
                {
                    "normalized_question": context.normalized_question,
                    "facts": [fact.model_dump() for fact in context.subgraph.facts],
                },
            )
        return context
