"""Pipeline step for subgraph retrieval."""

from graphrag_pipeline.context import PipelineContext
from graphrag_pipeline.pipeline.base import PipelineStep

from .retriever import build_kg_graph, load_kg_artifacts, retrieve_subgraph


class SubgraphRetrievalStep(PipelineStep):
    name = "subgraph_retrieval"

    def __init__(self, top_k: int = 10, include_two_hop: bool = True) -> None:
        self.top_k = top_k
        self.include_two_hop = include_two_hop

    @staticmethod
    def _query_candidates(context: PipelineContext) -> list[str]:
        candidates: list[str | None] = [
            context.metadata.get("retrieval_query") if isinstance(context.metadata, dict) else None,
            context.normalized_question,
            context.metadata.get("standalone_rewrite")
            if isinstance(context.metadata, dict)
            else None,
            context.raw_question,
        ]

        ordered_queries: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            if not isinstance(item, str):
                continue
            query = item.strip()
            if not query:
                continue
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered_queries.append(query)

        return ordered_queries

    def run(self, context: PipelineContext) -> PipelineContext:
        query_candidates = self._query_candidates(context)
        if not query_candidates:
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

        selected_query = query_candidates[0]
        selected_subgraph = retrieve_subgraph(
            selected_query,
            context.linked_entities,
            context.entities,
            context.relations,
            context.triples,
            context.provenance,
            context.graph,
            top_k=self.top_k,
            include_two_hop=self.include_two_hop,
        )

        if not selected_subgraph.facts:
            for query in query_candidates[1:]:
                candidate_subgraph = retrieve_subgraph(
                    query,
                    context.linked_entities,
                    context.entities,
                    context.relations,
                    context.triples,
                    context.provenance,
                    context.graph,
                    top_k=self.top_k,
                    include_two_hop=self.include_two_hop,
                )
                if candidate_subgraph.facts:
                    selected_query = query
                    selected_subgraph = candidate_subgraph
                    break

        context.subgraph = selected_subgraph
        context.metadata["retrieval_query_attempts"] = query_candidates
        context.metadata["retrieval_query_used"] = selected_query
        context.metadata.setdefault("steps", []).append(self.name)
        if context.metadata.get("debug"):
            print(
                "[retrieval]",
                {
                    "retrieval_query_attempts": query_candidates,
                    "retrieval_query_used": selected_query,
                    "normalized_question": context.normalized_question,
                    "facts": [fact.model_dump() for fact in context.subgraph.facts],
                },
            )
        return context
