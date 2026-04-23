"""Shared global context passed through the pipeline."""

from __future__ import annotations

from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from .types import (
    AnswerResult,
    ConversationMessage,
    Entity,
    LinkedEntity,
    ProvenanceRecord,
    Relation,
    SubgraphResult,
    Triple,
)


class PipelineContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str | None = None
    input_path: str | None = None
    kg_dir: str | None = None
    raw_text: str | None = None
    raw_question: str | None = None
    normalized_question: str | None = None
    conversation_id: str | None = None
    task_id: str | None = None
    collection: str | None = None
    conversation_messages: list[ConversationMessage] = Field(default_factory=list)

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    triples: list[Triple] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)

    linked_entities: list[LinkedEntity] = Field(default_factory=list)
    subgraph: SubgraphResult = Field(default_factory=SubgraphResult)
    answer_result: AnswerResult = Field(default_factory=AnswerResult)

    graph: nx.MultiDiGraph | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
