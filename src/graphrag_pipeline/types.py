"""Shared typed models used across the pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Entity(BaseModel):
    entity_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    entity_type: str | None = None


class Relation(BaseModel):
    relation_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)


class ProvenanceRecord(BaseModel):
    provenance_id: str
    source_path: str | None = None
    doc_id: str | None = None
    passage_id: str | None = None
    snippet: str | None = None
    offset_start: int | None = None
    offset_end: int | None = None
    extractor: str | None = None
    confidence: float | None = None


class Triple(BaseModel):
    triple_id: str
    head_id: str
    relation_id: str
    tail_id: str
    provenance_id: str | None = None
    confidence: float | None = None


class LinkedEntity(BaseModel):
    mention: str
    entity_id: str
    canonical_name: str
    score: float | None = None


class RetrievedFact(BaseModel):
    triple_id: str
    score: float
    head: str
    relation: str
    tail: str
    provenance_id: str | None = None


class SubgraphResult(BaseModel):
    node_ids: list[str] = Field(default_factory=list)
    triple_ids: list[str] = Field(default_factory=list)
    facts: list[RetrievedFact] = Field(default_factory=list)


class AnswerResult(BaseModel):
    answer: str | None = None
    reasoning: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class StructuredResponse(BaseModel):
    title: str = "GraphRAG Response"
    sections: list[dict[str, Any]] = Field(default_factory=list)
