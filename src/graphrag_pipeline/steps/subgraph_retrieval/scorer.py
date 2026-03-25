"""Scoring for candidate edges."""

from __future__ import annotations

import re


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.strip().lower().split())


def _match_phrase(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", flags=re.IGNORECASE)
    return pattern.search(text) is not None


def _is_definition_question(question: str) -> bool:
    normalized = _normalize_text(question)
    if not normalized:
        return False
    if normalized.startswith(("what is ", "who is ", "what's ", "who's ")):
        return True
    if normalized.startswith("define ") or normalized.startswith("definition of "):
        return True
    if normalized.startswith("what does ") and " mean" in normalized:
        return True
    return False


def score_candidates(
    question: str,
    candidates: list[dict[str, object]],
    anchor_ids: set[str],
) -> list[tuple[dict[str, object], float]]:
    if not question.strip():
        raise ValueError("Question is empty; cannot score subgraph candidates.")

    scored: list[tuple[dict[str, object], float]] = []

    relation_weights = {
        "offers": -0.2,
        "related to": -0.4,
        "associated with": -0.4,
        "also called": -0.4,
        "also known as": -0.4,
        "is also called": -0.6,
        "is also known as": -0.6,
        "is": 0.8,
        "refers to": 0.6,
        "defined as": 0.7,
        "used for": 0.5,
        "helps": 0.5,
        "provides": 0.5,
        "managed by": 0.4,
        "part of": 0.4,
    }
    alias_like_relations = {
        "is also called",
        "is also known as",
        "also called",
        "also known as",
    }
    definitional_relations = {"is", "refers to", "defined as", "means", "stands for"}
    definition_question = _is_definition_question(question)

    for edge in candidates:
        head_id = str(edge.get("head_id") or "")
        tail_id = str(edge.get("tail_id") or "")
        relation_name = str(edge.get("relation_name") or "")
        head_name = str(edge.get("head_name") or "")
        tail_name = str(edge.get("tail_name") or "")
        normalized_relation = _normalize_text(relation_name)
        normalized_head = _normalize_text(head_name)
        normalized_tail = _normalize_text(tail_name)
        hop_raw = edge.get("hop")
        hop = 1
        if isinstance(hop_raw, int):
            hop = hop_raw
        elif isinstance(hop_raw, float):
            hop = int(hop_raw)
        elif isinstance(hop_raw, str) and hop_raw.strip().isdigit():
            hop = int(hop_raw.strip())
        confidence = edge.get("confidence")
        confidence_val = float(confidence) if isinstance(confidence, (int, float)) else 0.0

        score = 0.0
        if head_id in anchor_ids or tail_id in anchor_ids:
            score += 2.0
        if head_id in anchor_ids and tail_id in anchor_ids:
            score += 1.0

        if relation_name and _match_phrase(question, relation_name):
            score += 0.6
        elif normalized_relation.endswith("s"):
            singular_relation = normalized_relation[:-1]
            if singular_relation and _match_phrase(question, singular_relation):
                score += 0.3
        if head_name and _match_phrase(question, head_name):
            score += 0.4
        if tail_name and _match_phrase(question, tail_name):
            score += 0.4

        if normalized_relation in relation_weights:
            score += relation_weights[normalized_relation]

        if normalized_head and normalized_head == normalized_tail:
            if normalized_relation in alias_like_relations:
                score -= 3.0

        if definition_question and normalized_relation in definitional_relations:
            score += 0.7

        score += 0.3 * confidence_val

        if hop >= 2:
            score -= 0.5

        scored.append((edge, score))

    scored.sort(key=lambda item: (-item[1], str(item[0].get("triple_id"))))
    return scored
