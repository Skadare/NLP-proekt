"""Scoring for candidate edges."""

from __future__ import annotations

import re


def _match_phrase(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", flags=re.IGNORECASE)
    return pattern.search(text) is not None


def score_candidates(
    question: str,
    candidates: list[dict[str, object]],
    anchor_ids: set[str],
) -> list[tuple[dict[str, object], float]]:
    if not question.strip():
        raise ValueError("Question is empty; cannot score subgraph candidates.")

    scored: list[tuple[dict[str, object], float]] = []

    for edge in candidates:
        head_id = str(edge.get("head_id") or "")
        tail_id = str(edge.get("tail_id") or "")
        relation_name = str(edge.get("relation_name") or "")
        head_name = str(edge.get("head_name") or "")
        tail_name = str(edge.get("tail_name") or "")
        hop = int(edge.get("hop") or 1)
        confidence = edge.get("confidence")
        confidence_val = float(confidence) if isinstance(confidence, (int, float)) else 0.0

        score = 0.0
        if head_id in anchor_ids or tail_id in anchor_ids:
            score += 2.0
        if head_id in anchor_ids and tail_id in anchor_ids:
            score += 1.0

        if relation_name and _match_phrase(question, relation_name):
            score += 0.6
        if head_name and _match_phrase(question, head_name):
            score += 0.4
        if tail_name and _match_phrase(question, tail_name):
            score += 0.4

        score += 0.3 * confidence_val

        if hop >= 2:
            score -= 0.5

        scored.append((edge, score))

    scored.sort(key=lambda item: (-item[1], str(item[0].get("triple_id"))))
    return scored
