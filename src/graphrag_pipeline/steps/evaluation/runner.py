"""Minimal evaluation runner for mt-rag-benchmark integration."""

from __future__ import annotations

import argparse
import heapq
import json
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

from graphrag_pipeline.context import PipelineContext
from graphrag_pipeline.pipeline.runner import PipelineRunner
from graphrag_pipeline.steps.answering.step import AnsweringStep
from graphrag_pipeline.steps.standardization.step import StandardizationStep
from graphrag_pipeline.steps.subgraph_retrieval.step import SubgraphRetrievalStep
from graphrag_pipeline.types import ConversationMessage

from .mtrag_adapter import build_generation_record, build_retrieval_record, map_subgraph_to_contexts


SAMPLE_PRESET_TASKS = {
    "smoke": 8,
    "dev": 64,
    "stable": 160,
}

COLLECTION_STEM_BY_NAME = {
    "mt-rag-clapnq-elser-512-100-20240503": "clapnq",
    "mt-rag-govt-elser-512-100-20240611": "govt",
    "mt-rag-fiqa-beir-elser-512-100-20240501": "fiqa",
    "mt-rag-ibmcloud-elser-512-100-20240502": "cloud",
}

_RETRIEVAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

_TEXT_REPLACEMENTS = {
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",
    "‚Äôs": "'s",
    "‚Äô": "'",
}


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    lines = [json.dumps(record, ensure_ascii=True) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_checker_input_if_sliced(
    *,
    output_dir: Path,
    file_name: str,
    input_path: Path,
    tasks: list[dict[str, object]],
    max_tasks: int,
) -> Path:
    if max_tasks <= 0:
        return input_path

    sliced_input = output_dir / file_name
    _write_jsonl(sliced_input, tasks)
    return sliced_input


def _format_duration(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _notify_send(title: str, message: str) -> None:
    if shutil.which("notify-send") is None:
        return
    try:
        subprocess.run(["notify-send", title, message], check=False)
    except Exception:
        return


def _extract_question(task: dict[str, object]) -> str:
    input_messages = task.get("input", [])
    if isinstance(input_messages, list):
        for item in reversed(input_messages):
            if isinstance(item, dict) and item.get("speaker") == "user":
                text = item.get("text")
                if isinstance(text, str):
                    return text
    raise ValueError("Task is missing a user question in input.")


def _extract_conversation(task: dict[str, object]) -> list[ConversationMessage]:
    messages = task.get("input", [])
    if not isinstance(messages, list):
        return []

    conversation: list[ConversationMessage] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        speaker = item.get("speaker")
        text = item.get("text")
        if not isinstance(speaker, str) or not speaker:
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        metadata = item.get("metadata")
        conversation.append(
            ConversationMessage(
                speaker=speaker,
                text=text,
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        )

    return conversation


def _extract_collection(task: dict[str, object]) -> str | None:
    collection = task.get("Collection")
    if isinstance(collection, str) and collection:
        return collection
    collection = task.get("collection")
    if isinstance(collection, str) and collection:
        return collection
    return None


def _extract_task_id(task: dict[str, object]) -> str | None:
    task_id = task.get("task_id")
    if isinstance(task_id, str) and task_id:
        return task_id
    return None


def _normalize_text(text: str) -> str:
    normalized = text
    for src, dst in _TEXT_REPLACEMENTS.items():
        normalized = normalized.replace(src, dst)
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _important_tokens(text: str) -> set[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return set()

    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    expanded: set[str] = set()
    for token in tokens:
        if len(token) <= 2 or token in _RETRIEVAL_STOPWORDS:
            continue
        expanded.add(token)
        if token.endswith("s") and len(token) > 3:
            expanded.add(token[:-1])
    return expanded


def _ordered_tokens(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    tokens = re.findall(r"[a-z0-9]+", normalized)
    ordered: list[str] = []
    for token in tokens:
        if len(token) <= 2 or token in _RETRIEVAL_STOPWORDS:
            continue
        ordered.append(token)
        if token.endswith("s") and len(token) > 3:
            ordered.append(token[:-1])
    return ordered


def _phrase_windows(tokens: list[str], size: int) -> set[str]:
    if len(tokens) < size or size <= 0:
        return set()
    return {" ".join(tokens[idx : idx + size]) for idx in range(len(tokens) - size + 1)}


def _score_passage_text(query: str, title: str, text: str) -> float:
    query_tokens = _important_tokens(query)
    if not query_tokens:
        return 0.0

    title_norm = _normalize_text(title)
    text_norm = _normalize_text(text)
    haystack = f"{title_norm} {text_norm}".strip()
    if not haystack:
        return 0.0

    title_tokens = _important_tokens(title_norm)
    text_tokens = _important_tokens(text_norm)
    passage_tokens = title_tokens.union(text_tokens)
    overlap = query_tokens.intersection(passage_tokens)
    if not overlap:
        return 0.0

    score = float(len(overlap))
    score += 2.5 * len(overlap) / max(len(query_tokens), 1)

    for token in overlap:
        if token in title_tokens:
            score += 0.9
        if token in text_tokens:
            score += 0.2
        if len(token) >= 8:
            score += 0.25

    query_norm = _normalize_text(query)
    if title_norm and title_norm in query_norm:
        score += 2.0

    if query_norm and query_norm in haystack:
        score += 5.0

    query_token_list = _ordered_tokens(query)
    query_bigrams = _phrase_windows(query_token_list, 2)
    if query_bigrams:
        title_bigrams = _phrase_windows(_ordered_tokens(title_norm), 2)
        text_bigrams = _phrase_windows(_ordered_tokens(text_norm), 2)
        title_overlap = query_bigrams.intersection(title_bigrams)
        text_overlap = query_bigrams.intersection(text_bigrams)
        score += 1.2 * len(title_overlap)
        score += 0.5 * len(text_overlap)

    if len(overlap) == 1 and len(query_tokens) >= 4:
        score -= 0.8

    return score


def _retrieve_contexts_from_corpus(
    *,
    mtrag_root: Path,
    collection: str,
    query: str,
    top_k: int,
    query_variants: list[str] | None = None,
    cache: dict[tuple[str, str], list[dict[str, object]]] | None = None,
) -> list[dict[str, object]]:
    if top_k <= 0:
        return []

    variant_key = "||".join(query_variants or [])
    cache_key = (collection, f"{query}::{variant_key}")
    if cache is not None and cache_key in cache:
        return [dict(item) for item in cache[cache_key]]

    stem = COLLECTION_STEM_BY_NAME.get(collection)
    if stem is None:
        return []

    zip_path = mtrag_root / "corpora" / "passage_level" / f"{stem}.jsonl.zip"
    if not zip_path.exists() or not zip_path.is_file():
        return []

    heap: list[tuple[float, str, dict[str, object]]] = []
    import zipfile

    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        if not members:
            return []
        with archive.open(members[0]) as handle:
            for raw_line in handle:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    continue
                document_id = payload.get("id") or payload.get("_id")
                text = payload.get("text")
                title = payload.get("title")
                url = payload.get("url")
                if not isinstance(document_id, str) or not document_id:
                    continue
                if not isinstance(text, str) or not text.strip():
                    continue

                scoring_queries = [query]
                if query_variants:
                    scoring_queries.extend(query_variants)

                score = 0.0
                for idx, query_item in enumerate(scoring_queries):
                    weight = 1.0 if idx == 0 else 0.85
                    score = max(
                        score,
                        weight
                        * _score_passage_text(
                            query_item,
                            title if isinstance(title, str) else "",
                            text,
                        ),
                    )
                if score <= 0.0:
                    continue

                context = {
                    "document_id": document_id,
                    "source": "",
                    "score": score,
                    "text": text,
                    "title": title if isinstance(title, str) else "",
                    "url": url if isinstance(url, str) else "",
                }
                item = (score, document_id, context)
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                else:
                    if item > heap[0]:
                        heapq.heapreplace(heap, item)

    ranked = [item[2] for item in sorted(heap, key=lambda x: (-x[0], x[1]))]
    if cache is not None:
        cache[cache_key] = [dict(item) for item in ranked]
    return ranked


def _merge_contexts(
    primary: list[dict[str, object]],
    secondary: list[dict[str, object]],
    *,
    top_k: int,
) -> list[dict[str, object]]:
    def context_score(item: dict[str, object]) -> float:
        raw = item.get("score")
        if isinstance(raw, (int, float)):
            return float(raw)
        return 0.0

    merged: dict[str, dict[str, object]] = {}
    for context in [*primary, *secondary]:
        doc_id = context.get("document_id")
        if not isinstance(doc_id, str) or not doc_id:
            continue
        score_raw = context.get("score")
        score = float(score_raw) if isinstance(score_raw, (int, float)) else 0.0
        existing = merged.get(doc_id)
        if existing is None:
            merged[doc_id] = dict(context)
            merged[doc_id]["score"] = score
            continue

        existing_score_raw = existing.get("score")
        existing_score = (
            float(existing_score_raw) if isinstance(existing_score_raw, (int, float)) else 0.0
        )
        if score > existing_score:
            existing["score"] = score
        if not existing.get("text") and context.get("text"):
            existing["text"] = context.get("text")
        if not existing.get("title") and context.get("title"):
            existing["title"] = context.get("title")
        if not existing.get("url") and context.get("url"):
            existing["url"] = context.get("url")

    ranked = sorted(
        merged.values(),
        key=lambda item: (
            -context_score(item),
            str(item.get("document_id") or ""),
        ),
    )
    if top_k > 0:
        ranked = ranked[:top_k]
    return ranked


def _extract_conversation_id(task: dict[str, object]) -> str | None:
    conversation_id = task.get("conversation_id")
    if isinstance(conversation_id, str) and conversation_id:
        return conversation_id

    task_id = _extract_task_id(task)
    if task_id is None:
        return None

    if "<::>" in task_id:
        prefix, _, _ = task_id.partition("<::>")
        return prefix or task_id
    return task_id


def _resolve_kg_dir_for_collection(base_kg_dir: str, collection: str | None) -> str:
    base_path = Path(base_kg_dir)
    if collection is None:
        return base_kg_dir

    stem = COLLECTION_STEM_BY_NAME.get(collection)
    candidate_dirs: list[Path] = []
    if stem is not None:
        candidate_dirs.append(base_path / stem)
    candidate_dirs.append(base_path / collection)

    for candidate in candidate_dirs:
        if not candidate.exists() or not candidate.is_dir():
            continue
        if (candidate / "entities.jsonl").exists():
            return str(candidate)

    return base_kg_dir


def _sample_tasks(
    tasks: list[dict[str, object]],
    *,
    max_tasks: int,
    sample_mode: str,
    seed: int,
) -> list[dict[str, object]]:
    if max_tasks <= 0 or len(tasks) <= max_tasks:
        return list(tasks)

    if sample_mode == "head":
        return tasks[:max_tasks]
    if sample_mode != "stratified":
        raise ValueError("--sample-mode must be one of: head, stratified")

    grouped: dict[str, dict[str, list[dict[str, object]]]] = {}
    for task in tasks:
        collection = _extract_collection(task)
        key = collection if collection is not None else "unknown"
        conversation_id = _extract_conversation_id(task)
        conversation_key = conversation_id if conversation_id is not None else "unknown"
        grouped.setdefault(key, {}).setdefault(conversation_key, []).append(task)

    rng = random.Random(seed)

    ordered_keys = sorted(grouped.keys())
    conversations_by_collection: dict[str, list[str]] = {}
    cursor_by_collection: dict[str, int] = {}
    for collection_key in ordered_keys:
        conversation_map = grouped[collection_key]
        conversation_keys = list(conversation_map.keys())
        rng.shuffle(conversation_keys)
        conversations_by_collection[collection_key] = conversation_keys
        cursor_by_collection[collection_key] = 0

    sampled: list[dict[str, object]] = []

    while len(sampled) < max_tasks:
        advanced = False
        for collection_key in ordered_keys:
            conversation_keys = conversations_by_collection[collection_key]
            if not conversation_keys:
                continue

            start = cursor_by_collection[collection_key]
            consumed = False
            for offset in range(len(conversation_keys)):
                idx = (start + offset) % len(conversation_keys)
                conversation_key = conversation_keys[idx]
                queue = grouped[collection_key][conversation_key]
                if not queue:
                    continue
                sampled.append(queue.pop(0))
                cursor_by_collection[collection_key] = (idx + 1) % len(conversation_keys)
                advanced = True
                consumed = True
                break

            if not consumed:
                continue
            if len(sampled) >= max_tasks:
                break
        if not advanced:
            break

    return sampled


def _filter_tasks_by_ids(
    tasks: list[dict[str, object]],
    selected_task_ids: list[str],
) -> list[dict[str, object]]:
    if not selected_task_ids:
        return []

    order = {task_id: idx for idx, task_id in enumerate(selected_task_ids)}
    selected: list[tuple[int, dict[str, object]]] = []
    for task in tasks:
        task_id = _extract_task_id(task)
        if task_id is None:
            continue
        rank = order.get(task_id)
        if rank is None:
            continue
        selected.append((rank, task))

    selected.sort(key=lambda item: item[0])
    return [task for _, task in selected]


def _build_runner(*, include_answering: bool, provider: str, model: str) -> PipelineRunner:
    steps = [
        StandardizationStep(provider=provider, model=model),
        SubgraphRetrievalStep(),
    ]
    if include_answering:
        steps.append(AnsweringStep(provider=provider, model=model))
    return PipelineRunner(steps=steps)


def _run_tasks(
    tasks: list[dict[str, object]],
    *,
    kg_dir: str,
    mtrag_root: Path | None,
    include_answering: bool,
    provider: str,
    model: str,
    top_k: int,
    debug_task_id: str | None,
    corpus_context_cache: dict[tuple[str, str], list[dict[str, object]]] | None,
    progress_every: int,
    phase_label: str,
    retrieval_strategy: str,
) -> list[dict[str, object]]:
    runner = _build_runner(include_answering=include_answering, provider=provider, model=model)
    retrieval_step = runner.steps[1]
    if isinstance(retrieval_step, SubgraphRetrievalStep):
        retrieval_step.top_k = top_k

    records: list[dict[str, object]] = []
    total_tasks = len(tasks)
    started_at = time.time()

    for index, task in enumerate(tasks, start=1):
        question = _extract_question(task)
        collection = _extract_collection(task)
        resolved_kg_dir = _resolve_kg_dir_for_collection(kg_dir, collection)
        conversation_id = task.get("conversation_id")
        task_id = task.get("task_id")
        context = PipelineContext(
            raw_question=question,
            kg_dir=resolved_kg_dir,
            conversation_id=conversation_id if isinstance(conversation_id, str) else None,
            task_id=task_id if isinstance(task_id, str) else None,
            collection=collection,
            conversation_messages=_extract_conversation(task),
        )
        context = runner.run(context)
        graph_contexts = map_subgraph_to_contexts(
            context.subgraph,
            context.provenance,
            context.triples,
            top_k=top_k,
        )
        contexts = graph_contexts

        query_for_corpus = context.metadata.get("retrieval_query_used")
        if not isinstance(query_for_corpus, str) or not query_for_corpus.strip():
            query_for_corpus = context.normalized_question or question

        query_variants: list[str] = []
        for candidate in [
            context.normalized_question,
            context.raw_question,
            context.metadata.get("standalone_rewrite"),
            context.metadata.get("retrieval_query"),
        ]:
            if not isinstance(candidate, str):
                continue
            value = candidate.strip()
            if not value:
                continue
            if value == query_for_corpus or value in query_variants:
                continue
            query_variants.append(value)

        corpus_contexts: list[dict[str, object]] = []
        if (
            retrieval_strategy in {"hybrid", "corpus"}
            and mtrag_root is not None
            and collection is not None
        ):
            corpus_contexts = _retrieve_contexts_from_corpus(
                mtrag_root=mtrag_root,
                collection=collection,
                query=query_for_corpus,
                top_k=top_k,
                query_variants=query_variants,
                cache=corpus_context_cache,
            )

        if retrieval_strategy == "graph":
            contexts = graph_contexts
        elif retrieval_strategy == "corpus":
            contexts = corpus_contexts or graph_contexts
        else:
            contexts = _merge_contexts(graph_contexts, corpus_contexts, top_k=top_k)

        task_id = task.get("task_id") if isinstance(task, dict) else None
        if debug_task_id and task_id == debug_task_id:
            print("[evaluation debug] task_id:", task_id)
            print("[evaluation debug] question:", question)
            print("[evaluation debug] collection:", collection)
            print("[evaluation debug] kg_dir:", resolved_kg_dir)
            print("[evaluation debug] retrieval_strategy:", retrieval_strategy)
            print("[evaluation debug] normalized_question:", context.normalized_question)
            print("[evaluation debug] query_for_corpus:", query_for_corpus)
            print("[evaluation debug] query_variants:", query_variants)
            print("[evaluation debug] graph_contexts:", graph_contexts[:5])
            print("[evaluation debug] corpus_contexts:", corpus_contexts[:5])
            print("[evaluation debug] subgraph facts:", len(context.subgraph.facts))
            for fact in context.subgraph.facts[:5]:
                print(
                    "[evaluation debug] fact:",
                    {
                        "triple_id": fact.triple_id,
                        "score": fact.score,
                        "provenance_id": fact.provenance_id,
                    },
                )
            provenance_map = {record.provenance_id: record for record in context.provenance}
            for fact in context.subgraph.facts[:5]:
                prov_id = fact.provenance_id
                record = provenance_map.get(prov_id) if prov_id else None
                if record is None:
                    continue
                print(
                    "[evaluation debug] provenance:",
                    {
                        "provenance_id": record.provenance_id,
                        "passage_id": record.passage_id,
                        "doc_id": record.doc_id,
                        "source_path": record.source_path,
                    },
                )
            print("[evaluation debug] contexts:", contexts[:5])

        if include_answering:
            answer_text = context.answer_result.answer or ""
            records.append(build_generation_record(task, contexts, answer_text))
        else:
            records.append(build_retrieval_record(task, contexts))

        if progress_every > 0 and (index % progress_every == 0 or index == total_tasks):
            elapsed = time.time() - started_at
            average = elapsed / index if index > 0 else 0.0
            remaining = max(total_tasks - index, 0)
            eta = average * remaining
            print(
                "[evaluation progress]"
                f" phase={phase_label} processed={index}/{total_tasks}"
                f" elapsed={_format_duration(elapsed)}"
                f" eta={_format_duration(eta)}"
            )

    return records


def _run_mtrag_eval(
    *,
    mtrag_root: Path,
    retrieval_predictions: Path,
    generation_predictions: Path | None,
    output_dir: Path,
) -> None:
    _ensure_mtrag_retrieval_layout(mtrag_root)

    retrieval_out = output_dir / "retrieval_eval.jsonl"
    subprocess.run(
        [
            "python",
            str(mtrag_root / "scripts" / "evaluation" / "run_retrieval_eval.py"),
            "--input_file",
            str(retrieval_predictions),
            "--output_file",
            str(retrieval_out),
        ],
        check=True,
    )

    if generation_predictions is None:
        return

    generation_out = output_dir / "generation_eval.jsonl"
    subprocess.run(
        [
            "python",
            str(mtrag_root / "scripts" / "evaluation" / "run_generation_eval.py"),
            "-i",
            str(generation_predictions),
            "-o",
            str(generation_out),
            "--provider",
            "hf",
            "--judge_model",
            "ibm-granite/granite-3.3-8b-instruct",
        ],
        check=True,
    )


def _ensure_symlink_or_copy_dir(*, source: Path, destination: Path) -> None:
    source = source.resolve()

    if destination.is_symlink() and not destination.exists():
        destination.unlink()

    if destination.exists():
        return
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Expected source directory missing: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source, destination, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass

    shutil.copytree(source, destination)


def _ensure_mtrag_retrieval_layout(mtrag_root: Path) -> None:
    """Create compatibility paths expected by upstream retrieval script."""

    root = mtrag_root.resolve()
    mtrag_human = root / "mtrag-human"
    human_dir = root / "human"

    if not human_dir.exists():
        _ensure_symlink_or_copy_dir(source=mtrag_human, destination=human_dir)

    retrieval_tasks = human_dir / "retrieval_tasks"
    retrieval_tasks_convid = human_dir / "retrieval_tasks_convid"
    if not retrieval_tasks_convid.exists():
        _ensure_symlink_or_copy_dir(source=retrieval_tasks, destination=retrieval_tasks_convid)


def run_evaluation() -> None:
    parser = argparse.ArgumentParser(description="Run minimal MTRAG evaluation adapter.")
    parser.add_argument("--mtrag-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval_out"))
    parser.add_argument("--kg-dir", type=str, required=True)
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--retrieval-input", type=Path, default=None)
    parser.add_argument("--generation-input", type=Path, default=None)
    parser.add_argument("--debug-task-id", type=str, default=None)
    parser.add_argument("--sample-mode", type=str, default="head")
    parser.add_argument("--sample-preset", type=str, default="none")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--retrieval-strategy", type=str, default="hybrid")
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--run-eval", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > 10:
        raise ValueError("--top-k must be in range [1, 10] for MT-RAG format compatibility.")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be >= 0")

    if args.sample_preset not in {"none", *SAMPLE_PRESET_TASKS.keys()}:
        raise ValueError("--sample-preset must be one of: none, smoke, dev, stable")
    if args.retrieval_strategy not in {"hybrid", "graph", "corpus"}:
        raise ValueError("--retrieval-strategy must be one of: hybrid, graph, corpus")

    max_tasks = args.max_tasks
    if args.sample_preset != "none" and max_tasks <= 0:
        max_tasks = SAMPLE_PRESET_TASKS[args.sample_preset]

    default_rag = args.mtrag_root / "mtrag-human" / "generation_tasks" / "RAG.jsonl"
    retrieval_input = args.retrieval_input or default_rag
    generation_input = args.generation_input or default_rag
    retrieval_tasks = _sample_tasks(
        _load_jsonl(retrieval_input),
        max_tasks=max_tasks,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_predictions = args.output_dir / "retrieval_predictions.jsonl"
    corpus_context_cache: dict[tuple[str, str], list[dict[str, object]]] = {}

    retrieval_records = _run_tasks(
        retrieval_tasks,
        kg_dir=args.kg_dir,
        mtrag_root=args.mtrag_root,
        include_answering=False,
        provider=args.provider,
        model=args.model,
        top_k=args.top_k,
        debug_task_id=args.debug_task_id,
        corpus_context_cache=corpus_context_cache,
        progress_every=args.progress_every,
        phase_label="retrieval",
        retrieval_strategy=args.retrieval_strategy,
    )
    _write_jsonl(retrieval_predictions, retrieval_records)
    retrieval_checker_input = _write_checker_input_if_sliced(
        output_dir=args.output_dir,
        file_name="retrieval_input_sliced.jsonl",
        input_path=retrieval_input,
        tasks=retrieval_tasks,
        max_tasks=max_tasks,
    )

    selected_task_ids = [
        task_id for task in retrieval_tasks if (task_id := _extract_task_id(task)) is not None
    ]

    generation_predictions: Path | None = None
    if not args.skip_generation:
        generation_tasks = _filter_tasks_by_ids(_load_jsonl(generation_input), selected_task_ids)
        if not generation_tasks:
            generation_tasks = _sample_tasks(
                _load_jsonl(generation_input),
                max_tasks=max_tasks,
                sample_mode=args.sample_mode,
                seed=args.seed,
            )
        generation_predictions_path = args.output_dir / "generation_predictions.jsonl"
        generation_records = _run_tasks(
            generation_tasks,
            kg_dir=args.kg_dir,
            mtrag_root=args.mtrag_root,
            include_answering=True,
            provider=args.provider,
            model=args.model,
            top_k=args.top_k,
            debug_task_id=args.debug_task_id,
            corpus_context_cache=corpus_context_cache,
            progress_every=args.progress_every,
            phase_label="generation",
            retrieval_strategy=args.retrieval_strategy,
        )
        _write_jsonl(generation_predictions_path, generation_records)
        generation_predictions = generation_predictions_path
        generation_checker_input = _write_checker_input_if_sliced(
            output_dir=args.output_dir,
            file_name="generation_input_sliced.jsonl",
            input_path=generation_input,
            tasks=generation_tasks,
            max_tasks=max_tasks,
        )
    else:
        generation_checker_input = generation_input

    format_checker = args.mtrag_root / "scripts" / "evaluation" / "format_checker.py"
    subprocess.run(
        [
            "python",
            str(format_checker),
            "--input_file",
            str(retrieval_checker_input),
            "--prediction_file",
            str(retrieval_predictions),
            "--mode",
            "retrieval_taska",
        ],
        check=True,
    )

    if generation_predictions is not None:
        subprocess.run(
            [
                "python",
                str(format_checker),
                "--input_file",
                str(generation_checker_input),
                "--prediction_file",
                str(generation_predictions),
                "--mode",
                "generation_taskb",
            ],
            check=True,
        )

        subprocess.run(
            [
                "python",
                str(format_checker),
                "--input_file",
                str(generation_checker_input),
                "--prediction_file",
                str(generation_predictions),
                "--mode",
                "rag_taskc",
            ],
            check=True,
        )

    if args.run_eval:
        _run_mtrag_eval(
            mtrag_root=args.mtrag_root,
            retrieval_predictions=retrieval_predictions,
            generation_predictions=None,
            output_dir=args.output_dir,
        )

    if args.notify:
        _notify_send("Done", "Command finished successfully")


if __name__ == "__main__":
    run_evaluation()
