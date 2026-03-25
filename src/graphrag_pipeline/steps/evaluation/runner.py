"""Minimal evaluation runner for mt-rag-benchmark integration."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from graphrag_pipeline.context import PipelineContext
from graphrag_pipeline.pipeline.runner import PipelineRunner
from graphrag_pipeline.steps.answering.step import AnsweringStep
from graphrag_pipeline.steps.standardization.step import StandardizationStep
from graphrag_pipeline.steps.subgraph_retrieval.step import SubgraphRetrievalStep

from .mtrag_adapter import build_generation_record, build_retrieval_record, map_subgraph_to_contexts


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


def _extract_question(task: dict[str, object]) -> str:
    input_messages = task.get("input", [])
    if isinstance(input_messages, list):
        for item in reversed(input_messages):
            if isinstance(item, dict) and item.get("speaker") == "user":
                text = item.get("text")
                if isinstance(text, str):
                    return text
    raise ValueError("Task is missing a user question in input.")


def _build_runner(*, include_answering: bool, provider: str, model: str) -> PipelineRunner:
    steps = [StandardizationStep(), SubgraphRetrievalStep()]
    if include_answering:
        steps.append(AnsweringStep(provider=provider, model=model))
    return PipelineRunner(steps=steps)


def _run_tasks(
    tasks: list[dict[str, object]],
    *,
    kg_dir: str,
    include_answering: bool,
    provider: str,
    model: str,
    top_k: int,
    debug_task_id: str | None,
) -> list[dict[str, object]]:
    runner = _build_runner(include_answering=include_answering, provider=provider, model=model)
    records: list[dict[str, object]] = []

    for task in tasks:
        question = _extract_question(task)
        context = PipelineContext(raw_question=question, kg_dir=kg_dir)
        context = runner.run(context)
        contexts = map_subgraph_to_contexts(
            context.subgraph,
            context.provenance,
            context.triples,
            top_k=top_k,
        )

        task_id = task.get("task_id") if isinstance(task, dict) else None
        if debug_task_id and task_id == debug_task_id:
            print("[evaluation debug] task_id:", task_id)
            print("[evaluation debug] question:", question)
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

    return records


def _run_mtrag_eval(
    *,
    mtrag_root: Path,
    retrieval_predictions: Path,
    generation_predictions: Path | None,
    output_dir: Path,
) -> None:
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


def run_evaluation() -> None:
    parser = argparse.ArgumentParser(description="Run minimal MTRAG evaluation adapter.")
    parser.add_argument("--mtrag-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval_out"))
    parser.add_argument("--kg-dir", type=str, required=True)
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--retrieval-input", type=Path, default=None)
    parser.add_argument("--generation-input", type=Path, default=None)
    parser.add_argument("--debug-task-id", type=str, default=None)
    parser.add_argument("--run-eval", action="store_true")
    args = parser.parse_args()

    retrieval_input = args.retrieval_input or (
        args.mtrag_root / "scripts" / "evaluation" / "sample_data" / "taskac_input.jsonl"
    )
    generation_input = args.generation_input or (
        args.mtrag_root / "scripts" / "evaluation" / "sample_data" / "taskb_input.jsonl"
    )
    retrieval_tasks = _load_jsonl(retrieval_input)
    if args.max_tasks > 0:
        retrieval_tasks = retrieval_tasks[: args.max_tasks]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_predictions = args.output_dir / "retrieval_predictions.jsonl"

    retrieval_records = _run_tasks(
        retrieval_tasks,
        kg_dir=args.kg_dir,
        include_answering=False,
        provider=args.provider,
        model=args.model,
        top_k=args.top_k,
        debug_task_id=args.debug_task_id,
    )
    _write_jsonl(retrieval_predictions, retrieval_records)

    generation_predictions: Path | None = None
    if not args.skip_generation:
        generation_tasks = _load_jsonl(generation_input)
        if args.max_tasks > 0:
            generation_tasks = generation_tasks[: args.max_tasks]
        generation_predictions_path = args.output_dir / "generation_predictions.jsonl"
        generation_records = _run_tasks(
            generation_tasks,
            kg_dir=args.kg_dir,
            include_answering=True,
            provider=args.provider,
            model=args.model,
            top_k=args.top_k,
            debug_task_id=args.debug_task_id,
        )
        _write_jsonl(generation_predictions_path, generation_records)
        generation_predictions = generation_predictions_path

    format_checker = args.mtrag_root / "scripts" / "evaluation" / "format_checker.py"
    subprocess.run(
        [
            "python",
            str(format_checker),
            "--input_file",
            str(retrieval_input),
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
                str(generation_input),
                "--prediction_file",
                str(generation_predictions),
                "--mode",
                "generation_taskb",
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


if __name__ == "__main__":
    run_evaluation()
