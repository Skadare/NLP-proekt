"""CLI entrypoint for the GraphRAG scaffold."""

from __future__ import annotations

import json
import sys

import typer

from .context import PipelineContext
from .pipeline.runner import PipelineRunner
from .result import build_structured_response
from .steps.kg_gen.command import run_command as run_kg_build_command
from .steps.standardization.step import StandardizationStep
from .steps.evaluation.runner import run_evaluation as run_mtrag_evaluation

app = typer.Typer(help="GraphRAG pipeline CLI")


@app.command("kg-build")
def kg_build(
    input: str = typer.Option(..., "--input", help="Path to text file."),
    kg_root: str = typer.Option("data/kg", "--kg-root", help="Root folder for KG artifacts."),
    kg_name: str | None = typer.Option(None, "--kg-name", help="Optional explicit KG folder name."),
    provider: str = typer.Option(
        "openai", "--provider", help="Model provider, openai or deepseek."
    ),
    model: str = typer.Option("gpt-4o-mini", "--model", help="Model name for kg-gen extraction."),
    chunk_size: int = typer.Option(5000, "--chunk-size", help="Chunk size for text processing."),
    cluster: bool = typer.Option(True, "--cluster/--no-cluster", help="Enable kg-gen clustering."),
) -> None:
    """Build a KG from a text file."""
    try:
        summary = run_kg_build_command(
            input_path=input,
            kg_root=kg_root,
            kg_name=kg_name,
            provider=provider,
            model=model,
            chunk_size=chunk_size,
            cluster=cluster,
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(summary, indent=2))


@app.command("normalize")
def normalize(
    question: str = typer.Option(..., "--question", help="Question to normalize."),
    kg_dir: str = typer.Option(..., "--kg-dir", help="Path to KG artifact directory."),
    provider: str = typer.Option(
        "openai", "--provider", help="Model provider for normalization, openai or deepseek."
    ),
    model: str = typer.Option("gpt-4o-mini", "--model", help="Model name for normalization."),
) -> None:
    """Normalize a question against a KG."""
    context = PipelineContext(raw_question=question, kg_dir=kg_dir)
    step = StandardizationStep(provider=provider, model=model)
    try:
        result = step.run(context)
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    payload = {
        "raw_question": result.raw_question,
        "llm_normalized_question": result.metadata.get("llm_normalized_question"),
        "normalized_question": result.normalized_question,
        "linked_entities": [item.model_dump() for item in result.linked_entities],
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command("retrieve")
def retrieve(
    question: str = typer.Option(..., "--question", help="Question to retrieve for."),
    kg_dir: str = typer.Option(..., "--kg-dir", help="Path to KG artifact directory."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
) -> None:
    """Retrieve a subgraph for a question."""
    context = PipelineContext(raw_question=question, kg_dir=kg_dir)
    if debug:
        context.metadata["debug"] = True
    try:
        runner = PipelineRunner.default()
        context = runner.run(context)
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    provenance_map = {record.provenance_id: record for record in context.provenance}
    provenance_records = []
    for fact in context.subgraph.facts:
        if fact.provenance_id is None:
            continue
        record = provenance_map.get(fact.provenance_id)
        if record is None:
            continue
        provenance_records.append(record.model_dump())

    payload = {
        "raw_question": context.raw_question,
        "normalized_question": context.normalized_question,
        "linked_entities": [item.model_dump() for item in context.linked_entities],
        "subgraph": {
            "node_ids": context.subgraph.node_ids,
            "triple_ids": context.subgraph.triple_ids,
        },
        "facts": [fact.model_dump() for fact in context.subgraph.facts],
        "provenance_records": provenance_records,
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command("answer")
def answer(
    question: str = typer.Option(..., "--question", help="Question to answer."),
    kg_dir: str = typer.Option(..., "--kg-dir", help="Path to KG artifact directory."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
) -> None:
    """Answer a question from a KG."""
    context = PipelineContext(raw_question=question, kg_dir=kg_dir)
    if debug:
        context.metadata["debug"] = True
    try:
        runner = PipelineRunner.default()
        context = runner.run(context)
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    payload = {
        "raw_question": context.raw_question,
        "normalized_question": context.normalized_question,
        "linked_entities": [item.model_dump() for item in context.linked_entities],
        "subgraph": {
            "node_ids": context.subgraph.node_ids,
            "triple_ids": context.subgraph.triple_ids,
        },
        "answer": context.answer_result.answer,
        "reasoning": context.answer_result.reasoning,
        "evidence_ids": context.answer_result.evidence_ids,
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command("evaluate")
def evaluate(
    dataset: str = typer.Option("mtrag", "--dataset", help="Benchmark name."),
    mtrag_root: str | None = typer.Option(
        None, "--mtrag-root", help="Path to mt-rag-benchmark repository."
    ),
    kg_dir: str | None = typer.Option(None, "--kg-dir", help="Path to KG artifact directory."),
    output_dir: str = typer.Option("data/eval_out", "--output-dir", help="Output directory."),
    provider: str = typer.Option("openai", "--provider", help="Model provider."),
    model: str = typer.Option("gpt-4o-mini", "--model", help="Model name."),
    top_k: int = typer.Option(10, "--top-k", help="Number of top facts to return."),
    max_tasks: int = typer.Option(0, "--max-tasks", help="Limit number of tasks."),
    skip_generation: bool = typer.Option(False, "--skip-generation", help="Skip generation."),
    retrieval_input: str | None = typer.Option(
        None, "--retrieval-input", help="Custom IBM-schema retrieval input JSONL."
    ),
    generation_input: str | None = typer.Option(
        None, "--generation-input", help="Custom IBM-schema generation input JSONL."
    ),
    debug_task_id: str | None = typer.Option(
        None, "--debug-task-id", help="Log debug output for a single task id."
    ),
    run_eval: bool = typer.Option(False, "--run-eval", help="Run retrieval evaluation."),
) -> None:
    """Run evaluation."""
    if dataset != "mtrag":
        typer.echo(f"Error: Unsupported dataset: {dataset}", err=True)
        raise typer.Exit(code=1)
    if mtrag_root is None or kg_dir is None:
        typer.echo("Error: --mtrag-root and --kg-dir are required for mtrag evaluation.", err=True)
        raise typer.Exit(code=1)

    argv = [
        "mtrag-eval",
        "--mtrag-root",
        mtrag_root,
        "--kg-dir",
        kg_dir,
        "--output-dir",
        output_dir,
        "--provider",
        provider,
        "--model",
        model,
        "--top-k",
        str(top_k),
        "--max-tasks",
        str(max_tasks),
    ]
    if skip_generation:
        argv.append("--skip-generation")
    if retrieval_input is not None:
        argv.extend(["--retrieval-input", retrieval_input])
    if generation_input is not None:
        argv.extend(["--generation-input", generation_input])
    if debug_task_id is not None:
        argv.extend(["--debug-task-id", debug_task_id])
    if run_eval:
        argv.append("--run-eval")

    original_argv = sys.argv
    try:
        sys.argv = argv
        run_mtrag_evaluation()
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        sys.argv = original_argv


@app.command("run")
def run(
    question: str = typer.Option(..., "--question", help="Question to process."),
    kg_dir: str = typer.Option(..., "--kg-dir", help="Path to KG artifact directory."),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
) -> None:
    """Run the full interactive pipeline."""
    context = PipelineContext(raw_question=question, kg_dir=kg_dir)
    if debug:
        context.metadata["debug"] = True
    runner = PipelineRunner.default()
    result = runner.run(context)
    typer.echo(json.dumps(build_structured_response(result).model_dump(), indent=2))
