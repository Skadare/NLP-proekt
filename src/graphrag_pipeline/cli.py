"""CLI entrypoint for the GraphRAG scaffold."""

from __future__ import annotations

import json

import typer

from .context import PipelineContext
from .pipeline.runner import PipelineRunner
from .result import build_structured_response
from .steps.kg_gen.command import run_command as run_kg_build_command
from .steps.standardization.step import StandardizationStep

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
) -> None:
    """Retrieve a subgraph for a question."""
    raise NotImplementedError(
        f"Subgraph retrieval is not implemented yet for KG directory: {kg_dir}"
    )


@app.command("answer")
def answer(
    question: str = typer.Option(..., "--question", help="Question to answer."),
    kg_dir: str = typer.Option(..., "--kg-dir", help="Path to KG artifact directory."),
) -> None:
    """Answer a question from a KG."""
    raise NotImplementedError(
        f"Answer generation is not implemented yet for KG directory: {kg_dir}"
    )


@app.command("evaluate")
def evaluate(dataset: str = typer.Option("mtrag", "--dataset", help="Benchmark name.")) -> None:
    """Run evaluation."""
    raise NotImplementedError(f"Evaluation is not implemented yet for dataset: {dataset}")


@app.command("run")
def run(
    question: str = typer.Option(..., "--question", help="Question to process."),
    kg_dir: str = typer.Option(..., "--kg-dir", help="Path to KG artifact directory."),
) -> None:
    """Run the full interactive pipeline."""
    context = PipelineContext(raw_question=question, kg_dir=kg_dir)
    runner = PipelineRunner.default()
    result = runner.run(context)
    typer.echo(json.dumps(build_structured_response(result).model_dump(), indent=2))
