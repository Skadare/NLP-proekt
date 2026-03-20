"""CLI entrypoint for the GraphRAG scaffold."""

from __future__ import annotations

import json

import typer

from .context import PipelineContext
from .pipeline.runner import PipelineRunner
from .result import build_structured_response

app = typer.Typer(help="GraphRAG pipeline CLI")


@app.command("kg-build")
def kg_build(input: str = typer.Option(..., "--input", help="Path to text file.")) -> None:
    """Build a KG from a text file."""
    raise NotImplementedError(f"KG build is not implemented yet for input: {input}")


@app.command("normalize")
def normalize(
    question: str = typer.Option(..., "--question", help="Question to normalize."),
    kg_dir: str = typer.Option(..., "--kg-dir", help="Path to KG artifact directory."),
) -> None:
    """Normalize a question against a KG."""
    raise NotImplementedError(
        f"Question normalization is not implemented yet for KG directory: {kg_dir}"
    )


@app.command("retrieve")
def retrieve(
    question: str = typer.Option(..., "--question", help="Question to retrieve for."),
    kg_dir: str = typer.Option(..., "--kg-dir", help="Path to KG artifact directory."),
) -> None:
    """Retrieve a subgraph for a question."""
    raise NotImplementedError(f"Subgraph retrieval is not implemented yet for KG directory: {kg_dir}")


@app.command("answer")
def answer(
    question: str = typer.Option(..., "--question", help="Question to answer."),
    kg_dir: str = typer.Option(..., "--kg-dir", help="Path to KG artifact directory."),
) -> None:
    """Answer a question from a KG."""
    raise NotImplementedError(f"Answer generation is not implemented yet for KG directory: {kg_dir}")


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
