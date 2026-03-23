"""CLI-facing entrypoint for the KG generation step."""

from __future__ import annotations

from datetime import UTC, datetime
import os
from pathlib import Path

from .extractor import extract_graph_from_text
from .serializer import save_artifacts
from graphrag_pipeline.utils.env import load_dotenv, required_key_for_provider


def _default_kg_name(input_path: str) -> str:
    stem = Path(input_path).stem.replace(" ", "_")
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{timestamp}"


def run_command(
    input_path: str,
    *,
    kg_root: str = "data/kg",
    kg_name: str | None = None,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    chunk_size: int = 5000,
    cluster: bool = True,
) -> dict[str, object]:
    """Build a KG from a text file and persist it as JSON artifacts."""
    load_dotenv()

    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if not input_file.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    text = input_file.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Input file is empty: {input_path}")

    key_name = required_key_for_provider(provider)
    if key_name is not None and not os.getenv(key_name):
        raise RuntimeError(
            f"Missing required API key for provider '{provider}'. "
            f"Set environment variable {key_name} (or add it to .env), "
            "or choose another provider."
        )

    output_name = kg_name or _default_kg_name(input_path=input_path)
    output_dir = Path(kg_root) / output_name

    try:
        artifacts = extract_graph_from_text(
            text,
            source_path=str(input_file),
            provider=provider,
            model=model,
            chunk_size=chunk_size,
            cluster=cluster,
        )
    except Exception as exc:  # pragma: no cover - defensive runtime mapping
        message = str(exc)
        if "api_key client option must be set" in message:
            raise RuntimeError(
                f"Provider '{provider}' requires an API key. "
                "Set the matching environment variable before running kg-build."
            ) from exc
        raise
    file_paths = save_artifacts(str(output_dir), artifacts)

    summary = {
        "output_dir": str(output_dir),
        "entities": len(artifacts.entities),
        "relations": len(artifacts.relations),
        "triples": len(artifacts.triples),
        "provenance": len(artifacts.provenance),
        "artifacts": file_paths,
        "metadata": artifacts.metadata,
    }
    return summary
