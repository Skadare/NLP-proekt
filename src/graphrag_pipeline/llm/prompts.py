"""Prompt template accessors."""

from __future__ import annotations

from pathlib import Path


def get_prompt(name: str) -> str:
    template_dir = Path(__file__).resolve().parents[1] / "templates"
    template_path = template_dir / f"{name}.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text(encoding="utf-8").strip()
