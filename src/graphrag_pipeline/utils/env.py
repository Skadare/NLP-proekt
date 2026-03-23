"""Environment loading helpers."""

from __future__ import annotations

import os
from pathlib import Path


def required_key_for_provider(provider: str) -> str | None:
    normalized = provider.lower()
    if normalized == "openai":
        return "OPENAI_API_KEY"
    if normalized == "deepseek":
        return "DEEPSEEK_API_KEY"
    return None


def parse_dotenv_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs into environment when unset."""
    if not env_path.exists() or not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue

        if len(value) >= 2 and (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]

        os.environ[key] = value


def load_dotenv() -> None:
    """Load `.env` from cwd and repository root if present."""
    repo_root_env = Path(__file__).resolve().parents[3] / ".env"
    cwd_env = Path.cwd() / ".env"

    paths: list[Path] = []
    if cwd_env.exists():
        paths.append(cwd_env)
    if repo_root_env.exists() and repo_root_env != cwd_env:
        paths.append(repo_root_env)

    for env_path in paths:
        parse_dotenv_file(env_path)
