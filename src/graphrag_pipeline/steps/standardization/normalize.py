"""Question normalization using LLM providers."""

from __future__ import annotations

import json
import os
from urllib import error, request

from graphrag_pipeline.utils.env import load_dotenv, required_key_for_provider


def _provider_endpoint(provider: str) -> str:
    normalized = provider.lower()
    if normalized == "openai":
        return "https://api.openai.com/v1/chat/completions"
    if normalized == "deepseek":
        return "https://api.deepseek.com/chat/completions"
    raise ValueError(f"Unsupported provider for normalization: {provider}")


def normalize_question(
    question: str,
    *,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
) -> str:
    """Normalize a question with an LLM before alias replacement."""
    if not question.strip():
        raise ValueError("Question is empty; cannot normalize.")

    load_dotenv()
    key_name = required_key_for_provider(provider)
    if key_name is None:
        raise ValueError(f"Unsupported provider for normalization: {provider}")

    api_key = os.getenv(key_name)
    if not api_key:
        raise RuntimeError(
            f"Missing API key for provider '{provider}'. Set {key_name} in environment or .env."
        )

    endpoint = _provider_endpoint(provider)
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You normalize user questions for retrieval. "
                    "Keep meaning identical, keep language, remove minor grammar noise, "
                    "and do not add new facts or entities. "
                    "Return only the normalized question text."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    }

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with request.urlopen(req, timeout=60) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Normalization request failed ({exc.code}): {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Normalization request failed: {exc.reason}") from exc

    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return question

    message = choices[0].get("message")
    if not isinstance(message, dict):
        return question

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    return question
