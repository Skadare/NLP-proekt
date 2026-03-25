"""LLM client abstraction placeholder."""

from __future__ import annotations

import os

from graphrag_pipeline.utils.env import load_dotenv, required_key_for_provider


class LLMClient:
    """Minimal LLM client interface for completion calls."""

    def complete(
        self,
        *,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        normalized = provider.lower()
        if normalized != "openai":
            raise ValueError(f"Unsupported provider: {provider}")

        if not model.strip():
            raise ValueError("Model name is required.")

        load_dotenv()
        key_name = required_key_for_provider(provider)
        if key_name is None:
            raise ValueError(f"Unsupported provider: {provider}")

        api_key = os.getenv(key_name)
        if not api_key:
            raise RuntimeError(
                f"Missing API key for provider '{provider}'. Set {key_name} in environment or .env."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI SDK is not installed. Install 'openai' to use provider 'openai'."
            ) from exc

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenAI response missing content.") from exc

        if isinstance(content, str) and content.strip():
            return content.strip()

        raise RuntimeError("OpenAI response missing content.")
