import sys
import types

import pytest

from graphrag_pipeline.llm.client import LLMClient


def test_openai_complete_success(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    class FakeChoice:
        def __init__(self, content: str) -> None:
            self.message = FakeMessage(content)

    class FakeResponse:
        def __init__(self, content: str) -> None:
            self.choices = [FakeChoice(content)]

    class FakeCompletions:
        def create(self, *, model: str, messages: list[dict[str, str]]):
            calls["model"] = model
            calls["messages"] = messages
            return FakeResponse("ok")

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            calls["api_key"] = api_key
            self.chat = FakeChat()

    fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr("graphrag_pipeline.llm.client.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = LLMClient()
    result = client.complete(
        provider="openai",
        model="gpt-4o-mini",
        system_prompt="system",
        user_prompt="user",
    )

    assert result == "ok"
    assert calls["api_key"] == "test-key"
    assert calls["model"] == "gpt-4o-mini"
    assert calls["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]


def test_openai_missing_api_key(monkeypatch) -> None:
    monkeypatch.setattr("graphrag_pipeline.llm.client.load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = LLMClient()
    with pytest.raises(RuntimeError, match="Missing API key"):
        client.complete(
            provider="openai",
            model="gpt-4o-mini",
            system_prompt="system",
            user_prompt="user",
        )


def test_openai_unsupported_provider() -> None:
    client = LLMClient()
    with pytest.raises(ValueError, match="Unsupported provider"):
        client.complete(
            provider="deepseek",
            model="gpt-4o-mini",
            system_prompt="system",
            user_prompt="user",
        )


def test_openai_empty_model_name(monkeypatch) -> None:
    monkeypatch.setattr("graphrag_pipeline.llm.client.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = LLMClient()
    with pytest.raises(ValueError, match="Model name is required"):
        client.complete(
            provider="openai",
            model="  ",
            system_prompt="system",
            user_prompt="user",
        )
