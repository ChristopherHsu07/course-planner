"""A small client for a local Ollama server, and the ``LLM`` interface.

The rest of the agent only depends on ``LLM.chat``, so tests pass in a
scripted fake instead of a real model. ``OllamaClient`` talks to Ollama's
``/api/chat`` over ``httpx`` (already a project dependency).
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence

import httpx

from . import config


class LLMError(RuntimeError):
    """The model server could not be reached or returned an error."""


class LLM(Protocol):
    """Anything that can answer a chat request."""

    def chat(self, messages: Sequence[dict], tools: Optional[list[dict]] = None,
             format: Optional[dict] = None,
             think: Optional[bool] = None) -> dict:
        """Send ``messages`` and return the assistant message dict.

        The message has ``role``, ``content`` and, when the model wants to
        use tools, ``tool_calls``: a list of ``{"function": {"name": str,
        "arguments": dict}}``. ``format`` is a JSON schema that constrains the
        reply (structured output). ``think`` turns reasoning on or off for
        models that support it; ``None`` leaves the model's default.
        """
        ...


class OllamaClient:
    """Chat with a model served by Ollama (``ollama serve``)."""

    def __init__(self, model: str = config.MODEL,
                 host: str = config.OLLAMA_HOST,
                 num_ctx: int = config.NUM_CTX,
                 timeout: float = config.TIMEOUT):
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self._http = httpx.Client(timeout=timeout)

    def check(self) -> None:
        """Raise ``LLMError`` with a fix-it message unless the model is ready."""
        try:
            r = self._http.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(
                f"Cannot reach Ollama at {self.host} ({e.__class__.__name__}).\n"
                "Install it with `brew install ollama` (or from ollama.com), "
                "then start it with `ollama serve`.") from e
        names = {m.get("name", "") for m in r.json().get("models", [])}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names:
            raise LLMError(
                f"Model {self.model!r} is not downloaded. Run "
                f"`ollama pull {self.model}`. Installed: "
                f"{', '.join(sorted(names)) or 'none'}.")

    def chat(self, messages: Sequence[dict], tools: Optional[list[dict]] = None,
             format: Optional[dict] = None,
             think: Optional[bool] = None) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            # Temperature 0: tool use and extraction should be repeatable.
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }
        if tools:
            payload["tools"] = tools
        if format is not None:
            payload["format"] = format
        if think is not None:
            payload["think"] = think
        try:
            r = self._http.post(f"{self.host}/api/chat", json=payload)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LLMError(f"Ollama returned {e.response.status_code}: "
                           f"{e.response.text[:300]}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama request failed: {e!r}") from e
        message = r.json().get("message")
        if not isinstance(message, dict):
            raise LLMError("Ollama reply had no message")
        return message
