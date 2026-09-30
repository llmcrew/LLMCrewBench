"""Model transports: any OpenAI-compatible endpoint, plus offline fakes.

Targets unsloth-studio (``:8888``), llama.cpp server (``:8080``), vLLM (``:8000``),
Ollama's compatibility endpoint (``:11434``) and ComfyUI-style cloud proxies
unchanged — they all speak ``POST /chat/completions`` with SSE streaming.

Streaming is the default because TTFT is a reported measurement; the reply
object carries the timing even when only the text is used.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from .core import ModelReply

Message = dict[str, Any]


class ProviderError(RuntimeError):
    """Transport or protocol failure — distinct from a wrong model answer."""


class Provider(Protocol):
    name: str

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        stream: bool = True,
        wall_clock: float | None = None,
    ) -> ModelReply: ...

    def tokenize(self, text: str) -> int | None: ...

    def describe(self) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- #
# OpenAI-compatible HTTP
# --------------------------------------------------------------------------- #


@dataclass
class OpenAICompatibleProvider:
    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = "local-model"
    api_key: str = "not-needed"
    timeout: float = 1800.0
    name: str = "openai-compatible"
    extra_body: dict[str, Any] = field(default_factory=dict)
    include_usage: bool = True
    """Ask for token usage inside the stream (``stream_options``).

    Without it most runtimes report no ``usage`` at all while streaming, and the
    context admission check would then be unverifiable. If a server rejects the
    option, the provider drops it once and retries, rather than failing the run.
    """

    _token_cache: dict[str, int] = field(default_factory=dict, repr=False)
    _tokenize_supported: bool | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- transport --------------------------------------------------------- #

    def _url(self, path: str) -> str:
        return self.base_url.rstrip("/") + path

    def _request(self, path: str, payload: dict[str, Any] | None, method: str = "POST"):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(self._url(path), data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.api_key and self.api_key != "not-needed":
            req.add_header("Authorization", f"Bearer {self.api_key}")
        return urllib.request.urlopen(req, timeout=self.timeout)

    def list_models(self) -> list[str]:
        try:
            with self._request("/models", None, method="GET") as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ProviderError(f"{self.base_url} unreachable: {exc}") from exc
        return [m.get("id", "") for m in data.get("data", [])]

    # -- inference --------------------------------------------------------- #

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        stream: bool = True,
        wall_clock: float | None = None,
    ) -> ModelReply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if seed is not None:
            payload["seed"] = seed
        if stream and self.include_usage:
            payload["stream_options"] = {"include_usage": True}
        payload.update(self.extra_body)

        started = time.perf_counter()
        try:
            if stream:
                reply = self._chat_stream(payload, started, wall_clock)
            else:
                reply = self._chat_blocking(payload, started)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:600]
            if self.include_usage and "stream_options" in body:
                # The runtime does not know the option: drop it and retry once.
                self.include_usage = False
                return self.chat(
                    messages,
                    tools,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    stream=stream,
                    wall_clock=wall_clock,
                )
            raise ProviderError(f"HTTP {exc.code} from {self._url('/chat/completions')}: {body}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"{self.base_url} unreachable: {exc}") from exc
        return reply

    def _chat_blocking(self, payload: dict[str, Any], started: float) -> ModelReply:
        with self._request("/chat/completions", payload) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return self._parse_completion(data, time.perf_counter() - started)

    def _chat_stream(
        self,
        payload: dict[str, Any],
        started: float,
        wall_clock: float | None = None,
    ) -> ModelReply:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        ttft: float | None = None
        usage: dict[str, Any] = {}
        finish_reason: str | None = None

        deadline = (started + wall_clock) if wall_clock else None
        with self._request("/chat/completions", payload) as resp:
            for raw_line in resp:
                if deadline and time.perf_counter() > deadline:
                    # A single runaway generation must not hold the GPU for the
                    # whole sweep: the task is reported as unverified instead.
                    raise ProviderError(
                        f"предел времени на запрос {wall_clock:.0f} с исчерпан "
                        f"({len(''.join(text_parts))} символов получено)"
                    )
                line = raw_line.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    event = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta") or choice.get("message") or {}
                    piece = delta.get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - started
                        text_parts.append(piece)
                    think = delta.get("reasoning_content") or delta.get("reasoning")
                    if think:
                        if ttft is None:
                            ttft = time.perf_counter() - started
                        reasoning_parts.append(think)
                    for tc in delta.get("tool_calls") or []:
                        if ttft is None:
                            ttft = time.perf_counter() - started
                        idx = tc.get("index", 0)
                        slot = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]

        total = time.perf_counter() - started
        return self._finish_reply(
            "".join(text_parts),
            "".join(reasoning_parts) or None,
            [tool_calls[i] for i in sorted(tool_calls)],
            usage,
            ttft,
            total,
            finish_reason,
        )

    def _parse_completion(self, data: dict[str, Any], total: float) -> ModelReply:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            tool_calls.append(
                {"id": tc.get("id", ""), "name": fn.get("name", ""), "arguments": fn.get("arguments", "")}
            )
        return self._finish_reply(
            message.get("content") or "",
            message.get("reasoning_content") or message.get("reasoning"),
            tool_calls,
            data.get("usage") or {},
            None,
            total,
            choice.get("finish_reason"),
        )

    def _finish_reply(
        self,
        text: str,
        reasoning: str | None,
        tool_calls: list[dict[str, Any]],
        usage: dict[str, Any],
        ttft: float | None,
        total: float,
        finish_reason: str | None,
    ) -> ModelReply:
        details = usage.get("completion_tokens_details") or {}
        return ModelReply(
            text=text,
            reasoning=reasoning,
            tool_calls=tool_calls,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            ttft_s=ttft,
            total_s=total,
            finish_reason=finish_reason,
            raw={"usage": usage},
        )

    # -- tokenizer --------------------------------------------------------- #

    def tokenize(self, text: str) -> int | None:
        """Exact token count when the runtime exposes it (llama.cpp, vLLM).

        Returns ``None`` when the endpoint is absent, so callers fall back to a
        character-based estimate and record that the count is approximate.
        """
        if not text:
            return 0
        key = f"{len(text)}:{hash(text) & 0xFFFFFFFF}"
        with self._lock:
            if key in self._token_cache:
                return self._token_cache[key]
        if self._tokenize_supported is False:
            return None
        root = self.base_url.rstrip("/")
        root = root[: -len("/v1")] if root.endswith("/v1") else root
        body = {"content": text, "model": self.model}
        for path in ("/tokenize", f"{root}/tokenize"):
            if path in ("/tokenize",) and root == self.base_url.rstrip("/"):
                continue
            try:
                with self._request(path, body) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                tokens = data.get("tokens")
                count = len(tokens) if isinstance(tokens, list) else data.get("count")
                if not isinstance(count, int):
                    raise ValueError(f"unexpected /tokenize payload: {data}")
                with self._lock:
                    self._tokenize_supported = True
                    self._token_cache[key] = count
                return count
            except Exception:
                continue
        with self._lock:
            self._tokenize_supported = False
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "tokenize_endpoint": self._tokenize_supported,
        }


# --------------------------------------------------------------------------- #
# Offline providers (self-check and unit tests)
# --------------------------------------------------------------------------- #


@dataclass
class CallableProvider:
    """A provider backed by a Python function — no network, fully in-process.

    ``fn(messages, tools) -> ModelReply | str``. Used by the self-check that
    proves the generators and evaluators agree, and by the test suite.
    """

    fn: Callable[[list[Message], list[dict[str, Any]] | None], ModelReply | str]
    name: str = "callable"
    calls: list[list[Message]] = field(default_factory=list)

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        stream: bool = True,
        wall_clock: float | None = None,
    ) -> ModelReply:
        self.calls.append([dict(m) for m in messages])
        started = time.perf_counter()
        out = self.fn(messages, tools)
        if isinstance(out, ModelReply):
            return out
        return ModelReply(text=out, total_s=time.perf_counter() - started)

    def tokenize(self, text: str) -> int | None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name}


@dataclass
class ScriptedProvider:
    """Returns queued replies in order — for multi-turn tool-loop tests."""

    replies: list[ModelReply | str]
    name: str = "scripted"
    calls: list[list[Message]] = field(default_factory=list)

    def chat(self, messages, tools=None, **kwargs) -> ModelReply:
        self.calls.append([dict(m) for m in messages])
        if not self.replies:
            return ModelReply(text="")
        out = self.replies.pop(0)
        return out if isinstance(out, ModelReply) else ModelReply(text=out)

    def tokenize(self, text: str) -> int | None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name}


@dataclass
class RecordingProvider:
    """Wraps a provider and records every exchange to disk."""

    inner: Provider
    sink: Callable[[dict[str, Any]], None]
    name: str = "recording"

    def chat(self, messages, tools=None, **kwargs) -> ModelReply:
        reply = self.inner.chat(messages, tools, **kwargs)
        self.sink(
            {
                "messages": messages,
                "tools": tools,
                "text": reply.text,
                "reasoning": reply.reasoning,
                "tool_calls": reply.tool_calls,
                "prompt_tokens": reply.prompt_tokens,
                "completion_tokens": reply.completion_tokens,
                "ttft_s": reply.ttft_s,
                "total_s": reply.total_s,
                "finish_reason": reply.finish_reason,
            }
        )
        return reply

    def tokenize(self, text: str) -> int | None:
        return self.inner.tokenize(text)

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "inner": self.inner.describe()}


def estimate_tokens(text: str) -> int:
    """Fallback size estimate when no tokenizer endpoint is available.

    Deliberately conservative (over-counts slightly) so a context that fits by
    this measure fits in reality.
    """
    return max(1, round(len(text) / 3.2))


def count_tokens(provider: Provider, text: str, cache: dict[str, int] | None = None) -> tuple[int, bool]:
    """``(tokens, exact)`` — exact when the runtime can tokenize."""
    if cache is not None:
        key = f"{len(text)}:{hash(text) & 0xFFFFFFFF}"
        if key in cache:
            return cache[key], cache.get(key + ":exact", True)  # type: ignore[return-value]
    count = None
    try:
        count = provider.tokenize(text)
    except Exception:
        count = None
    if count is None:
        est = estimate_tokens(text)
        if cache is not None:
            cache[f"{len(text)}:{hash(text) & 0xFFFFFFFF}"] = est
            cache[f"{len(text)}:{hash(text) & 0xFFFFFFFF}:exact"] = False  # type: ignore[assignment]
        return est, False
    if cache is not None:
        cache[f"{len(text)}:{hash(text) & 0xFFFFFFFF}"] = count
    return count, True


def messages_token_estimate(messages: Iterable[Message]) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        for tc in m.get("tool_calls") or []:
            total += estimate_tokens(json.dumps(tc, ensure_ascii=False))
    return total
