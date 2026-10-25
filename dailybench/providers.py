"""Model transports: any OpenAI-compatible endpoint, plus offline fakes.

Targets unsloth-studio (``:8888``), llama.cpp server (``:8080``), vLLM (``:8000``),
Ollama's compatibility endpoint (``:11434``) and ComfyUI-style cloud proxies
unchanged — they all speak ``POST /chat/completions`` with SSE streaming.

Streaming is the default because TTFT is a reported measurement; the reply
object carries the timing even when only the text is used.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Protocol

from .core import ModelReply

Message = dict[str, Any]


class ProviderError(RuntimeError):
    """Transport or protocol failure — distinct from a wrong model answer.

    ``infra`` is true when the endpoint itself failed (unreachable, timeout,
    5xx/408/429) rather than the model protocol (4xx, broken terminal state).
    The CLI ends the run with a non-zero code only when every chat request
    failed for infrastructure reasons; a protocol failure stays a task error.
    """

    def __init__(self, message: str, *, infra: bool = True) -> None:
        super().__init__(message)
        self.infra = infra


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
        tool_choice: str | None = None,
        contract: "WireContract | None" = None,
    ) -> ModelReply: ...

    def tokenize(self, text: str) -> int | None: ...

    def describe(self) -> dict[str, Any]: ...

    def probe_images(self) -> bool | None:
        """Whether the endpoint accepts an ``image_url`` content part.

        ``True``/``False`` are a measured answer; ``None`` means the probe
        itself could not decide (the transport was down). The runner turns a
        ``False`` into a skipped, not-verified row instead of a fake accuracy.
        """
        ...


#: Fields the harness owns per request. ``fixed`` sends them all and verifies
#: them on the final payload; ``native`` leaves sampling and thinking to the
#: server and keeps only the answer limit, which the window still has to fit.
WIRE_REQUIRED = ("model", "messages", "max_tokens", "stream")
WIRE_FIXED = ("temperature", "top_p")

#: 1×1 transparent PNG: the smallest valid image part for a capability probe.
_PROBE_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@dataclass(frozen=True)
class WireContract:
    """What a request is allowed to decide for itself — and what it must say.

    The declared sampling/reasoning used to live only in the config: the pi
    adapter dropped temperature and top_p, the chat path never sent a reasoning
    budget, and a report could not tell a fixed measurement from the server's
    defaults. The contract is built once per task, sent on the wire, verified
    against the final payload (``extra_body`` included) and recorded next to it.
    """

    mode: str = "fixed"
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    output_limit: int = 4096
    reasoning_budget: int | None = None
    stop: tuple[str, ...] = ()
    template_version: str = ""
    template_sha256: str | None = None
    reasoning_dropped: bool = False
    """The stand refused ``reasoning_budget``; the retry went without it."""

    def payload_fields(self, *, include_limit: bool = True) -> dict[str, Any]:
        """The keys this contract owns, ready to merge into the body.

        ``include_limit=False`` is for pi, which carries the answer limit as
        ``maxTokens`` on the model definition and turns it into ``max_tokens``
        itself.
        """
        fields: dict[str, Any] = {"max_tokens": int(self.output_limit)} if include_limit else {}
        if self.stop:
            fields["stop"] = list(self.stop)
        if self.mode != "fixed":
            return fields
        fields.update({"temperature": float(self.temperature), "top_p": float(self.top_p)})
        if self.seed is not None:
            fields["seed"] = int(self.seed)
        if self.reasoning_budget is not None:
            fields["reasoning_budget"] = int(self.reasoning_budget)
        return fields

    def required(self) -> tuple[str, ...]:
        keys = list(WIRE_REQUIRED)
        if self.mode == "fixed":
            keys.extend(WIRE_FIXED)
            if self.reasoning_budget is not None:
                keys.append("reasoning_budget")
            if self.seed is not None:
                keys.append("seed")
            if self.stop:
                keys.append("stop")
        return tuple(keys)

    def missing(self, payload: dict[str, Any]) -> list[str]:
        return [key for key in self.required() if key not in payload]

    def as_dict(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """The record kept in the transcript: declared, then what really went out."""
        record: dict[str, Any] = {
            "mode": self.mode,
            "output_limit": int(self.output_limit),
            "stop": list(self.stop),
            "template_version": self.template_version,
            "template_sha256": self.template_sha256,
            "reasoning_dropped": self.reasoning_dropped,
            "verified": "payload" if payload is not None else "declared",
        }
        for key in ("temperature", "top_p", "seed", "reasoning_budget"):
            record[key] = payload.get(key) if payload is not None else getattr(self, key)
        return record


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
    _images: bool | None = field(default=None, repr=False)
    _images_probed: bool = field(default=False, repr=False)
    image_probe_note: str = field(default="", repr=False)
    """Why the image probe ended the way it did; travel with the transcript."""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def probe_images(self) -> bool | None:
        """Send one tiny image part once, and cache the answer.

        A text-only endpoint rejects the content array with a 4xx, which is a
        protocol answer (``False``), not an outage. An ``infra`` failure says
        nothing about images, so it stays ``None`` and the task runs normally.
        Image bytes are fixed here rather than taken from a fixture: the probe
        is about the endpoint, and the same 1×1 PNG is sent to every model.
        """
        with self._lock:
            if self._images_probed:
                return self._images
            self._images_probed = True
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Ответь одним словом."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_PROBE_PNG}"},
                },
            ],
        }
        try:
            self.chat(
                [message],
                max_new_tokens=8,
                stream=False,
                wall_clock=min(60.0, self.timeout or 60.0),
            )
        except ProviderError as exc:
            self._images = None if getattr(exc, "infra", True) else False
            self.image_probe_note = str(exc)[:240]
        except Exception as exc:  # pragma: no cover - defensive
            self._images = None
            self.image_probe_note = f"{type(exc).__name__}: {exc}"[:240]
        else:
            self._images = True
            self.image_probe_note = "endpoint принял image_url"
        return self._images

    # -- transport --------------------------------------------------------- #

    def _url(self, path: str) -> str:
        """Absolute URLs pass through: a tokenizer can live at the server root."""
        if path.startswith(("http://", "https://")):
            return path
        return self.base_url.rstrip("/") + path

    @property
    def root_url(self) -> str:
        """The server root — management paths live above ``/v1``.

        llama.cpp serves ``/props`` (and every other management path) at the
        root, so joining it to ``base_url`` would ask ``/v1/props`` and 404.
        """
        root = self.base_url.rstrip("/")
        return root[: -len("/v1")] if root.endswith("/v1") else root

    def props_url(self) -> str:
        return self.root_url + "/props"

    def props(self) -> dict[str, Any] | None:
        """``GET /props`` at the root: what the server actually loaded.

        ``None`` when the runtime does not expose it; the caller then reports
        the window as unverified rather than assuming the declared value.
        """
        try:
            with self._request(self.props_url(), None, method="GET") as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _request(
        self,
        path: str,
        payload: dict[str, Any] | None,
        method: str = "POST",
        *,
        deadline: float | None = None,
        wall_clock: float | None = None,
    ):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(self._url(path), data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.api_key and self.api_key != "not-needed":
            req.add_header("Authorization", f"Bearer {self.api_key}")
        timeout = self.timeout
        if deadline is not None:
            # The socket timeout is the *remaining* budget, not the whole one:
            # connect and the first read share the same absolute deadline.
            timeout = min(timeout, self._left(deadline, wall_clock))
        return urllib.request.urlopen(req, timeout=timeout)

    def _left(self, deadline: float, wall_clock: float | None, received: int = 0) -> float:
        """Remaining seconds before ``deadline``; raises once it has passed."""
        left = deadline - time.monotonic()
        if left <= 0:
            raise ProviderError(
                f"предел времени на запрос {wall_clock or 0:g} с исчерпан "
                f"({received} символов получено)"
            )
        return left

    def _arm(self, resp: Any, deadline: float | None, wall_clock: float | None, received: int = 0) -> None:
        """Re-arm the socket for the rest of the budget before every read.

        ``urlopen(timeout=...)`` only bounds one socket operation, so a server
        that trickles bytes could otherwise stay ahead of it forever. The
        timeout is therefore re-set to the remaining budget on each iteration,
        and the caller checks ``monotonic`` after the read as well.
        """
        if deadline is None:
            return
        left = self._left(deadline, wall_clock, received)
        sock = getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None)
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.settimeout(left)

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
        tool_choice: str | None = None,
        contract: WireContract | None = None,
    ) -> ModelReply:
        contract = contract or WireContract(
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            output_limit=max_new_tokens,
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        payload.update(contract.payload_fields())
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if stream and self.include_usage:
            payload["stream_options"] = {"include_usage": True}
        payload.update(self.extra_body)

        # The final body is the contract: a field that silently failed to reach
        # it (an adapter dropping it, ``extra_body`` overwriting it) turns the
        # declared conditions into an unverified claim, so it is a protocol error
        # rather than a quietly different measurement.
        missing = contract.missing(payload)
        if missing:
            raise ProviderError(
                f"wire payload без полей контракта: {', '.join(missing)}", infra=False
            )

        # Elapsed metrics may use ``perf_counter``; every absolute deadline is
        # in the ``monotonic`` domain, because ``_left``/``_arm`` compare
        # against it. Mixing the two origins made a short ceiling last as long
        # as the platform's clock offset.
        started = time.perf_counter()
        deadline = (time.monotonic() + wall_clock) if wall_clock else None
        try:
            if stream:
                reply = self._chat_stream(payload, started, deadline, wall_clock, contract)
            else:
                reply = self._chat_blocking(payload, started, deadline, wall_clock, contract)
        except urllib.error.HTTPError as exc:
            body = ""
            with contextlib.suppress(Exception):
                body = exc.read().decode("utf-8", "replace")[:600]
            retry: WireContract | None = None
            if self.include_usage and "stream_options" in body:
                # The runtime does not know the option: drop it and retry once.
                # The retry is a *new* request but not a new budget: it gets
                # only the time left on this call's deadline, so a rejected
                # first attempt cannot silently double the task's wall clock.
                self.include_usage = False
                retry = contract
            elif contract.reasoning_budget is not None and "reasoning_budget" in body:
                # A stand without the field keeps its own thinking cap; the
                # transcript records that the fixed budget was refused.
                retry = replace(contract, reasoning_budget=None, reasoning_dropped=True)
            if retry is not None:
                retry_budget = wall_clock
                if deadline is not None:
                    retry_budget = deadline - time.monotonic()
                    if retry_budget <= 0:
                        raise ProviderError(
                            f"предел времени на запрос {wall_clock or 0:g} с исчерпан (повтор)"
                        ) from exc
                return self.chat(
                    messages,
                    tools,
                    stream=stream,
                    wall_clock=retry_budget,
                    tool_choice=tool_choice,
                    contract=retry,
                )
            raise ProviderError(
                f"HTTP {exc.code} from {self._url('/chat/completions')}: {body}",
                infra=exc.code >= 500 or exc.code in (408, 429),
            ) from exc
        except TimeoutError as exc:
            # A socket timeout is the deadline, not a half-answer: it happens
            # when the server trickles slower than the remaining budget.
            raise ProviderError(
                f"предел времени на запрос {wall_clock or 0:g} с исчерпан (сокет)"
            ) from exc
        except (ConnectionError, http.client.HTTPException) as exc:
            raise ProviderError(f"обрыв соединения: {exc.__class__.__name__}: {exc}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise ProviderError(
                    f"предел времени на запрос {wall_clock or 0:g} с исчерпан (сокет)"
                ) from exc
            raise ProviderError(f"{self.base_url} unreachable: {exc}") from exc
        return reply

    def _chat_blocking(
        self,
        payload: dict[str, Any],
        started: float,
        deadline: float | None,
        wall_clock: float | None,
        contract: WireContract,
    ) -> ModelReply:
        chunks: list[bytes] = []
        received = 0
        with self._request(
            "/chat/completions", payload, deadline=deadline, wall_clock=wall_clock
        ) as resp:
            while True:
                # ``read1`` returns after one socket read: ``read(n)`` would
                # loop internally and reset the timeout on every byte, so a
                # one-byte-at-a-time server could outlive the ceiling.
                self._arm(resp, deadline, wall_clock, received)
                piece = resp.read1(65536)
                if not piece:
                    break
                chunks.append(piece)
                received += len(piece)
                if deadline is not None:
                    self._left(deadline, wall_clock, received)
        try:
            data = json.loads(b"".join(chunks).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderError(f"ответ не является JSON: {exc}") from exc
        return self._parse_completion(data, time.perf_counter() - started, contract, payload)

    def _chat_stream(
        self,
        payload: dict[str, Any],
        started: float,
        deadline: float | None,
        wall_clock: float | None,
        contract: WireContract,
    ) -> ModelReply:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        reasoning_key: str | None = None
        tool_calls: dict[int, dict[str, Any]] = {}
        ttft: float | None = None
        usage: dict[str, Any] = {}
        finish_reason: str | None = None
        received = 0

        def _event(raw_line: bytes) -> bool:
            """Consume one SSE line; ``True`` when the stream is complete."""
            nonlocal usage, finish_reason, ttft, reasoning_key
            line = raw_line.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                return False
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                return True
            try:
                event = json.loads(chunk)
            except json.JSONDecodeError:
                # A keep-alive or comment line is not a terminal state.
                return False
            if not isinstance(event, dict):
                return False
            if event.get("error"):
                # An SSE-level error is a terminal event: without this the
                # stream ended as an empty successful reply (R14).
                detail = event["error"]
                if not isinstance(detail, str):
                    detail = detail.get("message") or json.dumps(detail, ensure_ascii=False)
                raise ProviderError(f"SSE error: {str(detail)[:600]}")
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
                    key = "reasoning_content" if "reasoning_content" in delta else "reasoning"
                    think = delta.get(key)
                    if think:
                        if ttft is None:
                            ttft = time.perf_counter() - started
                        reasoning_key = key
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
            return False

        pending = b""
        done = False
        with self._request(
            "/chat/completions", payload, deadline=deadline, wall_clock=wall_clock
        ) as resp:
            while True:
                self._arm(resp, deadline, wall_clock, received)
                # ``read1`` = one socket read, so the timeout is re-armed for
                # every trickled byte instead of being reset inside a buffer.
                piece = resp.read1(65536)
                if not piece:
                    break
                received += len(piece)
                if deadline is not None:
                    self._left(deadline, wall_clock, received)
                pending += piece
                while b"\n" in pending:
                    raw_line, pending = pending.split(b"\n", 1)
                    if _event(raw_line):
                        done = True
                        break
                if done:
                    break
            if not done and pending:
                _event(pending)

        if finish_reason is None:
            # EOF without a terminal marker: the runtime died mid-generation.
            # Accepting it as a reply turned a cut answer into a model verdict.
            received = len("".join(text_parts))
            raise ProviderError(
                f"SSE-поток оборвался без finish_reason ({received} символов получено)"
            )
        total = time.perf_counter() - started
        reasoning = "".join(reasoning_parts) or None
        # The reasoning goes back on the next assistant tool message under the
        # key the stand used: DeepSeek rejects the continuation without it.
        protocol = {reasoning_key: reasoning} if reasoning and reasoning_key else {}
        return self._finish_reply(
            "".join(text_parts),
            reasoning,
            [tool_calls[i] for i in sorted(tool_calls)],
            usage,
            ttft,
            total,
            finish_reason,
            payload,
            contract,
            protocol,
        )

    def _parse_completion(
        self, data: dict[str, Any], total: float, contract: WireContract, payload: dict[str, Any]
    ) -> ModelReply:
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise ProviderError(f"ответ без choices: {json.dumps(data, ensure_ascii=False)[:300]}")
        choice = choices[0]
        if not choice.get("finish_reason"):
            raise ProviderError(f"ответ без finish_reason: {json.dumps(data, ensure_ascii=False)[:300]}")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ProviderError(f"ответ без message: {json.dumps(data, ensure_ascii=False)[:300]}")
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            tool_calls.append(
                {"id": tc.get("id", ""), "name": fn.get("name", ""), "arguments": fn.get("arguments", "")}
            )
        protocol = {
            key: value
            for key, value in message.items()
            if key not in ("role", "content", "tool_calls") and isinstance(value, str) and value
        }
        return self._finish_reply(
            message.get("content") or "",
            message.get("reasoning_content") or message.get("reasoning"),
            tool_calls,
            data.get("usage") or {},
            None,
            total,
            choice.get("finish_reason"),
            payload,
            contract,
            protocol,
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
        payload: dict[str, Any],
        contract: WireContract,
        protocol: dict[str, Any],
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
            raw={
                "usage": usage,
                "wire": contract.as_dict(payload),
                "protocol": protocol,
            },
        )

    # -- tokenizer --------------------------------------------------------- #

    def tokenize(self, text: str) -> int | None:
        """Exact token count when the runtime exposes it (llama.cpp, vLLM).

        Returns ``None`` when the endpoint is absent, so callers fall back to a
        character-based estimate and record that the count is approximate.

        Two body shapes and two locations are tried: vLLM answers ``/tokenize``
        (server root, outside the ``/v1`` prefix) to a ``prompt`` field, while
        llama.cpp's compatibility layer takes ``content``; a stand that serves
        only ``/v1/tokenize`` (a proxy in front of the runtime) is tried too.
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
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        # A proxy can be pickier about the model spelling than the chat path
        # (``MiniMax-M3`` is served, ``minimax-m3`` tokenizes): try both cases.
        names = list(dict.fromkeys((self.model, self.model.lower(), self.model.capitalize())))
        candidates = [
            (f"{root}/tokenize", {"model": name, "prompt": text}) for name in names
        ] + [
            (f"{root}/tokenize", {"model": name, "content": text}) for name in names
        ] + [
            (f"{self.base_url.rstrip('/')}/tokenize", {"model": name, "prompt": text}) for name in names
        ] + [
            (f"{self.base_url.rstrip('/')}/tokenize", {"model": name, "content": text}) for name in names
        ]
        seen: set[tuple[str, str, tuple[str, ...]]] = set()
        for path, body in candidates:
            shape = (path, str(body.get("model")), tuple(sorted(body)))
            if shape in seen:
                continue
            seen.add(shape)
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
    images: bool | None = None
    calls: list[list[Message]] = field(default_factory=list)

    def probe_images(self) -> bool | None:
        return self.images

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
        tool_choice: str | None = None,
        contract: WireContract | None = None,
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
    images: bool | None = None
    calls: list[list[Message]] = field(default_factory=list)

    def probe_images(self) -> bool | None:
        return self.images

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

    def probe_images(self) -> bool | None:
        return self.inner.probe_images()

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "inner": self.inner.describe()}


#: Reserved prompt cost of one presented image. A 768×512 PNG packs into a few
#: hundred vision tokens; the base64 characters are not counted because they are
#: an encoding, not text (treating 200 KB of base64 as tokens crushed the answer
#: budget of every scenario).
IMAGE_TOKEN_ESTIMATE = 800


def estimate_tokens(text: str) -> int:
    """Fallback size estimate when no tokenizer endpoint is available.

    Deliberately conservative (over-counts) so a context that fits by this
    measure fits in reality.

    The divisor is 2.5 rather than the usual 3.2-4: measured against the live
    runtimes on Russian and code prompts the true ratio was 3.04 symbols per
    token, and an estimate *below* the real count makes the answer budget
    overshoot the window (HTTP 400 on the last token). Over-counting is safe:
    it only leaves the answer a little more room.
    """
    return max(1, round(len(text) / 2.5))


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
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    total += estimate_tokens(str(part.get("text") or ""))
                elif part.get("type") == "image_url":
                    total += IMAGE_TOKEN_ESTIMATE
        for tc in m.get("tool_calls") or []:
            total += estimate_tokens(json.dumps(tc, ensure_ascii=False))
    return total
