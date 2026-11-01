"""Hardware measurements: VRAM sampling and the speed profile.

Quality and speed are reported separately: nothing here contributes points.
Every number is recorded with the conditions it was taken under, because the
16 GB admission argument only holds if the peak is measured during the real
workload (a filled context *and* generation), not just after load.
"""

from __future__ import annotations

import ipaddress
import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from .config import RunConfig
from .providers import Provider

_SMI = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")


def gpu_inventory() -> list[dict[str, Any]]:
    """Name, total VRAM and driver for every visible GPU (``[]`` when none)."""
    if not _SMI:
        return []
    try:
        out = subprocess.run(
            [
                _SMI,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in (out.stdout or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "vram_total_mib": int(float(parts[1])), "driver": parts[2]})
    return gpus


def used_mib() -> int | None:
    """Currently used VRAM of GPU 0, in MiB."""
    if not _SMI:
        return None
    try:
        out = subprocess.run(
            [_SMI, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        return int(float((out.stdout or "0").strip().splitlines()[0]))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


@dataclass
class VramSampler:
    """Background VRAM sampler. Reports idle, peak and the sample count."""

    interval_s: float = 0.2
    samples: list[int] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="vram-sampler", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            value = used_mib()
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def idle_mib(self) -> int | None:
        return self.samples[0] if self.samples else None

    @property
    def peak_mib(self) -> int | None:
        return max(self.samples) if self.samples else None

    def report(self, base_mib: int | None = None) -> dict[str, Any]:
        base = base_mib if base_mib is not None else self.idle_mib
        return {
            "vram_idle_mib": base,
            "vram_peak_mib": self.peak_mib,
            "vram_delta_mib": (self.peak_mib - base) if (self.peak_mib is not None and base is not None) else None,
            "vram_samples": len(self.samples),
        }


# --------------------------------------------------------------------------- #
# Speed profile
# --------------------------------------------------------------------------- #

FILLER = (
    "The deployment record describes a service instance, its shard assignment, "
    "the node that currently serves it and the region the broker runs in. "
)


def synthetic_prompt(target_tokens: int, chars_per_token: float = 3.2, nonce: str = "") -> str:
    """A neutral prompt of roughly ``target_tokens`` tokens.

    ``nonce`` goes at the very front: runtimes cache prompt prefixes
    (llama.cpp ``--cache-prompt``, most desktop runtimes by default), and a repeated
    would be served from that cache — TTFT near zero and prompt throughput that
    measures nothing. Every measured prompt therefore starts differently.

    The character ratio is only a first guess: a real tokenizer can pack
    repetitive text several times denser, so callers that need an exact size use
    :func:`fit_prompt_to_tokens`.
    """
    target_chars = int(target_tokens * chars_per_token)
    repeats = max(1, target_chars // len(FILLER))
    body = FILLER * repeats
    header = f"[measurement {nonce}]\n\n" if nonce else ""
    return f"{header}{body}\n\nIgnore the text above. Reply with the single word: READY"


def count_prompt_tokens(provider: Provider, text: str) -> int | None:
    """Exact token count when the runtime offers ``/tokenize``, else ``None``."""
    try:
        return provider.tokenize(text)
    except Exception:
        return None


def measure_prompt_tokens(provider: Provider, text: str) -> tuple[int | None, str]:
    """Real input size for ``text``: tokenizer endpoint, else a live request.

    Returns ``(tokens, how)``. The usage of a real request costs one prefill, so
    it is only used when the runtime exposes no ``/tokenize``.
    """
    exact = count_prompt_tokens(provider, text)
    if exact:
        return exact, "endpoint /tokenize"
    try:
        reply = provider.chat(
            [{"role": "user", "content": text}],
            None,
            max_new_tokens=1,
            temperature=0.0,
            top_p=1.0,
            seed=0,
        )
    except Exception as exc:  # pragma: no cover - transport dependent
        return None, f"не удалось измерить: {exc}"
    if not reply.prompt_tokens:
        return None, "runtime не сообщил размер входа"
    return reply.prompt_tokens, "usage реального запроса"


def calibrate_chars_per_token(provider: Provider, sample: str) -> tuple[float | None, str]:
    """Measure chars-per-token for this tokenizer on this style of text.

    Character-based sizing is a guess: the same 90 000 characters can be 16 500
    tokens to one tokenizer and 36 000 to another, and a document sized by the
    wrong constant either tests a smaller context than it claims or overflows the
    window. Returns ``(ratio, how)``.
    """
    if not sample:
        return None, "пустой образец"
    tokens, how = measure_prompt_tokens(provider, sample)
    if not tokens:
        return None, how
    return len(sample) / tokens, how


def fit_prompt_to_tokens(
    provider: Provider,
    target_tokens: int,
    *,
    nonce: str = "",
    tolerance: float = 0.05,
    attempts: int = 3,
    sampler: VramSampler | None = None,
) -> tuple[str, int | None, float]:
    """Build a prompt that really is ``target_tokens`` long, measured.

    Starts from the character estimate, then rescales using the size the
    runtime actually reported. Returns ``(prompt, actual_tokens, ratio)``.
    """
    ratio = 3.2
    prompt = synthetic_prompt(target_tokens, ratio, nonce)
    actual = count_prompt_tokens(provider, prompt)
    for _ in range(max(1, attempts)):
        if actual is not None and abs(actual - target_tokens) <= max(64, target_tokens * tolerance):
            break
        if actual:
            ratio = max(1.0, len(prompt) / actual)
        prompt = synthetic_prompt(target_tokens, ratio, nonce)
        actual = count_prompt_tokens(provider, prompt)
        if actual is None:
            break
    return prompt, actual, ratio


@dataclass
class SpeedResult:
    label: str
    runs: int
    ttft_s: float | None
    prompt_tokens: float
    completion_tokens: float | None
    prompt_tok_s: float | None
    generation_tok_s: float | None
    total_s: float
    vram_peak_mib: int | None = None
    spread: dict[str, float | None] = field(default_factory=dict)
    sized: str = "estimate"
    overhead_s: float | None = None
    measured_on: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "runs": self.runs,
            "ttft_s": self.ttft_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prompt_tok_s": self.prompt_tok_s,
            "generation_tok_s": self.generation_tok_s,
            "total_s": self.total_s,
            "vram_peak_mib": self.vram_peak_mib,
            "spread": self.spread,
            "sized": self.sized,
            "overhead_s": self.overhead_s,
            "measured_on": self.measured_on,
        }


# localhost and 0.0.0.0 match sandbox_run.LOOPBACK_HOSTS; ipaddress covers 127/8 and ::1.
_LOOPBACK_NAMES = frozenset({"localhost", "0.0.0.0"})
_SPEED_TOLERANCE = 0.01


def _endpoint_url(provider: Provider, cfg: RunConfig) -> str:
    return str(getattr(provider, "base_url", None) or cfg.base_url or "")


def _is_loopback(url: str) -> bool:
    host = (urlparse(url).hostname or "").strip().lower()
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _vram_measured_on(url: str) -> str:
    """nvidia-smi runs here. A remote endpoint's VRAM is not this sample."""
    return "endpoint" if _is_loopback(url) else "harness_host"


def _overhead_ttft(provider: Provider, cfg: RunConfig) -> float | None:
    """TTFT of an empty prompt: queue and RTT, not prefill."""
    try:
        reply = provider.chat(
            [{"role": "user", "content": ""}],
            None,
            max_new_tokens=1,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            seed=0,
        )
    except Exception:
        return None
    ttft = reply.ttft_s
    return float(ttft) if ttft is not None else None


def _fit_speed_prompt(
    provider: Provider, tokens: int, nonce: str, *, tokenizer: bool
) -> tuple[str, str, int | None]:
    """Fit ``long_<N>`` (and the short twin) to the tokenizer, else estimate."""
    if not tokenizer:
        return synthetic_prompt(tokens, nonce=nonce), "estimate", None
    prompt, actual, _ratio = fit_prompt_to_tokens(
        provider, tokens, nonce=nonce, tolerance=_SPEED_TOLERANCE, attempts=8
    )
    limit = max(1.0, tokens * _SPEED_TOLERANCE)
    # ``fit_prompt_to_tokens`` may stop 64 tokens off. A speed claim is ±1%.
    for _ in range(6):
        if actual is None:
            return synthetic_prompt(tokens, nonce=nonce), "estimate", None
        if abs(actual - tokens) <= limit:
            return prompt, "tokenizer", actual
        ratio = max(1.0, len(prompt) / actual)
        prompt = synthetic_prompt(tokens, ratio, nonce)
        actual = count_prompt_tokens(provider, prompt)
    if actual is None:
        return synthetic_prompt(tokens, nonce=nonce), "estimate", None
    return prompt, "tokenizer", actual


def _prefill_tok_s(tokens: float, ttft: float | None, overhead: float | None) -> float | None:
    if ttft is None or overhead is None or ttft <= 0:
        return None
    window = ttft - overhead
    if window <= 1e-6:
        return None
    return tokens / window


def _measure_once(
    provider: Provider,
    prompt: str,
    cfg: RunConfig,
    sampler: VramSampler | None,
    *,
    overhead_s: float | None,
    known_prompt_tokens: int | None,
):
    sampler = sampler or VramSampler()
    sampler.start()
    reply = provider.chat(
        [{"role": "user", "content": prompt}],
        None,
        max_new_tokens=cfg.max_new_tokens if cfg.max_new_tokens <= 256 else 128,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        seed=0,
    )
    sampler.stop()
    usage_prompt = reply.prompt_tokens
    usage_completion = reply.completion_tokens
    if usage_prompt is not None:
        prompt_tokens = float(usage_prompt)
    elif known_prompt_tokens is not None:
        prompt_tokens = float(known_prompt_tokens)
    else:
        from .providers import estimate_tokens

        prompt_tokens = float(estimate_tokens(prompt))
    ttft = reply.ttft_s
    total = reply.total_s
    if usage_completion is None:
        generation = None
        completion: float | None = None
    else:
        generation_window = max(total - (ttft or 0.0), 1e-6) if total else None
        generation = (float(usage_completion) / generation_window) if generation_window else None
        completion = float(usage_completion)
    return {
        "ttft_s": ttft,
        "total_s": total,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "prompt_tok_s": _prefill_tok_s(prompt_tokens, ttft, overhead_s),
        "generation_tok_s": generation,
        "vram_peak_mib": sampler.peak_mib,
    }


def speed_profile(
    provider: Provider,
    cfg: RunConfig,
    *,
    short_tokens: int = 1024,
    long_tokens: int | None = None,
    repeats: int = 3,
    warmup: int = 1,
) -> list[SpeedResult]:
    """TTFT and throughput on a short and a long prompt.

    The prompt is fitted with ``/tokenize`` when the runtime has one (±1%);
    otherwise the character estimate is used and the row is labeled
    ``sized="estimate"``. Prefill throughput subtracts a measured empty-prompt
    TTFT (queue and RTT), recorded as ``overhead_s``. Generation throughput is
    omitted when ``completion_tokens`` is missing — a length estimate is not a
    speed. Medians over ``repeats`` runs; the min/max spread is reported so a
    throttled or cached run is visible. VRAM is sampled on this host: a remote
    endpoint is labeled ``measured_on="harness_host"``.
    """
    if long_tokens is None:
        long_tokens = max(1024, min(cfg.context_window - cfg.max_new_tokens - 512, 16384))
    profiles = [("short", short_tokens), (f"long_{long_tokens}", long_tokens)]
    tokenizer = count_prompt_tokens(provider, "проверка") is not None
    measured_on = _vram_measured_on(_endpoint_url(provider, cfg))
    results: list[SpeedResult] = []
    for label, tokens in profiles:
        # A fresh nonce per request: no repetition may be served from the
        # prompt cache of the one before it. Fit that nonce, or the counted
        # length is not the length that was sent.
        def one(nonce: str, probe_overhead: float | None) -> dict[str, Any]:
            prompt, sized, counted = _fit_speed_prompt(
                provider, tokens, nonce, tokenizer=tokenizer
            )
            row = _measure_once(
                provider,
                prompt,
                cfg,
                None,
                overhead_s=probe_overhead,
                known_prompt_tokens=counted,
            )
            row["sized"] = sized
            return row

        for index in range(max(0, warmup)):
            one(f"{label}-warmup-{index}", None)
        # After warmup: a cold first token must not be subtracted from a warm run.
        overhead_s = _overhead_ttft(provider, cfg)
        rows = [one(f"{label}-run-{index}", overhead_s) for index in range(max(1, repeats))]
        ttfts = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
        gen = [r["generation_tok_s"] for r in rows if r["generation_tok_s"] is not None]
        pre = [r["prompt_tok_s"] for r in rows if r["prompt_tok_s"] is not None]
        peaks = [r["vram_peak_mib"] for r in rows if r["vram_peak_mib"] is not None]
        completions = [r["completion_tokens"] for r in rows if r["completion_tokens"] is not None]
        sized = "estimate" if any(r["sized"] == "estimate" for r in rows) else "tokenizer"
        results.append(
            SpeedResult(
                label=label,
                runs=len(rows),
                ttft_s=statistics.median(ttfts) if ttfts else None,
                prompt_tokens=statistics.median([r["prompt_tokens"] for r in rows]),
                completion_tokens=statistics.median(completions) if completions else None,
                prompt_tok_s=statistics.median(pre) if pre else None,
                generation_tok_s=statistics.median(gen) if gen else None,
                total_s=statistics.median([r["total_s"] or 0.0 for r in rows]),
                vram_peak_mib=max(peaks) if peaks else None,
                spread={
                    "ttft_min_s": min(ttfts) if ttfts else None,
                    "ttft_max_s": max(ttfts) if ttfts else None,
                    "generation_tok_s_min": min(gen) if gen else None,
                    "generation_tok_s_max": max(gen) if gen else None,
                },
                sized=sized,
                overhead_s=overhead_s,
                measured_on=measured_on,
            )
        )
    return results


def weights_sha256(path: str, chunk: int = 1 << 20) -> str:
    """SHA-256 of a model file — the identity of the exact weights compared."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def wait_for_gpu_idle(target_mib: int, timeout_s: float = 120.0) -> bool:
    """Wait until VRAM usage drops — used between model swaps in a comparison."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        value = used_mib()
        if value is not None and value <= target_mib:
            return True
        time.sleep(1.0)
    return False
