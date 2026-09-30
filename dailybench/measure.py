"""Hardware measurements: VRAM sampling and the speed profile.

Quality and speed are reported separately: nothing here contributes points.
Every number is recorded with the conditions it was taken under, because the
16 GB admission argument only holds if the peak is measured during the real
workload (a filled context *and* generation), not just after load.
"""

from __future__ import annotations

import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

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
    ttft_s: float
    prompt_tokens: float
    completion_tokens: float
    prompt_tok_s: float
    generation_tok_s: float
    total_s: float
    vram_peak_mib: int | None = None
    spread: dict[str, float] = field(default_factory=dict)

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
        }


def _measure_once(provider: Provider, prompt: str, cfg: RunConfig, sampler: VramSampler | None):
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
    prompt_tokens = reply.prompt_tokens
    completion_tokens = reply.completion_tokens
    ttft = reply.ttft_s
    total = reply.total_s
    if prompt_tokens is None:
        from .providers import estimate_tokens

        prompt_tokens = estimate_tokens(prompt)
    if completion_tokens is None:
        from .providers import estimate_tokens

        completion_tokens = estimate_tokens(reply.text or "")
    generation_window = max(total - (ttft or 0.0), 1e-6) if total else None
    return {
        "ttft_s": ttft,
        "total_s": total,
        "prompt_tokens": float(prompt_tokens),
        "completion_tokens": float(completion_tokens),
        "prompt_tok_s": (prompt_tokens / ttft) if ttft else None,
        "generation_tok_s": (completion_tokens / generation_window) if generation_window else None,
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

    Prefill throughput uses TTFT (prompt tokens over time-to-first-token);
    generation throughput excludes prefill. Medians over ``repeats`` runs; the
    min/max spread is reported so a throttled or cached run is visible.
    """
    if long_tokens is None:
        long_tokens = max(1024, min(cfg.context_window - cfg.max_new_tokens - 512, 16384))
    profiles = [("short", short_tokens), (f"long_{long_tokens}", long_tokens)]
    results: list[SpeedResult] = []
    for label, tokens in profiles:
        # A fresh nonce per request: no repetition may be served from the
        # prompt cache of the one before it.
        for index in range(max(0, warmup)):
            _measure_once(provider, synthetic_prompt(tokens, nonce=f"{label}-warmup-{index}"), cfg, None)
        rows = [
            _measure_once(provider, synthetic_prompt(tokens, nonce=f"{label}-run-{index}"), cfg, None)
            for index in range(max(1, repeats))
        ]
        ttfts = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
        gen = [r["generation_tok_s"] for r in rows if r["generation_tok_s"] is not None]
        pre = [r["prompt_tok_s"] for r in rows if r["prompt_tok_s"] is not None]
        peaks = [r["vram_peak_mib"] for r in rows if r["vram_peak_mib"] is not None]
        results.append(
            SpeedResult(
                label=label,
                runs=len(rows),
                ttft_s=statistics.median(ttfts) if ttfts else float("nan"),
                prompt_tokens=statistics.median([r["prompt_tokens"] for r in rows]),
                completion_tokens=statistics.median([r["completion_tokens"] for r in rows]),
                prompt_tok_s=statistics.median(pre) if pre else float("nan"),
                generation_tok_s=statistics.median(gen) if gen else float("nan"),
                total_s=statistics.median([r["total_s"] or 0.0 for r in rows]),
                vram_peak_mib=max(peaks) if peaks else None,
                spread={
                    "ttft_min_s": min(ttfts) if ttfts else float("nan"),
                    "ttft_max_s": max(ttfts) if ttfts else float("nan"),
                    "generation_tok_s_min": min(gen) if gen else float("nan"),
                    "generation_tok_s_max": max(gen) if gen else float("nan"),
                },
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
