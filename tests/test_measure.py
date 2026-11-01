"""Behaviour of the hardware measurements: sizing, VRAM sampling, speed profile.

Nothing here scores a model, so the tests pin the numbers the harness reports
about the machine instead: a sampler that never sees the peak, or a speed
profile that reports a median over the wrong runs, would silently invalidate
every memory and speed claim in the report.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import pytest

from dailybench import core, measure
from dailybench.config import RunConfig
from dailybench.providers import CallableProvider


# --------------------------------------------------------------------------- #
# Prompt sizing
# --------------------------------------------------------------------------- #


def test_synthetic_prompt_carries_the_nonce_at_the_front():
    plain = measure.synthetic_prompt(256)
    assert not plain.startswith("[measurement")
    assert plain.endswith("Reply with the single word: READY")

    first = measure.synthetic_prompt(256, nonce="short-run-0")
    second = measure.synthetic_prompt(256, nonce="short-run-1")
    assert first.startswith("[measurement short-run-0]")
    assert first != second, "разный nonce — разный префикс, иначе ответ придёт из кэша промпта"
    assert first == measure.synthetic_prompt(256, nonce="short-run-0"), "функция детерминирована"


def test_synthetic_prompt_grows_with_the_target_and_the_ratio():
    small = measure.synthetic_prompt(64)
    large = measure.synthetic_prompt(1024)
    assert len(large) > len(small)
    assert len(measure.synthetic_prompt(100, chars_per_token=4.0)) > len(
        measure.synthetic_prompt(100, chars_per_token=1.0)
    )


def test_fit_prompt_to_tokens_converges_on_a_fake_tokenizer(monkeypatch):
    """The loop rescales from the reported count, not from an estimate."""
    calls: list[str] = []

    def fake_tokenize(provider, text: str) -> int:
        calls.append(text)
        return max(1, len(text) // 3)  # 3 characters per token, unlike the 3.2 default

    monkeypatch.setattr(measure, "count_prompt_tokens", fake_tokenize)
    provider = CallableProvider(fn=lambda messages, tools: "ok")

    prompt, actual, ratio = measure.fit_prompt_to_tokens(provider, 1000, nonce="fit", tolerance=0.05)

    assert actual is not None
    assert abs(actual - 1000) <= max(64, 1000 * 0.05), f"не сошлось: {actual}"
    assert abs(ratio - 3.0) < 0.5, "коэффициент должен подтянуться к реальному"
    assert len(calls) >= 2, "нужен хотя бы один пересчёт после первой оценки"


def test_fit_prompt_to_tokens_stops_when_the_runtime_cannot_count(monkeypatch):
    monkeypatch.setattr(measure, "count_prompt_tokens", lambda provider, text: None)
    provider = CallableProvider(fn=lambda messages, tools: "ok")
    prompt, actual, ratio = measure.fit_prompt_to_tokens(provider, 512)
    assert actual is None
    assert prompt and ratio == 3.2


def test_measure_prompt_tokens_prefers_the_tokenizer_endpoint():
    provider = CallableProvider(fn=lambda messages, tools: "ok")
    provider.tokenize = lambda text: 42  # type: ignore[method-assign]
    assert measure.measure_prompt_tokens(provider, "текст") == (42, "endpoint /tokenize")


def test_measure_prompt_tokens_falls_back_to_the_request_usage():
    provider = CallableProvider(
        fn=lambda messages, tools: core.ModelReply(text="ok", prompt_tokens=50, completion_tokens=1)
    )
    assert measure.measure_prompt_tokens(provider, "текст") == (50, "usage реального запроса")


def test_measure_prompt_tokens_reports_a_runtime_that_says_nothing():
    provider = CallableProvider(fn=lambda messages, tools: core.ModelReply(text="ok"))
    tokens, how = measure.measure_prompt_tokens(provider, "текст")
    assert tokens is None
    assert how == "runtime не сообщил размер входа"


def test_calibrate_chars_per_token():
    provider = CallableProvider(fn=lambda messages, tools: "ok")
    provider.tokenize = lambda text: 20  # type: ignore[method-assign]
    sample = "x" * 100
    ratio, how = measure.calibrate_chars_per_token(provider, sample)
    assert ratio == 5.0 and how == "endpoint /tokenize"
    assert measure.calibrate_chars_per_token(provider, "") == (None, "пустой образец")


def test_count_prompt_tokens_swallows_a_broken_endpoint():
    class Broken:
        def tokenize(self, text):
            raise RuntimeError("no /tokenize here")

    assert measure.count_prompt_tokens(Broken(), "текст") is None


# --------------------------------------------------------------------------- #
# VRAM sampling
# --------------------------------------------------------------------------- #


def test_vram_sampler_records_idle_peak_and_delta(monkeypatch):
    values = [100, 200, 300, 400]

    def fake_used_mib() -> int:
        return values.pop(0) if values else 400

    monkeypatch.setattr(measure, "used_mib", fake_used_mib)
    sampler = measure.VramSampler(interval_s=0.01)
    sampler.start()
    sampler.start()  # idempotent: a second start must not spawn a second thread
    time.sleep(0.1)
    sampler.stop()

    assert sampler.samples, "сэмплер не успел снять ни одного образца"
    assert sampler.idle_mib == 100
    assert sampler.peak_mib == 400
    report = sampler.report()
    assert report["vram_idle_mib"] == 100
    assert report["vram_peak_mib"] == 400
    assert report["vram_delta_mib"] == 300
    assert report["vram_samples"] == len(sampler.samples)
    assert sampler.report(base_mib=150)["vram_delta_mib"] == 250


def test_vram_sampler_reports_nothing_when_never_started():
    sampler = measure.VramSampler()
    assert sampler.idle_mib is None and sampler.peak_mib is None
    report = sampler.report()
    assert report["vram_peak_mib"] is None and report["vram_delta_mib"] is None
    sampler.stop()  # safe without start


# --------------------------------------------------------------------------- #
# Speed profile
# --------------------------------------------------------------------------- #


def _reply_provider(ttfts: list[float]):
    values = ttfts * 20
    iterator = iter(values)

    def fn(messages, tools):
        # Пустой промпт — замер накладных (очередь/RTT), не точка профиля.
        if messages and messages[0].get("content") == "":
            return core.ModelReply(
                text="ok", prompt_tokens=1, completion_tokens=1, ttft_s=0.0, total_s=0.01
            )
        ttft = next(iterator)
        return core.ModelReply(
            text="ok", prompt_tokens=100, completion_tokens=50, ttft_s=ttft, total_s=ttft + 1.0
        )

    return CallableProvider(fn=fn)


def test_speed_profile_reports_medians_and_spread(monkeypatch):
    monkeypatch.setattr(measure, "used_mib", lambda: 512)
    provider = _reply_provider([0.4, 0.6, 0.5])
    cfg = RunConfig()

    results = measure.speed_profile(provider, cfg, short_tokens=32, long_tokens=32, repeats=3, warmup=0)

    assert [row.label for row in results] == ["short", "long_32"]
    short = results[0]
    assert short.runs == 3
    assert short.ttft_s == pytest.approx(0.5)
    assert short.prompt_tokens == 100 and short.completion_tokens == 50
    assert short.prompt_tok_s == pytest.approx(200.0)
    assert short.generation_tok_s == pytest.approx(50.0)
    assert short.vram_peak_mib == 512
    assert short.spread["ttft_min_s"] == pytest.approx(0.4)
    assert short.spread["ttft_max_s"] == pytest.approx(0.6)
    assert short.spread["generation_tok_s_min"] == pytest.approx(50.0)
    assert short.spread["generation_tok_s_max"] == pytest.approx(50.0)


def test_speed_profile_counts_warmups_separately(monkeypatch):
    monkeypatch.setattr(measure, "used_mib", lambda: 512)
    provider = _reply_provider([1.0])
    measure.speed_profile(provider, RunConfig(), short_tokens=8, long_tokens=8, repeats=1, warmup=2)
    contents = [call[0]["content"] for call in provider.calls]
    assert len(contents) == 2 * (2 + 1 + 1)
    assert contents[2] == "" and contents[6] == "", "накладные TTFT — после прогрева, до зачётного прогона"


def test_speed_profile_estimates_sizes_when_the_runtime_reports_none(monkeypatch):
    monkeypatch.setattr(measure, "used_mib", lambda: 512)
    provider = CallableProvider(
        fn=lambda messages, tools: core.ModelReply(text="готово", total_s=1.0)
    )
    results = measure.speed_profile(provider, RunConfig(), short_tokens=32, long_tokens=32, repeats=1, warmup=0)

    row = results[0]
    assert row.ttft_s is None, "без TTFT медиана не определена (None, не NaN)"
    assert row.prompt_tokens >= 1, "размер входа без токенизатора берётся из оценки"
    assert row.as_dict().get("sized") == "estimate"
    assert row.completion_tokens is None
    assert row.generation_tok_s is None, "без usage скорость генерации не выдумывается по длине текста"


def test_long_prompt_is_fitted_with_the_tokenizer(monkeypatch):
    """long_<N> is the tokenizer's N (±1%), not the 3.2 character guess."""
    monkeypatch.setattr(measure, "used_mib", lambda: 8)
    target = 2000

    def tokenize(text: str) -> int:
        return max(1, len(text) // 4)

    sent: list[str] = []

    def fn(messages, tools):
        text = messages[0]["content"]
        sent.append(text)
        if text == "":
            return core.ModelReply(
                text="ok", prompt_tokens=1, completion_tokens=1, ttft_s=0.05, total_s=0.06
            )
        counted = tokenize(text)
        return core.ModelReply(
            text="ok", prompt_tokens=counted, completion_tokens=16, ttft_s=0.4, total_s=0.9
        )

    provider = CallableProvider(fn=fn)
    provider.tokenize = tokenize  # type: ignore[method-assign]
    results = measure.speed_profile(
        provider,
        RunConfig(base_url="http://127.0.0.1:9/v1"),
        short_tokens=32,
        long_tokens=target,
        repeats=1,
        warmup=0,
    )

    long_prompts = [text for text in sent if text.startswith(f"[measurement long_{target}")]
    assert long_prompts, "длинный промпт не отправлен"
    assert abs(tokenize(long_prompts[0]) - target) <= target * 0.01
    long = results[1].as_dict()
    assert long["label"] == f"long_{target}"
    assert long.get("sized") == "tokenizer"
    assert abs(long["prompt_tokens"] - target) <= target * 0.01


def test_unconverged_speed_prompt_is_labelled_estimate(monkeypatch):
    """A tokenizer that never lands in tolerance is an estimate, not a measurement."""

    class Stuck:
        def chat(self, messages, tools, **kwargs):
            return core.ModelReply(
                text="ok", prompt_tokens=1, completion_tokens=1, ttft_s=0.1, total_s=0.2
            )

    monkeypatch.setattr(measure, "used_mib", lambda: 4)
    monkeypatch.setattr(measure, "fit_prompt_to_tokens", lambda *a, **k: ("prompt", 100, 3.2))
    monkeypatch.setattr(measure, "count_prompt_tokens", lambda provider, text: 100)
    results = measure.speed_profile(
        Stuck(), RunConfig(), short_tokens=32, long_tokens=2048, repeats=1, warmup=0,
    )
    assert results[1].sized == "estimate"




def test_prefill_rate_subtracts_the_empty_prompt_ttft(monkeypatch):
    monkeypatch.setattr(measure, "used_mib", lambda: 4)

    def fn(messages, tools):
        if messages[0]["content"] == "":
            return core.ModelReply(
                text="ok", prompt_tokens=1, completion_tokens=1, ttft_s=0.2, total_s=0.21
            )
        return core.ModelReply(
            text="ok", prompt_tokens=100, completion_tokens=40, ttft_s=1.2, total_s=2.2
        )

    results = measure.speed_profile(
        CallableProvider(fn=fn),
        RunConfig(base_url="http://127.0.0.1:9/v1"),
        short_tokens=16,
        long_tokens=16,
        repeats=1,
        warmup=0,
    )
    short, long = results[0].as_dict(), results[1].as_dict()
    assert short.get("overhead_s") == pytest.approx(0.2)
    assert long.get("overhead_s") == pytest.approx(0.2)
    # 100 / (1.2 − 0.2) = 100; без вычитания было бы ≈ 83.3
    assert short["prompt_tok_s"] == pytest.approx(100.0)
    assert long["prompt_tok_s"] == pytest.approx(100.0)


def test_vram_row_names_the_harness_host_unless_the_endpoint_is_loopback(monkeypatch):
    monkeypatch.setattr(measure, "used_mib", lambda: 4)
    provider = CallableProvider(
        fn=lambda messages, tools: core.ModelReply(
            text="ok", prompt_tokens=8, completion_tokens=2, ttft_s=0.3, total_s=0.5
        )
    )

    def where(base_url: str) -> str | None:
        rows = measure.speed_profile(
            provider,
            RunConfig(base_url=base_url),
            short_tokens=16,
            long_tokens=16,
            repeats=1,
            warmup=0,
        )
        return rows[0].as_dict().get("measured_on")

    assert where("https://gpu.example:8000/v1") == "harness_host"
    assert where("http://127.0.0.1:9/v1") == "endpoint"
    assert where("http://localhost:9/v1") == "endpoint"
    assert where("http://[::1]:9/v1") == "endpoint"

def test_speed_result_round_trips_to_a_dict():
    row = measure.SpeedResult(
        label="short",
        runs=1,
        ttft_s=0.5,
        prompt_tokens=100.0,
        completion_tokens=50.0,
        prompt_tok_s=200.0,
        generation_tok_s=50.0,
        total_s=1.5,
        vram_peak_mib=512,
        spread={"ttft_min_s": 0.5},
    )
    data = row.as_dict()
    assert data["label"] == "short" and data["vram_peak_mib"] == 512 and data["spread"]["ttft_min_s"] == 0.5


# --------------------------------------------------------------------------- #
# Weights identity
# --------------------------------------------------------------------------- #


def test_weights_sha256_matches_hashlib(tmp_path: Path):
    payload = bytes(range(256)) * 512
    path = tmp_path / "weights.gguf"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert measure.weights_sha256(str(path)) == expected
    assert measure.weights_sha256(str(path), chunk=7) == expected, "размер блока не меняет хеш"


# --------------------------------------------------------------------------- #
# nvidia-smi probing
# --------------------------------------------------------------------------- #


def _make_smi(tmp_path: Path, *, inventory: str, used: str) -> Path:
    """A nvidia-smi stand-in.

    Windows cannot CreateProcess a shell script (WinError 193). A ``.cmd``
    also splits ``--query-gpu=memory.used`` on ``=``, so the launcher hands
    ``%*`` to Python, which writes CRLF the way nvidia-smi does.
    """
    if os.name == "nt":
        body = tmp_path / "nvidia-smi-body.py"
        body.write_text(
            "import sys\n"
            f"used = {used!r}\n"
            f"inventory = {inventory!r}\n"
            "arg = sys.argv[1] if len(sys.argv) > 1 else ''\n"
            "data = used if arg == '--query-gpu=memory.used' else inventory\n"
            "sys.stdout.buffer.write((data + '\\r\\n').encode('utf-8'))\n",
            encoding="utf-8",
        )
        script = tmp_path / "nvidia-smi.cmd"
        script.write_text(
            f'@"{sys.executable}" "{body}" %*\r\n',
            encoding="utf-8",
        )
        return script
    script = tmp_path / "nvidia-smi"
    script.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        f'  --query-gpu=memory.used) printf "%s\\r\\n" "{used}" ;;\n'
        f'  *) printf "%s\\r\\n" "{inventory}" ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def test_gpu_inventory_and_used_mib_parse_the_smi_output(tmp_path: Path, monkeypatch):
    smi = _make_smi(tmp_path, inventory="NVIDIA RTX 5070 Ti, 16303, 616.56", used="1234")
    monkeypatch.setattr(measure, "_SMI", str(smi))

    assert measure.gpu_inventory() == [
        {"name": "NVIDIA RTX 5070 Ti", "vram_total_mib": 16303, "driver": "616.56"}
    ]
    assert measure.used_mib() == 1234


def test_gpu_inventory_ignores_incomplete_lines(tmp_path: Path, monkeypatch):
    smi = _make_smi(tmp_path, inventory="only-a-name", used="10")
    monkeypatch.setattr(measure, "_SMI", str(smi))
    assert measure.gpu_inventory() == []


def test_used_mib_returns_none_on_unparseable_output(tmp_path: Path, monkeypatch):
    smi = _make_smi(tmp_path, inventory="gpu, 1, 2", used="N/A")
    monkeypatch.setattr(measure, "_SMI", str(smi))
    assert measure.used_mib() is None


def test_smi_probes_return_nothing_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(measure, "_SMI", None)
    assert measure.gpu_inventory() == []
    assert measure.used_mib() is None


def test_wait_for_gpu_idle_returns_as_soon_as_usage_drops(monkeypatch):
    monkeypatch.setattr(measure, "used_mib", lambda: 10)
    assert measure.wait_for_gpu_idle(100, timeout_s=0.2) is True


def test_wait_for_gpu_idle_times_out_when_usage_never_drops(monkeypatch):
    monkeypatch.setattr(measure, "used_mib", lambda: 900)
    started = time.time()
    assert measure.wait_for_gpu_idle(100, timeout_s=0.05) is False
    assert time.time() - started < 10
