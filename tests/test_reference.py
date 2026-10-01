"""Reference-configuration behaviour: loading, launch lines and preflight.

The policy is the contract every model is served under, so these tests check
that a written file owns it (an override is honoured), that every runtime
template is actually reproducible, and that preflight calls an unverifiable
item unverified instead of passing it.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest

import httpstub
from dailybench import measure, reference
from dailybench.config import RunConfig
from dailybench.providers import ProviderError


POLICY_OVERRIDE = {
    "temperature": 0.3,
    "top_p": 0.9,
    "answer_budget_default": 1024,
    "answer_budget_artifact": 2048,
    "reasoning_budget": 512,
    "request_wall_clock": 120,
    "max_total_tokens": 2048,
    "max_tool_calls": 5,
    "context_window": 8192,
}


def _write_reference(tmp_path: Path, models: list[dict], policy: dict | None = POLICY_OVERRIDE) -> Path:
    payload: dict = {"schema": "dailybench/reference/1", "models": models}
    if policy is not None:
        payload["policy"] = policy
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _model(base_url: str = "http://127.0.0.1:9/v1", **over) -> dict:
    entry = {
        "id": "test-entry",
        "label": "Test entry",
        "runtime": "llama.cpp",
        "base_url": base_url,
        "model": "test-model",
        "quant": "f16",
        "context_window": 32768,
        "kv_type": "f16",
        "capabilities": {"thinking": True, "tools": True},
    }
    entry.update(over)
    return entry


@contextlib.contextmanager
def stub(**kwargs):
    base_url, server = httpstub.start(**kwargs)
    try:
        yield base_url, server
    finally:
        httpstub.stop(server)


# --------------------------------------------------------------------------- #
# Loading the file
# --------------------------------------------------------------------------- #


def test_model_entries_and_resolve_read_a_written_file(tmp_path: Path):
    path = _write_reference(
        tmp_path,
        [
            _model(id="custom-a", model="a-model", quant="Q4_K_M", weights="/models/a.gguf",
                   capabilities={"thinking": True, "tools": False}),
            _model(id="custom-b", model="b-model"),
        ],
    )
    entries = reference.model_entries(path)
    assert [entry.id for entry in entries] == ["custom-a", "custom-b"]
    assert reference.resolve("custom-a", path).entry["model"] == "a-model"

    with pytest.raises(KeyError):
        reference.resolve("no-such-config", path)


def test_apply_to_run_config_honours_the_file_policy(tmp_path: Path):
    # No context_window on the entry: the policy's window is the only one to use.
    window_from_policy = _model(id="custom-a", model="a-model", quant="Q4_K_M", kv_type="q8_0")
    window_from_policy.pop("context_window")
    path = _write_reference(
        tmp_path,
        [
            window_from_policy,
            _model(id="custom-b", capabilities={"tools": False}),
        ],
    )
    resolved = reference.resolve("custom-a", path)
    cfg = reference.apply_to_run_config(RunConfig(), resolved)

    override = POLICY_OVERRIDE
    assert cfg.reference == "custom-a"
    assert cfg.model == "a-model" and cfg.quant == "Q4_K_M"
    assert cfg.base_url == resolved.entry["base_url"]
    assert cfg.temperature == override["temperature"]
    assert cfg.top_p == override["top_p"]
    assert cfg.context_window == override["context_window"] == cfg.n_ctx
    assert cfg.max_new_tokens == override["answer_budget_default"]
    # Политика применяется; при unbounded_budget лимит равен окну минус резерв,
    # таблица категорий остаётся доступной как запасной режим.
    assert cfg.budget_for("C") == cfg.context_window - cfg.context_reserve
    cfg.unbounded_budget = False
    assert cfg.budget_for("C") == override["answer_budget_artifact"]
    assert cfg.max_tool_calls == override["max_tool_calls"]
    assert cfg.max_total_tokens == override["max_total_tokens"]
    assert cfg.request_wall_clock == override["request_wall_clock"]
    assert cfg.kv_type == "q8_0"
    assert cfg.offload == "none"
    assert "-c 8192" in cfg.launch and "-ctk q8_0" in cfg.launch
    assert cfg.unsupported == ()

    absent = reference.apply_to_run_config(RunConfig(), reference.resolve("custom-b", path))
    assert absent.unsupported == ("D",)


def test_missing_policy_falls_back_to_the_shipped_defaults(tmp_path: Path):
    path = _write_reference(tmp_path, [_model()], policy=None)
    entry = reference.resolve("test-entry", path)
    assert entry.policy == reference.POLICY


def test_save_reference_round_trips(tmp_path: Path):
    target = tmp_path / "nested" / "reference.json"
    assert reference.save_reference(target) == target
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema"] == "dailybench/reference/1"
    assert payload["models"] and payload["policy"] and payload["runtimes"]
    assert [entry.id for entry in reference.model_entries(target)] == [
        entry.id for entry in reference.model_entries()
    ]
    saved = reference.resolve("qwen3.8-27b-gsq-rco-iq3xxs", target)
    shipped = reference.resolve("qwen3.8-27b-gsq-rco-iq3xxs")
    assert saved.entry["model"] == shipped.entry["model"] and saved.entry["quant"] == shipped.entry["quant"]


# --------------------------------------------------------------------------- #
# Launch lines
# --------------------------------------------------------------------------- #


def test_launch_line_substitutes_every_runtime_template():
    for runtime in reference.RUNTIMES:
        entry = reference.ReferenceConfig(
            entry={
                "id": f"launch-{runtime}",
                "runtime_id": runtime,
                "model": "my/model-name",
                "quant": "Q4_K_M",
                "weights": "/models/my.gguf",
                "context_window": 32768,
                "kv_type": "q8_0",
            }
        )
        launch = entry.launch_line()
        assert launch, f"{runtime}: шаблон загрузки не найден"
        assert "<" not in launch, f"{runtime}: плейсхолдер не подставлен"
        assert "32768" in launch, f"{runtime}: в строке нет окна"
        assert "q8_0" in launch, f"{runtime}: в строке нет типа KV"
        assert "my/model-name" in launch

    llama = reference.ReferenceConfig(
        entry={"runtime_id": "llama.cpp", "model": "m", "quant": "Q", "weights": "/w.gguf",
               "context_window": 8192, "kv_type": "q4_0"}
    ).launch_line()
    assert "/w.gguf" in llama and "-n 4096" in llama and "--reasoning-budget 2048" in llama

    vllm = reference.ReferenceConfig(
        entry={"runtime_id": "vllm", "model": "m", "context_window": 4096, "kv_type": "f16"}
    ).launch_line()
    assert vllm.startswith("vllm serve m") and "--max-model-len 4096" in vllm


def test_runtime_id_is_derived_from_the_label_when_not_explicit():
    labelled = reference.ReferenceConfig(entry={"runtime": "llama.cpp форк PrismML (prism-b10709)"})
    assert labelled.runtime_id == "llama.cpp"
    studio = reference.ReferenceConfig(entry={"runtime": "unsloth-studio v0.1.808-beta"})
    assert studio.runtime_id == "unsloth-studio"
    explicit = reference.ReferenceConfig(entry={"runtime": "что-то своё", "runtime_id": "vllm"})
    assert explicit.runtime_id == "vllm"
    unknown = reference.ReferenceConfig(entry={"runtime": "no-such-runtime"})
    assert unknown.runtime_id == "" and unknown.launch_line() == ""


def test_reference_config_properties_and_capabilities():
    entry = reference.ReferenceConfig(entry={"id": "x", "capabilities": {"tools": False}})
    assert entry.id == "x" and entry.label == "x"
    assert entry.runtime == "llama.cpp", "runtime по умолчанию"
    assert entry.tools_capability is False
    assert entry.unsupported_categories() == ("D",)
    assert reference.ReferenceConfig(entry={"capabilities": {"tools": "probe"}}).unsupported_categories() == ()
    assert reference.ReferenceConfig(entry={"capabilities": {"tools": True}}).unsupported_categories() == ()
    assert reference.ReferenceConfig(entry={}).unsupported_categories() == ()


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


@pytest.fixture()
def endpoint():
    base_url, server = httpstub.start(
        tool_call={"name": "search_code", "arguments": {"query": "max_model_len"}}
    )
    yield base_url, server
    httpstub.stop(server)


def _probes(report: reference.PreflightReport) -> dict[str, reference.Probe]:
    return {probe.name: probe for probe in report.probes}


def test_preflight_rejects_an_unreachable_endpoint():
    entry = reference.ReferenceConfig(entry=_model(base_url="http://127.0.0.1:9/v1"))
    report = reference.preflight(entry)
    assert report.verdict == "rejected" and not report.ok
    assert _probes(report)["endpoint"].status == "fail"


def test_preflight_rejects_a_model_missing_from_the_endpoint(endpoint):
    base_url, _ = endpoint
    entry = reference.ReferenceConfig(entry=_model(base_url=base_url, model="ghost-model"))
    report = reference.preflight(entry)
    assert _probes(report)["model_listed"].status == "fail"
    assert report.verdict == "rejected"


def test_preflight_reports_missing_props_as_an_unverified_window(endpoint):
    base_url, _ = endpoint
    entry = reference.ReferenceConfig(entry=_model(base_url=base_url))
    report = reference.preflight(entry)
    names = _probes(report)
    assert names["model_listed"].status == "ok"
    assert names["context_window"].status == "warn", "без /props окно нельзя подтвердить"
    assert "не предоставляет /props" in names["context_window"].detail
    assert names["tool_calling"].status == "ok"
    assert "search_code" in names["tool_calling"].detail
    assert names["reasoning"].status == "ok", "стаб отделяет reasoning от ответа"
    assert names["weights"].status == "warn"
    assert names["full_window"].status == "n/a"
    assert report.verdict == "admitted with unverified items"
    assert report.ok


def test_preflight_reads_props_when_the_runtime_serves_them():
    props = {
        "default_generation_settings": {"n_ctx": 32768},
        "chat_template": "{{ messages }}",
        "chat_template_caps": {"supports_tools": True},
        "model_path": "/models/x.gguf",
        "model_alias": "alias",
        "build_info": "b1",
    }
    with stub(props=props, tool_call={"name": "read_file", "arguments": {"path": "a"}}) as (base_url, _):
        report = reference.preflight(reference.ReferenceConfig(entry=_model(base_url=base_url)))
    names = _probes(report)
    assert names["context_window"].status == "ok"
    assert names["chat_template"].status == "ok"
    assert names["template_capabilities"].status == "ok"
    assert report.metrics["model_path"] == "/models/x.gguf"
    assert report.metrics["model_alias"] == "alias"
    assert report.metrics["build_info"] == "b1"


def test_preflight_fails_when_props_report_a_smaller_window():
    props = {"default_generation_settings": {"n_ctx": 8192}, "chat_template": "t"}
    with stub(props=props) as (base_url, _):
        entry = _model(base_url=base_url, context_window=32768)
        report = reference.preflight(reference.ReferenceConfig(entry=entry))
    assert _probes(report)["context_window"].status == "fail"
    assert report.verdict == "rejected"


def test_preflight_warns_when_the_template_cannot_call_tools():
    props = {"default_generation_settings": {"n_ctx": 32768}, "chat_template": "t",
             "chat_template_caps": {"supports_tools": False}}
    with stub(props=props) as (base_url, _):
        report = reference.preflight(reference.ReferenceConfig(entry=_model(base_url=base_url)))
    assert _probes(report)["template_capabilities"].status == "warn"


def test_preflight_warns_when_props_omit_the_window():
    props = {"chat_template": "t"}
    with stub(props=props) as (base_url, _):
        report = reference.preflight(reference.ReferenceConfig(entry=_model(base_url=base_url)))
    assert _probes(report)["context_window"].status == "warn"
    assert "не сообщил n_ctx" in _probes(report)["context_window"].detail


def test_preflight_marks_a_declared_tool_absence_as_not_applicable(endpoint):
    base_url, _ = endpoint
    entry = _model(base_url=base_url, capabilities={"thinking": True, "tools": False})
    report = reference.preflight(reference.ReferenceConfig(entry=entry))
    assert _probes(report)["tool_calling"].status == "n/a"


def test_preflight_hashes_weights_when_asked(endpoint, tmp_path: Path):
    base_url, _ = endpoint
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"weights")
    entry = _model(base_url=base_url, weights=str(weights))
    report = reference.preflight(reference.ReferenceConfig(entry=entry), hash_weights=True)
    assert _probes(report)["weights"].status == "ok"
    assert report.metrics["weights_sha256"] == measure.weights_sha256(str(weights))

    missing = reference.preflight(
        reference.ReferenceConfig(entry=_model(base_url=base_url, weights=str(tmp_path / "none.gguf")))
    )
    assert _probes(missing)["weights"].status == "warn"


def test_preflight_full_window_probe_against_the_stub(endpoint, monkeypatch):
    base_url, _ = endpoint
    monkeypatch.setattr(measure, "used_mib", lambda: 8000)
    monkeypatch.setattr(
        reference, "gpu_inventory", lambda: [{"name": "RTX", "vram_total_mib": 16303, "driver": "1"}]
    )
    report = reference.preflight(reference.ReferenceConfig(entry=_model(base_url=base_url)), full=True)

    names = _probes(report)
    full = names["full_window"]
    assert full.status == "ok", full.detail
    # window - answer reserve - margin, measured against the runtime tokenizer
    assert report.metrics["prompt_tokens_full_window"] == pytest.approx(28160, rel=0.1)
    assert report.metrics["vram_peak_mib"] == 8000
    assert names["kv_allocation"].status == "warn", "нулевой прирост VRAM — это загрузка, а не запрос"
    assert names["vram_headroom"].status == "ok"
    assert report.ok


class _RejectingProvider:
    """An endpoint whose chat always refuses the input, like an over-long prompt."""

    name = "rejecting"

    def list_models(self) -> list[str]:
        return ["test-model"]

    def chat(self, messages, tools=None, **kwargs):
        raise ProviderError("context length exceeded: input is too long for this window")

    def tokenize(self, text):
        return None

    def describe(self):
        return {"provider": self.name}


def test_preflight_full_window_reports_a_rejected_input():
    entry = reference.ReferenceConfig(entry=_model())
    report = reference.preflight(entry, provider=_RejectingProvider(), full=True)

    names = _probes(report)
    assert names["full_window"].status == "fail"
    assert "runtime отверг вход" in names["full_window"].detail
    assert report.verdict == "rejected"
