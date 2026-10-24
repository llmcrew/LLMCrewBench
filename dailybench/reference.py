"""Reference configuration: the invariant policy plus one entry per model.

The suite is only comparable if every model is served under the same contract.
This module holds that contract in one place, as data:

* the **policy** — sampling, budgets, context, offload and measurement rules that
  are identical for every model, with the reason for each;
* the **runtime launch lines** — the exact server command for llama.cpp and the
  exact ``unsloth studio run`` command for the primary runtime, so a configuration
  can be reproduced
  instead of described;
* the **model entries** — one per configuration to compare (model + quant +
  KV type + context), including what must be verified before the run.

``preflight`` turns the entry into evidence: it asks the running endpoint what
it actually loaded, probes tool calling and reasoning, and (optionally) fills the
whole context window while sampling VRAM. Anything it cannot verify is reported
as unverified rather than assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import core
from .config import DEFAULT_CONTEXT_WINDOW, RunConfig
from .measure import VramSampler, gpu_inventory, weights_sha256
from .providers import OpenAICompatibleProvider, ProviderError, Provider
from .tasks.tools import schemas as tool_schemas

DEFAULT_PATH = Path("configs") / "reference.json"

#: Invariants applied to every model. Values live here; prose lives in the
#: specification document (section 9).
POLICY: dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": "derived per task from the master seed (same seed for every model)",
    "system_prompt": "one neutral system message, identical for every model",
    "answer_budget_default": 4096,
    "answer_budget_artifact": 8192,
    "answer_budget_note": ("unbounded by default: the limit is the window minus the prompt; the "
                           "per-category table below is the fallback mode"),
    "reasoning_budget": 2048,
    "reasoning_budget_note": "server-side thinking cap (llama.cpp); vLLM stands keep their own default",
    "reasoning_effort_note": "not forced: the stand decides, the spend is documented instead",
    "request_wall_clock": 1800,
    "request_wall_clock_note": "ceiling per request; a task that hits it is reported as unverified, never as a wrong answer",
    "task_wall_clock": 3600,
    "task_wall_clock_note": ("ceiling per task across all turns and the stream_options retry; "
                             "each request gets no more than the remaining time"),
    "max_total_tokens": 16384,
    "max_tool_calls": 12,
    "context_window": DEFAULT_CONTEXT_WINDOW,
    "prompt_cache": "each measured prompt carries a fresh nonce, so no run is served from another run's prompt cache",
    "offload": "none — weights and KV cache fully on the GPU; partial offload is a different configuration",
    "rope_scaling": "none — no YaRN/linear scaling, the model must be run inside its trained window",
    "context_shift": "disabled — the runtime must reject an over-long prompt rather than silently drop the oldest tokens",
    "speculative_decoding": "disabled",
    "grammar_or_json_schema": "not used",
    "parallel_slots": 1,
    "quantization": "the largest quant that still fits the full 32K window on the GPU without offload",
    "kv_cache": "declared per entry; measured here: q8_0 doubles generation speed at a 32K window, so it is the reference value, not a default",
    "thinking": "enabled where the model has it, with the same token budget as everything else",
    "agent": ("categories C, K and L are solved by the pi coding agent in a working copy: the agent reads files, "
              "edits them and runs their tests, and the artefact is graded from disk. The harness is pinned "
              "(pi 1.0.0, tools read/bash/edit/write, --no-session, --no-context-files, --no-approve, --offline) "
              "and the per-category answer ceiling from this policy is passed to it as the model's max output"),
    "repeat_policy": "one scored pass per task; temperature=0 is not bitwise deterministic, so a re-run disagreement is reported instead of the better attempt",
}

#: Runtime launch references. Flags are the ones the servers actually accept.
RUNTIMES: dict[str, dict[str, str]] = {
    "llama.cpp": {
        "launch": (
            "llama-server -m <weights.gguf> --host 127.0.0.1 --port 8080 \\\n"
            "  -c <window> -ngl all -fa on -ctk <kv> -ctv <kv> \\\n"
            "  -np 1 --no-context-shift --jinja \\\n"
            "  --reasoning auto --reasoning-format deepseek --reasoning-budget <reasoning-budget> \\\n"
            "  --temp 0 --top-p 1 --seed -1 -n <answer-budget> \\\n"
            "  --metrics --props --alias <model-name>"
        ),
        "verify": [
            "GET /props → n_ctx, model_path, chat_template: подтверждает, что загружено именно то, что объявлено",
            "-ngl all вместе с пиком VRAM подтверждает отсутствие CPU offload",
            "--no-context-shift: без него сервер молча вытесняет старые токены вместо отказа",
            "--reasoning-format deepseek: reasoning приходит в reasoning_content и не попадает в content",
            "--reasoning-budget: без него thinking съедает весь бюджет ответа — измерено: C-1 уходила в length с пустым content, с бюджетом 2048 набрала 10/10",
        ],
        "notes": [
            "Флаги сверены с tools/server/README.md актуального llama.cpp (ggml-org/llama.cpp).",
            "Если версия не имеет какого-то флага (например --reasoning-budget), он просто не ставится: harness измеряет результат, а не доверяет конфигу.",
        ],
    },
    "unsloth-studio": {
        "launch": (
            "# предпочтительно: API приложения — тот же путь, что у самого Studio\n"
            "POST /api/inference/load\n"
            '  {"model_path": "<model>", "gguf_variant": "<quant>",\n'
            '   "max_seq_length": <window>, "n_parallel": 1, "gpu_memory_mode": "auto",\n'
            '   "cache_type_kv": "<kv>", "force_reload": true,\n'
            '   "llama_extra_args": ["-fa", "on", "--jinja", "--reasoning-format", "deepseek",\n'
            '                       "--no-context-shift", "--temp", "0", "--top-p", "1"]}\n'
            "\n"
            "# либо одноразовый запуск сервера целиком\n"
            "unsloth studio run --model <model> --gguf-variant <quant> \\\n"
            "  --max-seq-length <window> --gpu-memory-mode auto --parallel 1 \\\n"
            "  -fa on -ctk <kv> -ctv <kv> --jinja --no-context-shift \\\n"
            "  --reasoning-format deepseek --reasoning-budget <reasoning-budget> --temp 0 --top-p 1"
        ),
        "verify": [
            "GET /props → n_ctx должен совпасть с объявленным окном: gpu_memory_mode=auto умеет сам урезать контекст под VRAM",
            "GET /props → chat_template_caps: supports_tools / supports_tool_calls подтверждают разбор вызовов инструментов",
            "POST /api/inference/estimate-memory — оценка памяти до загрузки",
            "--no-context-shift: без него сервер молча вытесняет старые токены вместо отказа",
            "--reasoning-format deepseek: reasoning приходит в reasoning_content и не попадает в content",
            "--reasoning-budget: без него thinking съедает весь бюджет ответа — измерено: C-1 уходила в length с пустым content, с бюджетом 2048 набрала 10/10",
        ],
        "notes": [
            "Измерено на этой машине: `gpu_memory_mode=auto` даёт TTFT 0.98 с против 5.92 с у `manual` на том же окне (префилл в 6 раз быстрее).",
            "Измерено: KV `q8_0` даёт 39-40 tok/s генерации против 20 на `f16` при окне 32768 — тип KV объявляется в записи, а не подразумевается.",
            "Flash attention на генерацию здесь не влияет (39 против 40 tok/s в пределах разброса), но входит в эталонную строку.",
            "`manual` в CLI принимает -ngl только целым числом; в режиме `auto` слои и контекст подбирает сам Studio.",
            "Сервер не предоставляет /tokenize: размер входа берётся из usage реального запроса.",
        ],
    },
    "vllm": {
        "launch": (
            "vllm serve <model> --host 127.0.0.1 --port 8000 \\\n"
            "  --max-model-len <window> --gpu-memory-utilization 0.95 \\\n"
            "  --kv-cache-dtype <kv> --enforce-eager"
        ),
        "verify": [
            "GET /v1/models и размер KV-кэша, напечатанный при старте → проверка бюджета VRAM",
            "POST /tokenize доступен: точный размер входа до запуска сценария",
        ],
        "notes": ["Держать --max-model-len равным context_window, иначе длинный сценарий отвергается молча."],
    },
}

#: One entry per configuration to compare. Copy ``generic-gguf`` for anything
#: not listed; never edit an entry mid-comparison — a change makes it a new
#: configuration and it must be re-run in full.
#:
#: The comparison group is one model family with different weights, served by
#: the same runtime on the same machine: ``UD-Q3_K_XL`` is the baseline that is
#: actually in use, ``GSQ-RCO`` is the quant under test, ``UD-Q4_K_XL`` shows
#: what a larger quant costs in VRAM and speed.
MODELS: list[dict[str, Any]] = [
    {"id": "qwen3.8-27b-ud-iq2-s-128k", "label": "Qwen3.8 27B · UD-IQ2_S · окно 131072", "runtime": "unsloth-studio v0.1.811-beta", "base_url": "http://127.0.0.1:8888/v1", "model": "unsloth/Qwen3.8-27B-GGUF", "quant": "UD-IQ2_S", "weights": "", "context_window": 131072, "kv_type": "q4_0", "capabilities": {"thinking": True, "tools": True}, "notes": ["Полное окно 131072 достижимо только с 4-битным KV: при f16 он весит 32.5 ГиБ, при q8_0 — 17.2, при q4_0 — 8.96 (посчитано по метаданным GGUF: 4 KV-головы, голова 256, 64 слоя).", "Измерено: загрузка с 131072 и q4_0 занимает 14446 MiB."]},
    {"id": "bonsai-2-27b-pq2-128k", "label": "Ternary-Bonsai-2 27B Abliterated · PQ2_0 · окно 131072", "runtime": "llama.cpp форк PrismML (prism-b10709-9a9394a)", "base_url": "http://127.0.0.1:8090/v1", "model": "bonsai-2-27b-pq2", "quant": "PQ2_0 (2.13 bpw, тернарные веса)", "weights": "", "context_window": 131072, "kv_type": "q4_0", "capabilities": {"thinking": True, "tools": "probe"}, "notes": ["Требует форк PrismML: типы тензоров PQ2_0=142 и PTQ1_0=143 лежат за GGML_TYPE_COUNT любой стоковой сборки (в дереве Studio 67, в mainline 43) — сток отказывается их читать.", "Измерено: загрузка с 131072 и q4_0 KV — 12324 MiB; генерация 78–79 tok/s, prompt до 9322 tok/s, TTFT 0.27 с.", "MTP-спекудяция с этим abliterated-файлом не работает: «Hadamard-latent table 'token_embd.weight' is read without the inverse transform»; карточка модели подтверждает, что без спекуляции текст байт-в-байт тот же."]},
    {"id": "qwen3.8-27b-ud-q3-k-xl", "label": "Qwen3.8 27B · UD-Q3_K_XL (база)", "runtime": "unsloth-studio v0.1.808-beta", "base_url": "http://127.0.0.1:8888/v1", "model": "unsloth/Qwen3.8-27B-GGUF", "quant": "UD-Q3_K_XL", "weights": "", "context_window": 32768, "kv_type": "q8_0", "capabilities": {"thinking": True, "tools": True}, "notes": ["Базовая линия сравнения — та конфигурация, что используется сейчас.", "Launch-строка: --max-seq-length 32768 при --gpu-memory-mode manual (auto умеет урезать окно под VRAM), -ngl 999 (все слои) без CPU offload, --jinja, --reasoning-format deepseek, --no-context-shift, --parallel 1.", "Указать --weights, чтобы в отчёт попал SHA-256 именно этого файла."], "verified": {"tools": "preflight: модель вызвала search_code с корректными аргументами", "reasoning": "preflight: размышление отделено от ответа", "usage": "preflight: prompt/completion tokens приходят в потоковом ответе", "window": "preflight --full: принят вход 28116 токенов при окне 32768"}},
    {"id": "qwen3.8-27b-ud-iq1-s", "label": "Qwen3.8 27B · UD-IQ1_S (5.77 GiB)", "runtime": "unsloth-studio v0.1.808-beta", "base_url": "http://127.0.0.1:8888/v1", "model": "unsloth/Qwen3.8-27B-GGUF", "quant": "UD-IQ1_S", "weights": "", "context_window": 32768, "kv_type": "q8_0", "capabilities": {"thinking": True, "tools": "probe"}, "notes": ["Самый лёгкий из скачанных квантов: проверяет, где начинается обвал качества.", "Тот же репозиторий и тот же рантайм, что у базы: сравниваются только веса."]},
    {"id": "qwen3.8-27b-ud-iq2-s", "label": "Qwen3.8 27B · UD-IQ2_S (7.80 GiB)", "runtime": "unsloth-studio v0.1.808-beta", "base_url": "http://127.0.0.1:8888/v1", "model": "unsloth/Qwen3.8-27B-GGUF", "quant": "UD-IQ2_S", "weights": "", "context_window": 32768, "kv_type": "q8_0", "capabilities": {"thinking": True, "tools": "probe"}, "notes": ["Компромисс между IQ1 и IQ3: интересно, удержится ли reasoning и длинный контекст.", "Тот же репозиторий и тот же рантайм, что у базы: сравниваются только веса."]},
    {"id": "qwen3.8-27b-ud-iq3-s", "label": "Qwen3.8 27B · UD-IQ3_S (11.21 GiB)", "runtime": "unsloth-studio v0.1.808-beta", "base_url": "http://127.0.0.1:8888/v1", "model": "unsloth/Qwen3.8-27B-GGUF", "quant": "UD-IQ3_S", "weights": "", "context_window": 32768, "kv_type": "q8_0", "capabilities": {"thinking": True, "tools": "probe"}, "notes": ["Прямой конкурент базы UD-Q3_K_XL при меньшем файле.", "Тот же репозиторий и тот же рантайм, что у базы: сравниваются только веса."]},
    {"id": "qwen3.8-27b-ud-iq2-xxs", "label": "Qwen3.8 27B · UD-IQ2_XXS (8.39 GiB, копия из кэша LM Studio)", "runtime": "unsloth-studio v0.1.808-beta", "base_url": "http://127.0.0.1:8888/v1", "model": "C:\\Users\\spgs\\.lmstudio\\models\\unsloth\\Qwen3.8-27B-GGUF", "quant": "UD-IQ2_XXS", "weights": "", "context_window": 32768, "kv_type": "q8_0", "capabilities": {"thinking": True, "tools": "probe"}, "notes": ["Файл скачан в каталог LM Studio, но грузится тем же рантаймом unsloth-studio: сравниваются веса, не приложение.", "Если Studio не примет этот путь как идентификатор модели, preflight/загрузка это покажут, и строка останется честно непройденной."]},
    {"id": "qwen3.8-27b-gsq-rco-iq3xxs", "label": "Qwen3.8 27B · GSQ-RCO IQ3_XXS (сравнение)", "runtime": "unsloth-studio v0.1.808-beta + GSQ", "base_url": "http://127.0.0.1:8888/v1", "model": "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", "quant": "IQ3_XXS", "weights": "", "kv_type": "q8_0", "capabilities": {"thinking": True, "tools": "probe"}, "notes": ["Главный вопрос выпуска: сколько баллов из 100 теряется против UD-Q3_K_XL и сколько при этом освобождается VRAM.", "Тот же рантайм и порт, что у базы: сравниваются веса, а не сервер.", "GSQ требует своей сборки рантайма — версия фиксируется в поле runtime."]},
    {"id": "unsloth-lfm2.5-8b-a1b", "label": "LFM2.5 8B A1B · UD-Q8_K_XL", "runtime": "unsloth-studio v0.1.808-beta", "base_url": "http://127.0.0.1:8888/v1", "model": "unsloth/LFM2.5-8B-A1B-GGUF", "quant": "UD-Q8_K_XL", "weights": "", "context_window": 32768, "kv_type": "q8_0", "capabilities": {"thinking": "probe", "tools": "probe"}, "notes": ["Малая MoE-модель: показывает, как сьют ведёт себя на быстрой конфигурации и где узкое место — модель или тест.", "Thinking и tools проверяются preflight'ом; непроверенная категория не превращается в нули."]},
    {"id": "gemma-4-12b-agentic", "label": "Gemma 4 12B agentic · Q8_0", "runtime": "unsloth-studio v0.1.808-beta", "base_url": "http://127.0.0.1:8888/v1", "model": "yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF", "quant": "Q8_0", "weights": "", "context_window": 32768, "kv_type": "q8_0", "capabilities": {"thinking": "probe", "tools": "probe"}, "notes": ["Другое семейство и заявленная agentic-настройка: проверяет, что сьют не подогнан под Qwen.", "Sliding-window attention у Gemma снижает реальный размер KV: пик VRAM измеряется, а не считается."]},
    {"id": "qwen3-14b-uncensored-q5km", "label": "Qwen3 14B Uncensored · Q5_K_M", "runtime": "unsloth-studio v0.1.808-beta", "base_url": "http://127.0.0.1:8888/v1", "model": "mradermacher/Qwen3-14B-Uncensored-GGUF", "quant": "Q5_K_M", "weights": "", "context_window": 32768, "kv_type": "q8_0", "capabilities": {"thinking": "probe", "tools": "probe"}, "notes": ["Меньшая модель того же семейства: показывает цену размера, а не кванта.", "Квант крупнее (Q5_K_M) при меньшем числе параметров — сравнение с 27B IQ3-класса особенно показательно для вывода про VRAM."]},
    {"id": "qwen3.8-27b-uncensored-iq3xs", "label": "Qwen3.8 27B Uncensored (MTP) · IQ3_XS", "runtime": "unsloth-studio v0.1.808-beta", "base_url": "http://127.0.0.1:8888/v1", "model": "HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF", "quant": "IQ3_XS", "weights": "", "context_window": 32768, "kv_type": "q8_0", "capabilities": {"thinking": True, "tools": "probe"}, "notes": ["Третий квант того же размера: IQ3_XS против UD-Q3_K_XL и GSQ-RCO IQ3_XXS.", "MTP-сборка: если speculative decoding включён загрузчиком, это меняет скорость — фиксировать в отчёте."]},
    {"id": "generic-gguf", "label": "Шаблон для любой другой GGUF-модели", "runtime": "unsloth-studio", "base_url": "http://127.0.0.1:8888/v1", "model": "<model-name>", "quant": "<quant>", "weights": "", "context_window": DEFAULT_CONTEXT_WINDOW, "kv_type": "q8_0", "capabilities": {"thinking": "probe", "tools": "probe"}, "notes": ["Скопировать запись, заполнить model/quant/weights, прогнать preflight --full.", "Менять только эти поля: политика одинакова для всех, её изменение обесценивает сравнение."]},
]


# --------------------------------------------------------------------------- #
# Loading and applying
# --------------------------------------------------------------------------- #


@dataclass
class ReferenceConfig:
    entry: dict[str, Any]
    policy: dict[str, Any] = field(default_factory=lambda: dict(POLICY))

    @property
    def id(self) -> str:
        return str(self.entry.get("id", ""))

    @property
    def label(self) -> str:
        return str(self.entry.get("label") or self.id)

    @property
    def runtime(self) -> str:
        return str(self.entry.get("runtime", "llama.cpp"))

    @property
    def runtime_id(self) -> str:
        """Which launch template applies.

        ``runtime`` is the human label (it carries the exact build, e.g.
        ``"llama.cpp (unsloth-studio v0.1.808-beta)"``); ``runtime_id`` is the
        template key. When a label starts with a known key it is used directly,
        so a new engine variant does not silently lose its launch line.
        """
        explicit = str(self.entry.get("runtime_id") or "")
        if explicit:
            return explicit
        label = self.runtime.casefold()
        for key in sorted(RUNTIMES, key=len, reverse=True):
            if label.startswith(key.casefold()):
                return key
        return ""

    @property
    def tools_capability(self) -> Any:
        return (self.entry.get("capabilities") or {}).get("tools", True)

    def launch_line(self) -> str:
        template = RUNTIMES.get(self.runtime_id, {}).get("launch", "")
        return (
            template.replace("<model-key>", str(self.entry.get("model", "")))
            .replace("<model-name>", str(self.entry.get("model", "")))
            .replace("<weights.gguf>", str(self.entry.get("weights") or "<weights.gguf>"))
            .replace("<quant>", str(self.entry.get("quant", "Q4_K_M")))
            .replace("<kv>", str(self.entry.get("kv_type", "q8_0")))
            .replace("<window>", str(self.entry.get("context_window") or self.policy.get("context_window", DEFAULT_CONTEXT_WINDOW)))
            .replace("<reasoning-budget>", str(self.policy.get("reasoning_budget", 2048)))
            .replace("<answer-budget>", str(self.policy.get("answer_budget_default", 4096)))
            .replace("<model>", str(self.entry.get("model", "")))
        )

    def unsupported_categories(self) -> tuple[str, ...]:
        """Categories this entry declares as not applicable.

        ``False`` means "this configuration cannot do it" (the category is
        reported as not verified instead of a zero the model did not earn);
        ``"probe"`` means "let preflight decide", and the run keeps the default.
        """
        return ("D",) if self.tools_capability is False else ()


def load_reference(path: Path | str | None = None) -> dict[str, Any]:
    target = Path(path) if path else default_path()
    if target.is_file():
        return json.loads(target.read_text(encoding="utf-8"))
    return {"schema": "dailybench/reference/1", "policy": POLICY, "runtimes": RUNTIMES, "models": MODELS}


#: Environment override for the reference file: a clone keeps its own entries
#: (machine paths, ports, keys) in a local copy instead of editing the tracked one.
PATH_ENV = "DAILYBENCH_REFERENCE"


def default_path() -> Path:
    from .config import _env

    return Path(_env(PATH_ENV)) if _env(PATH_ENV) else DEFAULT_PATH


def model_entries(path: Path | str | None = None) -> list[ReferenceConfig]:
    if path is None:
        # The data file is the source of truth: the same table is also a Python
        # constant, so falling back to it keeps the file local-only-friendly.
        path = default_path()
    data = load_reference(path)
    entries = data.get("models") or MODELS
    policy = data.get("policy") or POLICY
    return [ReferenceConfig(entry=dict(entry), policy=dict(policy)) for entry in entries]


def resolve(name: str, path: Path | str | None = None) -> ReferenceConfig:
    entries = model_entries(path)
    for entry in entries:
        if entry.id == name:
            return entry
    known = ", ".join(entry.id for entry in entries)
    raise KeyError(f"unknown reference config {name!r}; known: {known}")


def apply_to_run_config(cfg: RunConfig, reference: ReferenceConfig) -> RunConfig:
    """Fill a run configuration from a reference entry, keeping the policy."""
    entry = reference.entry
    policy = reference.policy
    updated = replace(
        cfg,
        base_url=str(entry.get("base_url") or cfg.base_url),
        model=str(entry.get("model") or cfg.model),
        quant=str(entry.get("quant") or cfg.quant),
        runtime=str(entry.get("runtime") or cfg.runtime),
        weights_path=str(entry.get("weights") or ""),
        kv_type=str(entry.get("kv_type") or ""),
        context_window=int(entry.get("context_window") or policy["context_window"]),
        n_ctx=int(entry.get("context_window") or policy["context_window"]),
        temperature=float(policy["temperature"]),
        top_p=float(policy["top_p"]),
        max_new_tokens=int(policy.get("answer_budget_default", 4096)),
        # Artefact categories are graded on the artefact, so they get the larger
        # budget: thinking plus a whole file rarely fits 4K.
        max_new_tokens_by_category={
            "C": int(policy.get("answer_budget_artifact", 8192)),
            "K": int(policy.get("answer_budget_visual_sim", policy.get("answer_budget_artifact", 8192))),
            "L": int(policy.get("answer_budget_visual_art", policy.get("answer_budget_artifact", 8192))),
        },
        request_wall_clock=float(policy.get("request_wall_clock", 600)),
        task_wall_clock=float(policy.get("task_wall_clock", 3600)),
        max_total_tokens=int(policy["max_total_tokens"]),
        max_tool_calls=int(policy["max_tool_calls"]),
        offload=str(entry.get("offload") or "none"),
        reference=reference.id,
        launch=reference.launch_line(),
        unsupported=reference.unsupported_categories(),
        # An entry may opt out of the agent harness (a model whose endpoint has
        # no tool protocol, or a comparison against earlier chat runs).
        agent=bool(entry.get("agent", cfg.agent)),
    )
    return updated


def save_reference(path: Path | str) -> Path:
    """Write the shipped default reference file (used once, then edited by hand)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {"schema": "dailybench/reference/1", "policy": POLICY, "runtimes": RUNTIMES, "models": MODELS},
            ensure_ascii=False,
            indent=2,
            default=core.json_default,
        ),
        encoding="utf-8",
    )
    return target


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


@dataclass
class Probe:
    name: str
    status: str  # "ok" | "warn" | "fail" | "n/a"
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass
class PreflightReport:
    reference_id: str
    label: str
    probes: list[Probe] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    launch: str = ""

    @property
    def verdict(self) -> str:
        if any(probe.status == "fail" for probe in self.probes):
            return "rejected"
        if any(probe.status == "warn" for probe in self.probes):
            return "admitted with unverified items"
        return "admitted"

    @property
    def ok(self) -> bool:
        return self.verdict != "rejected"

    def render(self) -> str:
        icons = {"ok": "ok", "warn": "не подтверждено", "fail": "ПРОВАЛ", "n/a": "н/п"}
        lines = [f"# Преflight — {self.label} (`{self.reference_id}`)", "", f"**Вердикт: {self.verdict}**", ""]
        lines.append("| Проверка | Статус | Детали |")
        lines.append("| --- | --- | --- |")
        for probe in self.probes:
            lines.append(f"| {probe.name} | {icons.get(probe.status, probe.status)} | {probe.detail} |")
        if self.metrics:
            lines.append("")
            lines.append("| Метрика | Значение |")
            lines.append("| --- | --- |")
            for key, value in self.metrics.items():
                lines.append(f"| {key} | {value} |")
        if self.launch:
            lines.append("")
            lines.append("Загрузка конфигурации:")
            lines.append("")
            lines.append("```")
            lines.append(self.launch)
            lines.append("```")
        return "\n".join(lines)


def _props(provider: OpenAICompatibleProvider) -> dict[str, Any] | None:
    """``/props`` exists on llama.cpp; other runtimes answer 404."""
    try:
        with provider._request("/props", None, method="GET") as response:  # noqa: SLF001 - deliberate probe
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


TOOL_PROBE_PROMPT = (
    "Найди, где в репозитории задаётся max_model_len, и сообщи путь и значение. "
    "Ничего не изменяй. Используй доступные инструменты."
)


def preflight(
    reference: ReferenceConfig,
    *,
    provider: Provider | None = None,
    full: bool = False,
    hash_weights: bool = False,
) -> PreflightReport:
    """Verify that the running endpoint matches the declared configuration."""
    entry = reference.entry
    base_url = str(entry.get("base_url") or "http://127.0.0.1:8080/v1")
    model = str(entry.get("model") or "")
    provider = provider or OpenAICompatibleProvider(base_url=base_url, model=model, timeout=1800.0)
    report = PreflightReport(reference_id=reference.id, label=reference.label, launch=reference.launch_line())

    gpus = gpu_inventory()
    if gpus:
        report.metrics["gpu"] = f"{gpus[0]['name']} ({gpus[0]['vram_total_mib']} MiB)"

    # 1. endpoint and model
    try:
        models = provider.list_models()
    except ProviderError as exc:
        report.probes.append(Probe("endpoint", "fail", str(exc)))
        return report
    if model and model not in models:
        report.probes.append(
            Probe("model_listed", "fail", f"{model!r} отсутствует на {base_url}; доступны: {models}")
        )
    else:
        report.probes.append(Probe("model_listed", "ok", f"{model!r} доступна на {base_url}"))

    # 2. runtime-reported load parameters (llama.cpp /props)
    props = _props(provider) if isinstance(provider, OpenAICompatibleProvider) else None
    if props:
        generation = props.get("default_generation_settings") or {}
        n_ctx = generation.get("n_ctx") or props.get("n_ctx")
        declared = int(entry.get("context_window") or 0)
        if n_ctx is None:
            report.probes.append(Probe("context_window", "warn", "runtime не сообщил n_ctx"))
        elif declared and int(n_ctx) < declared:
            report.probes.append(
                Probe("context_window", "fail", f"загружено n_ctx={n_ctx}, объявлено {declared}")
            )
        else:
            report.probes.append(Probe("context_window", "ok", f"n_ctx={n_ctx} (объявлено {declared})"))
        model_path = props.get("model_path")
        if model_path:
            report.metrics["model_path"] = model_path
        template = props.get("chat_template")
        report.probes.append(
            Probe(
                "chat_template",
                "ok" if template else "warn",
                "шаблон из метаданных модели" if template else "runtime не вернул шаблон",
            )
        )
        caps = props.get("chat_template_caps")
        if caps:
            supported = sorted(str(name) for name, value in caps.items() if value)
            report.metrics["chat_template_caps"] = ", ".join(supported) or "нет"
            tools_ok = bool(caps.get("supports_tools") or caps.get("supports_tool_calls"))
            report.probes.append(
                Probe(
                    "template_capabilities",
                    "ok" if tools_ok else "warn",
                    "шаблон модели поддерживает вызовы инструментов"
                    if tools_ok
                    else "шаблон модели не описывает tool calls — категорию D считать не проверенной "
                    "(run --unsupported D), а не проваленной",
                )
            )
        if props.get("model_alias"):
            report.metrics["model_alias"] = props["model_alias"]
        if props.get("build_info"):
            report.metrics["build_info"] = props["build_info"]
    else:
        report.probes.append(
            Probe(
                "context_window",
                "warn",
                "runtime не предоставляет /props: окно и KV-cache подтверждаются только командой загрузки",
            )
        )

    # 3. tool calling probe — decides whether category D is verified or skipped
    tools_declared = reference.tools_capability
    if tools_declared is False:
        report.probes.append(
            Probe("tool_calling", "n/a", "конфигурация объявляет отсутствие tool calling — категория D не проверяется")
        )
    else:
        try:
            reply = _tool_probe(provider)
        except ProviderError as exc:
            report.probes.append(Probe("tool_calling", "warn", f"проба не выполнена: {exc}"))
        else:
            if reply.tool_calls:
                names = ", ".join(str(call.get("name")) for call in reply.tool_calls)
                report.probes.append(Probe("tool_calling", "ok", f"модель вызвала инструмент: {names}"))
            else:
                report.probes.append(
                    Probe(
                        "tool_calling",
                        "warn",
                        "модель не вызвала ни одного инструмента на прямой просьбе — "
                        "категорию D считать не проверенной, а не проваленной (run --unsupported D)",
                    )
                )
            report.metrics["tool_probe_finish_reason"] = reply.finish_reason or "n/a"

    # 4. reasoning separation
    try:
        thinking_reply = _thinking_probe(provider)
        if thinking_reply is None:
            report.probes.append(Probe("reasoning", "warn", "проба не выполнена"))
        elif (thinking_reply.reasoning or "").strip():
            report.probes.append(
                Probe(
                    "reasoning",
                    "ok",
                    f"reasoning отделён от ответа ({thinking_reply.reasoning_tokens or '?'} токенов размышления)",
                )
            )
        else:
            report.probes.append(
                Probe("reasoning", "n/a", "модель либо не размышляет, либо runtime не отделяет reasoning от ответа")
            )
    except ProviderError as exc:
        report.probes.append(Probe("reasoning", "warn", f"проба не выполнена: {exc}"))

    # 5. weights identity
    weights = str(entry.get("weights") or "")
    if hash_weights and weights and Path(weights).is_file():
        report.metrics["weights_sha256"] = weights_sha256(weights)
        report.probes.append(Probe("weights", "ok", "SHA-256 посчитан от указанного файла"))
    elif weights and not Path(weights).is_file():
        report.probes.append(Probe("weights", "warn", f"файл {weights} не найден — SHA-256 не посчитан"))
    else:
        report.probes.append(
            Probe("weights", "warn", "не указан путь к файлу весов: идентичность весов не подтверждена")
        )

    # 6. full-window probe: the admission claim itself
    if full:
        report.probes.extend(_window_probes(entry, provider, report.metrics))
    else:
        report.probes.append(
            Probe("full_window", "n/a", "не выполнялась (--full): допуск по 32K не подтверждён измерением")
        )
    return report


def _tool_probe(provider: Provider) -> core.ModelReply:
    return provider.chat(
        [{"role": "user", "content": TOOL_PROBE_PROMPT}],
        tool_schemas(),
        max_new_tokens=256,
        temperature=0.0,
        top_p=1.0,
        seed=1,
        wall_clock=float(POLICY["request_wall_clock"]),
    )


def _thinking_probe(provider: Provider) -> core.ModelReply | None:
    return provider.chat(
        [{"role": "user", "content": "Сколько будет 17 умножить на 23? Ответь только числом."}],
        None,
        max_new_tokens=2048,
        temperature=0.0,
        top_p=1.0,
        seed=2,
        wall_clock=float(POLICY["request_wall_clock"]),
    )


def _window_probes(
    entry: dict[str, Any],
    provider: Provider,
    metrics: dict[str, Any],
) -> list[Probe]:
    """Fill the declared window and sample VRAM during prefill and generation.

    The prompt is sized against the runtime's own tokenizer when it exposes one,
    and rescaled once from the reported usage when it does not: a character
    estimate can pack two to three times denser than intended and would turn the
    admission claim into a smaller test than it says.
    """
    from .measure import count_prompt_tokens, fit_prompt_to_tokens, synthetic_prompt, used_mib

    probes: list[Probe] = []
    window = int(entry.get("context_window") or 32768)
    reserve = int(POLICY["answer_budget_default"])
    target = max(1024, window - reserve - 512)
    tokenizer = count_prompt_tokens(provider, "проверка")
    metrics["tokenizer_endpoint"] = "есть" if tokenizer is not None else "нет (размер берётся из usage)"

    baseline = used_mib()
    sampler = VramSampler()
    sampler.start()

    prompt, exact, ratio = fit_prompt_to_tokens(
        provider, target, nonce=f"preflight-{entry.get('id')}", tolerance=0.05
    )
    metrics["chars_per_token_initial"] = round(ratio, 2)

    reply: core.ModelReply | None = None
    error: ProviderError | None = None
    for attempt in range(2):
        try:
            reply = provider.chat(
                [{"role": "user", "content": prompt}],
                None,
                max_new_tokens=128,
                temperature=0.0,
                top_p=1.0,
                seed=3,
                wall_clock=float(POLICY["request_wall_clock"]),
            )
        except ProviderError as exc:
            error = exc
            break
        actual = reply.prompt_tokens or exact
        if actual and actual < 0.9 * target:
            # The runtime does not tokenize for us: rescale from the real usage
            # and measure again, so the peak VRAM belongs to a full window.
            ratio = max(1.0, len(prompt) / actual)
            prompt = synthetic_prompt(target, ratio, nonce=f"preflight-{entry.get('id')}-retry")
            exact = count_prompt_tokens(provider, prompt)
            continue
        break
    sampler.stop()

    if error is not None:
        detail = f"запрос длиной ~{target} токенов не выполнен: {str(error)[:300]}"
        if any(marker in str(error).casefold() for marker in ("context", "too long", "exceed", "length")):
            detail += " — runtime отверг вход вместо тихого усечения (поведение верное, допуск не пройден)"
        probes.append(Probe("full_window", "fail", detail))
        return probes

    assert reply is not None
    prompt_tokens = reply.prompt_tokens or exact
    if prompt_tokens:
        metrics["chars_per_token_measured"] = round(len(prompt) / prompt_tokens, 2)
    metrics["prompt_tokens_full_window"] = prompt_tokens if prompt_tokens is not None else "runtime не сообщил"
    metrics["vram_before_mib"] = baseline
    metrics["vram_peak_mib"] = sampler.peak_mib
    if sampler.peak_mib is not None and baseline is not None:
        metrics["vram_delta_mib"] = sampler.peak_mib - baseline
        if sampler.peak_mib - baseline < 100:
            # A filled window that adds almost nothing means the KV cache was
            # allocated when the model loaded. The peak is then a property of the
            # load, not of the request — say so instead of implying that the
            # measurement covered the context.
            probes.append(
                Probe(
                    "kv_allocation",
                    "warn",
                    f"прирост VRAM при заполнении окна всего {sampler.peak_mib - baseline} MiB: "
                    "KV-cache выделен при загрузке, поэтому пик отражает загрузку, а не длину запроса",
                )
            )

    if prompt_tokens is None:
        probes.append(
            Probe(
                "full_window",
                "warn",
                f"запрос длиной ~{target} токенов выполнен, но runtime не сообщил точный размер входа",
            )
        )
    elif prompt_tokens + reserve > window:
        probes.append(
            Probe(
                "full_window",
                "fail",
                f"фактический вход {prompt_tokens} + резерв {reserve} > окна {window}: допуск не пройден",
            )
        )
    elif prompt_tokens < 0.8 * target:
        probes.append(
            Probe(
                "full_window",
                "fail",
                f"вход достиг только {prompt_tokens} токенов при цели {target}: окно не заполнено, "
                "допуск по 32K не подтверждён",
            )
        )
    else:
        probes.append(
            Probe(
                "full_window",
                "ok",
                f"вход {prompt_tokens} токенов (цель {target}) + резерв {reserve} укладывается в окно {window}; "
                f"пик VRAM {sampler.peak_mib} MiB",
            )
        )

    vram_total = gpu_inventory()
    if vram_total and sampler.peak_mib is not None:
        headroom = vram_total[0]["vram_total_mib"] - sampler.peak_mib
        metrics["vram_headroom_mib"] = headroom
        if headroom < 512:
            probes.append(
                Probe(
                    "vram_headroom",
                    "warn",
                    f"запас VRAM всего {headroom} MiB: рост батча или другой драйвер приведут к offload",
                )
            )
        else:
            probes.append(Probe("vram_headroom", "ok", f"запас {headroom} MiB при заполненном окне"))
    return probes
