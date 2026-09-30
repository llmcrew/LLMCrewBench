"""Command line interface.

    python -m dailybench doctor
    python -m dailybench list --seed 20260915
    python -m dailybench generate --seed 20260915 --out suite/20260915
    python -m dailybench selfcheck
    python -m dailybench run --base-url http://127.0.0.1:1234/v1 --model qwen3.8-27b \\
        --quant "Q4_K_M" --name qwen-plain --speed
    python -m dailybench report runs/qwen-plain runs/qwen-gsq
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from . import core, reference
from .config import DEFAULT_SEED, RunConfig
from .measure import (
    SpeedResult,
    gpu_inventory,
    speed_profile,
    used_mib,
    weights_sha256,
)
from .providers import OpenAICompatibleProvider, Provider, ProviderError
from .report import render_markdown, summarise, write_reports
from .runner import RunResult, Runner
from .selfcheck import save_suite, selfcheck, write_selfcheck_report
from .suite import build_suite, fit_suite_sizing, public_prompts


def _provider_from(cfg: RunConfig) -> Provider:
    """Provider for a resolved run configuration.

    Always build this from the resolved config, never from the raw argparse
    namespace: a reference entry overrides base_url and model, and a provider
    built from the defaults would talk to the wrong endpoint.
    """
    return OpenAICompatibleProvider(
        base_url=cfg.base_url,
        model=cfg.model,
        api_key=cfg.api_key,
        timeout=cfg.request_timeout,
    )


def _config_from(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig(
        base_url=getattr(args, "base_url", RunConfig.base_url),
        model=getattr(args, "model", RunConfig.model),
        api_key=getattr(args, "api_key", "not-needed"),
        request_timeout=getattr(args, "timeout", 1800.0),
        seed=args.seed,
        quant=getattr(args, "quant", "") or "",
        runtime=getattr(args, "runtime", "") or "",
        weights_path=getattr(args, "weights", "") or "",
        n_ctx=getattr(args, "context_window", 32768),
        kv_type=getattr(args, "kv_type", "") or "",
        offload=getattr(args, "offload", "none"),
        sql_backend=args.sql_backend,
    )
    if args.only:
        cfg.only = tuple(part.strip() for part in args.only.split(",") if part.strip())
    if getattr(args, "max_new_tokens", None):
        cfg.max_new_tokens = args.max_new_tokens
    if getattr(args, "context_window", None):
        cfg.context_window = args.context_window
    if getattr(args, "no_system_prompt", False):
        cfg.system_prompt = ""
    if getattr(args, "config", ""):
        # The reference entry owns the contract (model, quant, KV type, context
        # window, budgets); transport and selection stay on the command line.
        resolved = reference.resolve(args.config, getattr(args, "configs_path", "") or None)
        cfg = reference.apply_to_run_config(cfg, resolved)
    if getattr(args, "unsupported", ""):
        extra = tuple(part.strip().upper() for part in args.unsupported.split(",") if part.strip())
        cfg.unsupported = tuple(dict.fromkeys(cfg.unsupported + extra))
    return cfg


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_doctor(args: argparse.Namespace) -> int:
    info: dict[str, object] = {"python": sys.version.split()[0]}
    gpus = gpu_inventory()
    info["gpus"] = gpus
    info["vram_used_mib_now"] = used_mib()
    info["go"] = shutil.which("go") or shutil.which("go.exe") or "missing"
    info["docker"] = shutil.which("docker") or "missing"
    try:
        import sqlite3

        info["sqlite"] = sqlite3.sqlite_version
    except Exception as exc:  # pragma: no cover
        info["sqlite"] = f"unavailable: {exc}"

    cfg = _config_from(args)
    endpoint: dict[str, object] = {"base_url": cfg.base_url, "model": cfg.model}
    if cfg.reference:
        endpoint["reference"] = cfg.reference
    try:
        provider = _provider_from(cfg)
        models = provider.list_models()
        endpoint["models"] = models
        endpoint["reachable"] = True
        endpoint["tokenize_endpoint"] = provider.tokenize("проверка токенизатора") is not None
        if cfg.model not in models:
            endpoint["warning"] = f"model {cfg.model!r} is not in the endpoint's list"
    except ProviderError as exc:
        endpoint["reachable"] = False
        endpoint["error"] = str(exc)
    info["endpoint"] = endpoint

    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0 if endpoint.get("reachable") else 1


def cmd_list(args: argparse.Namespace) -> int:
    cfg = _config_from(args)
    tasks = build_suite(cfg)
    print(f"{'ID':6} {'Баллы':>6} {'Символов':>9} {'~токенов':>9}  Название")
    for entry in public_prompts(tasks):
        print(
            f"{entry['id']:6} {entry['max_points']:>6.0f} {entry['prompt_chars']:>9} "
            f"{(entry['estimated_tokens'] or 0):>9}  {entry['title']}"
        )
    total = sum(task.max_points for task in tasks)
    print(f"\nЗаданий: {len(tasks)}, сумма баллов: {total:.0f}, seed={cfg.seed}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    cfg = _config_from(args)
    written = save_suite(cfg, args.out)
    for name, path in written.items():
        print(f"{name}: {path}")
    print("\nФайл prompts.json можно публиковать; oracles.json содержит ответы.")
    return 0


def cmd_selfcheck(args: argparse.Namespace) -> int:
    cfg = _config_from(args)
    tasks = build_suite(cfg)
    if args.limit:
        tasks = tasks[: args.limit]
    report = selfcheck(cfg, tasks)
    print(report.render())
    if args.report:
        print(f"\nОтчёт: {write_selfcheck_report(args.report, report)}")
    return 0 if report.ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _config_from(args)
    provider = _provider_from(cfg)

    try:
        models = provider.list_models()
    except ProviderError as exc:
        print(f"endpoint {cfg.base_url} недоступен: {exc}", file=sys.stderr)
        return 2
    if cfg.model not in models:
        print(f"модель {cfg.model!r} отсутствует на {cfg.base_url}; доступны: {models}", file=sys.stderr)
        return 2

    if args.weights:
        print("считаю SHA-256 весов...", flush=True)
        cfg.weights_sha256 = weights_sha256(args.weights)

    # The run directory is decided before calibration so that every artefact of
    # the run lands in one place.
    name = args.name or f"{datetime.now():%Y%m%d-%H%M%S}-{cfg.model.replace('/', '_')}"
    cfg.out_dir = Path(args.out_dir or "runs") / name
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "calibrate", True):
        sizing = fit_suite_sizing(cfg, provider)
        print(f"калибровка длинного контекста: {sizing['note']}", flush=True)
        if sizing.get("calibrated"):
            (cfg.out_dir / "sizing.json").write_text(
                json.dumps(sizing, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    tasks = build_suite(cfg)
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print("нечего запускать: фильтр --only не совпал ни с одним заданием", file=sys.stderr)
        return 2

    runner = Runner(cfg, provider)
    started = datetime.now()
    print(
        f"Запуск {len(tasks)} заданий: model={cfg.model} quant={cfg.quant or 'n/a'} "
        f"seed={cfg.seed} out={cfg.out_dir}"
        + (f" reference={cfg.reference}" if cfg.reference else "")
        + (f" не проверяются: {','.join(cfg.unsupported)}" if cfg.unsupported else ""),
        flush=True,
    )

    runs: list = []
    for index, task in enumerate(tasks, start=1):
        run = runner.run_task(task)
        runs.append(run)
        marker = "ERR" if run.score.error else f"{run.score.earned:.0f}/{run.score.possible:.0f}"
        extra = f" tokens={run.transcript.prompt_tokens()}" if run.transcript.prompt_tokens() else ""
        print(f"[{index:>2}/{len(tasks)}] {task.id:<6} {marker}{extra}", flush=True)

    speed_rows: list[dict] = []
    if args.speed:
        print("Профиль скорости (1K и длинный prompt)...", flush=True)
        for row in speed_profile(provider, cfg, long_tokens=args.speed_long, repeats=args.speed_repeats):
            speed_rows.append(row.as_dict())
            print(
                f"  {row.label}: TTFT {row.ttft_s:.2f}s, prompt {row.prompt_tok_s:.0f} tok/s, "
                f"generation {row.generation_tok_s:.1f} tok/s, peak VRAM {row.vram_peak_mib} MiB",
                flush=True,
            )

    result = RunResult(
        cfg=cfg,
        runs=runs,
        provider_info=provider.describe(),
        suite_digest=build_suite_digest(cfg),
        speed=speed_rows,
        hardware=_hardware(),
    )
    result.save()
    report_path = cfg.out_dir / "report.md"
    report_path.write_text(render_markdown(_load_for_render(cfg.out_dir)), encoding="utf-8")

    summary = summarise([score.as_dict() for score in result.scores], len(tasks))
    print(f"\nИтог: {summary['verdict']}")
    print(f"Прошло: {(datetime.now() - started)}")
    print(f"Отчёт: {report_path}")
    if summary["errors"]:
        print(f"Прогонов с ошибками: {summary['errors']} (см. tasks.json)", file=sys.stderr)
    return 0


def cmd_speed(args: argparse.Namespace) -> int:
    cfg = _config_from(args)
    provider = _provider_from(cfg)
    rows: list[SpeedResult] = speed_profile(
        provider, cfg, long_tokens=args.speed_long, repeats=args.speed_repeats
    )
    for row in rows:
        print(json.dumps(row.as_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    paths = write_reports(args.directories)
    for path in paths:
        print(path)
    for directory in args.directories:
        data = json.loads((Path(directory) / "scores.json").read_text(encoding="utf-8"))
        summary = summarise(data)
        print(f"{directory}: {summary['verdict']}, заданий с ошибкой: {summary['errors']}")
    return 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def cmd_configs(args: argparse.Namespace) -> int:
    """List the reference configurations and whether their runtime is running."""
    if args.write:
        path = reference.save_reference(args.write)
        print(f"записан эталонный файл: {path}")
        return 0

    entries = reference.model_entries(args.configs_path or None)
    print(f"{'ID':24} {'Runtime':10} {'Model':22} {'Quant':18} {'KV':6} {'Tools':7} Окно")
    for entry in entries:
        capabilities = entry.entry.get("capabilities") or {}
        tools = capabilities.get("tools", "probe")
        tools_text = {True: "да", False: "нет"}.get(tools, "probe")
        print(
            f"{entry.id:24} {entry.runtime:10} {str(entry.entry.get('model', '')):22} "
            f"{str(entry.entry.get('quant', '')):18} {str(entry.entry.get('kv_type', '')):6} "
            f"{tools_text:7} {entry.entry.get('context_window')}"
        )
    print()
    print("Политика (одинакова для всех):")
    for key, value in reference.POLICY.items():
        if key.endswith("_note"):
            continue
        print(f"  {key} = {value}")
    print()
    print("Примеры загрузки:")
    for name, runtime in reference.RUNTIMES.items():
        print(f"\n[{name}]\n{runtime['launch']}")
    print("\nПеред прогоном: python -m dailybench preflight --config <ID> --full")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    resolved = reference.resolve(args.config, args.configs_path or None)
    report = reference.preflight(
        resolved,
        full=args.full,
        hash_weights=args.hash_weights,
    )
    print(report.render())
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report.render() + "\n", encoding="utf-8")
        print(f"\nОтчёт: {target}")
    return 0 if report.ok else 1


def cmd_rescore(args: argparse.Namespace) -> int:
    from .rescore import rescore_run

    failed = 0
    for directory in args.directories:
        try:
            marker = rescore_run(directory)
        except FileNotFoundError as exc:
            print(f"{directory}: {exc}", file=sys.stderr)
            failed += 1
            continue
        changed = marker["changed"]
        print(f"{directory}: пересчитано заданий {marker['tasks']}, изменений {len(changed)}")
        for row in changed:
            if "earned_before" in row:
                print(
                    f"  {row['task_id']}: {row['earned_before']} → {row['earned_after']} "
                    f"(допуск {row['admission_before']} → {row['admission_after']})"
                )
            else:
                print(f"  {row['task_id']}: {row['status']}")
    return 1 if failed else 0


def build_suite_digest(cfg: RunConfig) -> dict:
    from .suite import suite_digest

    return suite_digest(build_suite(cfg))


def _hardware() -> dict:
    gpus = gpu_inventory()
    if not gpus:
        return {"gpu": "unknown", "vram_used_mib_now": used_mib()}
    return {**gpus[0], "vram_used_mib_now": used_mib()}


def _load_for_render(directory: Path) -> dict:
    from .report import load_run

    return load_run(directory)


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dailybench", description="LLMCrew DailyBench-16")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser, *, endpoint: bool = False) -> None:
        target.add_argument("--seed", type=int, default=DEFAULT_SEED, help="master seed of the suite")
        target.add_argument("--only", default="", help="категории или ID заданий через запятую (A,B / IF-1,R-7)")
        target.add_argument("--limit", type=int, default=0, help="ограничить число заданий (для дымового прогона)")
        target.add_argument("--sql-backend", default="sqlite", choices=("sqlite", "postgres"))
        if endpoint:
            target.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
            target.add_argument("--model", default="local-model")
            target.add_argument("--api-key", default="not-needed")
            target.add_argument("--timeout", type=float, default=1800.0)
            target.add_argument("--quant", default="", help="метка кванта для отчёта, напр. IQ4_XS")
            target.add_argument("--runtime", default="", help="runtime и версия, напр. llama.cpp b4321")
            target.add_argument("--kv-type", default="", help="формат KV-cache, напр. q8_0")
            target.add_argument("--offload", default="none", help="описание offload (none — обязательный зачёт)")
            target.add_argument("--context-window", type=int, default=32768)
            target.add_argument("--max-new-tokens", type=int, default=4096)
            target.add_argument("--weights", default="", help="путь к файлу весов для SHA-256")
            target.add_argument("--no-system-prompt", action="store_true")

    doctor = sub.add_parser("doctor", help="проверить GPU, toolchain и endpoint")
    add_common(doctor, endpoint=True)
    doctor.add_argument("--config", default="", help="id эталонной конфигурации")
    doctor.add_argument("--configs-path", default="")
    doctor.set_defaults(func=cmd_doctor)

    listing = sub.add_parser("list", help="показать задания сьюта без запуска")
    add_common(listing)
    listing.set_defaults(func=cmd_list)

    generate = sub.add_parser("generate", help="записать fixtures сьюта в каталог")
    add_common(generate)
    generate.add_argument("--out", required=True)
    generate.set_defaults(func=cmd_generate)

    check = sub.add_parser("selfcheck", help="проверить генераторы и evaluator'ы без GPU")
    add_common(check)
    check.add_argument("--report", default="")
    check.set_defaults(func=cmd_selfcheck)

    run = sub.add_parser("run", help="прогнать сьют на модели")
    add_common(run, endpoint=True)
    run.add_argument("--config", default="", help="id эталонной конфигурации из configs/reference.json")
    run.add_argument("--configs-path", default="", help="путь к файлу эталонных конфигураций")
    run.add_argument("--unsupported", default="", help="категории, которые эта конфигурация не проверяет (например D)")
    run.add_argument("--name", default="")
    run.add_argument("--out-dir", default="")
    run.add_argument("--speed", action="store_true", help="снять профиль скорости и VRAM")
    run.add_argument("--no-calibrate", dest="calibrate", action="store_false",
                     help="не калибровать chars-per-token по реальному токенизатору перед прогоном")
    run.add_argument("--speed-long", type=int, default=0)
    run.add_argument("--speed-repeats", type=int, default=3)
    run.set_defaults(func=cmd_run)

    configs = sub.add_parser("configs", help="показать эталонные конфигурации и политику")
    configs.add_argument("--configs-path", default="")
    configs.add_argument("--write", default="", help="записать эталонный файл по указанному пути")
    configs.set_defaults(func=cmd_configs)

    preflight = sub.add_parser("preflight", help="проверить, что runtime соответствует эталонной конфигурации")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--configs-path", default="")
    preflight.add_argument("--full", action="store_true", help="заполнить всё окно контекста и снять пик VRAM")
    preflight.add_argument("--hash-weights", action="store_true", help="посчитать SHA-256 файла весов")
    preflight.add_argument("--report", default="", help="куда записать отчёт preflight")
    preflight.set_defaults(func=cmd_preflight)

    speed = sub.add_parser("speed", help="только профиль скорости и VRAM")
    add_common(speed, endpoint=True)
    speed.add_argument("--config", default="")
    speed.add_argument("--configs-path", default="")
    speed.add_argument("--speed-long", type=int, default=0)
    speed.add_argument("--speed-repeats", type=int, default=3)
    speed.set_defaults(func=cmd_speed)

    report = sub.add_parser("report", help="пересобрать отчёты и сравнение")
    report.add_argument("directories", nargs="+")
    report.set_defaults(func=cmd_report)

    rescore = sub.add_parser(
        "rescore", help="пересчитать баллы сохранённого прогона текущими evaluator'ами (без вызова модели)"
    )
    rescore.add_argument("directories", nargs="+")
    rescore.set_defaults(func=cmd_rescore)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BrokenPipeError:
        # `dailybench configs | head` closes the pipe early; that is not an error.
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
