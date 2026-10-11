"""Command line interface.

    python -m dailybench doctor
    python -m dailybench list --seed 20260915
    python -m dailybench generate --seed 20260915 --out suite/20260915
    python -m dailybench selfcheck
    python -m dailybench run --base-url http://127.0.0.1:1234/v1 --model qwen3.8-27b \\
        --quant "Q4_K_M" --name qwen-plain --speed
    python -m dailybench bench --config qwen3.8-27b-ud-q3-k-xl --speed
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
        base_url=getattr(args, "base_url", "") or None,
        model=getattr(args, "model", "") or None,
        api_key=getattr(args, "api_key", "") or None,
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
    if getattr(args, "agent", True) is False:
        cfg.agent = False
    if getattr(args, "agent_categories", ""):
        cfg.agent_categories = tuple(
            part.strip().upper() for part in args.agent_categories.split(",") if part.strip()
        )
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
    from . import agent
    from .tasks.coding_go import _go_binary

    info["go"] = _go_binary() or "missing"
    info["agent"] = {
        "pi": agent.binary() or "missing",
        "install": agent.install_command(),
        "categories": list(RunConfig().agent_categories),
        "tools": list(agent.AGENT_TOOLS),
    }
    info["docker"] = shutil.which("docker") or "missing"
    try:
        import sqlite3

        info["sqlite"] = sqlite3.sqlite_version
    except Exception as exc:  # pragma: no cover
        info["sqlite"] = f"unavailable: {exc}"

    try:
        from . import browser, visual

        usable, reason = browser.availability()
        info["visual_block"] = {
            "usable": usable,
            "reason": reason or "ok",
            "three": str(browser.VENDOR / "three.module.js"),
            "cache": str(visual.cache_dir()),
        }
    except Exception as exc:  # pragma: no cover - depends on the machine
        info["visual_block"] = {"usable": False, "reason": f"{exc.__class__.__name__}: {exc}"}

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
    total = sum(task.max_points for task in tasks if task.category in core.SCORED_CATEGORIES)
    showcase = [task for task in tasks if task.category in core.VISUAL_CATEGORIES]
    note = f", витрина: {len(showcase)} заданий ({sum(t.max_points for t in showcase):.0f} вне счёта)" if showcase else ""
    print(f"\nЗаданий: {len(tasks)}, сумма баллов: {total:.0f}{note}, seed={cfg.seed}")
    if args.prompts:
        for task in tasks:
            print("\n" + "-" * 78)
            title = task.title
            for prefix in (f"{task.id} — ", f"{task.id} - "):
                if title.startswith(prefix):
                    title = title[len(prefix):]
                    break
            print(f"{task.id} — {title} ({task.max_points:.0f} баллов)")
            print(task.prompt)
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

    resuming = getattr(args, "resume", False)
    saved_config = cfg.out_dir / "config.json"
    if resuming and saved_config.is_file():
        # The calibration defines the long-context document sizes, hence the
        # bytes of the tasks: reuse the saved numbers, or a fresh (slightly
        # different) ratio would regenerate a different suite and the merge
        # guard below would reject it.
        restored = RunConfig.from_dict(json.loads(saved_config.read_text(encoding="utf-8")))
        cfg.chars_per_token = restored.chars_per_token
        cfg.task_ratios = dict(restored.task_ratios)
        print(
            f"калибровка взята из сохранённого config.json: {cfg.chars_per_token:.4f} символов/токен",
            flush=True,
        )

    if getattr(args, "calibrate", True) and not (resuming and saved_config.is_file()):
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

    saved_runs: list = []
    if getattr(args, "resume", False):
        tasks, saved_runs = _resume_tasks(cfg, tasks)
        print(
            f"Догрузка: {len(tasks)} заданий поверх {len(saved_runs)} сохранённых"
            if tasks
            else f"Все задания уже сохранены ({len(saved_runs)}); только отчёт и профиль скорости",
            flush=True,
        )

    runner = Runner(cfg, provider)
    started = datetime.now()
    print(
        f"Запуск {len(tasks)} заданий: model={cfg.model} quant={cfg.quant or 'n/a'} "
        f"seed={cfg.seed} out={cfg.out_dir}"
        + (f" reference={cfg.reference}" if cfg.reference else "")
        + (f" не проверяются: {','.join(cfg.unsupported)}" if cfg.unsupported else ""),
        flush=True,
    )

    runs: list = list(saved_runs)
    result = RunResult(
        cfg=cfg,
        runs=runs,
        provider_info=provider.describe(),
        suite_digest=build_suite_digest(cfg),
        hardware=_hardware(),
    )
    if getattr(args, "resume", False):
        saved_speed = cfg.out_dir / "speed.json"
        if saved_speed.is_file():
            result.speed = json.loads(saved_speed.read_text(encoding="utf-8"))
    for index, task in enumerate(tasks, start=1):
        run = runner.run_task(task)
        _merge_run(runs, run)
        marker = "ERR" if run.score.error else f"{run.score.earned:.0f}/{run.score.possible:.0f}"
        extra = f" tokens={run.transcript.prompt_tokens()}" if run.transcript.prompt_tokens() else ""
        # Persist after every task, not only at the end: a single long run is
        # hours of GPU time, and a crash (or a debugger stop) at task 90 must
        # not lose 89 scored tasks.
        result.save()
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

    if speed_rows or not result.speed:
        result.speed = speed_rows
    if resuming:
        # Resumed tasks were appended at the tail; restore the suite order the
        # finished run must have — the report reads this list top to bottom.
        order = {task.id: index for index, task in enumerate(build_suite(cfg))}
        runs.sort(key=lambda item: order.get(item.task.id, 1 << 30))
    result.save()
    report_path = cfg.out_dir / "report.md"
    report_path.write_text(render_markdown(_load_for_render(cfg.out_dir)), encoding="utf-8")
    html_path: Path | None = None
    if getattr(args, "html", True):
        from .htmlreport import write_report_html

        try:
            html_path = write_report_html(cfg.out_dir)
        except Exception as exc:  # pragma: no cover - a broken frame must not lose the run
            print(f"HTML-отчёт не собрался: {type(exc).__name__}: {exc}", file=sys.stderr)

    summary = summarise([score.as_dict() for score in result.scores], result.suite_digest)
    print(f"\nИтог: {summary['verdict']}")
    print(f"Прошло: {(datetime.now() - started)}")
    print(f"Отчёт: {report_path}")
    if html_path:
        print(f"HTML-отчёт: {html_path}")
        print(f"живые сцены: make serve DIR={cfg.out_dir.parent}")
    if summary["errors"]:
        print(f"Прогонов с ошибками: {summary['errors']} (см. tasks.json)", file=sys.stderr)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """One command for the whole cycle: runtime up, preflight, run, report.

    ``run`` assumes a model is already serving; this composes the steps the
    reference workflow needs in order, so a single model can be measured with
    one command and the same code path the sweep uses.
    """
    from . import launch, reference

    cfg = _config_from(args)
    entry: dict = {}
    if args.config:
        entry = dict(reference.resolve(args.config, args.configs_path or None).entry)

    if entry and not args.no_launch:
        ok, note, key = launch.up(entry)
        print(f"runtime: {note}", flush=True)
        if not ok:
            print(
                f"не удалось поднять рантайм; продолжаю с endpoint'ом {cfg.base_url}",
                file=sys.stderr,
            )
        elif key and not args.api_key:
            args.api_key = key

    if args.config and not args.no_preflight:
        report = reference.preflight(reference.resolve(args.config, args.configs_path or None))
        print(report.render(), flush=True)
        if not report.ok:
            print("preflight отклонил конфигурацию: прогон не начат", file=sys.stderr)
            return 1

    return cmd_run(build_parser().parse_args(_bench_run_argv(args)))


def _bench_run_argv(args: argparse.Namespace) -> list[str]:
    """Translate the bench namespace into the equivalent ``run`` arguments.

    The run itself stays in one place: ``bench`` only decides which arguments
    reach it, so the two commands can never drift apart in behaviour.
    """
    argv = [
        "run",
        "--base-url", args.base_url,
        "--model", args.model,
        "--api-key", args.api_key,
        "--timeout", str(args.timeout),
        "--seed", str(args.seed),
        "--sql-backend", args.sql_backend,
        "--context-window", str(args.context_window),
        "--max-new-tokens", str(args.max_new_tokens),
        "--offload", args.offload,
    ]
    for flag, value in (
        ("--quant", args.quant),
        ("--runtime", args.runtime),
        ("--kv-type", args.kv_type),
        ("--weights", args.weights),
        ("--only", args.only),
        ("--name", args.name),
        ("--out-dir", args.out_dir),
        ("--unsupported", args.unsupported),
        ("--config", args.config),
        ("--configs-path", args.configs_path),
    ):
        if value:
            argv += [flag, str(value)]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.speed:
        argv += ["--speed", "--speed-long", str(args.speed_long), "--speed-repeats", str(args.speed_repeats)]
    if not args.calibrate:
        argv += ["--no-calibrate"]
    if args.no_system_prompt:
        argv += ["--no-system-prompt"]
    if not args.html:
        argv += ["--no-html"]
    if not args.agent:
        argv += ["--no-agent"]
    if args.agent_categories:
        argv += ["--agent-categories", args.agent_categories]
    if args.resume:
        argv += ["--resume"]
    return argv


def cmd_visual(args: argparse.Namespace) -> int:
    """Render one candidate page exactly as the visual check would.

    The debugging loop for the visual block: point it at an HTML file a model
    produced (or at a reference page) and read back the probe, the frame after
    the requested number of steps, and the frame statistics the image check
    looks at.
    """
    from . import browser, visual

    if not args.stdout_input and not args.page:
        print("укажи HTML-файл или --stdin", file=sys.stderr)
        return 2
    try:
        payload = sys.stdin.read() if args.stdout_input else Path(args.page).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"не читается {args.page}: {exc}", file=sys.stderr)
        return 2
    usable, reason = browser.availability()
    if not usable:
        print(f"браузер недоступен: {reason}", file=sys.stderr)
        return 2
    html = visual.extract_html(payload) or payload
    static_ok, static_detail = visual.looks_like_document(html)
    print(f"документ: {'ок' if static_ok else 'нет'} — {static_detail}")
    result = browser.render(html, steps=args.steps, dt=args.dt)
    print(f"рендер: {'ок' if result.ok else 'ошибка'} {result.reason}".rstrip())
    for line in result.errors[:5]:
        print(f"  ошибка страницы: {line}")
    for line in result.external[:5]:
        print(f"  внешний запрос: {line}")
    print("probe:", json.dumps(result.probe, ensure_ascii=False, sort_keys=True))
    if args.steps:
        print("после шагов:", json.dumps(browser.probe_of(result, True), ensure_ascii=False, sort_keys=True))
    print("кадр:", json.dumps(result.pixels, ensure_ascii=False, sort_keys=True))
    print(f"воспроизводимость кадра: {'да' if result.same_frame else 'нет'}")
    if args.shot:
        path = Path(args.shot)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(result.shot or b"")
        print(f"снимок: {path}")
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


def cmd_publish(args: argparse.Namespace) -> int:
    """Собрать витринный отчёт в git-каталог: графика и three.js внутри, runs/ вне.

    ``runs/`` воспроизводим и в git не нужен, но отчёт должен быть воспроизводим
    из репозитория как артефакт: ``publish`` кладёт в ``reports/<имя>/`` готовый
    ``index.html``, библиотеку three.js, plotly и копии страниц и кадров, ничего
    не читая из ``runs/`` после сборки.
    """
    from .htmlreport import write_html, write_report_html

    runs = [Path(directory) for directory in args.directories]
    for directory in runs:
        if not (directory / "tasks.json").is_file():
            raise FileNotFoundError(f"{directory}: нет tasks.json (прогон не сохранён)")
    out = Path(args.out) if args.out else Path("reports") / (
        runs[0].name if len(runs) == 1 else "comparison"
    )
    selected = [name.strip() for name in (getattr(args, "compare", "") or "").split(",") if name.strip()]
    page = write_html(runs, out / "index.html", compare=selected or None)
    print(f"отчёт: {page}")
    # Every published run also gets its own page inside the artefact: a reader who
    # wants one model must not have to open the comparison and hunt for it.
    for directory in runs:
        per_run = write_report_html(directory, out / "runs" / directory.name / "index.html")
        print(f"прогон: {per_run}")
    print(f"каталог: {out} — его и коммитить; runs/ остаётся локальным")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    paths = write_reports(args.directories)
    if not getattr(args, "no_html", False):
        from .htmlreport import write_html, write_report_html

        selected = [name.strip() for name in (getattr(args, "compare", "") or "").split(",") if name.strip()]
        for directory in args.directories:
            try:
                paths.append(write_report_html(directory))
            except Exception as exc:  # pragma: no cover - defensive
                print(f"{directory}: страница не собралась: {exc}", file=sys.stderr)
        paths.append(
            write_html(
                args.directories,
                Path(args.html) if args.html else None,
                compare=selected or None,
            )
        )
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


def _resume_tasks(cfg: RunConfig, tasks: list) -> tuple[list, list]:
    """Tasks still needing a fresh attempt, plus the saved runs worth keeping.

    A saved task is kept only when its transcript is *complete*: a reboot or a
    wall-clock timeout kills the agent mid-flight and leaves a truncated record,
    and re-running one task beats shipping half an artefact. The suite digest
    must match, or the merged run would mix two different task generations.
    """
    from .report import load_run  # noqa: F401 - symmetry with the render path
    from .runner import TaskRun, transcript_from_dict
    from .suite import build_suite, suite_digest

    directory = cfg.out_dir
    tasks_path = directory / "tasks.json"
    suite_path = directory / "suite.json"
    if not tasks_path.is_file() or not suite_path.is_file():
        return tasks, []

    saved_digest = json.loads(suite_path.read_text(encoding="utf-8"))
    full = build_suite(cfg)
    current = suite_digest(full)
    if saved_digest.get("sha256") != current.get("sha256"):
        raise SystemExit(
            f"{directory}: сьют разошёлся с сохранённым "
            f"({str(current.get('sha256'))[:12]} ≠ {str(saved_digest.get('sha256'))[:12]}) — "
            "догрузка смешала бы два сьюта"
        )

    by_id = {task.id: task for task in full}
    saved = _saved_entries(directory)
    kept: list = []
    damaged: set[str] = set()
    for entry in saved:
        task_id = str((entry.get("task") or {}).get("id"))
        task = by_id.get(task_id)
        if task is None:
            continue
        transcript = transcript_from_dict(entry.get("transcript") or {}, task)
        score = core.Score.from_dict(entry["score"])
        if _truncated(transcript):
            damaged.add(task_id)
            continue
        kept.append(TaskRun(task=task, transcript=transcript, score=score))

    kept_ids = {run.task.id for run in kept}
    missing = [task for task in tasks if task.id not in kept_ids]
    order = {task.id: index for index, task in enumerate(full)}
    missing.sort(key=lambda task: order.get(task.id, 1 << 30))
    kept.sort(key=lambda run: order.get(run.task.id, 1 << 30))
    if damaged:
        print(f"повтор из-за обрыва: {', '.join(sorted(damaged, key=lambda i: order.get(i, 0)))}", flush=True)
    return missing, kept


def _saved_entries(directory: Path) -> list[dict]:
    """Records worth merging back, from ``tasks.json`` or its per-task mirror.

    ``tasks.json`` is rewritten whole at every checkpoint; a reboot exactly while
    it was being written used to leave a truncated file and resume died on
    ``json.loads``. Each record is also mirrored under ``transcripts/``, so a
    corrupt master is rebuilt from there instead of losing hours of GPU time.
    """
    tasks_path = directory / "tasks.json"
    entries: list[dict] = []
    seen: set[str] = set()
    if tasks_path.is_file():
        try:
            raw = json.loads(tasks_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"{tasks_path}: битый JSON ({exc}) — собираю из transcripts/", flush=True)
        else:
            for entry in raw:
                task_id = str((entry.get("task") or {}).get("id"))
                if task_id not in seen:
                    seen.add(task_id)
                    entries.append(entry)
    folder = directory / "transcripts"
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            task_id = str((entry.get("task") or {}).get("id"))
            if task_id and task_id not in seen:
                seen.add(task_id)
                entries.append(entry)
    return entries


def _truncated(transcript: core.Transcript) -> bool:
    """Was the agent killed rather than finished?

    The exit code is the marker: a clean ``pi`` run says ``exit_ok``, while a
    timeout, a SIGTERM or an aborted turn leaves it false or records the error.
    """
    if transcript.error:
        return True
    agent = (transcript.extra or {}).get("agent") or {}
    if agent and agent.get("exit_ok") is False:
        return True
    reasons = transcript.spend().get("finish_reasons") or []
    return any(str(reason) in ("error", "aborted") for reason in reasons)


def _merge_run(runs: list, run: Any) -> None:
    """Replace the saved result of the same task, or append the new one."""
    for index, existing in enumerate(runs):
        if existing.task.id == run.task.id:
            runs[index] = run
            return
    runs.append(run)


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
    parser = argparse.ArgumentParser(
        prog="dailybench",
        description="LLMCrew DailyBench-16 — generated, programmatically scored local-LLM suite",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser, *, endpoint: bool = False) -> None:
        target.add_argument("--seed", type=int, default=DEFAULT_SEED, help="master seed of the suite")
        target.add_argument("--only", default="", help="категории или ID заданий через запятую (A,B / IF-1,R-7)")
        target.add_argument("--limit", type=int, default=0, help="ограничить число заданий (для дымового прогона)")
        target.add_argument("--sql-backend", default="sqlite", choices=("sqlite", "postgres"))
        if endpoint:
            # Empty default = "take it from the environment" (see RunConfig),
            # so a clone never has to pass its endpoint on every command.
            target.add_argument("--base-url", default="")
            target.add_argument("--model", default="")
            target.add_argument("--api-key", default="")
            target.add_argument("--timeout", type=float, default=1800.0)
            target.add_argument("--quant", default="", help="метка кванта для отчёта, напр. IQ4_XS")
            target.add_argument("--runtime", default="", help="runtime и версия, напр. llama.cpp b4321")
            target.add_argument("--kv-type", default="", help="формат KV-cache, напр. q8_0")
            target.add_argument("--offload", default="none", help="описание offload (none — обязательный зачёт)")
            target.add_argument("--context-window", type=int, default=32768)
            target.add_argument("--max-new-tokens", type=int, default=4096)
            target.add_argument("--weights", default="", help="путь к файлу весов для SHA-256")
            target.add_argument("--no-system-prompt", action="store_true")
            target.add_argument(
                "--no-html",
                dest="html",
                action="store_false",
                help="не собирать HTML-отчёт прогона (остаётся только report.md)",
            )
            target.add_argument(
                "--no-agent",
                dest="agent",
                action="store_false",
                help="не поднимать pi для категорий C/K/L (обычный chat-цикл)",
            )
            target.add_argument(
                "--agent-categories",
                default="",
                help="какие категории отдать агенту (по умолчанию C,K,L)",
            )

    doctor = sub.add_parser("doctor", help="проверить GPU, toolchain и endpoint")
    add_common(doctor, endpoint=True)
    doctor.add_argument("--config", default="", help="id эталонной конфигурации")
    doctor.add_argument("--configs-path", default="")
    doctor.set_defaults(func=cmd_doctor)

    listing = sub.add_parser("list", help="показать задания сьюта без запуска")
    add_common(listing)
    listing.add_argument("--prompts", action="store_true", help="напечатать промпты целиком")
    listing.add_argument("--config", default="", help="id эталонной конфигурации (окно, бюджеты, политика)")
    listing.add_argument("--configs-path", default="")
    listing.set_defaults(func=cmd_list)

    generate = sub.add_parser("generate", help="записать fixtures сьюта в каталог")
    add_common(generate)
    generate.add_argument("--out", required=True)
    generate.add_argument("--config", default="", help="id эталонной конфигурации (окно влияет на long-context)")
    generate.add_argument("--configs-path", default="")
    generate.set_defaults(func=cmd_generate)

    check = sub.add_parser("selfcheck", help="проверить генераторы и evaluator'ы без GPU")
    add_common(check)
    check.add_argument("--report", default="")
    check.add_argument("--config", default="", help="id эталонной конфигурации (окно влияет на long-context)")
    check.add_argument("--configs-path", default="")
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
    run.add_argument(
        "--resume",
        action="store_true",
        help="догрузить недостающие задания в существующий каталог прогона "
        "(прерванные задачи пере-прогоняются, сохранённые не пересъёмлются)",
    )
    run.set_defaults(func=cmd_run)

    bench = sub.add_parser("bench", help="один запуск целиком: runtime, preflight, сьют, отчёт")
    add_common(bench, endpoint=True)
    bench.add_argument("--config", default="", help="id эталонной конфигурации из configs/reference.json")
    bench.add_argument("--configs-path", default="", help="путь к файлу эталонных конфигураций")
    bench.add_argument("--unsupported", default="", help="категории, которые эта конфигурация не проверяет")
    bench.add_argument("--name", default="")
    bench.add_argument("--out-dir", default="")
    bench.add_argument("--speed", action="store_true", help="снять профиль скорости и VRAM")
    bench.add_argument("--speed-long", type=int, default=0)
    bench.add_argument("--speed-repeats", type=int, default=3)
    bench.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    bench.add_argument("--no-launch", action="store_true", help="не поднимать рантайм, работать с запущенным")
    bench.add_argument("--no-preflight", action="store_true", help="не проверять конфигурацию перед прогоном")
    bench.add_argument("--resume", action="store_true", help="догрузить прерванный прогон в тот же каталог")
    bench.set_defaults(func=cmd_bench)

    visual = sub.add_parser("visual", help="отрисовать одну страницу так же, как это делает проверка K")
    visual.add_argument("page", nargs="?", default="", help="HTML-файл (или --stdin)")
    visual.add_argument("--stdin", dest="stdout_input", action="store_true", help="читать ответ со стандартного ввода")
    visual.add_argument("--steps", type=int, default=0, help="сколько шагов симуляции сделать перед снимком")
    visual.add_argument("--dt", type=float, default=1 / 60)
    visual.add_argument("--shot", default="", help="куда сохранить PNG")
    visual.set_defaults(func=cmd_visual)

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
    report.add_argument("--html", default="", help="куда положить HTML-отчёт (по умолчанию comparison.html рядом с прогонами)")
    report.add_argument("--no-html", action="store_true", help="только markdown, без HTML")
    report.add_argument(
        "--compare",
        default="",
        help="какие модели сравнить в comparison.html: метки через запятую "
             "(по умолчанию — все переданные прогоны)",
    )
    report.set_defaults(func=cmd_report)

    publish = sub.add_parser(
        "publish",
        help="собрать отчёт-артефакт в git-каталог (reports/): страницы, кадры и библиотеки внутри",
    )
    publish.add_argument("directories", nargs="+")
    publish.add_argument("--out", default="", help="каталог отчёта (по умолчанию reports/<имя прогона>)")
    publish.add_argument(
        "--compare",
        default="",
        help="какие модели сравнить: метки через запятую (по умолчанию — все переданные прогоны)",
    )
    publish.set_defaults(func=cmd_publish)

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
