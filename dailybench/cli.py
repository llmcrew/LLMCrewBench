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
from dataclasses import replace
import shutil
import sys
from datetime import datetime
from pathlib import Path

from . import core, generation, holdout, reference, schema
from .config import DEFAULT_CONTEXT_WINDOW, DEFAULT_SEED, RunConfig
from .measure import (
    SpeedResult,
    gpu_inventory,
    speed_profile,
    used_mib,
    weights_sha256,
)
from .providers import OpenAICompatibleProvider, Provider, ProviderError
from .report import render_markdown, summarise, write_reports
from .runner import RunResult, Runner, run_identity
from .adapters.bigcodebench import add_subcommands as add_adapter_commands
from .decontam import add_subcommands as add_decontam_commands
from .selfcheck import save_suite, selfcheck, write_selfcheck_report
from .suite import build_suite, fit_suite_sizing, public_prompts

#: Upper bound for a document handed to ``validate``. The biggest strict suite
#: exports to about 2 MB, so 16 MiB leaves room for YAML while stopping a file
#: that would otherwise be walked element by element.
MAX_DOCUMENT_BYTES = 16 * 1024 * 1024


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
        extra_body=dict(cfg.extra_body or {}),
    )


def _json_object(text: str) -> dict:
    if not text:
        return {}
    import json as _json

    value = _json.loads(text)
    if not isinstance(value, dict):
        raise SystemExit("--extra-body: нужен JSON-объект")
    return value


def _config_from(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig(
        base_url=getattr(args, "base_url", "") or None,
        model=getattr(args, "model", "") or None,
        api_key=getattr(args, "api_key", "") or None,
        request_timeout=getattr(args, "timeout", 1800.0),
        seed=args.seed,
        profile=getattr(args, "profile", core.DEFAULT_PROFILE),
        quant=getattr(args, "quant", "") or "",
        runtime=getattr(args, "runtime", "") or "",
        weights_path=getattr(args, "weights", "") or "",
        n_ctx=getattr(args, "context_window", DEFAULT_CONTEXT_WINDOW),
        max_new_tokens=getattr(args, "max_new_tokens", None) or 4096,
        # The flag is an upper bound on one answer, not a table value the window
        # branch may ignore: 259834 used to go out after ``--max-new-tokens 32``.
        answer_cap=getattr(args, "max_new_tokens", None) or 4096,
        # ``None`` unless the flag was typed: ``__post_init__`` then derives the
        # total from the answer budget instead of clamping it to 16384.
        max_total_tokens=getattr(args, "max_total_tokens", None),
        kv_type=getattr(args, "kv_type", "") or "",
        offload=getattr(args, "offload", "none"),
        sql_backend=args.sql_backend,
        extra_body=_json_object(getattr(args, "extra_body", "")),
    )
    if args.only:
        cfg.only = tuple(part.strip() for part in args.only.split(",") if part.strip())
    if getattr(args, "max_new_tokens", None):
        # ``max_new_tokens`` and the cap already went through the constructor,
        # so the derived total saw the enlarged answer.
        cfg.max_new_tokens = args.max_new_tokens
    if getattr(args, "generation_mode", ""):
        cfg.generation_mode = args.generation_mode
    if getattr(args, "reasoning_budget", None) is not None:
        cfg.reasoning_budget = args.reasoning_budget
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
    if getattr(args, "sandbox", True) is False:
        cfg.sandbox = False
    if getattr(args, "agent_categories", ""):
        cfg.agent_categories = tuple(
            part.strip().upper() for part in args.agent_categories.split(",") if part.strip()
        )
    if getattr(args, "unsupported", ""):
        extra = tuple(part.strip().upper() for part in args.unsupported.split(",") if part.strip())
        cfg.unsupported = tuple(dict.fromkeys(cfg.unsupported + extra))
    # Provenance lanes from ``bench``/``sweep``: what the runtime reported and
    # what the launcher really sent. Absent under plain ``run``.
    provenance: dict = {}
    provenance_path = getattr(args, "provenance", "")
    if provenance_path and Path(provenance_path).is_file():
        provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    cfg.observed = dict(getattr(args, "observed", None) or provenance.get("observed") or {})
    cfg.launch_executed = dict(getattr(args, "executed", None) or provenance.get("executed") or {})
    if provenance.get("weights_sha256") and not cfg.weights_sha256:
        cfg.weights_sha256 = str(provenance["weights_sha256"])
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
    for entry in public_prompts(tasks, args.include_private):
        print(
            f"{entry['id']:6} {entry['max_points']:>6.0f} {(entry.get('prompt_chars') or 0):>9} "
            f"{(entry.get('estimated_tokens') or 0):>9}  {entry.get('title') or entry['id']}"
        )
    total = sum(task.max_points for task in tasks if task.category in core.SCORED_CATEGORIES)
    showcase = [task for task in tasks if task.category in core.VISUAL_CATEGORIES]
    note = f", витрина: {len(showcase)} заданий ({sum(t.max_points for t in showcase):.0f} вне счёта)" if showcase else ""
    print(f"\nЗаданий: {len(tasks)}, сумма баллов: {total:.0f}{note}, seed={cfg.seed}")
    if args.prompts:
        shown = [task for task in tasks if args.include_private or not task.private]
        hidden = len(tasks) - len(shown)
        for task in shown:
            print("\n" + "-" * 78)
            title = task.title
            for prefix in (f"{task.id} — ", f"{task.id} - "):
                if title.startswith(prefix):
                    title = title[len(prefix):]
                    break
            marker = " · приватное" if task.private else ""
            print(f"{task.id} — {title} ({task.max_points:.0f} баллов){marker}")
            print(task.prompt)
        if hidden:
            print(f"\nПриватных заданий скрыто: {hidden} (добавьте --include-private)")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    cfg = _config_from(args)
    written = save_suite(cfg, args.out, include_private=args.include_private)
    for name, path in written.items():
        print(f"{name}: {path}")
    print(f"\nПрофиль: {cfg.profile}. Файл prompts.json можно публиковать; oracles.json содержит ответы.")
    return 0


def _read_document(path: Path) -> Any:
    """Read a JSON or YAML document by extension, then by content.

    The size is capped first: a YAML alias bomb is tiny on disk but expensive
    to walk, and a full strict suite is a couple of megabytes at most.
    """
    size = path.stat().st_size
    if size > MAX_DOCUMENT_BYTES:
        raise ValueError(
            f"{path}: документ больше {MAX_DOCUMENT_BYTES} байт ({size} байт)"
        )
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"{path}: не разбирается как YAML: {exc}") from exc
    if path.suffix.lower() != ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            import yaml

            try:
                return yaml.safe_load(text)
            except yaml.YAMLError as exc:
                raise ValueError(f"{path}: не разбирается ни как JSON, ни как YAML: {exc}") from exc
    return json.loads(text)


def _suite_file(path: Path) -> Path:
    if not path.is_dir():
        return path
    for name in ("suite.json", "suite.yaml", "suite.yml", "prompts.json", "prompts.yaml"):
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"в каталоге {path} нет suite.json/prompts.json")


def _report_errors(errors: list[str], subject: str) -> int:
    for error in errors:
        print(f"  {error}", file=sys.stderr)
    print(f"{subject}: {len(errors)} ошибок", file=sys.stderr)
    return 1


def cmd_validate(args: argparse.Namespace) -> int:
    if args.task:
        path = Path(args.task)
        try:
            doc = _read_document(path)
        except (OSError, ValueError) as exc:
            print(f"  {exc}", file=sys.stderr)
            return 1
        errors = schema.validate_task(doc, str(path))
        if errors:
            return _report_errors(errors, str(path))
        print(f"{path}: задание валидно")
        return 0

    if args.suite:
        try:
            path = _suite_file(Path(args.suite))
            doc = _read_document(path)
        except (OSError, ValueError) as exc:
            print(f"  {exc}", file=sys.stderr)
            return 1
        errors = schema.validate_suite_doc(doc, check_totals=not args.no_totals)
        if errors:
            return _report_errors(errors, str(path))
        tasks = doc.get("tasks") if isinstance(doc, dict) else []
        print(f"{path}: сьют валиден ({len(tasks)} заданий)")
        return 0

    cfg = _config_from(args)
    tasks = build_suite(cfg)
    errors: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        if task.id in seen:
            errors.append(f"task {task.id}: duplicate task id")
        seen.add(task.id)
        errors.extend(schema.validate_task(task, f"task {task.id}"))
    if errors:
        return _report_errors(errors, f"сгенерированный сьют {cfg.profile}")
    print(f"сгенерированный сьют {cfg.profile}: валиден ({len(tasks)} заданий)")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    cfg = _config_from(args)
    tasks = build_suite(cfg)
    include_private = bool(args.include_private)
    manifest = None
    hidden: dict[str, str] = {}
    if getattr(args, "holdout", ""):
        try:
            manifest = holdout.load_manifest(Path(args.holdout))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  {exc}", file=sys.stderr)
            return 1
        if not args.include_closed:
            hidden = holdout.hidden_ids(manifest)
    doc = schema.export_document(
        cfg.profile, tasks, include_private=include_private, hidden=hidden
    )
    errors = schema.validate_suite_doc(doc, check_totals=not cfg.only)
    if manifest is not None and not args.include_closed:
        errors.extend(holdout.check_export(doc, manifest))
    if errors:
        return _report_errors(errors, f"экспорт {cfg.profile}")

    if args.format == "yaml":
        import yaml

        text = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)
    else:
        text = json.dumps(doc, ensure_ascii=False, indent=2, default=core.json_default) + "\n"

    if args.out and args.out != "-":
        target = Path(args.out)
        if target.is_dir() or args.out.endswith(("/", "\\")):
            target.mkdir(parents=True, exist_ok=True)
            target = target / f"suite.{args.format}"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"{target}: {len(doc['tasks'])} заданий")
    else:
        print(text, end="" if text.endswith("\n") else "\n")
    return 0


def cmd_holdout_assign(args: argparse.Namespace) -> int:
    cfg = _config_from(args)
    tasks = build_suite(cfg)
    overrides: dict[str, str] = {}
    for item in args.pin or []:
        identifier, _, split = item.partition("=")
        if not identifier or split not in holdout.SPLITS:
            print(f"  --pin {item!r}: ожидалось ID=сплит из {', '.join(holdout.SPLITS)}", file=sys.stderr)
            return 1
        overrides[identifier] = split
    try:
        manifest = holdout.build_manifest(
            tasks,
            profile=cfg.profile,
            seed=cfg.seed,
            closed_share=args.closed_share,
            audit_share=args.audit_share,
            overrides=overrides,
        )
    except ValueError as exc:
        print(f"  {exc}", file=sys.stderr)
        return 1
    errors = holdout.validate_manifest(manifest)
    if errors:
        return _report_errors(errors, "манифест holdout")
    holdout.save_manifest(Path(args.out), manifest)
    sizes = ", ".join(f"{name}={len(manifest['splits'][name])}" for name in holdout.SPLITS)
    print(f"{args.out}: {sizes}, digest {manifest['digest']}")
    return 0


def cmd_holdout_check(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    try:
        manifest = holdout.load_manifest(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  {exc}", file=sys.stderr)
        return 1
    errors = holdout.validate_manifest(manifest, str(manifest_path))
    if errors:
        return _report_errors(errors, str(manifest_path))
    cfg = _config_from(args)
    tasks = build_suite(cfg)
    errors = holdout.check_manifest(manifest, tasks)
    doc = schema.export_document(
        cfg.profile, tasks, hidden=holdout.hidden_ids(manifest)
    )
    errors.extend(holdout.check_export(doc, manifest))
    if errors:
        return _report_errors(errors, str(manifest_path))
    print(f"{manifest_path}: сплиты и хеши совпадают, экспорт закрыт")
    return 0


def cmd_licenses_check(args: argparse.Namespace) -> int:
    explicit = [Path(item) for item in args.paths]
    if not explicit:
        failures = holdout.check_directory(Path(args.dir) if args.dir else None)
    else:
        failures = []
        for path in explicit:
            try:
                record = holdout.load_record(path)
            except (OSError, json.JSONDecodeError) as exc:
                failures.append((path, [f"{path}: {exc}"]))
                continue
            errors = holdout.validate_record(record, str(path))
            if isinstance(record, dict) and record.get("id") != path.stem:
                errors.append(f"{path}/id: {record.get('id')!r} не совпадает с именем файла")
            if errors:
                failures.append((path, errors))
    if failures:
        count = sum(len(errors) for _, errors in failures)
        for path, errors in failures:
            for error in errors:
                print(f"  {error}", file=sys.stderr)
        print(f"права: {count} ошибок в {len(failures)} записях", file=sys.stderr)
        return 1
    print(f"права: все записи пригодны к импорту ({len(explicit) if explicit else 'каталог'})")
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


def _executor_environment(cfg: RunConfig) -> dict[str, Any]:
    """The executor record a run of ``cfg`` writes, reasons included.

    One builder for the run and for the resume guard: the guard rebuilt a bare
    ``generation.environment()``, so a saved record that said ``container_run``
    (with the CA it mounted) never matched and every resume was refused.
    """
    from . import isolation

    if cfg.sandbox:
        # Builds a missing image here, once, before the first task; the runner
        # reuses the remembered answer. An unusable sandbox is recorded with its
        # own status: a failed build must not look like a deliberate host run.
        usable, reason, status = isolation.prepare()
    else:
        usable, reason, status = False, "", ""
    environment = generation.environment(container_ran=cfg.sandbox and usable)
    cfg.image_digest = str(environment.get("image_digest") or "")
    if not cfg.sandbox:
        environment["executor_status"] = "host_run"
        environment["executor_reason"] = "sandbox выключен (--no-sandbox)"
    elif not usable:
        environment["executor_status"] = status or "host_run"
        environment["executor_reason"] = f"контейнер недоступен ({reason}); агент пойдёт на хосте"
    # The CA is only a mount when a container really will start; a host run
    # verifies with the host store anyway.
    ca = isolation.ca_bundle() if cfg.sandbox and usable else None
    environment["ca_mounted"] = ca is not None
    environment["ca_bundle"] = str(ca or "")
    environment["ca_container_path"] = isolation.CONTAINER_CA if ca else ""
    environment["ca_reason"] = (
        "" if ca or not cfg.sandbox or not usable else isolation.CA_MISSING_REASON
    )
    return environment


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

    # Provenance of the executor is captured once per run: the digest joins
    # the identity below, so a rebuilt image is a different run, and the
    # report can state what the numbers were measured on. ``container_ran`` is
    # decided up front by the same probe the runner uses, so a run that will
    # execute in a container does not first claim to be a host run.
    from . import isolation

    environment = _executor_environment(cfg)
    if environment.get("status") != "verified":
        print(
            f"среда песочницы не проверена: {environment.get('reason') or 'нет данных'}",
            flush=True,
        )
    # A host CA is what makes the container's TLS verification mean anything.
    # Missing is a warning, never a stop: an http endpoint does not need one.
    if environment.get("ca_reason"):
        print(f"CA: {isolation.CA_MISSING_REASON}", flush=True)
    if environment.get("executor_status") != "container_run":
        print(
            f"исполнитель не подтверждён контейнером: "
            f"{environment.get('executor_reason') or environment.get('executor_status') or 'нет данных'}",
            flush=True,
        )

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
        tasks, saved_runs = _resume_tasks(cfg, tasks, environment)
        print(
            f"Догрузка: {len(tasks)} заданий поверх {len(saved_runs)} сохранённых"
            if tasks
            else f"Все задания уже сохранены ({len(saved_runs)}); только отчёт и профиль скорости",
            flush=True,
        )

    runner = Runner(cfg, provider, environment=environment)
    started = datetime.now()
    print(
        f"Запуск {len(tasks)} заданий: model={cfg.model} quant={cfg.quant or 'n/a'} "
        f"seed={cfg.seed} profile={cfg.profile} out={cfg.out_dir}"
        + (f" reference={cfg.reference}" if cfg.reference else "")
        + (f" не проверяются: {','.join(cfg.unsupported)}" if cfg.unsupported else ""),
        flush=True,
    )

    runs: list = list(saved_runs)
    attempted: list = []
    result = RunResult(
        cfg=cfg,
        runs=runs,
        provider_info=provider.describe(),
        suite_digest=build_suite_digest(cfg),
        hardware=_hardware(),
        environment=environment,
    )
    if getattr(args, "resume", False):
        saved_speed = cfg.out_dir / "speed.json"
        if saved_speed.is_file():
            result.speed = json.loads(saved_speed.read_text(encoding="utf-8"))
    for index, task in enumerate(tasks, start=1):
        run = runner.run_task(task)
        attempted.append(run)
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
    if not summary["eligible"]:
        kinds = ", ".join(
            f"{core.failure_label(kind)} x{count}" for kind, count in summary["infra_failures"].items()
        )
        reason = "удостоверение отозвано" if summary.get("integrity_failures") else "инфраструктура"
        print(f"прогон неранжируем: {reason} — {kinds}", file=sys.stderr)
    if not summary.get("rankable_with_baseline", True):
        print(
            f"профиль {cfg.profile} не ранжируется с baseline: он несёт свой набор "
            f"категорий и знаменатель {core.scored_total(cfg.profile):.0f}",
            file=sys.stderr,
        )
    infra = [run for run in attempted if run.transcript.extra.get("infra_error")]
    if attempted and len(infra) == len(attempted):
        # Transport/protocol failure is not a wrong answer: every chat request
        # died before the model could reply, so the score describes nothing.
        print(
            f"все chat-запросы упали инфраструктурно ({len(infra)}/{len(attempted)}): "
            "результат не отражает модель",
            file=sys.stderr,
        )
        return 2
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """One command for the whole cycle: runtime up, preflight, run, report.

    ``run`` assumes a model is already serving; this composes the steps the
    reference workflow needs in order, so a single model can be measured with
    one command and the same code path the sweep uses.

    The effective configuration is resolved exactly once here and handed to the
    launcher, the preflight and the run: resolving the window separately in each
    step was what let the same entry run at two different windows. A launcher
    failure ends the command — otherwise the previous model would be measured
    under the new entry's name.
    """
    from . import launch, reference

    cfg = _config_from(args)
    resolved = reference.resolve(args.config, args.configs_path or None) if args.config else None
    if resolved is not None:
        effective = resolved.effective()
        # The run must use the entry's own window, not the CLI default.
        args.context_window = effective["context_window"]

    if resolved is not None and not args.no_launch:
        ok, note, key = launch.up(effective)
        print(f"runtime: {note}", flush=True)
        if not ok:
            print(
                f"рантайм не поднялся ({note}); прогон не начат",
                file=sys.stderr,
            )
            return 2
        if key and not args.api_key:
            args.api_key = key

    observed: dict = {}
    if resolved is not None and not args.no_preflight:
        # ``full`` is not optional here: without the window probe the admission
        # claim (32K of context on the card) was never measured.
        report = reference.preflight(resolved, full=True, hash_weights=True)
        print(report.render(), flush=True)
        if not report.ok:
            print("preflight отклонил конфигурацию: прогон не начат", file=sys.stderr)
            return 1
        if not report.eligible:
            print(
                "preflight не подтвердил конфигурацию целиком ("
                + ", ".join(report.unverified)
                + "): результат нельзя ранжировать по весам",
                file=sys.stderr,
            )
            if not args.allow_unverified:
                print("прогон не начат; повторить можно с --allow-unverified", file=sys.stderr)
                return 3
        observed = dict(report.observed)

    run_args = build_parser().parse_args(_bench_run_argv(args))
    # The observed snapshot and the executed load come from the steps above, not
    # from the command line: passing them through the namespace keeps the report
    # from presenting the launch template as the executed command.
    run_args.observed = observed
    run_args.executed = dict(effective.get("executed") or {}) if resolved is not None else {}
    return cmd_run(run_args)


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
        "--profile", args.profile,
        "--sql-backend", args.sql_backend,
        "--context-window", str(args.context_window),
        "--generation-mode", args.generation_mode,
        "--reasoning-budget", str(args.reasoning_budget),
        "--offload", args.offload,
    ]
    for flag, value in (
        # ``None`` (flag not typed) stays unset, so the run derives the total
        # from its own answer budget instead of inheriting a stale default.
        ("--max-new-tokens", args.max_new_tokens),
        ("--max-total-tokens", args.max_total_tokens),
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
    if not getattr(args, "sandbox", True):
        argv += ["--no-sandbox"]
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
    from . import generation
    from .report import load_run

    # A broken manifest is quarantined: report it instead of reporting root
    # bytes that no longer belong to any sealed checkpoint.
    for directory in args.directories:
        try:
            load_run(directory)
        except generation.IntegrityError as exc:
            print(f"integrity: {exc}", file=sys.stderr)
            return 2
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
        data = load_run(directory)["scores"]
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
    if args.target == "sandbox":
        return _preflight_sandbox()
    if not args.config:
        print("нужен --config <ID> (или 'preflight sandbox')", file=sys.stderr)
        return 2
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


def _preflight_sandbox() -> int:
    """Check (and build) the sandbox image without a model or a config.

    This is the one place where a missing image is repaired *on demand* instead
    of being discovered by the first task, and it is what ``make
    preflight-sandbox`` calls: the same probe the run uses, with the same reason
    strings, so a green preflight is evidence for the run that follows.
    """
    from . import isolation

    isolation.forget_build_state()
    ca = isolation.ca_bundle()
    if ca is None:
        print(f"CA: {isolation.CA_MISSING_REASON}", file=sys.stderr, flush=True)
    else:
        print(f"CA: {ca} → {isolation.CONTAINER_CA} (read-only)", flush=True)
    usable, reason, status = isolation.prepare()
    if usable:
        print(
            f"песочница готова: образ {isolation.default_image()} "
            f"(runtime {isolation.default_runtime()})",
            flush=True,
        )
        return 0
    print(
        f"песочница недоступна ({status or 'host_run'}): {reason}",
        file=sys.stderr,
        flush=True,
    )
    return 2


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
            if "before" in row:
                before, after = row["before"], row["after"]
                print(
                    f"  {row['task_id']}: {before['status']} {before['earned']}/{before['possible']} "
                    f"→ {after['status']} {after['earned']}/{after['possible']} "
                    f"(допуск {before['admission']} → {after['admission']})"
                )
            else:
                print(f"  {row['task_id']}: {row['status']}")
    return 1 if failed else 0


def build_suite_digest(cfg: RunConfig) -> dict:
    from .suite import suite_digest

    # ``--only`` filters the run, not the suite identity.
    full = build_suite(replace(cfg, only=()) if cfg.only else cfg)
    return suite_digest(full, cfg.profile)


def _resume_tasks(
    cfg: RunConfig, tasks: list, environment: dict | None = None
) -> tuple[list, list]:
    """Tasks still needing a fresh attempt, plus the saved runs worth keeping.

    A saved task is kept only when its transcript is *complete*: a reboot or a
    wall-clock timeout kills the agent mid-flight and leaves a truncated record,
    and re-running one task beats shipping half an artefact. The suite digest
    and the full run identity (config contract, suite, harness version) must
    match, or the merged run would mix two different task generations.

    Mirrors under ``transcripts/`` are a recovery aid, not a second source of
    truth: a valid master already lists every kept task, so old mirrors must
    not add tasks back. Without a master there is no checkpoint to anchor them
    and the mirrors are ignored.
    """
    from .runner import TaskRun, transcript_from_dict
    from .suite import build_suite, suite_digest

    directory = cfg.out_dir
    tasks_path = directory / "tasks.json"
    suite_path = directory / "suite.json"
    if not tasks_path.is_file() or not suite_path.is_file():
        return tasks, []
    if environment is None:
        # Only direct callers reach this: the guard must compare against the
        # record a run would write, not a bare ``environment()`` without the
        # executor/CA facts the run adds.
        environment = _executor_environment(cfg)

    saved_digest = generation.read_json(directory, "suite.json")
    saved_profile = saved_digest.get("profile") or core.DEFAULT_PROFILE
    if saved_profile != cfg.profile:
        raise SystemExit(
            f"{directory}: сохранённый прогон собран в профиле {saved_profile}, "
            f"а сейчас {cfg.profile} — resume смешал бы версии сьюта"
        )
    # The digest is a property of the whole profile: a resume filtered to a few
    # ids must compare against the complete suite, not against its own slice.
    full_cfg = replace(cfg, only=()) if cfg.only else cfg
    full = build_suite(full_cfg)
    current = suite_digest(full, cfg.profile)
    # A limited resume carries a smaller denominator by design; the full one
    # must weigh exactly the profile's declared total.
    complete = len(tasks) >= len(full)
    if complete:
        expected_total = core.profile_total(cfg.profile)
        current_total = float(current.get("possible_total") or 0.0)
        if abs(current_total - expected_total) > 1e-9:
            raise SystemExit(
                f"{directory}: сьют весит {current_total:g} возможных баллов, а профиль "
                f"{cfg.profile} объявляет {expected_total:g} — resume смешал бы разные знаменатели"
            )
        saved_total = saved_digest.get("possible_total")
        if saved_total is not None and abs(float(saved_total) - current_total) > 1e-9:
            raise SystemExit(
                f"{directory}: в сохранённом прогоне {float(saved_total):g} возможных баллов, "
                f"в текущем сьюте {current_total:g} — resume смешал бы разные знаменатели"
            )
    if saved_digest.get("sha256") != current.get("sha256"):
        raise SystemExit(
            f"{directory}: сьют разошёлся с сохранённым "
            f"({str(current.get('sha256'))[:12]} ≠ {str(saved_digest.get('sha256'))[:12]}) — "
            "догрузка смешала бы два сьюта"
        )
    saved_config = directory / "config.json"
    if not saved_config.is_file():
        raise SystemExit(f"{directory}: нет config.json — сохранённый прогон нечем опознать")
    old = generation.read_json(directory, "config.json")
    try:
        generation.verify_environment(old, environment)
    except generation.ProvenanceError as exc:
        raise SystemExit(f"{directory}: {exc}") from exc
    # On a host without Docker the digest cannot be re-observed; the saved one
    # is carried into the identity so an unverifiable resume is not refused.
    if not cfg.image_digest:
        cfg.image_digest = str(old.get("image_digest") or "")
    saved_identity = old.get("identity")
    if not saved_identity:
        raise SystemExit(
            f"{directory}: в сохранённом config.json нет identity — "
            "прогон сделан до checkpoint-контракта, продолжать его нельзя"
        )
    identity = run_identity(cfg, str(current.get("sha256") or ""))
    if saved_identity != identity:
        changed = sorted(
            key
            for key in set(old) | set(cfg.__dataclass_fields__)
            if key not in {"identity", "run_id", "api_key_set", "out_dir", "only"}
            and str(old.get(key)) != str(getattr(cfg, key, None))
        )
        detail = f": {', '.join(changed)}" if changed else ""
        raise SystemExit(
            f"{directory}: resume смешал бы два прогона — identity {str(identity)[:12]} ≠ "
            f"{str(saved_identity)[:12]}{detail} (продолжать нельзя; запустите новый прогон с новым --name)"
        )

    by_id = {task.id: task for task in full}
    saved = _saved_entries(directory, identity)
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


def _saved_entries(directory: Path, identity: str) -> list[dict]:
    """Records of *this* run, from the master or — if it is corrupt — its mirrors.

    The master is written whole at every checkpoint and already contains every
    kept task. A stale mirror therefore must not be merged into a valid master;
    mirrors only stand in when the master itself cannot be parsed.
    """
    tasks_path = directory / "tasks.json"
    if not tasks_path.is_file():
        return []
    try:
        raw = generation.read_json(directory, "tasks.json")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"{tasks_path}: битый JSON ({exc}) — собираю из transcripts/", flush=True)
        return _mirror_entries(directory, identity)
    return [entry for entry in raw if entry.get("identity") == identity]


def _mirror_entries(directory: Path, identity: str) -> list[dict]:
    """Per-task mirrors written by the same run identity."""
    folder = directory / "transcripts"
    entries: list[dict] = []
    seen: set[str] = set()
    if not folder.is_dir():
        return entries
    for path in sorted(folder.glob("*.json")):
        try:
            entry = generation.read_json(directory, f"transcripts/{path.name}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        task_id = str((entry.get("task") or {}).get("id"))
        if task_id and task_id not in seen and entry.get("identity") == identity:
            seen.add(task_id)
            entries.append(entry)
    return entries


def _truncated(transcript: core.Transcript) -> bool:
    """Was the agent killed rather than finished?

    The exit code is the marker: a clean ``pi`` run says ``exit_ok``, while a
    timeout, a SIGTERM or an aborted turn leaves it false or records the error.
    A dead-last-turn (``finish_reason`` is empty — generation never completed,
    e.g. a transport drop or a kill between the exit marker and the record)
    also counts: resume must replay it instead of shipping half an artefact
    and near-zero latency.
    """
    if transcript.error:
        return True
    agent = (transcript.extra or {}).get("agent") or {}
    if agent and agent.get("exit_ok") is False:
        return True
    reasons = [str(reason) for reason in transcript.spend().get("finish_reasons") or []]
    if any(reason in ("error", "aborted") for reason in reasons):
        return True
    return bool(reasons) and reasons[-1] in ("", "None")


def _merge_run(runs: list, run: Any) -> None:
    """Replace the saved result of the same task, or append the new one."""
    for index, existing in enumerate(runs):
        if existing.task.id == run.task.id:
            runs[index] = run
            return
    runs.append(run)


def _hardware() -> dict:
    """Host sampler reading: the harness machine, not the serving one."""
    gpus = gpu_inventory()
    source = "client sampler (хост харнесса, не сервер)"
    if not gpus:
        return {"gpu": "unknown", "vram_used_mib_now": used_mib(), "source": source}
    return {**gpus[0], "vram_used_mib_now": used_mib(), "source": source}


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
        target.add_argument(
            "--profile",
            default=core.DEFAULT_PROFILE,
            choices=core.PROFILES,
            help="версия сьюта: baseline-20261020 (рейтинг), pilot (только пилотные), "
                 "stretch-v1 (baseline + новые сценарии, те же 100 баллов), "
                 "research-v1 (десять research-сценариев, 71 балл, + 6 VQA-сценариев вне счёта; не ранжируется с baseline)",
        )
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
            target.add_argument(
                "--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW,
                help="окно контекста в токенах (по умолчанию 262144; --config задаёт своё окно)",
            )
            target.add_argument(
                "--max-new-tokens", type=int, default=None,
                help="верхняя граница ответа; меньшее из неё и свободного окна уходит в запрос "
                     "(по умолчанию 4096)",
            )
            target.add_argument(
                "--max-total-tokens", type=int, default=None,
                help="потолок генерации на задание (ответы считаются суммой); по умолчанию "
                     "не меньше 16384 и не меньше --max-new-tokens",
            )
            target.add_argument(
                "--generation-mode", choices=("fixed", "native"), default="fixed",
                help="fixed — пришпилить sampling/seed/reasoning; native — отдать серверу",
            )
            target.add_argument(
                "--reasoning-budget", type=int, default=2048,
                help="серверный потолок thinking в токенах (только fixed)",
            )
            target.add_argument(
                "--extra-body", default="",
                help="JSON-объект, добавляемый в каждый payload (флаги сервера)",
            )
            target.add_argument("--weights", default="", help="путь к файлу весов для SHA-256")
            target.add_argument(
                "--provenance", default="",
                help="JSON от bench/sweep: observed-снимок runtime и выполненная загрузка",
            )
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
            target.add_argument(
                "--no-sandbox",
                dest="sandbox",
                action="store_false",
                help="исполнять агента на хосте, а не в контейнере",
            )

    doctor = sub.add_parser("doctor", help="проверить GPU, toolchain и endpoint")
    add_common(doctor, endpoint=True)
    doctor.add_argument("--config", default="", help="id эталонной конфигурации")
    doctor.add_argument("--configs-path", default="")
    doctor.set_defaults(func=cmd_doctor)

    listing = sub.add_parser("list", help="показать задания сьюта без запуска")
    add_common(listing)
    listing.add_argument(
        "--prompts",
        action="store_true",
        help="напечатать промпты целиком (приватные — только с --include-private)",
    )
    listing.add_argument("--include-private", action="store_true", help="показать приватные задания с промптами")
    listing.add_argument("--config", default="", help="id эталонной конфигурации (окно, бюджеты, политика)")
    listing.add_argument("--configs-path", default="")
    listing.set_defaults(func=cmd_list)

    generate = sub.add_parser("generate", help="записать fixtures сьюта в каталог")
    add_common(generate)
    generate.add_argument("--out", required=True)
    generate.add_argument("--include-private", action="store_true", help="писать промпты приватных заданий")
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
    bench.add_argument(
        "--allow-unverified", action="store_true",
        help="идти в прогон, даже если preflight не подтвердил конфигурацию (результат не ранжируется)",
    )
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
    preflight.add_argument(
        "target",
        nargs="?",
        default="",
        choices=("", "sandbox"),
        help="'sandbox' — собрать и проверить образ песочницы без конфигурации",
    )
    preflight.add_argument("--config", default="")
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
        "rescore",
        help="пересчитать баллы сохранённого прогона текущими evaluator'ами (без вызова модели); "
        "прогон, сделанный под другой сьют (например, до смены категории L), будет размечен "
        "как «нет задания в текущем сьюте» — сравнивать такие прогоны с новым сьютом нельзя",
    )
    rescore.add_argument("directories", nargs="+")
    rescore.set_defaults(func=cmd_rescore)

    validate = sub.add_parser("validate", help="проверить задание или сьют по схеме")
    add_common(validate)
    source = validate.add_mutually_exclusive_group()
    source.add_argument("--task", default="", help="JSON/YAML файл одного задания")
    source.add_argument("--suite", default="", help="JSON/YAML файл или каталог со сьютом")
    validate.add_argument("--no-totals", action="store_true", help="не сверять суммы баллов с профилем")
    validate.set_defaults(func=cmd_validate)

    export = sub.add_parser("export", help="выгрузить сьют по схеме в JSON или YAML")
    add_common(export)
    export.add_argument("--out", default="-", help="файл или каталог; по умолчанию stdout")
    export.add_argument("--format", choices=("json", "yaml"), default="json")
    export.add_argument("--include-private", action="store_true", help="включить приватные задания")
    export.add_argument("--holdout", default="", help="манифест holdout: audit и closed-eval остаются хешами")
    export.add_argument(
        "--include-closed",
        action="store_true",
        help="опубликовать закрытый текст даже при --holdout",
    )
    export.set_defaults(func=cmd_export)

    holdout_parser = sub.add_parser(
        "holdout", help="сплиты public-dev/audit/closed-eval: назначить и проверить"
    )
    holdout_actions = holdout_parser.add_subparsers(dest="holdout_action", required=True)

    holdout_assign = holdout_actions.add_parser(
        "assign", help="детерминированно назначить сплиты по task ID и записать манифест"
    )
    add_common(holdout_assign)
    holdout_assign.add_argument("out", help="куда записать JSON-манифест")
    holdout_assign.add_argument("--closed-share", type=float, default=holdout.CLOSED_SHARE)
    holdout_assign.add_argument("--audit-share", type=float, default=holdout.AUDIT_SHARE)
    holdout_assign.add_argument(
        "--pin", action="append", default=[], metavar="ID=СПЛИТ",
        help="закрепить задание в сплите (public-dev, audit, closed-eval)",
    )
    holdout_assign.set_defaults(func=cmd_holdout_assign)

    holdout_check = holdout_actions.add_parser(
        "check", help="сверить манифест с текущим сьютом и доказать, что экспорт закрыт"
    )
    add_common(holdout_check)
    holdout_check.add_argument("manifest", help="JSON-манифест")
    holdout_check.set_defaults(func=cmd_holdout_check)

    licenses = sub.add_parser("licenses", help="реестр прав на внешние материалы")
    license_actions = licenses.add_subparsers(dest="licenses_action", required=True)
    licenses_check = license_actions.add_parser(
        "check", help="отклонить записи без лицензии или с неverified статусом"
    )
    licenses_check.add_argument("paths", nargs="*", help="JSON-записи; по умолчанию каталог docs/licenses")
    licenses_check.add_argument("--dir", default="", help="альтернативный каталог реестра")
    licenses_check.set_defaults(func=cmd_licenses_check)

    add_decontam_commands(sub, add_common)
    add_adapter_commands(sub)
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
