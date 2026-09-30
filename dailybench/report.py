"""Reporting: per-run markdown, JSON totals and a cross-model comparison table.

Quality is reported per category and as a total; a configuration whose
long-context tasks failed the admission check is shown out of the short total
(85) with the reason, never folded into a 100-point number it did not earn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from . import core

SHORT_ORDER = core.ADMISSION_CATEGORIES


def load_run(directory: Path | str) -> dict[str, Any]:
    directory = Path(directory)
    scores_path = directory / "scores.json"
    if not scores_path.is_file():
        raise FileNotFoundError(f"{scores_path} not found — run the suite first")
    data: dict[str, Any] = {
        "dir": str(directory),
        "scores": json.loads(scores_path.read_text(encoding="utf-8")),
    }
    for name in ("config", "provider", "hardware", "suite", "speed", "tasks", "sizing", "rescored"):
        path = directory / f"{name}.json"
        if path.is_file():
            data[name] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _stop_reason(run_entry: dict[str, Any] | None) -> str:
    """Why generation stopped, plus the reasoning-heavy case called out."""
    if not run_entry:
        return "—"
    turns = (run_entry.get("transcript") or {}).get("turns") or []
    if not turns:
        return "—"
    reply = turns[-1].get("reply") or {}
    reason = reply.get("finish_reason") or "—"
    generated = reply.get("completion_tokens")
    reasoning = reply.get("reasoning_tokens")
    if reason == "length" and not (reply.get("text") or "").strip():
        return "length — бюджет исчерпан до видимого ответа"
    if reasoning:
        return f"{reason} ({generated} ток., из них reasoning {reasoning})"
    return reason


def summarise(scores: Iterable[dict[str, Any]], suite_tasks: int | None = None) -> dict[str, Any]:
    per_category: dict[str, dict[str, Any]] = {
        key: {
            "earned": 0.0,
            "possible": 0.0,
            "admitted": True,
            "tasks": 0,
            "errors": 0,
            "skipped": 0,
            "skip_reasons": [],
        }
        for key in core.CATEGORY_POINTS
    }
    for score in scores:
        category = score["category"]
        row = per_category.setdefault(
            category,
            {
                "earned": 0.0,
                "possible": 0.0,
                "admitted": True,
                "tasks": 0,
                "errors": 0,
                "skipped": 0,
                "skip_reasons": [],
            },
        )
        row["tasks"] += 1
        if score.get("skipped"):
            # Not verified is not a zero: it leaves both the numerator and the
            # denominator untouched, and the reason stays visible.
            row["skipped"] += 1
            row["skip_reasons"].append(score["skipped"])
            continue
        row["possible"] += score.get("possible", 0.0)
        if score.get("error"):
            row["errors"] += 1
        if score.get("admission") is False:
            row["admitted"] = False
        else:
            row["earned"] += score.get("earned", 0.0)

    long_context_admitted = per_category.get("E", {}).get("admitted", True)
    admitted_total = round(sum(per_category[key]["earned"] for key in core.CATEGORY_POINTS), 3)
    short_total = round(sum(per_category[key]["earned"] for key in SHORT_ORDER), 3)
    possible_total = round(sum(row["possible"] for row in per_category.values()), 3)
    errors = sum(row["errors"] for row in per_category.values())
    ran = sum(row["tasks"] for row in per_category.values())
    skipped_total = sum(row["skipped"] for row in per_category.values())

    if long_context_admitted:
        headline_earned, headline_possible = admitted_total, possible_total
    else:
        headline_earned = short_total
        headline_possible = round(sum(per_category[key]["possible"] for key in SHORT_ORDER), 3)

    full_suite = (
        ran >= suite_tasks if suite_tasks else possible_total >= core.FULL_TOTAL - 1e-9
    )
    verdict = f"{headline_earned:.0f}/{headline_possible:.0f}"
    if not long_context_admitted:
        verdict += " (из доступных баллов: long context не прошёл допуск)"
    if not full_suite:
        verdict += f" (частичный прогон: {possible_total:.0f} из {core.FULL_TOTAL:.0f} баллов сьюта)"
    if skipped_total:
        skipped_categories = [
            key for key, row in per_category.items() if row["skipped"] and row["skipped"] == row["tasks"]
        ]
        verdict += (
            f" (не проверялось: {', '.join(skipped_categories) or skipped_total} —"
            " результат не входит ни в числитель, ни в знаменатель)"
        )

    return {
        "per_category": per_category,
        "admitted_total": admitted_total,
        "short_total": short_total,
        "possible_total": possible_total,
        "full_suite": full_suite,
        "headline_earned": headline_earned,
        "headline_possible": headline_possible,
        "long_context_admitted": long_context_admitted,
        "errors": errors,
        "ran": ran,
        "skipped": skipped_total,
        "verdict": verdict,
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(run: dict[str, Any]) -> str:
    config = run.get("config", {})
    provider = run.get("provider", {})
    hardware = run.get("hardware", {})
    suite = run.get("suite", {})
    summary = summarise(run["scores"], suite.get("tasks"))

    lines: list[str] = []
    model = config.get("model", "unknown")
    quant = config.get("quant") or "n/a"
    lines.append(f"# DailyBench-16 — {model} ({quant})")
    lines.append("")
    lines.append("| Поле | Значение |")
    lines.append("| --- | --- |")
    lines.append(f"| Модель | `{model}` |")
    lines.append(f"| Квант | {quant} |")
    lines.append(f"| Runtime | {config.get('runtime') or provider.get('provider', 'n/a')} |")
    lines.append(f"| Endpoint | `{config.get('base_url', '')}` |")
    lines.append(f"| Веса SHA-256 | `{config.get('weights_sha256') or 'не указан'}` |")
    lines.append(f"| KV-cache | {config.get('kv_type') or 'n/a'} |")
    lines.append(f"| Окно контекста | {config.get('context_window')} |")
    sizing = run.get("sizing") or {}
    if sizing.get("calibrated"):
        lines.append(
            f"| Калибровка long context | {sizing['chars_per_token']:.2f} символов/токен "
            f"({sizing.get('how', 'измерено')}) |"
        )
    elif config.get("chars_per_token") and abs(float(config["chars_per_token"]) - 3.2) > 1e-9:
        lines.append(f"| Калибровка long context | {float(config['chars_per_token']):.2f} символов/токен (задано, не измерено) |")
    lines.append(f"| Offload | {config.get('offload', 'n/a')} |")
    lines.append(f"| Seed | {config.get('seed')} |")
    lines.append(f"| Бюджеты | max_new_tokens={config.get('max_new_tokens')}, "
                 f"max_total_tokens={config.get('max_total_tokens')}, max_tool_calls={config.get('max_tool_calls')} |")
    if hardware:
        lines.append(
            f"| GPU | {hardware.get('gpu') or hardware.get('name', 'n/a')} "
            f"({hardware.get('vram_total_mib', '?')} MiB, driver {hardware.get('driver', '?')}) |"
        )
    lines.append(f"| Suite SHA-256 | `{suite.get('sha256', 'n/a')}` |")
    if config.get("reference"):
        lines.append(f"| Reference config | `{config['reference']}` |")
    if config.get("unsupported"):
        lines.append(f"| Не проверялось | {', '.join(config['unsupported'])} |")
    lines.append("")
    if config.get("launch"):
        lines.append("Загрузка конфигурации (для воспроизведения):")
        lines.append("")
        lines.append("```")
        lines.append(str(config["launch"]))
        lines.append("```")
        lines.append("")
    lines.append(f"**Итог: {summary['verdict']}**")
    rescored = run.get("rescored") or {}
    if rescored:
        lines.append("")
        lines.append(
            f"> Баллы пересчитаны {rescored.get('rescored_at', '?')} по сохранённым ответам "
            f"(изменений: {len(rescored.get('changed') or [])}); модель повторно не вызывалась."
        )
    lines.append("")
    lines.append("| Категория | Баллы | Допуск | Ошибок прогона |")
    lines.append("| --- | ---: | --- | ---: |")
    for key in sorted(core.CATEGORY_POINTS):
        row = summary["per_category"].get(key)
        if not row or not row["tasks"]:
            continue
        title = core.CATEGORY_TITLES.get(key, key)
        if row["skipped"] == row["tasks"]:
            lines.append(f"| {key}. {title} | не проверялось | — | {row['errors']} |")
            continue
        score_text = f"{row['earned']:.0f} / {row['possible']:.0f}"
        if row["skipped"]:
            score_text += f" (не проверялось: {row['skipped']})"
        admitted = "да" if row["admitted"] else "нет"
        lines.append(f"| {key}. {title} | {score_text} | {admitted} | {row['errors']} |")
    lines.append("")
    lines.append("## Задания")
    lines.append("")
    lines.append("| ID | Балл | Стоп-причина | Ошибка прогона | Проваленные чеки |")
    lines.append("| --- | ---: | --- | --- | --- |")
    runs_by_id = {entry.get("task", {}).get("id"): entry for entry in (run.get("tasks") or [])}
    for score in run["scores"]:
        if score.get("skipped"):
            lines.append(f"| {score['task_id']} | не проверялось | — | — | {score['skipped']} |")
            continue
        failed = [c["name"] for c in score.get("checks", []) if not c["passed"]]
        lines.append(
            f"| {score['task_id']} | {score.get('earned', 0):.0f}/{score.get('possible', 0):.0f} | "
            f"{_stop_reason(runs_by_id.get(score['task_id']))} | "
            f"{score.get('error') or '—'} | {', '.join(failed) if failed else '—'} |"
        )
    lines.append("")

    speed = run.get("speed")
    if speed:
        lines.append("## Скорость и память (без баллов качества)")
        lines.append("")
        lines.append("| Профиль | TTFT, с | Prompt tok/s | Generation tok/s | Пик VRAM, MiB | Повторов |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in speed:
            lines.append(
                f"| {row.get('label')} | {_fmt(row.get('ttft_s'))} | {_fmt(row.get('prompt_tok_s'), 1)} | "
                f"{_fmt(row.get('generation_tok_s'), 1)} | {row.get('vram_peak_mib', 'n/a')} | {row.get('runs')} |"
            )
        lines.append("")
    return "\n".join(lines)


def compare_markdown(runs: list[dict[str, Any]]) -> str:
    """Side-by-side table across runs: quality first, then memory and speed."""
    lines = ["# Сравнение конфигураций", ""]
    header = ["Модель", "Квант", "Итог"]
    for key in sorted(core.CATEGORY_POINTS):
        header.append(f"{key}, %")
    header += ["Пик VRAM, MiB", "TTFT long, с", "Gen tok/s", "Ошибок"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    for run in runs:
        config = run.get("config", {})
        summary = summarise(run["scores"], (run.get("suite") or {}).get("tasks"))
        row = [config.get("model", "?"), config.get("quant") or "n/a", summary["verdict"]]
        for key in sorted(core.CATEGORY_POINTS):
            entry = summary["per_category"].get(key)
            if not entry or not entry["tasks"]:
                row.append("—")
            elif entry["skipped"] == entry["tasks"]:
                row.append("не проверялось")
            elif not entry["admitted"]:
                row.append("не допущен")
            elif entry["possible"]:
                row.append(f"{100 * entry['earned'] / entry['possible']:.0f}")
            else:
                row.append("—")
        speed = run.get("speed") or []
        long_row = next((s for s in speed if str(s.get("label", "")).startswith("long")), None)
        peak = long_row.get("vram_peak_mib") if long_row else None
        row.append(str(peak if peak is not None else "n/a"))
        row.append(_fmt(long_row.get("ttft_s") if long_row else None))
        row.append(_fmt(long_row.get("generation_tok_s") if long_row else None, 1))
        row.append(str(summary["errors"]))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("Пик VRAM и TTFT берутся из длинного профиля: это измерение при заполненном "
                 "контексте, а не после загрузки модели.")
    return "\n".join(lines)


def write_reports(run_dirs: list[Path | str]) -> list[Path]:
    """Render ``report.md`` for every run directory and a comparison when >1."""
    written: list[Path] = []
    runs = [load_run(directory) for directory in run_dirs]
    for run in runs:
        path = Path(run["dir"]) / "report.md"
        path.write_text(render_markdown(run), encoding="utf-8")
        written.append(path)
    if len(runs) > 1:
        target = Path(runs[0]["dir"]).parent / "comparison.md"
        target.write_text(compare_markdown(runs), encoding="utf-8")
        written.append(target)
    return written
