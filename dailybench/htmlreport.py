"""HTML-отчёт: решения моделей, графики и живая витрина.

Словарь отчёта простой и одинаковый везде: **сьют** — это задания, **решения
моделей** — то, что модели на них ответили. Поэтому и вкладки называются
«Решения моделей» и «Решения моделей по сьюту», а не абстрактные «Свод» и
«Графики».

Отчёт отвечает на три вопроса, каждый на своей вкладке:

``Решения``
    кто сколько набрал, где потерял и сколько на это потратил;
``Решения по сьюту``
    каждый пункт сьюта — строка, решения всех моделей рядом, с баллом, стоп-
    причиной и проваленной проверкой;
``Витрина``
    кадры K и L; клик открывает **живую страницу** решения в iframe (та же
    библиотека, что снимала кадр), кнопка — на весь экран для съёмки видео,
    есть покадровый шаг и сравнение решений нескольких моделей рядом.

Страница самодостаточна: plotly, three.js и кадры лежат рядом в `_vendor` и
`_showcase`, внешние запросы не нужны.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from . import core, gallery
from .report import load_run, summarise

SCORED = core.SCORED_CATEGORIES
SHOWCASE = core.VISUAL_CATEGORIES


def _cell(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


def _endpoint(url: str) -> str:
    """Host without the scheme, the /v1 suffix and the shared cluster domain."""
    short = str(url or "").replace("https://", "").replace("http://", "")
    short = short[:-3] if short.endswith("/v1") else short
    for suffix in (".apps.evodagpup.sberdevices.ru", ".sberdevices.ru"):
        if short.endswith(suffix):
            short = short[: -len(suffix)]
            break
    return short


def _short(title: str, task_id: str) -> str:
    for prefix in (f"{task_id} — ", f"{task_id} - "):
        if title.startswith(prefix):
            return title[len(prefix):]
    return title


def _tone(earned: float, possible: float) -> str:
    share = earned / (possible or 1)
    return "good" if share >= 0.8 else ("mid" if share >= 0.5 else "bad")


def _pct(earned: float, possible: float) -> float | None:
    return round(100 * earned / possible, 1) if possible else None


def _selector_of(run: dict[str, Any]) -> set[str]:
    """Every name a caller may plausibly use for this run."""
    config = run.get("config") or {}
    names = {_model_name(run), str(config.get("model") or ""), Path(str(run.get("dir") or "")).name}
    return {name for name in names if name}


def _select_runs(runs: list[dict[str, Any]], wanted: list[str]) -> list[dict[str, Any]]:
    """Runs matching the selectors, in the requested order.

    Accepting label, model id *and* directory name costs nothing and removes a
    real footgun: ``--compare qwen36`` (the name of the command that produced the
    run) used to fail because the run's label was "Qwen3.6 35B-A3B".
    """
    by_selector: dict[str, dict[str, Any]] = {}
    for run in runs:
        for selector in _selector_of(run):
            by_selector.setdefault(selector, run)
    missing = [name for name in wanted if name not in by_selector]
    if missing:
        raise ValueError(
            f"неизвестные модели для сравнения: {missing}; есть {sorted(by_selector)}"
        )
    picked: list[dict[str, Any]] = []
    for name in wanted:
        run = by_selector[name]
        if run not in picked:
            picked.append(run)
    return picked


def _suite_facts(runs: list[dict[str, Any]]) -> dict[str, int]:
    """Counts that describe the suite this report was built from.

    Three sources, most specific first: the run's own digest, then the suite
    rebuilt from the run's config (a full-suite run recorded before digests were
    stored), then the live registry. Hard-coding these was wrong the moment a
    category was added — the header claimed 83 tasks while the suite shipped 96,
    and a 100-point suite was advertised as 559.
    """
    # The registry holds the per-category weights (K's block total, L's per-task
    # weight); without it the showcase column under-counts silently.
    from .tasks import load_all

    load_all()
    per_task: list[dict[str, Any]] = []
    for run in runs:
        rows = (run.get("suite") or {}).get("per_task") or []
        if len(rows) > len(per_task):
            per_task = rows
    if not per_task:
        per_task = _rebuilt_digest(runs)
    scored = [row for row in per_task if row.get("category") in SCORED]
    showcase = [row for row in per_task if row.get("category") in SHOWCASE]
    return {
        "tasks": len(per_task),
        "scored": len(scored),
        "showcase": len(showcase),
        # Points belong to the *category*, not to each task: a per-task sum gave
        # 559 "points" for a 100-point suite. ``suite_points`` is the ceiling of
        # the whole scored half, so a showcase-only run can still say what it did
        # not test instead of implying a 0-point suite.
        "scored_points": int(sum(core.CATEGORY_POINTS.get(key, 0) for key in {r["category"] for r in scored})),
        "showcase_points": int(sum(_showcase_points(key) for key in {r["category"] for r in showcase})),
        "suite_points": int(core.FULL_TOTAL),
    }


def _rebuilt_digest(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The suite a run faced, rebuilt from its own recorded config.

    Cheap (no model, no fixtures beyond generation) and honest about partial
    runs: ``only`` and ``unsupported`` come from the config, so a 12-task smoke
    run reports 12 tasks instead of today's 96.
    """
    from .config import RunConfig
    from .suite import build_suite

    for run in runs:
        config = run.get("config") or {}
        if not config:
            continue
        try:
            cfg = RunConfig.from_dict(config)
            return sorted(
                ({"id": task.id, "category": task.category} for task in build_suite(cfg)),
                key=lambda row: row["id"],
            )
        except Exception:  # pragma: no cover - a config from an older version
            continue
    return []


def _showcase_points(category: str) -> float:
    """Points a showcase category carries.

    ``SHOWCASE_POINTS`` totals the whole block under ``K``; the art category
    ``L`` has no entry of its own (no leaderboard points), so a per-category
    lookup has to fall back to its registered weight instead of counting it as
    zero.
    """
    if category in core.SHOWCASE_POINTS:
        return float(core.SHOWCASE_POINTS[category])
    module = core.REGISTRY.get(category)
    return float(module.points) if module is not None else 0.0


def _suite_line(facts: dict[str, int], *, per_run: bool) -> str:
    """One honest sentence about the suite; identical in both report flavours.

    The showcase figure is the *block ceiling*, not a sum over the tasks that ran:
    a 2-task smoke run still faces the whole visual block's 225 frames' worth of
    scoring scale, and saying "225 вне счёта" next to 2 tasks is exact if the
    sentence says ceiling.
    """
    if not facts["tasks"]:
        return "Сьют неизвестен: в прогоне нет ни конфигурации, ни дайджеста."
    subject = "Прогон" if per_run else "Сьют"
    if facts["scored"]:
        scored = f"{facts['scored']} балльных ({facts['scored_points']} баллов качества)"
    else:
        # A showcase-only run has no scored task in it; "0 баллов качества"
        # would read as a score of zero rather than as nothing scored.
        scored = (
            f"балльных категорий нет (из {facts['suite_points']} баллов сьюта "
            "в этот прогон ничего не попало)"
        )
    showcase = (
        f"и {facts['showcase']} витринных (потолок блока {facts['showcase_points']} вне счёта)"
        if facts["showcase"]
        else "витринных нет"
    )
    return f"{subject} — {facts['tasks']} заданий, из них {scored} {showcase}."


def _spend_totals(run: dict[str, Any]) -> dict[str, float]:
    answer = reason = seconds = tool_calls = 0.0
    for entry in run.get("tasks") or []:
        spend = (entry.get("transcript") or {}).get("spend") or {}
        answer += spend.get("answer_tokens") or 0
        reason += spend.get("reasoning_tokens") or 0
        seconds += spend.get("generate_s") or 0
        tool_calls += spend.get("tool_calls") or 0
    total = answer + reason
    return {"answer": answer, "reasoning": reason, "seconds": seconds, "tool_calls": tool_calls,
            "total": total, "reasoning_share": reason / total if total else 0.0,
            "answer_per_min": answer / (seconds / 60) if seconds > 1 else 0.0}


def _model_name(run: dict[str, Any]) -> str:
    config = run.get("config") or {}
    label = config.get("label") or config.get("model") or "?"
    return str(label)


def _model_cards(summaries: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    """One card per model: score, showcase, spend, and where the points went."""
    cards = []
    for place, (run, summary) in enumerate(summaries, 1):
        config = run.get("config") or {}
        spend = _spend_totals(run)
        rows = run["scores"]
        # A published artefact keeps one page per run next to the comparison; link
        # it when it is there, so a reader does not have to guess the path.
        own = Path(str(run.get("dir") or "")).name
        link = (
            f"<div class='sub'><a class='link' href='runs/{_cell(own)}/index.html'>"
            "отчёт по этой модели →</a></div>"
        ) if own else ""
        failed = sum(1 for r in rows if r.get("earned", 0) < r.get("possible", 0) and not r.get("skipped"))
        zero = sum(1 for r in rows if not r.get("earned") and not r.get("skipped") and r.get("possible"))
        failed_cats = sorted({r["category"] for r in rows
                              if r.get("earned", 0) < r.get("possible", 0) and not r.get("skipped")})
        cards.append(
            f"<div class='card' data-model='{_cell(_model_name(run))}'>"
            f"<div class='rank'>{place}</div>"
            f"<h3>{_cell(_model_name(run))}</h3>"
            f"<div class='big {_tone(summary['headline_earned'], summary['headline_possible'])}'>"
            f"{summary['headline_earned']:.0f}<span class='of'>/{summary['headline_possible']:.0f}</span>"
            f"<span class='unit'>баллов качества</span></div>"
            f"<div class='meter'><i style='width:{100 * summary['headline_earned'] / (summary['headline_possible'] or 1):.0f}%'></i></div>"
            f"<div class='sub'>симуляции K: <b>{summary['showcase_earned']:.0f}/{summary['showcase_possible']:.0f}</b> · L — оценка человеком · "
            f"заданий с потерей: <b>{failed}</b> (нулевых {zero})</div>"
            f"<div class='sub'>категории с потерей: {_cell(', '.join(failed_cats) or 'нет')}</div>"
            f"<div class='sub spend'>токенов: <b>{spend['total'] / 1000:.0f}K</b> "
            f"(reasoning {100 * spend['reasoning_share']:.0f}%) · <b>{spend['seconds'] / 60:.0f} мин</b> "
            f"· балл/мин <b>{summary['headline_earned'] / max(spend['seconds'] / 60, 1e-9):.2f}</b></div>"
            f"<div class='sub'>{_cell(config.get('quant') or '')} · {_cell(_endpoint(config.get('base_url', '')))}</div>"
            f"{link}"
            "</div>"
        )
    return "".join(cards)


def _verdict_table(summaries: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    """Category matrix in percent, with a plain-language column title."""
    head = ["Модель", "Качество"]
    for key in SCORED:
        head.append(f"{key}<span class='th-sub'>{_cell(core.category_title(key))}</span>")
    for key in SHOWCASE:
        note = " · на глаз" if key in core.ART_CATEGORIES else ""
        head.append(f"{key}<span class='th-sub'>{_cell(core.category_title(key))}{note}</span>")
    head.append("Потрачено токенов")
    head.append("Минут генерации")

    body = []
    for run, summary in summaries:
        spend = _spend_totals(run)
        cells = [
            f"<td class='model'>{_cell(_model_name(run))}</td>",
            f"<td class='verdict {_tone(summary['headline_earned'], summary['headline_possible'])}'>"
            f"{summary['headline_earned']:.0f}/{summary['headline_possible']:.0f}</td>",
        ]
        for key in SCORED:
            row = summary["per_category"].get(key, {})
            pct = _pct(row.get("earned", 0), row.get("possible", 0))
            cells.append(f"<td class='num'>{'—' if pct is None else f'{pct:.0f}%'}</td>")
        for key in SHOWCASE:
            row = (summary.get("showcase") or {}).get(key, {})
            if key in core.ART_CATEGORIES:
                label = "оценка человеком" if row.get("tasks") else "—"
                if row.get("tasks") and row.get("skipped") == row.get("tasks"):
                    label = "не проверялось"
                cells.append(f"<td class='show'>{label}</td>")
                continue
            pct = _pct(row.get("earned", 0), row.get("possible", 0))
            cells.append(f"<td class='num show'>{'—' if pct is None else f'{pct:.0f}%'}</td>")
        cells.append(f"<td class='num'>{spend['total'] / 1000:.0f}K</td>")
        cells.append(f"<td class='num'>{spend['seconds'] / 60:.0f}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class='grid verdict-table'><thead><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _suite_table(summaries: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    """Suite × solutions: one row per task, one column per model.

    The matrix is the point of the report: the same task for everyone, so the
    differences are visible side by side instead of in three separate tables.
    """
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    order: list[tuple[str, str, str]] = []
    for run, _summary in summaries:
        name = _model_name(run)
        by_model.setdefault(name, {})
        for score in run["scores"]:
            task_id = score["task_id"]
            if task_id not in by_model[name]:
                by_model[name][task_id] = score
            if all(task_id != tid for tid, _, _ in order):
                entry = next((e for e in (run.get("tasks") or [])
                              if (e.get("task") or {}).get("id") == task_id), {})
                title = _short(str((entry.get("task") or {}).get("title", "")), task_id)
                order.append((task_id, score.get("category", ""), title))
    def order_key(row: tuple[str, str, str]) -> tuple[int, int, str]:
        task_id, category = row[0], row[1]
        number = int(task_id.split("-")[-1]) if task_id.split("-")[-1].isdigit() else 0
        return (SCORED.index(category) if category in SCORED else 99 + SHOWCASE.index(category), number, task_id)

    order.sort(key=order_key)

    header = "".join(f"<th>{_cell(_model_name(run))}</th>" for run, _ in summaries)
    rows = []
    for task_id, category, title in order:
        cells = []
        for run, _ in summaries:
            score = by_model.get(_model_name(run), {}).get(task_id)
            if not score:
                cells.append("<td class='cell miss'>—</td>")
                continue
            if score.get("skipped"):
                cells.append("<td class='cell skip' title='не проверялось'>н/п</td>")
                continue
            earned, possible = score.get("earned", 0), score.get("possible", 0)
            failed = [c["name"] for c in score.get("checks", []) if not c["passed"]]
            klass = "full" if earned == possible else ("zero" if not earned else "part")
            detail = "; ".join(f"{c['name']}: {c['detail']}" for c in score.get("checks", []) if not c["passed"])
            if category in core.ART_CATEGORIES:
                label = "отрисовано" if earned == possible and earned else "не отрисовалось"
                cells.append(f"<td class='cell {klass}' title='{_cell(detail)[:400]}'>{label}</td>")
                continue
            cells.append(
                f"<td class='cell {klass}' title='{_cell(detail)[:400]}'>"
                f"{earned:.1f}<span class='of'>/{possible:.0f}</span>"
                f"{'<span class=\"badge\">' + _cell(', '.join(failed)) + '</span>' if failed else ''}</td>"
            )
        block = "витрина" if category in SHOWCASE else ""
        rows.append(
            f"<tr><td class='task'><b>{_cell(task_id)}</b> {_cell(title)}"
            f"<span class='cat'>{_cell(core.category_title(category))}{' · ' + block if block else ''}</span></td>"
            + "".join(cells) + "</tr>"
        )
    legend = (
        "<p class='hint'>в ячейке — сумма проверок за задание у этой модели; цвет: зелёный — все проверки, "
        "жёлтый — часть, красный — ноль. Для блока K это баллы (симуляции проверяются расчётом), "
        "для блока L — только загрузка HTML; художественное качество оценивает человек по кадрам. "
        "Наведите курсор, чтобы увидеть, какая проверка не прошла.</p>"
    )
    return (f"<table class='grid suite-table'><thead><tr><th>задание сьюта</th>{header}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>{legend}")


def _task_rows(run: dict[str, Any]) -> str:
    tasks = {entry.get("task", {}).get("id"): entry for entry in (run.get("tasks") or [])}
    rows = []
    for score in run["scores"]:
        task_id = score["task_id"]
        entry = tasks.get(task_id) or {}
        title = _short(str(entry.get("task", {}).get("title", "")), task_id)
        spend = ((entry.get("transcript") or {}).get("spend") or {})
        failed = [(c["name"], c["detail"]) for c in score.get("checks", []) if not c["passed"]]
        stop = ""
        turns = ((entry.get("transcript") or {}).get("turns") or [])
        if turns:
            stop = (turns[-1].get("reply") or {}).get("finish_reason") or ""
        klass = "ok" if score.get("earned") == score.get("possible") else ("skip" if score.get("skipped") else "part")
        names = ", ".join(name for name, _ in failed)
        detail = "; ".join(f"<b>{_cell(name)}</b>: {_cell(text)}" for name, text in failed) or "—"
        if score.get("error"):
            detail = f"<b>ошибка</b>: {_cell(score['error'])}"
        if score.get("skipped"):
            detail = f"<b>не проверялось</b>: {_cell(score['skipped'])}"
        rows.append(
            f"<tr class='{klass}'><td>{_cell(task_id)}</td><td>{_cell(title)}</td>"
            f"<td class='num'>{score.get('earned', 0):.1f}/{score.get('possible', 0):.0f}</td>"
            f"<td class='num small'>{spend.get('answer_tokens') or 0}</td>"
            f"<td class='num small'>{spend.get('reasoning_tokens') or 0}</td>"
            f"<td class='num small'>{round(spend.get('generate_s') or 0, 1)}</td>"
            f"<td>{_cell(stop)}</td><td class='detail'>{detail}</td></tr>"
        )
    return "".join(rows)


def _provenance(runs: list[dict[str, Any]]) -> str:
    """Everything needed to reproduce the numbers, in one table per run.

    The markdown report carries this; the HTML used to show only the endpoint.
    A page people quote must say which weights, which window, which agent harness
    and which suite hash produced the score.
    """
    blocks: list[str] = []
    for run in runs:
        config = run.get("config") or {}
        hardware = run.get("hardware") or {}
        suite = run.get("suite") or {}
        sizing = run.get("sizing") or {}
        rows: list[tuple[str, str]] = [
            ("Модель", str(config.get("model") or "—")),
            ("Метка", str(config.get("label") or "—")),
            ("Квант", str(config.get("quant") or "—")),
            ("Рантайм", str(config.get("runtime") or "—")),
            ("Endpoint", str(config.get("base_url") or "—")),
            ("Окно контекста", str(config.get("context_window") or "—")),
            ("KV-cache", str(config.get("kv_type") or "—")),
            ("Offload", str(config.get("offload") or "none")),
            ("Seed сьюта", str(config.get("seed") or "—")),
            ("Бюджеты", f"max_new_tokens={config.get('max_new_tokens')}, "
                        f"max_total_tokens={config.get('max_total_tokens')}, "
                        f"max_tool_calls={config.get('max_tool_calls')}"),
        ]
        if config.get("agent"):
            rows.append((
                "Каркас артефактов",
                "агент pi, категории " + ", ".join(config.get("agent_categories") or ())
                + ": читает и правит рабочую копию, оценка по файлу на диске",
            ))
        rows.append(("Веса SHA-256", str(config.get("weights_sha256") or "не указан")))
        if config.get("reference"):
            rows.append(("Эталонная конфигурация", str(config["reference"])))
        if config.get("launch"):
            rows.append(("Строка запуска", str(config["launch"])))
        if sizing.get("calibrated"):
            rows.append(("Калибровка long context",
                         f"{sizing['chars_per_token']:.2f} символов/токен ({sizing.get('how', 'измерено')})"))
        if hardware:
            rows.append(("Железо", str(hardware.get("gpu") or hardware.get("name") or "—")))
        if suite.get("sha256"):
            rows.append(("Сьюта SHA-256", str(suite["sha256"])))
        if (run.get("rescored") or {}).get("rescored_at"):
            rows.append(("Пересчёт", str(run["rescored"]["rescored_at"])))

        body = "".join(
            f"<tr><th>{_cell(name)}</th><td>{_cell(value)}</td></tr>" for name, value in rows
        )
        heading = f"<h3>{_cell(_model_name(run))}</h3>" if len(runs) > 1 else ""
        blocks.append(f"{heading}<table class='grid prov'>{body}</table>")
    return "".join(blocks)


def _charts_payload(runs: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {"models": [], "scoredKeys": list(SCORED),
                               "scoredNames": [f"{k}. {core.category_title(k).split(' ')[0]}" for k in SCORED],
                               "showcaseKeys": list(core.SHOWCASE_CATEGORIES), "scored": [], "showcase": [],
                               "spend": [], "cost": []}
    for run in runs:
        summary = summarise(run["scores"], run.get("suite"))
        name = _model_name(run)
        payload["models"].append(name)
        payload["scored"].append([_pct(summary["per_category"].get(k, {}).get("earned", 0),
                                       summary["per_category"].get(k, {}).get("possible", 0)) for k in SCORED])
        showcase = summary.get("showcase") or {}
        payload["showcase"].append([_pct(showcase.get(k, {}).get("earned", 0),
                                         showcase.get(k, {}).get("possible", 0)) for k in core.SHOWCASE_CATEGORIES])
        spend = _spend_totals(run)
        payload["spend"].append({"model": name, "answer": spend["answer"], "reasoning": spend["reasoning"],
                                 "minutes": spend["seconds"] / 60})
        payload["cost"].append({"model": name, "score": summary["headline_earned"],
                                "minutes": spend["seconds"] / 60, "reasoning": spend["reasoning"] / 1000,
                                "per_min": summary["headline_earned"] / max(spend["seconds"] / 60, 1e-9)})
    return payload




def _verdict_text(category: str, failed: list[str]) -> str:
    """Keep HTML delivery separate from human judgement of artwork."""
    if category in core.ART_CATEGORIES:
        return "HTML не загрузился: " + ", ".join(failed) if failed else "HTML загружен · художественное качество оценивает человек"
    return "провалено: " + ", ".join(failed) if failed else "все проверки пройдены"


def _gallery(entries: list[dict[str, Any]]) -> str:
    cards = []
    for item in entries:
        score = item.get("score") or {}
        earned, possible = score.get("earned"), score.get("possible")
        failed = [c["name"] for c in score.get("checks", []) if not c["passed"]]
        status = f"{earned:.0f}/{possible:.0f}" if isinstance(earned, (int, float)) else "н/п"
        if item.get("category") in core.ART_CATEGORIES:
            status = "HTML" if earned and not failed else ("н/п" if score.get("skipped") else "ошибка HTML")
        tone = "full" if earned == possible else ("zero" if not earned else "part")
        page, frame = item.get("page") or "", item.get("frame") or ""
        spend = item.get("spend") or {}
        window_name = _cell(f"scene-{item['id']}-{item['model']}")
        if frame:
            media = (f"<button class='shot' data-page='{_cell(page)}' data-id='{_cell(item['id'])}'"
                     f" data-window='{window_name}' title='открыть живую сцену в новом окне'>"
                     f"<img src='{_cell(frame)}' loading='lazy' alt='{_cell(item['id'])}'>"
                     f"<span class='play'>живая сцена ↗</span></button>")
        else:
            why = "модель не дошла до HTML" if not page else "кадра нет"
            media = f"<div class='missing'>{_cell(why)}</div>"
        cards.append(
            f"<figure data-model='{_cell(item['model'])}' data-id='{_cell(item['id'])}' data-cat='{_cell(item['category'])}'>"
            f"<figcaption><span class='block'>{_cell(core.category_title(item['category']))}</span>"
            f"<b>{_cell(item['id'])}</b> {_cell(_short(item['title'], item['id']))}"
            f"<span class='score {tone}'>{_cell(status)}</span></figcaption>"
            f"<div class='who'>{_cell(item['model'])}</div>{media}"
            f"<div class='checks'>{_cell(_verdict_text(item['category'], failed))}"
            f"<span class='sub'> · {spend.get('answer_tokens') or 0} ток. ответ / {spend.get('reasoning_tokens') or 0} reasoning</span></div>"
            f"<div class='row'><button class='open' data-page='{_cell(page)}' data-window='{window_name}'>"
            f"сцена в новом окне</button>"
            f"<button class='open cmp-solo' data-page='{_cell(page)}'>сравнить</button></div>"
            f"</figure>"
        )
    return "".join(cards)


def render_html(
    run_dirs: list[Path | str],
    page_dir: Path | None = None,
    *,
    per_run: bool = False,
    compare: list[str] | None = None,
    page_name: str = "comparison.html",
) -> str:
    """Two report flavours out of one renderer.

    ``per_run``
        a single model's full report — the page a run writes next to its own
        ``report.md``. Same sections, same showcase, but the wording and the
        header describe *this run* instead of a field of models;
    ``compare``
        explicit models to put in the comparison when more runs are passed
        (``comparison.html``). A selector matches the run's label, its model id or
        its directory name — whichever the caller happens to know — and keeps the
        order given. An unknown selector is a loud error, because a silently
        dropped model would make a leaderboard that lies.
    """
    run_dirs = list(run_dirs)
    if per_run and len(run_dirs) != 1:
        raise ValueError(f"per_run отчёт ждёт один прогон, получено {len(run_dirs)}")
    runs = [load_run(directory) for directory in run_dirs]
    if compare:
        runs = _select_runs(runs, compare)
    page_dir = page_dir or Path(run_dirs[0]).parent

    gallery.ensure_vendor(page_dir)
    _vendor_plotly(page_dir / "_vendor")

    index: list[dict[str, Any]] = []
    for run in runs:
        run_dir = Path(run["dir"])
        target = page_dir / "_showcase" / run_dir.name
        entry = gallery.materialise(run_dir, target)
        for item in entry["tasks"]:
            item["model"] = entry["model"]
            item["page"] = str(target.relative_to(page_dir) / item["page"]) if item.get("page") else ""
            item["frame"] = str(target.relative_to(page_dir) / item["frame"]) if item.get("frame") else ""
        index.extend(entry["tasks"])

    summaries = [(run, summarise(run["scores"], run.get("suite"))) for run in runs]
    summaries.sort(key=lambda pair: -pair[1]["headline_earned"] / (pair[1]["headline_possible"] or 1))

    facts = _suite_facts(runs)
    single = len(runs) == 1
    if per_run:
        title = f"DailyBench — {_model_name(runs[0])}"
        lead = (f"{_suite_line(facts, per_run=True)} Задания одинаковы для всех моделей; "
                "расход токенов документируется, а не обрезается лимитом. "
                "Живые сцены витрины открываются в новом окне.")
    else:
        title = "DailyBench — решения моделей на сьюте" if not single else f"DailyBench — {_model_name(runs[0])}"
        lead = (f"{_suite_line(facts, per_run=single)} Задания одинаковы для всех моделей; "
                "расход токенов документируется, а не обрезается лимитом.")

    prov_tab = ('<div class="tab" data-panel="p-prov">Провенанс</div>' if per_run else "")
    prov_panel = (
        '<div class="panel" id="p-prov"><h2>Провенанс: чем именно получены числа</h2>'
        '<p class="hint">Веса, окно, бюджеты, каркас и хеш сьюта — чтобы результат можно было '
        'воспроизвести, а не только прочитать.</p>'
        + _provenance([run for run, _ in summaries])
        + "</div>"
    ) if per_run else ""

    charts = json.dumps(_charts_payload([run for run, _ in summaries]), ensure_ascii=False)
    compare_rows = "".join(
        f"<label class='pick'><input type='checkbox' class='cmp' value='{_cell(item['id'])}' "
        f"data-model='{_cell(item['model'])}'><span>{_cell(item['model'])}</span> {_cell(item['id'])}</label>"
        for item in sorted(index, key=lambda x: (reversed(x["id"].split("-")) and int(x["id"].split("-")[1]), x["model"]))
    )

    return f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>{_cell(title)}</title>
<style>
 :root {{ color-scheme: dark }}
 body {{ margin:0; background:#0d1117; color:#e6edf3; font:14px/1.55 -apple-system,"Segoe UI",Roboto,sans-serif }}
 header {{ padding:22px 30px 4px }}
 h1 {{ font-size:22px; margin:0 0 4px }}
 h2 {{ font-size:17px; margin:22px 0 8px; border-top:1px solid #30363d; padding-top:16px }}
 h3 {{ font-size:15px; margin:6px 0 6px; color:#e6edf3 }}
 .hint {{ color:#8b949e; margin:2px 0 }}
 .tabs {{ display:flex; gap:6px; padding:8px 30px 0; position:sticky; top:0; background:#0d1117; z-index:5; border-bottom:1px solid #21262d }}
 .tab {{ background:#161b22; border:1px solid #30363d; border-radius:8px 8px 0 0; padding:9px 16px; cursor:pointer; color:#c9d1d9; font-weight:600 }}
 .tab.active {{ background:#1f6feb22; border-color:#1f6feb; color:#e6edf3 }}
 .panel {{ display:none; padding:6px 30px 56px }} .panel.active {{ display:block }}
 .cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:12px; padding:10px 0 0; align-items:start }}
 .card {{ position:relative; background:#161b22; border:1px solid #30363d; border-radius:12px; padding:14px; min-width:0 }}
 .card .sub {{ overflow-wrap:anywhere }}
 .rank {{ position:absolute; right:12px; top:10px; color:#484f58; font-weight:700 }}
 .big {{ font-size:30px; font-weight:700; line-height:1.1 }}
 .big .of {{ color:#8b949e; font-size:18px; font-weight:500 }}
 .big .unit {{ display:block; font-size:11px; color:#8b949e; font-weight:400; letter-spacing:.03em }}
 .meter {{ height:5px; background:#21262d; border-radius:4px; margin:8px 0; overflow:hidden }}
 .meter i {{ display:block; height:100%; background:#7ee787 }}
 .good {{ color:#7ee787 }} .mid {{ color:#e3b341 }} .bad {{ color:#ff7b72 }}
 .sub {{ color:#8b949e; font-size:11.5px }} .spend b {{ color:#c9d1d9 }}
 table.grid {{ border-collapse:collapse; width:100% }}
 table.grid th, table.grid td {{ border:1px solid #21262d; padding:5px 8px; text-align:left; vertical-align:top }}
 table.grid th {{ background:#161b22; font-weight:600; white-space:nowrap; position:sticky; top:46px }}
 .th-sub {{ display:block; color:#8b949e; font-weight:400; font-size:10.5px }}
 td.num {{ text-align:right; white-space:nowrap }}
 td.model {{ font-weight:600; white-space:nowrap }}
 .verdict-table td.verdict {{ font-weight:700 }}
 .verdict-table td.show, .suite-table .cell {{ font-variant-numeric:tabular-nums }}
 .suite-table .task {{ width:34%; font-size:12.5px }}
 .suite-table .cat {{ display:block; color:#8b949e; font-size:11px }}
 .suite-table .cell {{ text-align:center; font-weight:600 }}
 .cell.full {{ color:#7ee787; background:#7ee7870d }}
 .cell.part {{ color:#e3b341; background:#e3b3410d }}
 .cell.zero {{ color:#ff7b72; background:#ff7b720d }}
 .cell.skip {{ color:#8b949e }}
 .cell .of {{ color:#8b949e; font-weight:400; font-size:11px }}
 .cell .badge {{ display:block; color:#8b949e; font-weight:400; font-size:10.5px }}
 .row {{ display:flex; gap:8px; margin-top:8px }}
 button.open, a.link {{ background:#21262d; border:1px solid #30363d; color:#c9d1d9; border-radius:6px;
   padding:4px 9px; font-size:12px; cursor:pointer; text-decoration:none }}
 .gallery {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:12px }}
 figure {{ margin:0; background:#161b22; border:1px solid #30363d; border-radius:10px; padding:10px }}
 img {{ width:100%; border-radius:6px; display:block }}
 figcaption {{ font-size:12.5px; margin-bottom:6px }}
 .block {{ color:#79c0ff; text-transform:uppercase; font-size:10px; letter-spacing:.05em; margin-right:6px }}
 .who {{ color:#c9d1d9; font-size:11.5px; margin-bottom:6px; font-weight:600 }}
 .score {{ margin-left:6px }} .score.full {{ color:#7ee787 }} .score.part {{ color:#e3b341 }} .score.zero {{ color:#ff7b72 }}
 .checks {{ color:#8b949e; font-size:11.5px; margin-top:6px }}
 .marks {{ display:flex; flex-wrap:wrap; gap:4px 8px; margin-top:6px }}
 .mark {{ font-size:10.5px; color:#8b949e; background:#0f1319; border:1px solid #21262d; border-radius:5px; padding:1px 6px }}
 .mark b {{ font-weight:700 }} .m1,.m2 {{ color:#ff7b72 }} .m3 {{ color:#e3b341 }} .m4,.m5 {{ color:#7ee787 }}
 .missing {{ padding:36px; text-align:center; color:#8b949e; border:1px dashed #30363d }}
 button.shot {{ all:unset; cursor:zoom-in; display:block; position:relative }}
 button.shot .play {{ position:absolute; left:8px; bottom:8px; background:#000000b3; border-radius:6px;
   padding:2px 8px; font-size:11px; color:#e6edf3 }}
 .toolbar {{ display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin:10px 0; color:#8b949e; font-size:12.5px }}
 .toolbar input[type=range] {{ width:170px }}
 dialog {{ background:#0d1117; color:#e6edf3; border:1px solid #30363d; border-radius:12px; padding:0; width:96vw; height:94vh; max-width:none }}
 dialog::backdrop {{ background:#000c }}
 .dlg-head {{ display:flex; justify-content:space-between; align-items:center; padding:10px 14px; border-bottom:1px solid #21262d }}
 .frames {{ display:flex; height:calc(94vh - 52px) }}
 .pane {{ flex:1; display:flex; flex-direction:column; border-right:1px solid #21262d; min-width:0 }}
 .pane:last-child {{ border-right:0 }}
 .pane iframe {{ flex:1; width:100%; border:0; background:#000 }}
 .pane.solo {{ align-items:center; justify-content:center; overflow:hidden }}
 /* Кадр сцены 1920x1080: вписываем целиком под панель, без обрезки и полей.
    Масштаб считается в JS (fitSolo) и применяется зумом обёртки — так он
    работает и для одной сцены, и для нескольких панелей рядом. */
 .pane.solo .stage {{ flex:none; transform-origin:center center; zoom:1 }}
 .pane.solo iframe {{ width:1920px; height:1080px; border:0; display:block; background:#000 }}
 .frames.many .pane iframe {{ width:100%; height:100%; }}
 .pane .cap {{ padding:6px 10px; font-size:12px; color:#8b949e; display:flex; justify-content:space-between }}
 .pick {{ display:inline-block; padding:1px 10px 1px 0; font-size:12px; color:#8b949e; white-space:nowrap }}
 .pick span {{ color:#c9d1d9 }}
 #cmp-list {{ display:flex; flex-wrap:wrap; gap:2px 14px; max-height:132px; overflow:auto;
   border:1px solid #21262d; border-radius:8px; padding:8px 10px; background:#0f1319; margin:6px 0 }}
 #cmp-list::before {{ content:'отметьте два и больше решения, затем «сравнить выбранные»'; flex-basis:100%; color:#6e7681 }}
 button.open:hover, a.link:hover {{ border-color:#1f6feb }}
 .chart {{ margin:6px 0 22px }}
 .warn {{ background:#3d2b00; color:#e3b341; border:1px solid #6b4b00; border-radius:8px; padding:8px 12px; margin:8px 0 }}
 code {{ background:#21262d; border-radius:4px; padding:1px 5px }}
</style></head>
<body>
<header>
 <div id="filehint" class="warn" hidden>Отчёт открыт как <code>file://</code>: живая витрина тянет ES-модули,
 которые браузер с такого origin не грузит (кадры работают). Запустите <code>make serve DIR=runs</code>
 и откройте <code>http://127.0.0.1:8765/{page_name}</code>.</div>
 <h1>{title}</h1>
 <p class="hint">{lead}</p>
</header>
<nav class="tabs">
 <div class="tab active" data-panel="p-summary">Решения моделей</div>
 <div class="tab" data-panel="p-suite">Решения моделей по сьюту</div>
 <div class="tab" data-panel="p-charts">Графики</div>
 <div class="tab" data-panel="p-showcase">Витрина решений</div>
 {prov_tab}
</nav>

<div class="panel active" id="p-summary">
 <h2>Кто сколько набрал</h2>
 <div class="cards">{_model_cards(summaries)}</div>
 <h2>Качество по категориям сьюта</h2>
 <p class="hint">A–J — балльные категории (проценты от возможного), K и L — витрина: A инстр., B рассужд., C код,
 D инструм., E контекст, F галлюц., G JSON, H диалог, I файлы, J ревью.</p>
 {_verdict_table(summaries)}
 <h2>Расход: токены и время (документируется, лимит — окно модели)</h2>
 <p class="hint">Ответ и reasoning считаются отдельно: 100K токенов reasoning без единой строки кода — это результат,
 который видно здесь, а не молчаливый потолок.</p>
 <table class="grid"><thead><tr><th>Модель</th><th>Токены ответа</th><th>Токены reasoning</th>
 <th>Минут генерации</th><th>Балл за минуту</th><th>Вызовов инструментов</th></tr></thead><tbody>
 {''.join(f"<tr><td class='model'>{_cell(_model_name(run))}</td>"
          f"<td class='num'>{_spend_totals(run)['answer']:.0f}</td>"
          f"<td class='num'>{_spend_totals(run)['reasoning']:.0f}</td>"
          f"<td class='num'>{_spend_totals(run)['seconds'] / 60:.1f}</td>"
          f"<td class='num'>{summary['headline_earned'] / max(_spend_totals(run)['seconds'] / 60, 1e-9):.2f}</td>"
          f"<td class='num'>{_spend_totals(run)['tool_calls']:.0f}</td></tr>" for run, summary in summaries)}
 </tbody></table>
 <h2>Как читать отчёт</h2>
 <ul class="hint">
   <li><b>Баллы качества</b> — только категории A–J, максимум 100. Витрина K и L в итог не входит.</li>
   <li><b>«Не проверялось»</b> — задание не удалось проверить на этой машине (например нет браузера); это не ноль.</li>
   <li><b>Стоп-причина</b> — почему генерация закончилась: <code>stop</code> — сама, <code>length</code> — упёрлась в окно,
       <code>tool_calls</code> — ждёт результат инструмента.</li>
 </ul>
 <h2>Провалы по моделям</h2>
 {''.join(f"<h3>{_cell(_model_name(run))}</h3><table class='grid tasks'><thead><tr><th>ID</th><th>Название</th>"
          f"<th>Балл</th><th>Ответ</th><th>Думанье</th><th>Сек</th><th>Стоп</th><th>Проваленная проверка</th></tr></thead>"
          f"<tbody>{_task_rows(run)}</tbody></table>" for run, _ in summaries)}
</div>

<div class="panel" id="p-suite">
 <h2>Сьют × решения моделей</h2>
 <p class="hint">Одна строка — одно задание сьюта, столбец — модель. Так видно, где именно расходятся модели,
 а не только итоговый балл.</p>
 <div class="toolbar">
   <label class="pick"><input type="checkbox" class="flt" value="scored" checked> только балльные A–J</label>
   <label class="pick"><input type="checkbox" class="flt" value="show" checked> витрина K–L</label>
   <label class="pick"><input type="checkbox" class="flt" value="onlyfail"> только строки с потерями</label>
   <button class="open" id="expand-all">развернуть все проверки</button>
 </div>
 <div id="suite-wrap">{_suite_table(summaries)}</div>
</div>

<div class="panel" id="p-charts">
 <h2>Качество по категориям</h2><div id="ch-scored" class="chart"></div>
 <h2>Витрина: K и L</h2><div id="ch-show" class="chart"></div>
 <h2>Расход токенов: ответ против reasoning</h2><div id="ch-spend" class="chart"></div>
 <h2>Цена результата: балл за минуту и минуты генерации</h2><div id="ch-cost" class="chart"></div>
 <h2>Reasoning против качества</h2><div id="ch-think" class="chart"></div>
</div>

<div class="panel" id="p-showcase">
 <h2>Кадры решений: клик открывает живую сцену в новом окне</h2>
 <p class="hint">Арт-блок (L) баллов не получает: у каждой сцены своя форма, а проверки лишь описывают кадр. Смотрите кадры и живые сцены — решение за человеком. Блок K остаётся балльным: его координаты сверяются с пересчётом.</p>
 <div class="toolbar">
   <span>показать:</span>
      <label class="pick"><input type="checkbox" class="flt" value="K" checked> K — симуляции (баллы)</label>
      <label class="pick"><input type="checkbox" class="flt" value="L" checked> L — арт (смотрим глазами)</label>
   <span>· размер <input type="range" id="size" min="240" max="900" step="20" value="380"></span>
   <button class="open" id="video-mode">режим съёмки</button>
   <button class="open" id="cmp-mode">сравнить выбранные</button>
 </div>
 <p class="hint">Режим съёмки: одна сцена на весь экран, ← / → листают задачи, пробел шагает время, Esc закрывает.
 Сравнение: отметьте решения двух и более моделей — они встанут рядом и будут шагать синхронно.</p>
 <div id="cmp-list">{compare_rows}</div>
 <div class="gallery" id="gallery">{_gallery(sorted(index, key=lambda x: (x['category'], int(x['id'].split('-')[1]), x['model'])))}</div>
</div>

{prov_panel}

<dialog id="viewer">
 <div class="dlg-head"><b id="viewer-title">решение</b>
   <span><button class="open" id="viewer-step">шаг</button>
   <button class="open" id="viewer-close">закрыть (Esc)</button></span></div>
 <div class="frames" id="viewer-frames"></div>
</dialog>

<script src="_vendor/plotly.min.js"></script>
<script>
const DATA = {charts};

document.querySelectorAll('.tab').forEach(tab => tab.onclick = () => {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  tab.classList.add('active');
  document.getElementById(tab.dataset.panel).classList.add('active');
  if (tab.dataset.panel === 'p-charts' && !window.__drawn) {{ window.__drawn = true; drawCharts(); }}
}});

const PALETTE = ['#79c0ff', '#7ee787', '#e3b341', '#ff7b72', '#d2a8ff', '#56d4dd'];
const LAYOUT = {{ paper_bgcolor:'#0d1117', plot_bgcolor:'#0d1117', font:{{color:'#c9d1d9', size:11}},
  margin:{{l:60,r:20,t:34,b:70}}, legend:{{orientation:'h', y:-0.25}}, barmode:'group' }};
const CONFIG = {{ displayModeBar:false, responsive:true }};
const AXTITLE = {{size:11, color:'#8b949e'}};

function drawCharts() {{
  Plotly.newPlot('ch-scored', DATA.models.map((m, i) => ({{
    type:'bar', name:m, x: DATA.scoredNames, y: DATA.scored[i],
    marker:{{color:PALETTE[i % PALETTE.length]}},
    text: DATA.scored[i].map(v => v === null ? '' : v.toFixed(0) + '%'), textposition:'outside',
    hovertemplate: m + ': %{{y:.0f}}% от возможного<extra></extra>',
  }})), Object.assign({{}}, LAYOUT, {{yaxis:{{title:{{text:'% от возможного', font:AXTITLE}}, range:[0,110]}}}}), CONFIG);

  Plotly.newPlot('ch-show', DATA.models.map((m, i) => ({{
    type:'bar', name:m, x:['K — симуляции (баллы)', 'L — арт (глазами)'], y: DATA.showcase[i],
    marker:{{color:PALETTE[i % PALETTE.length]}},
    text: DATA.showcase[i].map(v => v === null ? '' : v.toFixed(0) + '%'), textposition:'outside',
  }})), Object.assign({{}}, LAYOUT, {{yaxis:{{title:{{text:'% витрины', font:AXTITLE}}, range:[0,110]}}}}), CONFIG);

  Plotly.newPlot('ch-spend', [
    {{ type:'bar', name:'токены ответа', x: DATA.spend.map(s => s.model), y: DATA.spend.map(s => s.answer),
       marker:{{color:'#7ee787'}}, hovertemplate: '%{{x}}: %{{y:.0f}} токенов ответа<extra></extra>' }},
    {{ type:'bar', name:'токены reasoning', x: DATA.spend.map(s => s.model), y: DATA.spend.map(s => s.reasoning),
       marker:{{color:'#e3b341'}}, hovertemplate: '%{{x}}: %{{y:.0f}} токенов reasoning<extra></extra>' }},
  ], Object.assign({{}}, LAYOUT, {{barmode:'stack', yaxis:{{title:{{text:'токенов за прогон, K', font:AXTITLE}},
     tickformat:'.3s'}}}}), CONFIG);

  Plotly.newPlot('ch-cost', [
    {{ type:'bar', name:'балл за минуту', x: DATA.cost.map(c => c.model), y: DATA.cost.map(c => c.per_min),
       marker:{{color:'#56d4dd'}}, text: DATA.cost.map(c => c.per_min.toFixed(2)), textposition:'outside',
       yaxis:'y', hovertemplate: '%{{x}}: %{{y:.2f}} балла за минуту<extra></extra>' }},
    {{ type:'scatter', mode:'markers+text', name:'минут генерации', x: DATA.cost.map(c => c.model),
       y: DATA.cost.map(c => c.minutes), text: DATA.cost.map(c => c.minutes.toFixed(1) + ' мин'),
       textposition:'top center', marker:{{color:'#ff7b72', size:11}}, yaxis:'y2',
       hovertemplate: '%{{x}}: %{{y:.1f}} минут<extra></extra>' }},
  ], Object.assign({{}}, LAYOUT, {{
     yaxis:{{title:{{text:'балл за минуту', font:AXTITLE}}}},
     yaxis2:{{title:{{text:'минут генерации', font:AXTITLE}}, overlaying:'y', side:'right', showgrid:false}},
  }}), CONFIG);

  Plotly.newPlot('ch-think', [{{
    type:'scatter', mode:'markers+text', x: DATA.cost.map(c => c.reasoning), y: DATA.cost.map(c => c.score),
    text: DATA.cost.map(c => c.model), textposition:'bottom right',
    marker:{{size:13, color: PALETTE.slice(0, DATA.cost.length)}},
    hovertemplate: '%{{text}}: %{{x:.0f}}K токенов reasoning → %{{y:.0f}} баллов<extra></extra>',
  }}], Object.assign({{}}, LAYOUT, {{
    xaxis:{{title:{{text:'токенов reasoning, K', font:AXTITLE}}}},
    yaxis:{{title:{{text:'баллы качества', font:AXTITLE}}}},
  }}), CONFIG);
}}

const galleryEl = document.getElementById('gallery');
const viewer = document.getElementById('viewer');
const framesEl = document.getElementById('viewer-frames');
let current = [];

function applyFilters() {{
  const cats = new Set([...document.querySelectorAll('.flt[value="K"],.flt[value="L"]:checked')].map(i => i.value));
  document.querySelectorAll('#gallery figure').forEach(f => {{
    f.style.display = cats.has(f.dataset.cat) ? '' : 'none';
  }});
}}
document.querySelectorAll('.flt[value="K"],.flt[value="L"]').forEach(i => i.onchange = applyFilters);
document.getElementById('size').oninput = e => {{
  galleryEl.style.gridTemplateColumns = `repeat(auto-fill, minmax(${{e.target.value}}px, 1fr))`;
}};

function visibleCards() {{
  return [...document.querySelectorAll('#gallery figure')].filter(f => f.style.display !== 'none');
}}

function openViewer(items, index) {{
  current = items;
  framesEl.innerHTML = items.map((it, i) => `
    <div class="pane${{items.length === 1 ? ' solo' : ''}}">
      <div class="cap"><span>${{it.model}} · ${{it.id}}</span><span>${{it.score}}</span></div>
      ${{it.page ? `<div class="stage"><iframe data-i="${{i}}" src="${{it.page}}" sandbox="allow-scripts"></iframe></div>`
                 : '<div class="missing">модель не дошла до HTML</div>'}}
    </div>`).join('');
  document.getElementById('viewer-title').textContent =
    items.length === 1 ? `${{items[0].model}} · ${{items[0].id}}` : `сравнение: ${{items.length}} решения`;
  framesEl.classList.toggle('many', items.length > 1);
  viewer.showModal();
  fitSolo();
}}

function fitSolo() {{
  // Сцена рисуется в 1920x1080. Масштабируем её целиком под панель, чтобы кадр
  // не обрезался ни на маленьком ноутбуке, ни на большом мониторе: пустых
  // полей нет, картинка занимает максимум доступного места.
  const pane = framesEl.querySelector('.pane.solo');
  const stage = pane && pane.querySelector('.stage');
  if (!stage) return;
  const cap = pane.querySelector('.cap');
  const availH = pane.clientHeight - (cap ? cap.offsetHeight : 0);
  // zoom, а не scale: он сжимает и раскладку, поэтому панель не остаётся с
  // распухшим размером 1920x1080 за пределами экрана.
  const k = Math.min(pane.clientWidth / 1920, availH / 1080);
  stage.style.zoom = String(Math.max(0.05, k).toFixed(4));
}}
window.addEventListener('resize', fitSolo);
if (window.ResizeObserver) new ResizeObserver(fitSolo).observe(framesEl);

function stepAll() {{
  framesEl.querySelectorAll('iframe').forEach(fr => {{ try {{ fr.contentWindow.postMessage('step', '*'); }} catch (e) {{}} }});
}}

function cardItems(cards) {{
  return cards.map(f => ({{model: f.dataset.model, id: f.dataset.id,
    score: f.querySelector('.score').textContent,
    page: f.querySelector('button.shot')?.dataset.page || f.querySelector('button.open')?.dataset.page || ''}}));
}}

// Живая сцена живёт в собственном окне: она играет сама и остаётся, пока в
// отчёте листают дальше. Имя окна фиксировано задачей и моделью, поэтому
// повторный клик переиспользует уже открытое окно, а не плодит вкладки.
// Окно сцены — на весь экран; страница вписывает кадр 1920x1080 без обрезки.
function sceneSize() {{
  return {{
    w: Math.min(screen.availWidth || 1920, 2560),
    h: Math.max(540, (screen.availHeight || 1080) - 60),
  }};
}}
function openScene(page, name) {{
  if (!page) return null;
  const {{w, h}} = sceneSize();
  // noopener: a published answer must not reach back into the report window.
  const win = window.open(page, name || '', `width=${{w}},height=${{h}},noopener`);
  if (win) {{ win.focus(); }}
  return win;
}}
function openScenes(items) {{
  const many = items.length > 1;
  const {{w, h}} = sceneSize();
  items.forEach((it, i) => {{
    const place = many ? `,left=${{120 + i * 40}},top=${{80 + i * 30}}` : '';
    const win = window.open(it.page, many ? `cmp-${{it.id}}-${{it.model}}` : (it.window || ''), `width=${{w}},height=${{h}}${{place}},noopener`);
    if (win) win.focus();
  }});
}}
document.querySelectorAll('button.shot').forEach(btn => btn.onclick = () => openScene(btn.dataset.page, btn.dataset.window));
document.querySelectorAll('button.open[data-window]').forEach(btn => btn.onclick = () => openScene(btn.dataset.page, btn.dataset.window));
document.querySelectorAll('button.cmp-solo').forEach(btn => btn.onclick = () => openViewer(cardItems([btn.closest('figure')])));
document.getElementById('video-mode').onclick = () => {{
  const cards = visibleCards();
  if (cards.length) openViewer(cardItems([cards[0]]));
}};
document.getElementById('cmp-mode').onclick = () => {{
  const picked = [...document.querySelectorAll('.cmp:checked')].map(i => ({{id: i.value, model: i.dataset.model}}));
  if (picked.length < 2) {{ alert('выберите минимум два решения'); return; }}
  const cards = picked.map(p => [...document.querySelectorAll('#gallery figure')]
    .find(f => f.dataset.id === p.id && f.dataset.model === p.model)).filter(Boolean);
  if (cards.length) openScenes(cardItems(cards));
}};
document.getElementById('viewer-close').onclick = () => viewer.close();
document.getElementById('viewer-step').onclick = stepAll;
document.addEventListener('keydown', e => {{
  if (!viewer.open) return;
  if (e.key === 'Escape') viewer.close();
  if (e.key === ' ') {{ e.preventDefault(); stepAll(); }}
  if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {{
    if (current.length === 1) {{
      const cards = visibleCards();
      const at = cards.findIndex(f => f.dataset.id === current[0].id && f.dataset.model === current[0].model);
      const next = cards[(at + (e.key === 'ArrowRight' ? 1 : cards.length - 1)) % cards.length];
      if (next) openViewer(cardItems([next]));
    }}
  }}
}});

// --- suite table filters ------------------------------------------------------
document.querySelectorAll('.flt[value="scored"],.flt[value="show"]').forEach(i => i.onchange = () => {{
  const rows = document.querySelectorAll('.suite-table tbody tr');
  const showScored = document.querySelector('.flt[value="scored"]').checked;
  const showShow = document.querySelector('.flt[value="show"]').checked;
  rows.forEach(r => {{
    const isShow = r.querySelector('.task .cat').textContent.includes('витрина');
    r.style.display = (isShow ? showShow : showScored) ? '' : 'none';
  }});
}});
document.querySelector('.flt[value="onlyfail"]').onchange = e => {{
  document.querySelectorAll('.suite-table tbody tr').forEach(r => {{
    const losses = [...r.querySelectorAll('.cell')].some(c => !c.classList.contains('full') && !c.classList.contains('skip'));
    r.style.display = !e.target.checked || losses ? '' : 'none';
  }});
}};
document.getElementById('expand-all').onclick = () => {{
  document.querySelectorAll('.suite-table .cell .badge').forEach(b => {{
    b.style.display = b.style.display === 'block' ? 'none' : 'block';
  }});
}};

applyFilters();
if (location.protocol === 'file:') {{ document.getElementById('filehint').hidden = false; }}
</script>
</body></html>
"""


def _vendor_plotly(target: Path) -> None:
    """Plotly next to the page: charts must work without the network."""
    from .browser import REPO

    source = REPO / "vendor" / "plotly" / "plotly.min.js"
    if not source.is_file():
        return
    target.mkdir(parents=True, exist_ok=True)
    destination = target / "plotly.min.js"
    if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
        import shutil

        shutil.copyfile(source, destination)


def write_report_html(run_dir: Path | str, target: Path | None = None) -> Path:
    """One model's HTML report, written next to its own ``report.md``.

    The run directory is the unit that gets published, archived and linked, so the
    page that describes it lives there too. Assets go into ``_vendor`` and
    ``_showcase`` beside it — the same layout ``publish`` uses, so the folder can
    be served or zipped as is.
    """
    run_dir = Path(run_dir)
    out = Path(target) if target else run_dir / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_html([run_dir], page_dir=out.parent, per_run=True, page_name=out.name),
        encoding="utf-8",
    )
    return out


def write_html(
    run_dirs: list[Path | str],
    target: Path | None = None,
    *,
    compare: list[str] | None = None,
) -> Path:
    """Render the report next to the runs and copy the assets it needs.

    ``target`` — куда положить страницу: по умолчанию ``comparison.html`` рядом с
    прогонами, для публикации — ``reports/<имя>/index.html``. Всё, что нужно
    странице после сборки (кадры, страницы сцен, three.js, plotly), копируется в
    каталог отчёта: ``runs/`` после публикации можно не хранить.
    """
    runs = [Path(directory) for directory in run_dirs]
    out = Path(target) if target else runs[0].parent / "comparison.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_html(runs, page_dir=out.parent, page_name=out.name, compare=compare),
        encoding="utf-8",
    )
    return out
