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

Визуальные фигуры спецификации §7.2 собирает :mod:`dailybench.viz`, а этот
модуль только подаёт им посчитанные строки: радар и heatmap — на страницу
сравнения, вердикт-полоса, полнота, кластеры и диффы — на страницу прогона, в
порядке §7.2. Ни одна фигура не пересчитывает метрики.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from . import core, gallery, schema, stats, viz
from .report import (
    WEIGHT_SCHEMES,
    compare_summaries,
    contract_key,
    load_run,
    rankable,
    summarise,
)

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


def _task_sort_key(task_id: Any) -> tuple[int, str]:
    """Numeric second id segment, tolerant of a missing dash; the raw id ties."""
    text = str(task_id or "")
    head, _, rest = text.partition("-")
    tail = rest.split("-", 1)[0]
    return (int(tail) if head and tail.isdigit() else 0, text)


def _tone(earned: float, possible: float) -> str:
    share = earned / (possible or 1)
    return "good" if share >= 0.8 else ("mid" if share >= 0.5 else "bad")


def _pct(earned: float, possible: float) -> float | None:
    return round(100 * earned / possible, 1) if possible else None


def _ci_band(ci: tuple[float, float] | None, n: int) -> str:
    """A 0–100 % uncertainty bar, with its numbers in data attributes.

    Decoding is the point: a test (or a screen reader) reads ``data-ci-low`` and
    ``data-ci-high`` instead of guessing from pixels. Below the small-sample
    threshold the band is labelled "ориентировочно".
    """
    if not ci:
        return ""
    low, high = (max(0.0, ci[0]), min(100.0, ci[1]))
    note = stats.note(n)
    label = f"ДИ 95%: {low:.0f}–{high:.0f}% · n={n}" + (f" · {note}" if note else "")
    return (
        f"<span class='ci-band' data-ci-low='{low:.2f}' data-ci-high='{high:.2f}' data-n='{n}' "
        f"title='{_cell(label)}' style='--lo:{low:.1f}%;--width:{max(0.0, high - low):.1f}%'>"
        "<i></i></span>"
    )


def _caveat(summary: dict[str, Any]) -> str:
    """The verdict's caveats, without the score the headline already shows.

    ``summarise`` never changes a denominator silently: a failed or unverified
    admission, a partial run and skipped tasks are all spelled out in the
    verdict. The markdown report carries that sentence verbatim; the HTML must
    too, or a reader of the published page sees "85/90" with no reason.
    """
    verdict = str(summary.get("verdict") or "")
    return verdict.split(" ", 1)[1] if " " in verdict else ""


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
    by_selector: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        for selector in _selector_of(run):
            by_selector.setdefault(selector, []).append(run)
    missing = [name for name in wanted if name not in by_selector]
    if missing:
        raise ValueError(
            f"неизвестные модели для сравнения: {missing}; есть {sorted(by_selector)}"
        )
    picked: list[dict[str, Any]] = []
    used: dict[str, int] = {}
    for name in wanted:
        candidates = by_selector[name]
        # A repeated selector means "the next run with this label", so two runs
        # of one model can be put side by side explicitly.
        index = min(used.get(name, 0), len(candidates) - 1)
        used[name] = index + 1
        run = candidates[index]
        if not any(run is present for present in picked):
            picked.append(run)
    return picked


def _suite_facts(runs: list[dict[str, Any]]) -> dict[str, int]:
    """Counts that describe the suite this report was built from.

    Three sources, most specific first: the run's own digest, then the suite
    rebuilt from the run's config (a full-suite run recorded before digests were
    stored), then the live registry. Hard-coding these was wrong the moment a
    category was added — the header claimed 83 tasks while the suite shipped 96,
    and a 100-point suite was advertised as 559.

    Weights follow the run instead of today's baseline: the profile comes from
    the config (or the digest) and the per-task points from the digest when it
    recorded them. The category ceiling was also wrong for a filtered run — a
    ``--only IF-1`` report advertised A's whole 10 points for one 2-point task,
    and a ``pilot`` one an 80-point visual block where the profile offers 12.
    """
    # The registry holds the per-category weights (K's block total, L's per-task
    # weight); without it the showcase column under-counts silently.
    from .tasks import load_all

    load_all()
    per_task: list[dict[str, Any]] = []
    profile = core.DEFAULT_PROFILE
    for run in runs:
        digest = run.get("suite") or {}
        rows = digest.get("per_task") or []
        if len(rows) > len(per_task):
            per_task = rows
            profile = _run_profile(run, digest)
    if not per_task:
        per_task = _rebuilt_digest(runs)
    scored = [row for row in per_task if row.get("category") in SCORED]
    showcase = [row for row in per_task if row.get("category") in SHOWCASE]
    return {
        "tasks": len(per_task),
        "scored": len(scored),
        "showcase": len(showcase),
        # ``scored_points`` is the sum over the tasks that were actually offered;
        # a digest without weights falls back to the category ceiling, which is
        # exact for a full run. ``suite_points`` stays the ceiling of the whole
        # scored half, so a showcase-only run can still say what it did not test
        # instead of implying a 0-point suite.
        "scored_points": int(_scored_points(scored, profile)),
        "showcase_points": int(sum(_showcase_points(key, profile) for key in {r["category"] for r in showcase})),
        "suite_points": int(sum(core.points_for(key, profile) or 0 for key in SCORED)),
    }


def _run_profile(run: dict[str, Any], digest: dict[str, Any]) -> str:
    """The profile a run was built for; the baseline when it is unknown."""
    name = (run.get("config") or {}).get("profile") or digest.get("profile")
    return name if name in core.PROFILES else core.DEFAULT_PROFILE


def _scored_points(rows: list[dict[str, Any]], profile: str) -> float:
    """What the scored tasks on offer were worth.

    A digest that recorded ``max_points`` answers exactly (a filtered run only
    offered its own tasks' points). An older digest cannot, so the category
    ceiling — the profile's weight — is the best remaining answer.
    """
    weights = [row.get("max_points") for row in rows]
    if rows and all(isinstance(weight, (int, float)) for weight in weights):
        return float(sum(weights))
    return float(sum(core.points_for(row["category"], profile) or 0.0 for row in rows))


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
                ({"id": task.id, "category": task.category, "max_points": task.max_points}
                 for task in build_suite(cfg)),
                key=lambda row: row["id"],
            )
        except Exception:  # pragma: no cover - a config from an older version
            continue
    return []


def _showcase_points(category: str, profile: str = core.DEFAULT_PROFILE) -> float:
    """Points a showcase category carries.

    ``SHOWCASE_POINTS`` totals the whole block under ``K``; for the art
    category ``L`` the number is informational only. The block depends on the
    profile (pilot 12, stretch 120), so the caller passes the run's own. A
    category without a declared weight falls back to its registered one instead
    of counting as zero.
    """
    declared = core.points_for(category, profile)
    if declared is not None:
        return float(declared)
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


def _spend_totals(run: dict[str, Any]) -> dict[str, Any]:
    """Task transcripts summed; an unknown contribution is counted, not zeroed.

    A partial sum divided by a complete score reads as a cheap run, so
    ``answer``/``reasoning``/``total``/``seconds`` are ``None`` as soon as one
    measured task left the field unknown. ``unknown`` names the fields and
    ``unknown_tasks`` the number of affected tasks, for the «н/п» text.
    """
    answer = reason = seconds = tool_calls = 0.0
    unknown = {"answer": 0, "reasoning": 0, "seconds": 0}
    tasks = 0
    for entry in run.get("tasks") or []:
        spend = (entry.get("transcript") or {}).get("spend") or {}
        if not spend:
            continue
        tasks += 1
        unknown["answer"] += spend.get("answer_tokens") is None
        unknown["reasoning"] += spend.get("reasoning_tokens") is None
        unknown["seconds"] += spend.get("generate_s") is None
        answer += spend.get("answer_tokens") or 0
        reason += spend.get("reasoning_tokens") or 0
        seconds += spend.get("generate_s") or 0
        tool_calls += spend.get("tool_calls") or 0
    total_known = not unknown["answer"] and not unknown["reasoning"] and bool(tasks)
    total = answer + reason if total_known else None
    return {
        "answer": answer if tasks and not unknown["answer"] else None,
        "reasoning": reason if tasks and not unknown["reasoning"] else None,
        "seconds": seconds if tasks and not unknown["seconds"] else None,
        "tool_calls": tool_calls,
        "total": total,
        "reasoning_share": (reason / total) if total else None,
        "unknown": unknown,
        "unknown_tasks": sum(
            1
            for entry in run.get("tasks") or []
            if (spend := (entry.get("transcript") or {}).get("spend") or {})
            and any(spend.get(key) is None for key in ("answer_tokens", "reasoning_tokens", "generate_s"))
        ),
        "tasks": tasks,
    }


def _kv(value: float | None) -> str:
    return "н/п" if value is None else f"{value / 1000:.0f}K"


def _minutes(seconds: float | None, digits: int = 0) -> str:
    return "н/п" if seconds is None else f"{seconds / 60:.{digits}f} мин"


def _num(value: float | None, digits: int = 0) -> str:
    return "н/п" if value is None else f"{value:.{digits}f}"


def _per_min(summary: dict[str, Any], spend: dict[str, Any]) -> float | None:
    """Earned points per generation minute — ``None`` without a known time."""
    seconds = spend.get("seconds")
    if seconds is None or seconds / 60 < 1e-9:
        return None
    return summary["headline_earned"] / (seconds / 60)


def _spend_note(spend: dict[str, Any]) -> str:
    unknown = spend.get("unknown_tasks") or 0
    return f" <span class='th-sub'>(н/п у {unknown} из {spend['tasks']})</span>" if unknown else ""


def _task_spend(spend: dict[str, Any], key: str) -> str:
    """One task's number: «—» when no transcript, «н/п» when it went unreported."""
    if not spend:
        return "—"
    value = spend.get(key)
    if value is None:
        return "н/п"
    return f"{value:.1f}" if key == "generate_s" else f"{int(value)}"


def _json_script(payload: Any) -> str:
    """JSON that is safe inside a ``<script>`` block.

    ``json.dumps`` leaves ``<``/``>`` (so ``</script>`` closes the block early)
    and U+2028/U+2029 (line terminators in older JS) as-is. Escaping them keeps
    imported metadata from breaking out of the data block (R16).
    """
    text = json.dumps(payload, ensure_ascii=False)
    for raw, safe in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                      ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        text = text.replace(raw, safe)
    return text


def _run_id(run: dict[str, Any]) -> str:
    """Immutable identity of a run: its directory name, never a model label."""
    return Path(str(run.get("dir") or "")).name


def _model_name(run: dict[str, Any]) -> str:
    config = run.get("config") or {}
    label = config.get("label") or config.get("model") or "?"
    return str(label)


def _column_name(run: dict[str, Any]) -> str:
    """Column caption: the label alone is ambiguous for two runs of one model."""
    label = _model_name(run)
    return label if not _run_id(run) or _run_id(run) == label else f"{label} · {_run_id(run)}"


def _contract_key(run: dict[str, Any]) -> str:
    """Immutable contract identity of a run (see :func:`report.contract_key`)."""
    return contract_key(run)


def _contract_label(contract: str) -> str:
    return f"<code>{_cell(contract[:16])}</code>" if contract else "без контракта"


def _rank_key(pair: tuple[dict[str, Any], dict[str, Any]]) -> tuple[bool, float]:
    """Order inside one contract: earned points against the profile denominator.

    The denominator is fixed per profile, so a run that offered fewer points
    cannot outrank more earned ones by a smaller number. A failed admission
    shrinks its own headline denominator, so it sorts after the admitted runs
    of the same contract instead of pretending to lead.
    """
    summary = pair[1]
    profile = str(summary.get("profile") or core.DEFAULT_PROFILE)
    total = core.scored_total(profile) or 1.0
    return (summary.get("admission_state") == "failed", -summary["headline_earned"] / total)


def _group_contracts(
    summaries: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[tuple[str, list[tuple[dict[str, Any], dict[str, Any]]], list[tuple[dict[str, Any], dict[str, Any]]]]]:
    """Split runs by contract, and inside a contract into rankable and rest.

    A partial ``1/1`` used to sit above a full ``95/100`` because both were
    sorted by their own ratio. Ranking happens *after* this split: only the
    first list of each contract gets places, everything else is shown as an
    incomplete or ineligible measurement.
    """
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for pair in summaries:
        groups.setdefault(_contract_key(pair[0]), []).append(pair)
    ordered: list[tuple[str, list[tuple[dict[str, Any], dict[str, Any]]], list[tuple[dict[str, Any], dict[str, Any]]]]] = []
    for contract, pairs in groups.items():
        ranked = [pair for pair in pairs if rankable(pair[1])]
        rest = [pair for pair in pairs if not rankable(pair[1])]
        ranked.sort(key=_rank_key)
        rest.sort(
            key=lambda pair: (
                not pair[1].get("eligible"),
                -len([s for s in pair[0].get("scores") or [] if not s.get("skipped")]),
            )
        )
        ordered.append((contract, ranked, rest))
    # A contract with no rankable run never leads the page: a lone ``1/1``
    # partial used to outrank a complete suite merely by arriving first.
    ordered.sort(
        key=lambda item: (
            not item[1],
            _rank_key(min(item[1], key=_rank_key)) if item[1] else (False, 0.0),
        )
    )
    return ordered


def _coverage(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Intersection of the task ids and per-run coverage against it.

    Reading two ratios side by side is only meaningful over the tasks *both*
    runs actually presented; the report states that core instead of implying
    every number was measured on the same suite.
    """
    ids = {
        _run_id(run): {str(score.get("task_id")) for score in run.get("scores") or []}
        for run in runs
    }
    common = set.intersection(*ids.values()) if ids else set()
    per_run = {
        run_id: {
            "total": len(own),
            "common": len(own & common),
            "coverage": round(100 * len(own & common) / len(own), 1) if own else 0.0,
        }
        for run_id, own in ids.items()
    }
    return {"common": sorted(common), "per_run": per_run}


def _coverage_block(runs: list[dict[str, Any]]) -> str:
    if len(runs) < 2:
        return ""
    contract = {_contract_key(run) for run in runs}
    data = _coverage(runs)
    rows = "".join(
        f"<tr><td class='model'>{_cell(_model_name(run))}</td>"
        f"<td class='num'>{data['per_run'][_run_id(run)]['total']}</td>"
        f"<td class='num'>{data['per_run'][_run_id(run)]['common']}</td>"
        f"<td class='num'>{data['per_run'][_run_id(run)]['coverage']:.0f}%</td></tr>"
        for run in runs
    )
    note = (
        f"<p class='hint warn'>{len(contract)} разных контракта сьюта: таблицы ниже раздельные, "
        "общий ранг между ними не выставляется.</p>"
        if len(contract) > 1
        else ""
    )
    return (
        "<h2>Покрытие и общее ядро заданий</h2>"
        + note
        + f"<p class='hint'>Общих заданий у всех прогонов: <b>{len(data['common'])}</b>. "
        "Доля считается от собственного набора каждого прогона; неполный прогон не поднимается "
        "в общем ранге только за счёт меньшего знаменателя.</p>"
        f"<table class='grid coverage-table'><thead><tr><th>Модель</th><th>Заданий</th>"
        f"<th>В общем ядре</th><th>Покрытие</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _failure_badges(summary: dict[str, Any]) -> str:
    """Visible taxonomy of a run: infra first, model outcomes after."""
    kinds = summary.get("failures") or {}
    if not kinds:
        return ""
    badges = " ".join(
        f"<span class='badge {'infra' if core.is_infra_failure(kind) else 'model'}' "
        f"title='{_cell(core.failure_label(kind))}'>{_cell(core.failure_label(kind))} ×{count}</span>"
        for kind, count in sorted(kinds.items())
    )
    return f"<div class='sub failures'>{badges}</div>"


def _incomplete_table(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> str:
    """Runs that are not ranked: ineligible by infrastructure or incomplete."""
    rows = []
    for run, summary in pairs:
        contract = run.get("suite") or {}
        expected = len(contract.get("per_task") or []) or len(run.get("scores") or [])
        ran = summary.get("ran", 0) + summary.get("ran_showcase", 0)
        state = (
            "отказ в удостоверении" if summary.get("integrity_failures")
            else "инфраструктура" if not summary.get("eligible")
            else "неполный прогон"
        )
        if summary.get("skipped_scored"):
            state += f"; не проверялось заданий: {summary['skipped_scored']} (unsupported)"
        rows.append(
            f"<tr><td class='model'>{_cell(_model_name(run))}</td>"
            f"<td class='num'>{ran}/{expected}</td>"
            f"<td>{_cell(state)}</td>"
            f"<td>{_failure_badges(summary) or '<span class=\'sub\'>—</span>'}</td>"
            f"<td class='sub'>{_cell(_caveat(summary))}</td></tr>"
        )
    return (
        "<h3>Неполные и неранжируемые</h3>"
        "<p class='hint'>Эти числа показаны, но места в ранге не занимают: инфраструктурный "
        "провал ничего не говорит о модели, а неполный прогон — про свой урезанный знаменатель.</p>"
        "<table class='grid incomplete-table'><thead><tr><th>Модель</th><th>Заданий</th>"
        "<th>Состояние</th><th>Исходы</th><th>Причина</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _model_cards(summaries: list[tuple[dict[str, Any], dict[str, Any]]], *, per_run: bool = False, ranks: dict[str, int | None] | None = None) -> str:
    """One card per model: score, showcase, spend, and where the points went."""
    cards = []
    for fallback, (run, summary) in enumerate(summaries, 1):
        place = (ranks or {}).get(_run_id(run), fallback if ranks is None else None)
        config = run.get("config") or {}
        spend = _spend_totals(run)
        rows = run["scores"]
        # A published artefact keeps one page per run next to the comparison; link
        # it when it is there, so a reader does not have to guess the path.
        own = Path(str(run.get("dir") or "")).name
        link = (
            f"<div class='sub'><a class='link' href='runs/{_cell(own)}/index.html'>"
            "отчёт по этой модели →</a></div>"
        ) if own and not per_run else ""
        failed = sum(1 for r in rows if r.get("earned", 0) < r.get("possible", 0) and not r.get("skipped"))
        zero = sum(1 for r in rows if not r.get("earned") and not r.get("skipped") and r.get("possible"))
        failed_cats = sorted({r["category"] for r in rows
                              if r.get("earned", 0) < r.get("possible", 0) and not r.get("skipped")})
        ci = summary.get("headline_ci95")
        ci_line = ""
        if ci:
            note = stats.note(summary.get("headline_n") or 0)
            ci_line = (
                f"<div class='ci-scale'>{_ci_band(ci, summary.get('headline_n') or 0)}</div>"
                f"<div class='sub ci-caption'>ДИ 95%: {ci[0]:.0f}–{ci[1]:.0f}% · "
                f"n={summary.get('headline_n') or 0}" + (f" · {note}" if note else "") + "</div>"
            )
        cards.append(
            f"<div class='card' data-model='{_cell(_model_name(run))}'>"
            f"<div class='rank'>{'—' if place is None else place}</div>"
            f"<h3>{_cell(_model_name(run))}</h3>"
            + (
                ""
                if summary.get("eligible") and not summary.get("incomplete")
                else (
                    "<div class='sub state'>"
                    + (
                        "удостоверение отозвано" if summary.get("integrity_failures")
                        else "инфраструктура" if not summary.get("eligible")
                        else "неполный прогон — вне ранга"
                    )
                    + "</div>"
                )
            )
            + f"<div class='big {_tone(summary['headline_earned'], summary['headline_possible'])}'>"
            f"{summary['headline_earned']:.0f}<span class='of'>/{summary['headline_possible']:.0f}</span>"
            f"<span class='unit'>баллов качества</span></div>"
            f"<div class='meter'><i style='width:{100 * summary['headline_earned'] / (summary['headline_possible'] or 1):.0f}%'></i></div>"
            f"{ci_line}"
            f"{f'<div class=\'sub verdict\'>{_cell(_caveat(summary))}</div>' if _caveat(summary) else ''}"
            f"{_failure_badges(summary)}"
            f"<div class='sub'>симуляции K: <b>{summary['showcase_earned']:.0f}/{summary['showcase_possible']:.0f}</b> · L — оценка человеком · "
            f"заданий с потерей: <b>{failed}</b> (нулевых {zero})</div>"
            f"<div class='sub'>категории с потерей: {_cell(', '.join(failed_cats) or 'нет')}</div>"
            f"<div class='sub spend'>токенов: <b>{_kv(spend['total'])}</b>{_spend_note(spend)}"
            f" · reasoning {_num(None if spend['reasoning_share'] is None else 100 * spend['reasoning_share'])}%"
            f" · <b>{_minutes(spend['seconds'])}</b>"
            f" · балл/мин <b>{_num(_per_min(summary, spend), 2)}</b></div>"
            f"<div class='sub'>{_cell(config.get('quant') or '')} · {_cell(_endpoint(config.get('base_url', '')))}</div>"
            f"{link}"
            "</div>"
        )
    return "".join(cards)


def _comparison_block(runs: list[dict[str, Any]]) -> str:
    """Paired difference, category vector and weight sensitivity.

    ``runs`` is in the *caller's* order and the first entry is the baseline —
    the same rule and the same baseline as ``comparison.md``. Significance is
    read from the interval of the paired difference on shared task ids,
    stratified by family; whether the separate bands overlap is shown beside it
    so the conservative reading stays visible. Only a complete, infrastructure-
    clean run inside the baseline's contract takes part in the verdict; any
    other row is marked «не ранжируется» and does not count.
    """
    if len(runs) < 2:
        return ""
    summaries = [(run, summarise(run["scores"], run.get("suite"))) for run in runs]
    base_run, base = summaries[0]
    base_contract = _contract_key(base_run)
    rows = []
    sensitivities = []
    for run, summary in summaries[1:]:
        comparison = compare_summaries(
            base, summary, same_contract=_contract_key(run) == base_contract
        )
        overall = comparison["overall"]
        band = overall["delta_ci95"]
        shown = f"{band[0]:+.2f} … {band[1]:+.2f} п.п." if band else "n/a"
        delta = overall["delta"]
        delta_text = "n/a" if delta is None else f"{delta:+.2f}"
        if overall["ranked"]:
            verdict = "значимо" if overall["significant"] else "незначимо"
            tone = "good" if overall["significant"] else "sub"
        else:
            verdict = f"не ранжируется: {_cell(overall['unranked_reason'])}"
            tone = "sub"
        rows.append(
            f"<tr><td class='model'>{_cell(_model_name(run))}</td>"
            f"<td class='num'>{delta_text}</td>"
            f"<td class='num'>{_cell(shown)}</td>"
            f"<td class='num'>{_cell(_pct(base.get('headline_earned', 0), base.get('headline_possible', 0)))}</td>"
            f"<td class='num'>{_cell(_pct(summary.get('headline_earned', 0), summary.get('headline_possible', 0)))}</td>"
            f"<td class='num'>{overall['pairs'] or 'n/a'}</td>"
            f"<td class='num'>{'да' if overall['overlap'] else 'нет'}</td>"
            f"<td class='num {tone}'>{verdict}</td></tr>"
        )
        sensitivities.append((run, comparison["weight_sensitivity"]))
    sensitivity_rows = []
    for scheme in WEIGHT_SCHEMES:
        cells = []
        for run, sensitivity in sensitivities:
            entry = sensitivity["schemes"][scheme]
            delta = entry["delta"]
            unknown = entry["unknown"]
            cells.append(
                f"<td class='num'>{'' if delta is None else f'{delta:+.2f}'} "
                f"<span class='sub'>{_cell(entry['winner'] or 'n/a')}"
                + (f" · не измерено {unknown:.0f}% веса" if unknown else "")
                + "</span></td>"
            )
        sensitivity_rows.append(f"<tr><th>{_cell(scheme)}</th>{''.join(cells)}</tr>")
    vector_head = "".join(f"<th>{_cell(key)}</th>" for key in sorted(core.CATEGORY_POINTS))
    vector_rows = []
    for run, summary in summaries:
        cells = "".join(
            f"<td class='num'>{_cell(_pct(summary['per_category'].get(key, {}).get('earned', 0), summary['per_category'].get(key, {}).get('possible', 0)))}</td>"
            for key in sorted(core.CATEGORY_POINTS)
        )
        vector_rows.append(f"<tr><td class='model'>{_cell(_model_name(run))}</td>{cells}</tr>")
    stable = all(sensitivity["stable"] for _, sensitivity in sensitivities)
    return (
        '<h2>Парное сравнение: разность и её интервал</h2>'
        f"<p class='hint'>База — <b>{_cell(_model_name(base_run))}</b>, первый прогон в порядке "
        "вызова; та же база в comparison.md. Одинаковые задания сравниваются парами, "
        "стратифицированно по семействам; Δ — парная разность на общих задачах, а доли сторон от "
        "их собственных наборов стоят отдельными колонками. Решает 95 % интервал разности, а не "
        "перекрытие отдельных полос. Прогон вне контракта базы или неполный в вердикте не участвует.</p>"
        "<table class='grid'><thead><tr><th>Модель</th><th>Δ (парная), п.п.</th><th>95 % ДИ разности</th>"
        "<th>База, %</th><th>Модель, %</th><th>Общих задач</th><th>Полосы перекрываются</th>"
        "<th>Вердикт</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "<h2>Вектор категорий</h2>"
        "<p class='hint'>Один итог — смесь весов, поэтому публикуется и вектор по категориям, %.</p>"
        f"<table class='grid'><thead><tr><th>Модель</th>{vector_head}</tr></thead>"
        f"<tbody>{''.join(vector_rows)}</tbody></table>"
        "<h2>Чувствительность к весам</h2>"
        "<p class='hint'>Дельта под каждой политикой весов: знаменатель — полный вес профиля, "
        "поэтому непомеренная категория не даёт бесплатного прохода, а её доля веса подписана. "
        "Вывод "
        + ("устойчив к смеси.</p>" if stable else "зависит от смеси — это свойство весов, не модели.</p>")
        + "<table class='grid'><thead><tr><th>Политика</th>"
        + "".join(f"<th>{_cell(_model_name(run))}</th>" for run, _ in sensitivities)
        + f"</tr></thead><tbody>{''.join(sensitivity_rows)}</tbody></table>"
    )


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
            f"{summary['headline_earned']:.0f}/{summary['headline_possible']:.0f}"
            f"{f'<span class=\'th-sub\'>{_cell(_caveat(summary))}</span>' if _caveat(summary) else ''}</td>",
        ]
        for key in SCORED:
            row = summary["per_category"].get(key, {})
            if row.get("tasks") and row.get("skipped") == row.get("tasks"):
                cells.append("<td class='num'><span class='th-sub'>не проверялось</span></td>")
                continue
            if key == "E" and row.get("admission") in ("failed", "unverified"):
                # E is the only category whose admission can move the verdict:
                # its percentage alone would read as an ordinary zero.
                label = "не допущен" if row["admission"] == "failed" else "допуск не подтверждён"
                cells.append(
                    f"<td class='num admission {row['admission']}' "
                    f"title='{row.get('earned', 0):.0f}/{row.get('possible', 0):.0f}'>{label}</td>"
                )
                continue
            pct = _pct(row.get("earned", 0), row.get("possible", 0))
            band = _ci_band(row.get("ci95"), row.get("n") or 0)
            cells.append(
                f"<td class='num'>{'—' if pct is None else f'{pct:.0f}%'}"
                f"{f'<div class=\"ci-scale small\">{band}</div>' if band else ''}</td>"
            )
        for key in SHOWCASE:
            row = (summary.get("showcase") or {}).get(key, {})
            if key in core.ART_CATEGORIES:
                label = "оценка человеком" if row.get("tasks") else "—"
                if row.get("tasks") and row.get("skipped") == row.get("tasks"):
                    label = "не проверялось"
                cells.append(f"<td class='show'>{label}</td>")
                continue
            pct = _pct(row.get("earned", 0), row.get("possible", 0))
            band = _ci_band(row.get("ci95"), row.get("n") or 0)
            cells.append(
                f"<td class='num show'>{'—' if pct is None else f'{pct:.0f}%'}"
                f"{f'<div class=\"ci-scale small\">{band}</div>' if band else ''}</td>"
            )
        cells.append(f"<td class='num'>{_kv(spend['total'])}{_spend_note(spend)}</td>")
        cells.append(f"<td class='num'>{_minutes(spend['seconds'])}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class='grid verdict-table'><thead><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _suite_table(summaries: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    """Suite × solutions: one row per task, one column per run.

    The matrix is the point of the report: the same task for everyone, so the
    differences are visible side by side instead of in three separate tables.
    Columns are keyed by the immutable run directory: two runs of one model are
    two columns, not one overwriting the other.
    """
    by_run: dict[str, dict[str, dict[str, Any]]] = {}
    private_by_run = {_run_id(run): _private_ids(run) for run, _summary in summaries}
    order: list[tuple[str, str, str]] = []
    for run, _summary in summaries:
        run_id = _run_id(run)
        by_run.setdefault(run_id, {})
        for score in run["scores"]:
            task_id = score["task_id"]
            if task_id not in by_run[run_id]:
                by_run[run_id][task_id] = score
            if all(task_id != tid for tid, _, _ in order):
                entry = next((e for e in (run.get("tasks") or [])
                              if (e.get("task") or {}).get("id") == task_id), {})
                title = _short(str((entry.get("task") or {}).get("title", "")), task_id)
                order.append((task_id, score.get("category", ""), title))
    def order_key(row: tuple[str, str, str]) -> tuple[int, int, str]:
        task_id, category = row[0], row[1]
        number, _ = _task_sort_key(task_id)
        return (SCORED.index(category) if category in SCORED else 99 + SHOWCASE.index(category), number, task_id)

    order.sort(key=order_key)

    header = "".join(f"<th>{_cell(_column_name(run))}</th>" for run, _ in summaries)
    rows = []
    for task_id, category, title in order:
        cells = []
        for run, _ in summaries:
            score = by_run.get(_run_id(run), {}).get(task_id)
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
        badge = (" <span class='private' title='приватное задание: детали свёрнуты'>private</span>"
                 if any(task_id in ids for ids in private_by_run.values()) else "")
        rows.append(
            f"<tr><td class='task'><b>{_cell(task_id)}</b> {_cell(title)}{badge}"
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


def _private_ids(run: dict[str, Any]) -> set[str]:
    """Ids of the private layer, from the suite digest the run recorded.

    The digest carries ``private`` per task; a run recorded before that flag is
    simply treated as public. Rebuilding the suite is the last resort because
    it re-runs generators only to learn a boolean.
    """
    if "_private_ids" in run:  # the page asks three times per run
        return run["_private_ids"]
    known = (run.get("suite") or {}).get("per_task") or []
    ids = {str(row.get("id")) for row in known if isinstance(row, dict) and row.get("private") is True}
    if known and not ids and (run.get("config") or {}):
        ids = {task.id for task in _live_tasks(run).values() if task.private}
    run["_private_ids"] = ids
    return ids


def _task_rows(run: dict[str, Any]) -> str:
    tasks = {entry.get("task", {}).get("id"): entry for entry in (run.get("tasks") or [])}
    private_ids = _private_ids(run)
    live: dict[str, Any] | None = None
    rows = []
    for score in run["scores"]:
        task_id = score["task_id"]
        private = task_id in private_ids
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
        badge = ""
        if private:
            # The private layer may publish aggregates only: the row stays, the
            # oracle-bearing detail is collapsed, and the marker tells a reader
            # why. The reference is fetched from the rebuilt task when an older
            # snapshot did not store it.
            reference = (((task := entry.get("task") or {}).get("oracle") or {}).get("reference") or "")
            if not reference:
                if live is None:
                    live = _live_tasks(run)
                rebuilt = live.get(task_id)
                if rebuilt is not None:
                    reference = (rebuilt.oracle or {}).get("reference") or ""
            if reference:
                detail += f"<div class='oracle'><b>оракул</b>: {_cell(reference)}</div>"
            detail = (
                "<details class='private-detail'><summary>детали и оракул — раскрыть</summary>"
                f"{detail}</details>"
            )
            badge = " <span class='private' title='приватное задание: детали свёрнуты'>private</span>"
        rows.append(
            f"<tr class='{klass}'><td>{_cell(task_id)}</td><td>{_cell(title)}{badge}</td>"
            f"<td class='num'>{score.get('earned', 0):.1f}/{score.get('possible', 0):.0f}</td>"
            f"<td class='num small'>{_task_spend(spend, 'answer_tokens')}</td>"
            f"<td class='num small'>{_task_spend(spend, 'reasoning_tokens')}</td>"
            f"<td class='num small'>{_task_spend(spend, 'generate_s')}</td>"
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
        rows.append(("Веса SHA-256", str(config.get("weights_sha256") or "unknown")))
        observed = config.get("observed") or {}
        if observed:
            rows.append((
                "Наблюдённая конфигурация",
                "со слов runtime (/props); unknown — не измерено: "
                + ", ".join(
                    f"{key}={observed.get(key) or 'unknown'}" for key in
                    ("model_path", "quant", "n_ctx", "kv_type", "chat_template_sha256",
                     "build_info", "server_gpu")
                ),
            ))
        if config.get("reference"):
            rows.append(("Эталонная конфигурация", str(config["reference"])))
        if config.get("launch"):
            rows.append(("Шаблон запуска (declared)", str(config["launch"])))
        executed = config.get("launch_executed") or {}
        if executed:
            rows.append((
                "Выполненная загрузка (executed)",
                f"{executed.get('method') or 'метод неизвестен'}: "
                + json.dumps(executed.get("request") or {}, ensure_ascii=False),
            ))
        if sizing.get("calibrated"):
            rows.append(("Калибровка long context",
                         f"{sizing['chars_per_token']:.2f} символов/токен ({sizing.get('how', 'измерено')})"))
        if hardware:
            rows.append((
                "GPU сэмплера (хост харнесса)",
                str(hardware.get("gpu") or hardware.get("name") or "unknown"),
            ))
        if suite.get("sha256"):
            rows.append(("Сьюта SHA-256", str(suite["sha256"])))
        environment = run.get("environment") or {}
        if environment:
            rows.append(("Образ песочницы", str(environment.get("image") or "—")))
            rows.append(("Digest образа", str(environment.get("image_digest") or "—")))
            verified = environment.get("status") == "verified"
            container = environment.get("executor_status") == "container_run"
            # A verified image is not a run inside it: without this, the row said
            # "Среда: подтверждено" right above "host_run".
            status = (
                "подтверждено"
                if verified and container
                else "образ подтверждён, но не применялся"
                if verified
                else "не проверено"
            )
            if environment.get("reason"):
                status += f": {environment['reason']}"
            rows.append(("Образ песочницы: статус", status))
            runtime = str(environment.get("runtime") or "—")
            if environment.get("runtime_verified") is False:
                # The image digest says nothing about the requested runtime: a
                # name that is not registered on this host runs as plain runc,
                # so the row must not read as confirmed.
                runtime += " (не подтверждён)"
            rows.append(("Runtime песочницы", runtime))
            executor = str(environment.get("executor_status") or "—")
            if environment.get("executor_reason"):
                executor += f": {environment['executor_reason']}"
            rows.append(("Исполнитель", executor))
            rows.append(("SOURCE_DATE_EPOCH", str(environment.get("source_date_epoch") or "—")))
            packages = environment.get("sbom") or []
            if packages:
                rows.append(("SBOM", f"{len(packages)} пакетов"))
        private_ids = _private_ids(run)
        if private_ids:
            rows.append((
                "Приватный слой",
                f"{len(private_ids)} заданий: публикуются id, категория и хеши; "
                "их проверки и оракул в таблицах свёрнуты и видны только после раскрытия",
            ))
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
                               "scoredCi": [], "spend": [], "cost": []}
    for run in runs:
        summary = summarise(run["scores"], run.get("suite"))
        name = _model_name(run)
        payload["models"].append(name)
        payload["scored"].append([_pct(summary["per_category"].get(k, {}).get("earned", 0),
                                       summary["per_category"].get(k, {}).get("possible", 0)) for k in SCORED])
        payload["scoredCi"].append([
            (summary["per_category"].get(k, {}).get("ci95") and
             (lambda ci, pct: None if pct is None else
              (max(0.0, pct - ci[0]), max(0.0, ci[1] - pct)))(summary["per_category"][k]["ci95"], pct))
            for k, pct in zip(SCORED, payload["scored"][-1])
        ])
        showcase = summary.get("showcase") or {}
        payload["showcase"].append([_pct(showcase.get(k, {}).get("earned", 0),
                                         showcase.get(k, {}).get("possible", 0)) for k in core.SHOWCASE_CATEGORIES])
        spend = _spend_totals(run)
        minutes = None if spend["seconds"] is None else spend["seconds"] / 60
        payload["spend"].append({"model": name, "answer": spend["answer"], "reasoning": spend["reasoning"],
                                 "minutes": minutes})
        payload["cost"].append({"model": name, "score": summary["headline_earned"],
                                "minutes": minutes,
                                "reasoning": None if spend["reasoning"] is None else spend["reasoning"] / 1000,
                                "per_min": _per_min(summary, spend)})
    return payload




def _radar_payload(summaries: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    """One radar series per model, measured axes only."""
    series = []
    for run, summary in summaries:
        values = {}
        for key in SCORED:
            row = summary["per_category"].get(key) or {}
            values[key] = _pct(row.get("earned", 0), row.get("possible", 0))
        series.append({"name": _model_name(run), "values": values})
    return viz.radar(series, SCORED, labels={key: key for key in SCORED})


def _heatmap_payload(summaries: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    """Losses per family, aggregated from the scores themselves.

    The family comes from the taxonomy, not from a hand-written list, so a new
    category appears in the heatmap without touching the report.
    """
    families = sorted({
        schema.family_for(score.get("category", "")) or score.get("category", "?")
        for run, _ in summaries for score in run["scores"]
    })
    rows = []
    for run, _ in summaries:
        buckets: dict[str, dict[str, float]] = {
            family: {"earned": 0.0, "possible": 0.0, "tasks": 0, "measured": 0, "failed": 0}
            for family in families
        }
        for score in run["scores"]:
            family = schema.family_for(score.get("category", "")) or score.get("category", "?")
            bucket = buckets[family]
            bucket["tasks"] += 1
            if score.get("skipped"):
                continue
            # A skip leaves numerator and denominator alone, so the percentage
            # does not move — but the cell must still show that fewer tasks
            # were measured, otherwise the skip silently reads as full cover.
            bucket["measured"] += 1
            earned, possible = float(score.get("earned", 0) or 0), float(score.get("possible", 0) or 0)
            bucket["earned"] += earned
            bucket["possible"] += possible
            if earned < possible:
                bucket["failed"] += 1
        rows.append({
            "name": _model_name(run),
            "values": {
                family: {
                    "pct": _pct(bucket["earned"], bucket["possible"]) if bucket["measured"] else None,
                    "tasks": bucket["tasks"], "measured": bucket["measured"],
                    "failed": bucket["failed"],
                }
                for family, bucket in buckets.items()
            },
        })
    return viz.heatmap(rows, families)


def _failure_items(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Failed checks, errors and skip reasons as cluster points."""
    titles = {entry.get("task", {}).get("id"): entry.get("task", {}).get("title") for entry in run.get("tasks") or []}
    items: list[dict[str, Any]] = []
    for score in run["scores"]:
        task_id, category = score.get("task_id", "?"), score.get("category", "?")
        texts = [f"{check.get('name')}: {check.get('detail')}" for check in score.get("checks") or [] if not check.get("passed")]
        if score.get("error"):
            texts.append(f"ошибка: {score['error']}")
        if score.get("skipped"):
            texts.append(f"не проверялось: {score['skipped']}")
        for text in texts:
            items.append({
                "id": task_id, "model": _model_name(run), "category": category,
                "text": text, "count": 1, "title": titles.get(task_id) or "",
            })
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        key = (str(item["id"]), str(item["text"]))
        if key in merged:
            merged[key]["count"] += 1
        else:
            merged[key] = item
    return list(merged.values())


def _live_tasks(run: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the run's tasks from its own config seed, keyed by id."""
    from .config import RunConfig
    from .suite import build_suite

    try:
        return {task.id: task for task in build_suite(RunConfig.from_dict(run.get("config") or {}))}
    except Exception:  # старый/частичный конфиг не должен ронять отчёт
        return {}


def _diffs(run: dict[str, Any], *, limit: int = 4) -> str:
    """Reference/answer diff per visual task, when both frames exist.

    A reference frame is rendered through the same deterministic cache the
    showcase uses; without a browser the panel says «нет данных» instead of
    hiding the task.
    """
    from . import visual

    entries = {entry.get("task", {}).get("id"): entry for entry in run.get("tasks") or []}
    run_dir = Path(run["dir"])
    roots = gallery._allowed_roots(run_dir)
    # Older run snapshots did not store the oracle. The registry still rebuilds
    # the same task from the run's seed, so the reference frame is available
    # instead of the panel collapsing to "нет данных" for every task.
    live: dict[str, Any] | None = None
    blocks: list[str] = []
    for score in run["scores"]:
        if len(blocks) >= limit:
            break
        if score.get("category") not in SHOWCASE or score.get("skipped"):
            continue
        task_id = score.get("task_id")
        entry = entries.get(task_id) or {}
        task = entry.get("task") or {}
        oracle = task.get("oracle") or {}
        reference = oracle.get("reference") or ""
        visual_extra = ((entry.get("transcript") or {}).get("extra") or {}).get("visual") or {}
        candidate = gallery._shot_candidate(str(visual_extra.get("shot") or ""), run_dir)
        # Containment: a hand-edited tasks.json can point anywhere on the
        # machine, and the page embeds every byte it reads as a data URI.
        if candidate is None or not candidate.is_file() or not gallery._within(candidate, roots):
            continue
        shot = candidate
        meta = task.get("meta") or {}
        if not reference:
            if live is None:
                live = _live_tasks(run)
            rebuilt = live.get(task_id)
            if rebuilt is not None:
                reference = (rebuilt.oracle or {}).get("reference") or ""
                meta = meta or rebuilt.meta
                task = task or {"title": rebuilt.title}
        if not reference:
            continue
        steps = visual_extra.get("steps") or meta.get("steps")
        dt = visual_extra.get("dt") or 1 / 60
        html = visual.extract_html(reference)
        result, _ = visual._cached_render(html, steps, dt) if steps else (None, False)
        if result is None or not result.ok:
            blocks.append(
                f"<h3>{_cell(task_id)} · {_cell(_short(str(task.get('title') or ''), str(task_id)))}</h3>"
                + viz.visual_diff(Path("/nonexistent"), shot)
            )
            continue
        ref_path = visual.shot_path(html, steps, dt)
        if not gallery._within(ref_path, roots):
            ref_path = Path("/nonexistent")
        earned, possible = score.get("earned", 0), score.get("possible", 0)
        blocks.append(
            f"<h3>{_cell(task_id)} · {_cell(_short(str(task.get('title') or ''), str(task_id)))} "
            f"<span class='score {_tone(earned, possible)}'>{_cell(f'{earned:.0f}/{possible:.0f}')}</span></h3>"
            + viz.visual_diff(ref_path, shot)
        )
    if not blocks:
        return viz.visual_diff(Path("/nonexistent"), Path("/nonexistent"))
    return "".join(blocks)


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
            f"<span class='sub'> · {_task_spend(spend, 'answer_tokens')} ток. ответ / {_task_spend(spend, 'reasoning_tokens')} reasoning</span></div>"
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
    # The paired comparison keeps the caller's order: its first run is the
    # baseline, the same one comparison.md uses. Ranking re-sorts a *copy*.
    comparison_runs = list(runs)
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
    # Eligibility and completeness gate the ranking *before* any comparison:
    # only a full, infrastructure-clean run inside one contract gets a place.
    groups = _group_contracts(summaries)
    ranks: dict[str, int | None] = {}
    flat: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for _contract, ranked, rest in groups:
        # A place is only meaningful inside one contract, so the numbering
        # restarts there instead of continuing across incomparable suites.
        place = 1
        for pair in ranked:
            ranks[_run_id(pair[0])] = place
            place += 1
            flat.append(pair)
        flat.extend(rest)
    ranked_groups = groups
    all_runs = [run for run, _ in flat]
    summaries = flat
    cards = _model_cards(flat, per_run=per_run, ranks=ranks)
    comparison = "" if per_run else _comparison_block(comparison_runs)
    if per_run:
        ranking = cards
    else:
        sections = []
        for contract, ranked, rest in ranked_groups:
            head = (
                f"<h2>Сравнение по контракту {_contract_label(contract)}</h2>"
                if len(ranked_groups) > 1
                else ""
            )
            block = cards if len(ranked_groups) == 1 else _model_cards(
                ranked, ranks=ranks
            )
            body = block + (_verdict_table(ranked) if ranked else "")
            body += _incomplete_table(rest) if rest else ""
            if not ranked and not rest:
                body = "<p class='hint'>в этом контракте нет прогонов</p>"
            sections.append(head + body)
        ranking = _coverage_block(all_runs) + "".join(sections)

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

    if per_run:
        run = runs[0]
        summary = summaries[0][1]
        statuses = viz.task_statuses(run["scores"])
        viz_tab = '<div class="tab" data-panel="p-viz">Карты прогона</div>'
        viz_panel = (
            '<div class="panel" id="p-viz">'
            '<h2>Вердикт по цифрам</h2>'
            '<p class="hint">Три вывода и три риска собраны из баллов, ДИ и статусов ниже; '
            'ничего не написано рукой.</p>'
            + viz.verdict_strip(
                summary, statuses,
                titles={key: core.category_title(key) for key in core.CATEGORY_POINTS},
            )
            + '<h2>Полнота прогона и статусы</h2>'
            '<p class="hint">«Не проверялось» — не ноль: причины разные, поэтому у каждого статуса '
            'свой цвет, знак и подпись.</p>'
            + viz.completeness(statuses)
            + viz.status_legend(statuses)
            + '<h2>Семантические кластеры провалов</h2>'
            '<p class="hint">Похожие формулировки провалов ложатся рядом: char-триграммы → '
            'хеш-признаки → PCA 2D. Наведите на точку, чтобы прочитать задание.</p>'
            + viz.cluster_map(_failure_items(run))
            + '<h2>Визуальные диффы: эталон / ответ</h2>'
            '<p class="hint">Красный бокс — изменённые пиксели, врезка показывает их крупно; '
            'координаты и число пикселей подписаны.</p>'
            + _diffs(run)
            + '</div>'
        )
    else:
        viz_tab = '<div class="tab" data-panel="p-viz">Компетенции</div>'
        viz_panel = (
            '<div class="panel" id="p-viz">'
            '<h2>Радар компетенций</h2>'
            '<p class="hint">Ось — категория A–J, радиус — процент от возможного. '
            'Провал оси виден формой профиля; пустая ось помечена в подсказке.</p>'
            + _radar_payload(summaries)
            + '<h2>Heatmap ошибок (модель × семейство)</h2>'
            '<p class="hint">Ячейка несёт процент и сырые числа: провалы и «нет задач» '
            'различаются не только цветом.</p>'
            + _heatmap_payload(summaries)
            + '</div>'
        )

    charts = _json_script(_charts_payload([run for run, _ in summaries]))
    compare_rows = "".join(
        f"<label class='pick'><input type='checkbox' class='cmp' value='{_cell(item['id'])}' "
        f"data-model='{_cell(item['model'])}'><span>{_cell(item['model'])}</span> {_cell(item['id'])}</label>"
        for item in sorted(index, key=lambda x: (*_task_sort_key(x["id"]), x["model"]))
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
 .card .state {{ color:#e3b341; font-weight:600 }}
 .failures .badge {{ display:inline-block; border:1px solid #30363d; border-radius:8px; padding:0 6px; margin:2px 4px 0 0; font-size:11px }}
 .failures .badge.infra {{ color:#d2a8ff; border-color:#6e40c9 }}
 .failures .badge.model {{ color:#e3b341; border-color:#9e6a03 }}
 .incomplete-table td, .coverage-table td {{ vertical-align:top }}
 .big {{ font-size:30px; font-weight:700; line-height:1.1 }}
 .big .of {{ color:#8b949e; font-size:18px; font-weight:500 }}
 .big .unit {{ display:block; font-size:11px; color:#8b949e; font-weight:400; letter-spacing:.03em }}
 .meter {{ height:5px; background:#21262d; border-radius:4px; margin:8px 0; overflow:hidden }}
 .meter i {{ display:block; height:100%; background:#7ee787 }}
 .ci-scale {{ position:relative; height:8px; background:#21262d; border-radius:4px; margin:4px 0 5px; overflow:hidden }}
 .ci-scale.small {{ height:4px; margin:3px 0 0 }}
 .ci-band {{ position:absolute; left:var(--lo); width:var(--width); top:0; bottom:0;
   background:#79c0ff44; border-left:1px solid #79c0ff; border-right:1px solid #79c0ff }}
 .ci-caption {{ margin-bottom:4px }}
 .good {{ color:#7ee787 }} .mid {{ color:#e3b341 }} .bad {{ color:#ff7b72 }}
 .sub {{ color:#8b949e; font-size:11.5px }} .spend b {{ color:#c9d1d9 }}
 table.grid {{ border-collapse:collapse; width:100% }}
 table.grid th, table.grid td {{ border:1px solid #21262d; padding:5px 8px; text-align:left; vertical-align:top }}
 table.grid th {{ background:#161b22; font-weight:600; white-space:nowrap; position:sticky; top:46px }}
 .th-sub {{ display:block; color:#8b949e; font-weight:400; font-size:10.5px }}
 td.num {{ text-align:right; white-space:nowrap }}
 td.model {{ font-weight:600; white-space:nowrap }}
 .verdict-table td.verdict {{ font-weight:700 }}
 .verdict-table .admission {{ font-size:11.5px; white-space:nowrap; color:#e3b341 }}
 .verdict-table .admission.failed {{ color:#ff7b72 }}
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
 .private {{ color:#d2a8ff; border:1px solid #8957e5; border-radius:5px; padding:0 5px; font-size:10px; margin-left:4px }}
 details.private-detail summary {{ cursor:pointer; color:#8b949e; font-size:11.5px }}
 details.private-detail[open] summary {{ color:#c9d1d9; margin-bottom:3px }}
 .oracle {{ margin-top:3px; color:#8b949e }}
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
 .viz {{ margin:10px 0 16px; max-width:100% }}
 .viz svg {{ width:100%; height:auto; display:block }}
 .viz-legend {{ display:flex; flex-wrap:wrap; gap:4px 14px; font-size:11.5px; color:#8b949e; margin:4px 0 10px }}
 .viz-key i {{ display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:4px; vertical-align:baseline }}
 .hm rect {{ fill:#21262d }}
 .hm text {{ fill:#c9d1d9; font-variant-numeric:tabular-nums }}
 .hm.full rect {{ fill:#7ee787; fill-opacity:.28 }} .hm.full text {{ fill:#7ee787 }}
 .hm.part rect {{ fill:#e3b341; fill-opacity:.28 }} .hm.part text {{ fill:#e3b341 }}
 .hm.zero rect {{ fill:#ff7b72; fill-opacity:.32 }} .hm.zero text {{ fill:#ff7b72 }}
 .hm.none rect {{ fill:#21262d; stroke:#30363d; stroke-dasharray:3 3 }} .hm.none text {{ fill:#8b949e }}
 .viz-legend-svg rect.full {{ fill:#7ee787; fill-opacity:.4 }} .viz-legend-svg rect.part {{ fill:#e3b341; fill-opacity:.4 }}
 .viz-legend-svg rect.zero {{ fill:#ff7b72; fill-opacity:.4 }} .viz-legend-svg rect.none {{ fill:#21262d; stroke:#30363d; stroke-dasharray:3 3 }}
 .verdict-strip {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:10px 14px }}
</style></head>
<body>
<header>
 <div id="filehint" class="warn" hidden>Отчёт открыт как <code>file://</code>: живая витрина тянет ES-модули,
 которые браузер с такого origin не грузит (кадры работают). Запустите <code>make serve DIR=runs</code>
 и откройте <code>http://127.0.0.1:8765/{_cell(page_name)}</code>.</div>
 <h1>{_cell(title)}</h1>
 <p class="hint">{lead}</p>
</header>
<nav class="tabs">
 <div class="tab active" data-panel="p-summary">Решения моделей</div>
 {viz_tab}
 <div class="tab" data-panel="p-suite">Решения моделей по сьюту</div>
 <div class="tab" data-panel="p-charts">Графики</div>
 <div class="tab" data-panel="p-showcase">Витрина решений</div>
 {prov_tab}
</nav>

<div class="panel active" id="p-summary">
 <h2>Кто сколько набрал</h2>
 {f'<div class="cards">{cards}</div>' if per_run else ranking}
 {comparison}
 <h2>Качество по категориям сьюта</h2>
 <p class="hint">A–J — балльные категории (проценты от возможного), K и L — витрина: A инстр., B рассужд., C код,
 D инструм., E контекст, F галлюц., G JSON, H диалог, I файлы, J ревью.</p>
 {_verdict_table(summaries) if per_run else ''}
 <h2>Расход: токены и время (документируется, лимит — окно модели)</h2>
 <p class="hint">Ответ и reasoning считаются отдельно: 100K токенов reasoning без единой строки кода — это результат,
 который видно здесь, а не молчаливый потолок.</p>
 <table class="grid"><thead><tr><th>Модель</th><th>Токены ответа</th><th>Токены reasoning</th>
 <th>Минут генерации</th><th>Балл за минуту</th><th>Вызовов инструментов</th></tr></thead><tbody>
 {''.join(f"<tr><td class='model'>{_cell(_model_name(run))}</td>"
          f"<td class='num'>{_num(_spend_totals(run)['answer'])}{_spend_note(_spend_totals(run))}</td>"
          f"<td class='num'>{_num(_spend_totals(run)['reasoning'])}</td>"
          f"<td class='num'>{_minutes(_spend_totals(run)['seconds'], 1)}</td>"
          f"<td class='num'>{_num(_per_min(summary, _spend_totals(run)), 2)}</td>"
          f"<td class='num'>{_spend_totals(run)['tool_calls']:.0f}</td></tr>" for run, summary in summaries)}
 </tbody></table>
 <h2>Как читать отчёт</h2>
 <ul class="hint">
   <li><b>Баллы качества</b> — только категории A–J, максимум 100. Витрина K и L в итог не входит.</li>
   <li><b>«Не проверялось»</b> — задание не удалось проверить на этой машине (например нет браузера); это не ноль.</li>
   <li><b>Полоса неопределённости</b> — 95 % бутстрэп-ДИ по задачам. Значимость разницы решает
       парный тест по одинаковым заданиям и его собственный интервал: перекрытие отдельных полос её
       не отменяет. Порог n=30 — ориентир, а не допуск: при меньшей выборке метрика помечена
       «ориентировочно», а повторы и сиды не измеряются.</li>
   <li><b>Стоп-причина</b> — почему генерация закончилась: <code>stop</code> — сама, <code>length</code> — упёрлась в окно,
       <code>tool_calls</code> — ждёт результат инструмента.</li>
 </ul>
 <h2>Провалы по моделям</h2>
 {''.join(f"<h3>{_cell(_model_name(run))}</h3><table class='grid tasks'><thead><tr><th>ID</th><th>Название</th>"
          f"<th>Балл</th><th>Ответ</th><th>Думанье</th><th>Сек</th><th>Стоп</th><th>Проваленная проверка</th></tr></thead>"
          f"<tbody>{_task_rows(run)}</tbody></table>" for run, _ in summaries)}
</div>

{viz_panel}

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
 <div class="gallery" id="gallery">{_gallery(sorted(index, key=lambda x: (x['category'], *_task_sort_key(x['id']), x['model'])))}</div>
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
    error_y:{{type:'data', symmetric:false,
      array: DATA.scoredCi[i].map(c => c ? c[1] : 0),
      arrayminus: DATA.scoredCi[i].map(c => c ? c[0] : 0), color:'#8b949e', thickness:1}},
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
       marker:{{color:'#56d4dd'}}, text: DATA.cost.map(c => c.per_min === null ? 'н/п' : c.per_min.toFixed(2)), textposition:'outside',
       yaxis:'y', hovertemplate: '%{{x}}: %{{y:.2f}} балла за минуту<extra></extra>' }},
    {{ type:'scatter', mode:'markers+text', name:'минут генерации', x: DATA.cost.map(c => c.model),
       y: DATA.cost.map(c => c.minutes), text: DATA.cost.map(c => c.minutes === null ? 'н/п' : c.minutes.toFixed(1) + ' мин'),
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
  const cats = new Set([...document.querySelectorAll('.flt[value="K"],.flt[value="V"],.flt[value="L"]:checked')].map(i => i.value));
  document.querySelectorAll('#gallery figure').forEach(f => {{
    f.style.display = cats.has(f.dataset.cat) ? '' : 'none';
  }});
}}
document.querySelectorAll('.flt[value="K"],.flt[value="V"],.flt[value="L"]').forEach(i => i.onchange = applyFilters);
document.getElementById('size').oninput = e => {{
  galleryEl.style.gridTemplateColumns = `repeat(auto-fill, minmax(${{e.target.value}}px, 1fr))`;
}};

function visibleCards() {{
  return [...document.querySelectorAll('#gallery figure')].filter(f => f.style.display !== 'none');
}}

function openViewer(items, index) {{
  current = items;
  // Пользовательские строки (метка модели, id, путь) идут в DOM API, а не в
  // innerHTML: кавычка или `<` в импортированных metadata больше не может
  // переписать разметку viewer (R16).
  framesEl.replaceChildren();
  items.forEach((it, i) => {{
    const pane = document.createElement('div');
    pane.className = 'pane' + (items.length === 1 ? ' solo' : '');
    const cap = document.createElement('div');
    cap.className = 'cap';
    const who = document.createElement('span');
    who.textContent = `${{it.model}} · ${{it.id}}`;
    const score = document.createElement('span');
    score.textContent = it.score;
    cap.append(who, score);
    pane.append(cap);
    if (it.page) {{
      const stage = document.createElement('div');
      stage.className = 'stage';
      const frame = document.createElement('iframe');
      frame.dataset.i = String(i);
      frame.src = it.page;
      frame.setAttribute('sandbox', 'allow-scripts');
      stage.append(frame);
      pane.append(stage);
    }} else {{
      const missing = document.createElement('div');
      missing.className = 'missing';
      missing.textContent = 'модель не дошла до HTML';
      pane.append(missing);
    }}
    framesEl.append(pane);
  }});
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
    from .browser import PLOTLY

    source = PLOTLY / "plotly.min.js"
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
