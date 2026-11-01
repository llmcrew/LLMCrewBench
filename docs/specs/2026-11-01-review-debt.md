# Долг после ревью 2026-11-01

Ветка `spgsroot/benchmark-review-stabilize-speedup` закрыла находки из
`2026-11-01-review-findings.md`. Ниже то, что владелец принял как долг.

## Гейты (eng-gates `gate full`)

`gate full` по изменённым файлам (без `py.test`) не зелёный. В репозитории нет
`gates.toml`, ruff/pyright до ветки не применялись; большая часть ошибок
унаследована.

| Проверка | `ccb0863` | после ветки |
|---|---|---|
| ruff lint (ошибок) | 265 | 299 |
| ruff format (файлов к переформатированию) | 56 | 64 |
| pyright (ошибок) | 398 | 429 |

- `guard.config`: изменены `.github/workflows/ci.yml` (матрица Windows/3.10,
  pinned actions) и `pyproject.toml#tool.pytest` (маркер `docker`). Нужен
  человеческий `gate.py approve` после просмотра.
- `guard.suppress`: 13 новых платформенных `skipif`/`pytest.skip`/`pragma`
  (`tests/test_agent.py`, `tests/test_harness.py`, `scripts/sandbox_run.py`)
  без `gate-allow:`.
- `ub.crap`/`ub.dry`/`ub.mutate` не запускались: не установлены инструменты.

## Не проверено

- Полный `pytest tests` после слияния всех волн не запускался (решение
  владельца); каждая задача проверена своими тестами, общие файлы — после
  каждого слияния.
- Docker-сценарии песочницы (egress только к endpoint, DNS во `--internal`
  сети, сборка pinned-образов): демон не был запущен.
- CI на windows-latest и Python 3.10; цели `Makefile` (нет `make` на хосте).

## Принятые мелочи

- `tests/test_agent.py`: `.cmd`-лаунчер делает `chcp 65001` на общей консоли.
- `agent._JOB_ATTACH_ERRORS` не читается и не чистится.
- `generation._write_durable` пишет в текстовом режиме: CRLF на Windows ломает
  переиспользование hardlink (только диск).
- `--evaluate-cmd` BigCodeBench разбирается POSIX-правилами (пути Windows —
  в кавычках, см. README).
- Общий `GOCACHE` доверяет пользователю хоста; изоляция — задача песочницы.
- `TestNoHangWhenConsumerStops` (C-2) использует `NumGoroutine`.
- `browser._await_ready` опрашивает без паузы (ограничено бюджетом готовности).
