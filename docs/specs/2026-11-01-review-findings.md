# Ревью 2026-11-01: стабилизация, ускорение, логические противоречия

Спецификация исправлений. Каждая находка — требование; план
`docs/plans/2026-11-01-review-fixes.md` ссылается на ID.

## Решения владельца (обязательны)

| Тема | Решение |
|---|---|
| L3 pilot | Код под README: pilot читается по весам модулей (не /100), входит в `BASELINE_INCOMPARABLE_PROFILES`, отчёт пишет «не ранжируется с baseline» |
| L2 предел времени/токенов | Оба пути (между запросами и в сокете) → `resource_exhausted`: 0 баллов, задача в знаменателе, прогон остаётся ранжируемым. Тексты `reference.json`/`config.py` исправить («unverified» убрать) |
| S13 сеть песочницы | Полная изоляция: internal-сеть + egress только к endpoint + негативный тест egress; образы по digest, base-образ по digest, apt snapshot |
| S16 Windows | Полная поддержка: тесты портируемы, Makefile платформенно-нейтрален, Windows-job в CI |
| L16 reference.json | Убрать `--seed -1`; квант/KV явно помечены «измерено при 32K, при 262144 не проверено»; preflight предупреждает |
| Git | Коммиты прямо в текущую ветку `spgsroot/benchmark-review-stabilize-speedup`, без push |

## Логические противоречия

- **L1** `runner.py:48` — `retry_errors`/`allow_suite_drift` входят в `run_identity`; любой из флагов сбрасывает прогресс resume. `--allow-suite-drift` не записывает решение.
- **L2** Дедлайн задачи между запросами → `resource_exhausted` (`runner.py:692-695`), таймаут сокета → `infra_failure` (`providers.py:422`, `runner.py:911-919`), делает прогон нерейтинговым (`report.py:247`). Политика говорит «unverified».
- **L3** `core.py:159-160` pilot = /100 и рейтинговый, README и docstring `core.py:40-44` — наоборот. M08 в TECHNICAL_REVIEW ложно закрыт.
- **L4** `generation_mode` не входит в `contract_key` (`report.py:514`); fixed и native попадают в один лидерборд вопреки `PROTOCOL.md:22-26`.
- **L5** IF-8 (`instruction.py:1150-1157`) нормализует ответ в NFC до сравнения — правило NFC не проверяется.
- **L6** T-12 (`tools.py:2809-2815`) требует фраз «без повтор…», которых промпт (`tools.py:1839`) не просит.
- **L7** T-11 (`tools.py:2708`) регулярка `верс\w*[^\d]{0,12}N` режет «версия слота теперь 2»; id резервации не проверяется.
- **L8** I-5 (`retrieval.py:1438`) сравнивает `sorted(sources)` при требуемом порядке; лишний ключ-инъекция сохраняет 1.5/2.
- **L9** F-3 (`hallucination.py:899-909`) засчитывает ```json-блок как «без Markdown».
- **L10** `visual.py:1045-1054` скрытые пробы изображения по ИЛИ, контракт (`visual-contract.md:57`) — И.
- **L11** `visual.py:884` движение проходит, когда объект не виден.
- **L12** `browser.py:896` + `art.py:107` — base64 `<img>` получает балл L вопреки `art-contract.md:21`.
- **L13** `browser.py:104,1164` — 90-секундный бюджет скриншота недостижим (абсолютный cap 20 с проверяется раньше).
- **L14** `report.py:370-376` чувствительность research-v1 считается от baseline-100.
- **L15** README/README.ru: stretch «30 and 39» (фактически 36/39); список новых задач содержит T-8, не содержит F-3, C-7, N-2, N-9, T-10, T-11, I-5.
- **L16** `reference.json:28-29,36` — квант/KV под 32K при окне 262144; `--seed -1`.
- **L17** `cli.py:78` `--timeout` пишет `request_timeout` (1800), реальный предел `request_wall_clock` = 600 без флага.
- **L18** `report.py:997`, `htmlreport.py:611` — баллы V подписаны «Симуляции K»; README.ru: V «без баллов», код начисляет 6.
- **L19** `sandbox_run.py:10-13` timeout/OOM → `evaluator_failure`, `agent.py:1543` → `resource_exhausted`.
- **L20** `pyproject.toml:27-28` package-data ссылается на удалённый `shot.sh`.

## Стабильность

- **S1** Кодировки Windows: `review.py:1076-1088` (J-5 падает при генерации stretch/research без PYTHONUTF8 — воспроизведено), `adapters/bigcodebench.py:517`, `sandbox_run.py:525`.
- **S2** `tasks/visual_input.py:33` жёсткий `import PIL`, Pillow — только extra `vision`; `dailybench list` падает после `make setup`.
- **S3** `browser.py:621` `frame_stats` требует numpy+Pillow, extra `visual` их не содержит.
- **S4** `cli.py:1248` `_truncated`: любая `transcript.error` → перезапуск при каждом resume; endpoint без usage не сходится никогда.
- **S5** `agent.py:898,1594-1642,984` Ctrl-C на Windows не убивает дерево pi.
- **S6** `sandbox_run.py:542` `os.killpg` на Windows → `AttributeError`.
- **S7** `visual.py:733` кэш хранит `same_frame=False` навсегда; виртуальных часов нет (в `LLMCrewBenchRecoder (tools/record_clip.py):816` есть).
- **S8** `browser.py:1001,1239` исключение `PIXEL_PROBE` → `model_failure` 0/4, упавший Chromium переиспользуется.
- **S9** `generation.py:742-746` ключи manifest с `\` на Windows: hardlink не находится, POSIX-чтение → `IntegrityError`.
- **S10** `browser.py` нет двойного rAF и `document.fonts.ready` перед снимком.
- **S11** `coding_go.py:486-492,1375-1394`, `coding_python.py:1305` — скрытые тесты зависят от sleep/`NumGoroutine`/`all_tasks()`.
- **S12** `stats.py:236-239`, `report.py:394-404` bootstrap CI зависит от порядка строк.
- **S13** `isolation.py:453`, `sandbox_run.py:404`, `Dockerfile.agent:16,30-39` — сеть открыта, теги вместо digest, apt без версий.
- **S14** `sandbox_run.py:533,580-584` весь вывод в памяти; `docker rm` не выполняется, если упал `docker kill`.
- **S15** `setup.py:15` нет `dailybench.adapters` и схем.
- **S16** Тесты на Windows: `test_harness.py:12` `import fcntl`; `test_agent.py` sh-фейки; yaml/numpy/playwright/postgres без skip; CI без Windows; `Makefile:1` `.venv/bin/python`.
- **S17** Тесты «недоступного endpoint» на порту 9.
- **S18** Нет lock-файла, Python 3.10 не тестируется, actions на плавающих тегах.

## Скорость

- **P1** `review.py:1860-1874,2330-2331` J-5 гоняет тесты в подпроцессах при каждой сборке сьюта (~6 с), `fit_suite_sizing` собирает сьют дважды.
- **P2** `browser.py:1310`, `visual.py:924,1036` — 6 навигаций на K-задачу, `PIXEL_PROBE` не используется в оценке.
- **P3** `browser.py:286` copytree three.js на каждый рендер на Windows без symlink.
- **P4** `coding_go.py:760-770` GOCACHE во временном дереве.
- **P5** `sandbox_run.py:605,216-241` SBOM отдельным контейнером на каждую команду.
- **P6** `scripts/sweep.py:29-35,104` сборка сьюта ради подсчёта задач.
- **P7** `scripts/coder_loop.py:79-119` квадратичное перечитывание лога; дети не убиваются деревом.
- **P8** `tests/test_profiles.py`, `test_cli.py` — сьюты пересобираются в каждом тесте; нет session-фикстуры.
- **P9** `measure.py:262-310` — промпт без `/tokenize`, TTFT с очередью/RTT, скорость без usage по оценке длины.
