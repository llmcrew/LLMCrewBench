# Официальные адаптеры

Адаптер запускает **чужой pinned набор его же оценщиком** и публикует исходную метрику отдельно. Он не превращается в native-задание и не переписывает upstream score в `/100`. Первый и пока единственный адаптер — BigCodeBench-Hard/Instruct.

Общий контракт:

- идентичность (dataset, revision/sha, split, subset, версия scorer) зашита в код и попадает в `bundle.json`;
- upstream `task_id` сохраняются дословно;
- входы и промпты замораживаются до вызова модели;
- вердикт выдаёт только upstream evaluator; наш код лишь разбирает его файл результата;
- при отсутствии пакета, его pinned версии или docker пакет получает `status: unverified`, а метрика — `null`, не `0`;
- метрика внешняя: `pass@1` как доля (`0.5`), с `percent_of_100: false`.

## BigCodeBench

### Что закреплено

| Поле | Значение |
|---|---|
| датасет | `bigcode/bigcodebench-hard` |
| revision | `v0.1.4` (HF split marker) + sha `298d2cc7b96612e15e47313c3603ee124cee0c1f` |
| split | `instruct`, поле `instruct_prompt` |
| subset | `hard`, 148 задач |
| scorer | `bigcodebench==0.2.5`, `bigcodebench.evaluate` |
| исполнение | `--execution local`, `--parallel 1` |
| выборка | `--selective_evaluate <upstream ids через запятую>` |
| лицензия | Apache-2.0 на код и данные; запись и проверка: [`licenses/bigcodebench.json`](licenses/bigcodebench.json), [`licenses/README.md`](licenses/README.md) |

`v0.1.4` — имя split, поэтому оно закреплено вместе с git-revision: сам marker можно перепривязать. Пакет `0.2.5` — первый релиз, одновременно закрепляющий split `v0.1.4` и поддерживающий `--execution`/`--selective_evaluate`. Названный в задаче `bigcodebench==0.1.8.post1` этих ключей не знает: его CLI принимает только `--split/--subset`, а датасет он грузит из `v0.1.0_hf`, поэтому с protocol-командой он несовместим.

Batch size 1 — часть протокола: README предупреждает, что greedy-выборка зависит от размера batch. Один `task_id` — ровно одна строка `solution`; повтор `task_id` отклоняется. `--save_pass_rate False` убирает интерактивный вопрос upstream CLI, `pass@1` считается здесь из статусов `eval_results.json`.

### Команды

```bash
# 1. Заморозить всю hard-выборку (или первые N задач) без вызова модели.
.venv/bin/python -m dailybench adapter bigcodebench prepare --out runs/bcb-hard

# Для пилота из двух задач:
.venv/bin/python -m dailybench adapter bigcodebench prepare --out runs/bcb-hard --limit 2

# 2. Положить ответы модели: JSONL с task_id и solution, по строке на задачу.
cp runs/bcb-hard/predictions.example.jsonl runs/bcb-hard/model_predictions.jsonl

# 3. Прогнать pinned upstream evaluator и получить внешний pass@1.
.venv/bin/python -m dailybench adapter bigcodebench score \
  --out runs/bcb-hard --samples runs/bcb-hard/model_predictions.jsonl
```

`prepare` создаёт `bundle.json`, `tasks.json` (промпт, entry point, каноническое решение, хеши) и `predictions.example.jsonl`. `score` пишет `samples.jsonl`, лог `evaluate.log`, upstream `<samples>_eval_results.json` и обновляет bundle.

Для тестов и обёрток `--evaluate-cmd` заменяет argv-префикс `bigcodebench.evaluate`; при явном префиксе readiness-проверка не выполняется.

Коды выхода: `0` — `verified` или успешный `prepare`, `1` — `unverified`/ошибка входов. Невалидный или неполный прогон не выглядит завершённым.

### Результат

```json
{
  "status": "verified",
  "metric": {
    "name": "pass@1",
    "value": 0.5,
    "numerator": 1,
    "denominator": 2,
    "scale": "ratio",
    "percent_of_100": false,
    "source": "eval_results.json:status==pass"
  },
  "verdicts": [
    {"task_id": "BigCodeBench/289", "status": "pass", "details": {}},
    {"task_id": "BigCodeBench/500", "status": "fail", "details": {}}
  ]
}
```

`status: unverified` означает, что измерения нет: нет пакета нужной версии, docker, upstream-результата или задача покрыта неполно. `metric.value` в этом случае `null`, `reason` объясняет причину. `timeout` — upstream-вердикт задачи, он входит в знаменатель как неуспех, но не приравнивается к инфраструктурному сбою.

### Ограничения подписи

- Это hard subset, а не полный BigCodeBench из 1140 задач; `limit` — ещё меньший пилот.
- Локальный executor и собственный batch size 1 — не официальный leaderboard score.
- Внешний `pass@1` не складывается с native `/100` и не меняет веса профилей.
- Задачи импортируют сторонние библиотеки; их лицензии Apache-2.0 scorера не покрывает.

### Воспроизведение

`bundle.json` хранит dataset/sha/split, все `task_id`, версию и точную protocol-команду, хеши промптов и канонических решений. Для повторного разбора достаточно сохранённых `samples.jsonl` и `_eval_results.json`; заново вызывать модель не нужно. Тесты: `tests/test_adapter_bcb.py` (fixture из двух задач и fake CLI; live-тест пропускается без пакета и docker).
