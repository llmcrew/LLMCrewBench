# Телеметрия прогона

Журнал событий на задачу (`telemetry/<task>.jsonl`) и индекс
(`telemetry/index.json`) собираются харнессом, а не моделью. Назначение —
разделять стоимость, задержку и качество и не выдавать неизвестное за ноль.

## События

`attempt_start`, `request_sent` (бюджет, причина, размер prompt: точный или
оценка), `first_token`, `decode_chunk` (интервал, токены), `usage`
(prompt/completion/reasoning/cached), `tool_call_start|end|ok`,
`evaluator_start|end`, `browser_stage`, `task_end`. Поля: монотонный `seq`,
monotonic-секунды, имя задачи/категории.

## Агрегат

TTFT (только первый запрос), decode tok/s, answer/reasoning/cached, prompt
всего/последний, turns, tools, wall, queue/overhead, unbounded/budgeted,
занятость контекста. Правила:

- любое неизвестное слагаемое → агрегат `None`, не 0; `usage_known=False`;
- reasoning не вычитается дважды: `completion` уже включает reasoning;
  оценка по тексту помечается `reasoning_estimated`;
- `queue_s`/`overhead_s` — `None` при неизвестном;
- индекс: медианы TTFT/decode, доля неизвестного расхода, cache-share.

## Отчёт

Секция «Телеметрия»: счётчики бюджета, медианы, доли «н/п» по категориям,
корреляции score × расход/время/контекст (Spearman, n≥3), стоимость балла
только при полном измерении (`complete=false` — с причиной). Отсутствие
журнала — явная строка, не пустая таблица.

## Приватность

Значения `api_key` не пишутся; wire-подпись содержит только имена и бюджеты.
Телеметрия хешируется в commit-generation и входит в grading identity.
