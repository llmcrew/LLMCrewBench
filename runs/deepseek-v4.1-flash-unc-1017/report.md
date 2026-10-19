# DailyBench-16 — deepseek-v4.1-flash-uncensored-1017 (bf16/fp8)

| Поле | Значение |
| --- | --- |
| Модель | `deepseek-v4.1-flash-uncensored-1017` |
| Квант | bf16/fp8 |
| Runtime | vLLM |
| Endpoint | `https://giga-osa-deepseek-v41-flsh-unc1017.apps.evodagpup.sberdevices.ru/v1` |
| Веса SHA-256 | `не указан` |
| KV-cache | по умолчанию |
| Окно контекста | 262144 |
| Калибровка long context | 3.04 символов/токен (endpoint /tokenize) |
| Offload | none |
| Каркас артефактов | агент pi, категории C, K, L: читает и правит рабочую копию, оценка по файлу на диске (`--no-agent` переключает на чат) |
| Seed | 20260915 |
| Бюджеты | max_new_tokens=4096, max_total_tokens=16384, max_tool_calls=12 |
| GPU | unknown (? MiB, driver ?) |
| Suite SHA-256 | `3ea49b96cb8deff818707e7ae95f9825a4f0761e12c2f29550452cce48527f76` |
| Reference config | `deepseek-v4.1-flash-unc-1017` |

Загрузка конфигурации (для воспроизведения):

```
vllm serve deepseek-v4.1-flash-uncensored-1017 --host 127.0.0.1 --port 8000 \
  --max-model-len 262144 --gpu-memory-utilization 0.95 \
  --kv-cache-dtype по умолчанию --enforce-eager
```

**Итог: 100/100**

| Категория | Баллы | Допуск | Ошибок прогона |
| --- | ---: | --- | ---: |
| A. Instruction following | 10 / 10 | да | 0 |
| B. Reasoning | 15 / 15 | да | 0 |
| C. Coding | 20 / 20 | да | 0 |
| D. Tool calling | 14 / 14 | да | 0 |
| E. Long context | 10 / 10 | да | 0 |
| F. Hallucination resistance | 8 / 8 | да | 0 |
| G. Structured output | 8 / 8 | да | 0 |
| H. Multi-turn dialogue | 6 / 6 | да | 0 |
| I. Retrieval over files | 4 / 4 | да | 0 |
| J. Code review | 5 / 5 | да | 0 |

## Задания

| ID | Балл | Стоп-причина | Ошибка прогона | Проваленные чеки |
| --- | ---: | --- | --- | --- |
| IF-1 | 2/2 | stop (313 ток., из них reasoning 295) | — | — |
| IF-2 | 2/2 | stop (169 ток., из них reasoning 154) | — | — |
| IF-3 | 2/2 | stop (114 ток., из них reasoning 98) | — | — |
| IF-4 | 2/2 | stop (115 ток., из них reasoning 101) | — | — |
| IF-5 | 2/2 | stop (96 ток., из них reasoning 62) | — | — |
| R-1 | 1/1 | stop (713 ток., из них reasoning 710) | — | — |
| R-2 | 1/1 | stop (120 ток., из них reasoning 117) | — | — |
| R-3 | 1/1 | stop (256 ток., из них reasoning 223) | — | — |
| R-4 | 1/1 | stop (91 ток., из них reasoning 86) | — | — |
| R-5 | 1/1 | stop (129 ток., из них reasoning 126) | — | — |
| R-6 | 1/1 | stop (105 ток., из них reasoning 101) | — | — |
| R-7 | 1/1 | stop (237 ток., из них reasoning 232) | — | — |
| R-8 | 1/1 | stop (314 ток., из них reasoning 309) | — | — |
| R-9 | 1/1 | stop (115 ток., из них reasoning 112) | — | — |
| R-10 | 1/1 | stop (333 ток., из них reasoning 330) | — | — |
| R-11 | 1/1 | stop (216 ток., из них reasoning 212) | — | — |
| R-12 | 1/1 | stop (212 ток., из них reasoning 207) | — | — |
| R-13 | 1/1 | stop (97 ток., из них reasoning 92) | — | — |
| R-14 | 1/1 | stop (1731 ток., из них reasoning 1728) | — | — |
| R-15 | 1/1 | stop (481 ток., из них reasoning 475) | — | — |
| C-1 | 8/8 | stop (8624 ток., из них reasoning 4329) | — | — |
| C-2 | 8/8 | stop (9136 ток., из них reasoning 6471) | — | — |
| C-3 | 4/4 | stop (835 ток., из них reasoning 329) | — | — |
| T-1 | 2/2 | stop (124 ток., из них reasoning 34) | — | — |
| T-2 | 2/2 | stop (487 ток., из них reasoning 223) | — | — |
| T-3 | 2/2 | stop | — | — |
| T-4 | 2/2 | stop | — | — |
| T-5 | 2/2 | stop | — | — |
| T-6 | 2/2 | stop (189 ток., из них reasoning 46) | — | — |
| T-7 | 2/2 | stop (74 ток., из них reasoning 22) | — | — |
| LC-1 | 2/2 | stop (121 ток., из них reasoning 112) | — | — |
| LC-2 | 2/2 | stop (96 ток., из них reasoning 91) | — | — |
| LC-3 | 2/2 | stop (88 ток., из них reasoning 82) | — | — |
| LC-4 | 2/2 | stop (292 ток., из них reasoning 287) | — | — |
| LC-5 | 2/2 | stop (96 ток., из них reasoning 92) | — | — |
| F | 8/8 | stop (356 ток., из них reasoning 279) | — | — |
| G-1 | 2/2 | stop (73 ток., из них reasoning 40) | — | — |
| G-2 | 2/2 | stop (628 ток., из них reasoning 453) | — | — |
| G-3 | 2/2 | stop (150 ток., из них reasoning 80) | — | — |
| G-4 | 2/2 | stop (91 ток., из них reasoning 69) | — | — |
| H-1 | 2/2 | stop (39 ток., из них reasoning 32) | — | — |
| H-2 | 2/2 | stop (62 ток., из них reasoning 59) | — | — |
| H-3 | 2/2 | stop | — | — |
| I-1 | 2/2 | stop (337 ток., из них reasoning 289) | — | — |
| I-2 | 2/2 | stop (108 ток., из них reasoning 92) | — | — |
| J-1 | 3/3 | stop (541 ток., из них reasoning 403) | — | — |
| J-2 | 2/2 | stop (522 ток., из них reasoning 350) | — | — |
| K-1 | 4/4 | stop (7035 ток., из них reasoning 3500) | — | — |
| K-2 | 4/4 | stop (7912 ток., из них reasoning 3161) | — | — |
| K-3 | 4/4 | stop (11320 ток., из них reasoning 3732) | — | — |
| K-4 | 4/4 | stop (12552 ток., из них reasoning 5127) | — | — |
| K-5 | 4/4 | stop (13297 ток., из них reasoning 5129) | — | — |
| K-6 | 4/4 | stop (7830 ток., из них reasoning 2166) | — | — |
| K-7 | 4/4 | stop (10084 ток., из них reasoning 6203) | — | — |
| K-8 | 4/4 | stop (11751 ток., из них reasoning 4588) | — | — |
| K-9 | 4/4 | stop (8286 ток., из них reasoning 2820) | — | — |
| K-10 | 4/4 | stop (24814 ток., из них reasoning 13366) | — | — |
| K-11 | 4/4 | stop (11747 ток., из них reasoning 6030) | — | — |
| K-12 | 4/4 | stop (8539 ток., из них reasoning 3415) | — | — |
| K-13 | 4/4 | stop (13632 ток., из них reasoning 5971) | — | — |
| K-14 | 4/4 | stop (7030 ток., из них reasoning 2400) | — | — |
| K-15 | 4/4 | stop (13006 ток., из них reasoning 6292) | — | — |
| K-16 | 4/4 | stop (4410 ток., из них reasoning 1774) | — | — |
| K-17 | 4/4 | stop (7475 ток., из них reasoning 2073) | — | — |
| K-18 | 4/4 | stop (5361 ток., из них reasoning 2081) | — | — |
| K-19 | 4/4 | stop (49215 ток., из них reasoning 23267) | — | — |
| K-20 | 4/4 | stop (8106 ток., из них reasoning 3464) | — | — |
| L-1 | отрисовано | stop (59882 ток., из них reasoning 34855) | — | — |

## Витрина: картинки вне общего счёта

Симуляции K — 80 / 80 баллов вне итоговой суммы. L — свободная арт-работа: сохраняются страница и кадр, цвет, тональность и художественное качество оценивает человек.

### K. Visual simulations (Three.js)

Итог блока: 80 / 80

| ID | Баллы | Проваленные проверки |
| --- | ---: | --- |
| K-1 | 4/4 | — |
| K-2 | 4/4 | — |
| K-3 | 4/4 | — |
| K-4 | 4/4 | — |
| K-5 | 4/4 | — |
| K-6 | 4/4 | — |
| K-7 | 4/4 | — |
| K-8 | 4/4 | — |
| K-9 | 4/4 | — |
| K-10 | 4/4 | — |
| K-11 | 4/4 | — |
| K-12 | 4/4 | — |
| K-13 | 4/4 | — |
| K-14 | 4/4 | — |
| K-15 | 4/4 | — |
| K-16 | 4/4 | — |
| K-17 | 4/4 | — |
| K-18 | 4/4 | — |
| K-19 | 4/4 | — |
| K-20 | 4/4 | — |

### L. Free-form art

Без художественных баллов. Автоматически проверяется только загрузка HTML.

Снимки кадров: [`visual.html`](visual.html).

## Скорость и память (без баллов качества)

| Профиль | TTFT, с | Prompt tok/s | Generation tok/s | Пик VRAM, MiB | Повторов |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 0.10 | 6671.9 | 560.1 | None | 2 |
| long_4096 | 0.12 | 20718.4 | 575.7 | None | 2 |
