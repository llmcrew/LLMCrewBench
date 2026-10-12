# Как отрендерить ролик на другом устройстве

Архив самодостаточен: клипы уже сняты, пересъёмка не нужна.

## Быстро

```bash
unzip dailybench-video.zip        # или: tar --zstd -xf dailybench-video.tar.zst
cd dailybench-video
npm ci
npx remotion render DailyBench out/dailybench.mp4 --concurrency=6 --timeout=120000
# → out/dailybench.mp4, 1920x1080, 60 fps, 34:32, без звука
```

`public/clips/` (196 файлов, ~1,2 ГБ) уже внутри — ничего записывать не нужно.

## Требования

- **Node 20+** и npm.
- **Сеть один раз**: `npm install` + Remotion скачает свой chromium
  (`node_modules/.remotion/chrome-headless-shell`) при первом запуске.
  Шрифты (Inter, JetBrains Mono) `@remotion/google-fonts` тянет при бандлинге —
  тоже нужна сеть при первом рендере, дальше кэш.
- **ОЗУ ≥ 8 ГБ** свободных. Рендер идёт `--concurrency=8`; на ноутбуке
  ставь `--concurrency=3..4`, на сервере можно `12+`.
- Место: ~1,2 ГБ на `node_modules`/кэш + 220 МБ на результат.

Никаких ffmpeg/Homebrew/GPU в системе не требуется — Remotion приносит всё своё.
Из `runs/` (прогоны бенчмарка) внутри архива ничего не нужно: цифры уже посчитаны
в `src/data/runs.json`.

## Платформа

Проверено на macOS arm64. На Linux/Windows:

- сцены и клипы платформо-независимы (h264 mp4 + React);
- `bin/` (комплектный ffmpeg от Remotion под macOS-arm) **для рендера не нужен** —
  он только для `tools/record_*.py` (пересъёмка клипов). На другой ОС
  пересъёмка потребует любого `ffmpeg` в `PATH`;
- шрифты рендерятся как есть, подстановок системных нет.

## Пересъёмка клипов

Клипы снимаются с живых страниц витрины в `reports/comparison/_showcase/`:
страница исполняется в headless-Chrome, кадры двигает её собственный драйвер
(`__dbscene.seek` — окно движения разворачивается на все 23 секунды клипа).
60 fps: один кадр клипа — один шаг сцены, частота у клипа и у ролика одна
(`FPS` в `src/video/data.ts` и `record_clip.py` — пара).

```bash
python3 tools/record_all.py 6 23 0.6          # дополнить пропущенные
python3 tools/record_all.py 6 23 0.6 --force  # перезаписать все 196
python3 tools/probe_pages.py 6                # пересчитать плашки: жива ли страница
```

## Порядок

```bash
unzip dailybench-video.zip && cd dailybench-video
npm ci
npx remotion render DailyBench out/dailybench.mp4 --concurrency=6 --timeout=120000
```

Второй формат — тот же набор в `dailybench-video.tar.zst` (zstd есть в brew
и в linux-репозиториях; если нет — `zstd -d < архив.zst | tar -x`).

## Проверка

```bash
npx remotion still scene-board out/check-board.png --frame=1800      # таблица результатов
npx remotion still scene-showcase out/check-showcase.png --frame=3200
```

Ожидаемо: кадр витрины показывает раскрытую полосу одной модели, остальные
сжаты по краям; подпись снизу «решение <модель> · сцена <id> · N/4 · живая страница»

## Если что-то пошло не так

| Симптом | Причина | Что делать |
|---|---|---|
| `Target closed`, рендер встал | не хватает памяти, chromium падает | `--concurrency=2..4`, закрыть другие браузеры |
| `delayRender() ... not cleared after 28000ms` на кадре витрины | кадр тянет четыре видео сразу, дефолтный таймаут 28 с не успевает | `--timeout=120000` (и `--concurrency<=4`) |
| Шрифты «квадратиками» | не было сети при первом бандлинге | запустить рендер один раз с сетью |
| «не найден chrome-headless-shell» | не докачался chromium | `npx remotion browser ensure` |
| Очень медленно | витрина декодирует четыре видео на сцену | норма; накрытые полосы уже не рендерятся |

## Что внутри архива

```
src/            144K  сцены, таймлайн (SCENES), данные — весь ролик отсюда
public/clips/   1,2G  196 записей: 168 живых + 28 плашек «модель не справилась»
                      (18 сцен без страницы у GLM 5.3 — сервер упал в прогоне; 10 страниц, упавших до кадра)
docs/           8K    текст диктора (voiceover.md) и это руководство
tools/          24K   утилиты (для рендера не обязательны)
  build_runs_data.py    переносит цифры из runs/ в src/data/runs.json (нужен `runs/`;
                        в архиве данные уже готовы)
  format_labels.py      человеческие имена моделей
  record_clip.py, record_all.py — пересъёмка клипов
  probe_pages.py        — пересчёт плашек: страница падает до кадра или живёт
package.json, package-lock.json, tsconfig.json
```

Сборка архивов (из каталога `video/`; в архив кладётся корень
`dailybench-video/` без `node_modules`, `out`, `dist` и рабочих кадров
`public/clips/_frames`):

```bash
rsync -a --delete \
  --exclude node_modules --exclude out --exclude dist --exclude '.DS_Store' \
  --exclude public/clips/_frames \
  ./ /tmp/dailybench-video/
zip -r -q -X dist/dailybench-video.zip dailybench-video   # запускать из /tmp
tar --zstd -cf dist/dailybench-video.tar.zst -C /tmp dailybench-video
```

## Про `bin/`

`bin/` — комплектный ffmpeg от Remotion под macOS-arm64 (16 МБ). Для **рендера
не нужен**; пригодится только для пересъёмки клипов (`tools/record_*.py`) на macOS.
На Linux поставь любой `ffmpeg` в `PATH`.

## Что не нужно

- `runs/`, `reports/` — нужны только для пересъёмки клипов
  (`tools/record_all.py`) с исходными страницами решений.
