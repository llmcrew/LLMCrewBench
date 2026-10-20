# Как отрендерить ролик на другом устройстве

Архив самодостаточен: клипы уже сняты, пересъёмка не нужна. Вес — 1,5 ГБ
(17 частей по 90 МБ).
В git его нет: клипы и записи не публикуются. Рядом лежат только описание
(`video/archive/RESTORE.md`) и хеш (`video/archive/SHA256`); сам архив — части
по 90 МБ в `video/archive/` на машине записи, склейка и сверка — в RESTORE.md.

У стенда 1017 живой K и L-1, дальше L — плашки: в прогоне у него одно
L-задание, страницы L-2…L-29 есть только у стенда A.

## Быстро

```bash
# склейка частей архива — см. video/archive/RESTORE.md
cat <repo>/video/archive/dailybench-video.zip.part-* > dailybench-video.zip
unzip dailybench-video.zip
cd dailybench-video
npm ci
npx remotion render DailyBench out/dailybench.mp4 --concurrency=4 --timeout=180000
# → out/dailybench.mp4, ролик 1920x1080, 120 fps, ≈19 мин; сам рендер — часы
```

`public/clips/` (98 записей: 70 живых клипов и 28 плашек) уже внутри — ничего
записывать не нужно.

## Требования

- **Node 20+** и npm.
- **Сеть один раз**: `npm install` + Remotion скачает свой chromium
  (`node_modules/.remotion/chrome-headless-shell`) при первом запуске.
  Шрифты (Inter, JetBrains Mono) `@remotion/google-fonts` тянет при бандлинге —
  тоже нужна сеть при первом рендере, дальше кэш.
- **ОЗУ ≥ 8 ГБ** свободных. Клипы 120 fps тяжёлые: держи `--concurrency=4`
  (на ноутбуке 2..3, на сервере можно `12+`).
- Место: ~1 ГБ на `node_modules`/кэш + ~0,8 ГБ на результат.

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

Клипы снимаются с живых страниц витрины в
`reports/comparison-deepseek-1017/_showcase/`:
страница исполняется в headless-Chrome, кадры двигает её собственный драйвер
(`__dbscene.seek` — окно движения разворачивается на все 23 секунды клипа).
120 fps: один кадр клипа — один шаг сцены, частота у клипа и у ролика одна;
на раскрытии клип переигрывается с первого кадра за 8 с (ускорение ~2,8×),
и запас кадров держит это движение плавным
(`FPS` в `src/video/data.ts` и `record_clip.py` — пара). Окно коротких сцен
(меньше ~190 шагов) клип замедляет: живой показ играет его за 4–10 с, клип —
за 23; на сценах, где окно пустое (K-16) или в один шаг (K-19), виден
стоп-кадр — так и задумано, сцена живёт один-два шага.

```bash
python3 tools/record_all.py 6 23 0.6          # дополнить пропущенные
python3 tools/record_all.py 4 23 0.6 --force  # перезаписать все 98
python3 tools/probe_pages.py 6                # пересчитать плашки: жива ли страница
```

## Порядок

```bash
unzip dailybench-video.zip && cd dailybench-video
npm ci
npx remotion render DailyBench out/dailybench.mp4 --concurrency=4 --timeout=180000
```

Рендер 120 fps тяжёлый: витрина гоняет два клипа на сцену плюс раскрытие. Полоса в сетке режется вокруг своего показа (пока она раскрыта, её клип не декодируется); с этой правкой витрина идёт ~4,5 кадра/с и весь ролик — около 8 часов на M-серии. Без неё было бы ~сутки (1,2 кадра/с).

Второй формат — тот же набор в `dailybench-video.tar.zst`; в git он не входит,
лежит рядом с zip в `video/` на машине записи (zstd есть в brew и в
linux-репозиториях; если нет — `zstd -d < архив.zst | tar -x`).

## Проверка

```bash
npx remotion still scene-board out/check-board.png --frame=1800      # таблица результатов
npx remotion still scene-showcase out/check-showcase.png --frame=300
```

Ожидаемо: кадр витрины показывает раскрытую полосу одного стенда, вторая полоса
сжата по краю; снизу подпись «решение <стенд> · сцена <id> · N/2 · живая
страница», кадр раскрытия — из начала клипа, а не из середины.

## Если что-то пошло не так

| Симптом | Причина | Что делать |
|---|---|---|
| `Target closed`, рендер встал | не хватает памяти, chromium падает | `--concurrency=2..4`, закрыть другие браузеры |
| `delayRender() ... not cleared after 28000ms` на кадре витрины | кадр тянет несколько видео 120 fps сразу, дефолтный таймаут 28 с не успевает | `--timeout=180000` (и `--concurrency<=4`) |
| Шрифты «квадратиками» | не было сети при первом бандлинге | запустить рендер один раз с сетью |
| «не найден chrome-headless-shell» | не докачался chromium | `npx remotion browser ensure` |
| Очень медленно | витрина декодирует клипы 120 fps, накрытые полосы уже не рендерятся | норма; помогите `--concurrency` |

## Что внутри архива

```
src/            144K  сцены, таймлайн (SCENES), данные — весь ролик отсюда
public/clips/   1,5G  98 записей: 70 живых + 28 плашек «модель не справилась»
                      (у стенда 1017 L-2…L-29 — плашки: страниц нет)
docs/           56K   текст диктора (voiceover.md), это руководство и фреймворк
                      записи (recording/: рекордер, драйвер, гайд, грабли)
tools/          24K   утилиты (для рендера не обязательны)
  build_runs_data.py    переносит цифры из runs/ в src/data/runs.json (нужен `runs/`;
                        в архиве данные уже готовы)
  format_labels.py      человеческие имена моделей
  record_clip.py, record_all.py — пересъёмка клипов
  probe_pages.py        — пересчёт плашек: страница падает до кадра или живёт
package.json, package-lock.json, tsconfig.json
```

Сборка архивов (из каталога `video/`; в архив кладётся корень
`dailybench-video/` без `node_modules`, `out`, `dist`, рабочих кадров
`public/clips/_frames` и самого `archive/` — иначе архив дублирует себя):

```bash
rsync -a --delete \
  --exclude node_modules --exclude out --exclude dist --exclude '.DS_Store' \
  --exclude public/clips/_frames --exclude archive \
  --exclude dailybench-video.zip --exclude dailybench-video.tar.zst \
  ./ /tmp/dailybench-video/
cp -R ../docs/recording /tmp/dailybench-video/docs/recording   # из корня репо
cd /tmp && zip -r -q -X /tmp/dailybench-video.zip dailybench-video
shasum -a 256 /tmp/dailybench-video.zip | cut -d' ' -f1 > <repo>/video/archive/SHA256
# ^ этот же хеш - в archive/RESTORE.md (строка «ожидается»)
rm -f <repo>/video/archive/dailybench-video.zip.part-*
split -b 90m /tmp/dailybench-video.zip <repo>/video/archive/dailybench-video.zip.part-
# части в git не нужны: трекаются только RESTORE.md и SHA256
```

## Про `bin/`

`bin/` — комплектный ffmpeg от Remotion под macOS-arm64 (16 МБ). Для **рендера
не нужен**; пригодится только для пересъёмки клипов (`tools/record_*.py`) на macOS.
На Linux поставь любой `ffmpeg` в `PATH`.

## Что не нужно

- `runs/`, `reports/` — нужны только для пересъёмки клипов
  (`tools/record_all.py`) с исходными страницами решений.
