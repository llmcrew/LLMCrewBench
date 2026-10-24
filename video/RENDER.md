# Как отрендерить ролик на другом устройстве

Архив самодостаточен, но сохраняет предыдущую версию видеопроекта и её
23-секундный набор клипов. Это recovery-копия, а не поставка по новому
20-секундному контракту. Вес - 1,5 ГБ (17 частей по 90 МБ).
В git архива нет: клипы не публикуются. Описание восстановления и хеш -
`video/archive/RESTORE.md` и `video/archive/SHA256`; сами части лежат на машине записи.

Текущие страницы для записи берутся из
`reports/comparison-deepseek-1021/_showcase/`; исторический архив не обновляет
эту витрину и не означает повторный прогон модели.

## Перенос текущего ручного монтажа Resolve

Актуальная нарезка с коротким K и конфигом vLLM — отдельный таймлайн
`Flash vs 1017 — short K + vLLM — 60fps`, длительность **25:10,3**.
Готовое видео — `out/DailyBench-vllm-shortK-final.mp4`.
Импортируй `out/resolve/Flash-vs-1017-vllm-shortK-60fps.drt` либо открывай
проект `out/resolve/flashVS1017-vllm-shortK-final.drp`.

Вместе с ними переноси `out/edit/cache/`, `out/showcase-master.mp4` и
`out/edit/short-k/`: DRT/DRP не встраивают видео и WAV. На другой машине
перепривяжи эти папки через Relink Media. Старым таймлайнам полного проекта
дополнительно нужны их исходные записи. Покадровая нарезка сохранена в
`out/edit/short-k/current-cut.json`; резерв до изменений —
`out/resolve/flashVS1017-user-cut-before-vllm-k.drp`.

Команды Remotion ниже воспроизводят базовую длинную композицию, а не ручные
разрезы текущего таймлайна. Исходные 20-секундные захваты K не перезаписаны.

## Быстро

Базовая композиция Remotion и её экспорт — 60 fps. Доступные работы раскрываются на
20 секунд; неуспешным результатам полноэкранный ход не выделяется. Мастер-клипы сохраняются в исходных 120 fps:
Remotion выбирает кадры по времени, пересъёмка или перекодирование не нужны.

Если `public/clips/` перенесён прямым копированием из полной копии проекта с
текущим 20-секундным контрактом, склеивать и распаковывать архив не нужно:

```bash
cd video
npm ci
npm run preflight
npm run render -- --browser-executable "<chrome-path>"
```

Для текущего озвученного монтажа переносится также `public/audio/`:
там исходные WAV, музыкальные MP3 и производные стемы. Исторический recovery-
архив их не содержит и сам по себе не восстанавливает новый аудиомонтаж.
Для частичного переноса старых клипов этот путь не подходит.

```bash
# склейка частей архива - см. video/archive/RESTORE.md
cat <repo>/video/archive/dailybench-video.zip.part-* > dailybench-video.zip
unzip dailybench-video.zip
cd dailybench-video
npm ci
npm run preflight                       # манифест, маркеры и decode до рендера
npm run render -- --browser-executable "<chrome-path>"
# → out/DailyBench-voiceover-final.mp4, 1920×1080, 60 fps, диктор + музыка
```

## Полный мастер самостоятельной витрины

Чтобы выгрузить все сцены витрины как отдельный ролик (не preview и не
устаревший `DailyBench.mp4`), запускай из `video/`:

```bash
npm ci
npm run render:showcase -- --browser-executable "<chrome-path>"
```

Готовый файл: `video/out/showcase-master.mp4`. Параметры:
`--output <absolute-path>`, `--concurrency 8`, `--force`.
Chrome задаётся через `--browser-executable` либо `REMOTION_BROWSER_EXECUTABLE`;
без них используется стандартный браузер Remotion.

Экспорт загружает первичные `showcaseTimeline`, `showcase-timing.ts` и
`videoFramingStyle`. Все 49 сцен и 83 полноэкранных хода сохраняют длительности,
кадрирование, честные статусы и порядок. Мастер без аудио:
H.264/yuv420p, 1920×1080, 60 fps, 1913 секунд / 114 780 кадров.

Движущиеся шторы, фоновые половины, начало и конец fade подписей рендерятся
настоящим Remotion. Только неподвижный полноэкранный участок идёт через
FFmpeg: исходное живое видео плюс прозрачный PNG реальных React-подписей.
Кадрирование берётся из той же функции; скорость равна текущему `openRate`,
включая продолжение до конца возврата шторы (`20 / 20.75`).
Это не замена живого клипа статичным кадром. Для проверки отдельного случая:
`--only K-1,L-2 --output <absolute-path-to-preview.mp4>`.

Части лежат в `out/showcase-master.parts/<fingerprint>/`.
Fingerprint включает исходники, монтаж и размер/время изменения клипов.
Готовые части сверяются через ffprobe; незавершённые части перерендериваются.
Склейка — stream copy без повторного кодирования.

Экспортёр сам собирает Remotion bundle; если общий bundle уже собран другим
рендер-процессом, передай его путь или URL через `--serve-url <bundlePath/URL>`.
Для итоговой склейки нужны `ffmpeg` и `ffprobe` в `PATH`.

`public/clips/` содержит клипы и маркеры статусов текущего проекта. Их состав
определяется `showcase.json`; `.missing` - статусная плашка внутри
соответствующего MP4, а не отсутствие видеофайла. `npm run preflight` сверяет
манифесты, маркеры и media до render; `npm run render` вызывает его сам.

## Требования

- **Node 20+** и npm.
- **Сеть один раз**: `npm install` + Remotion скачает свой chromium
  (`node_modules/.remotion/chrome-headless-shell`) при первом запуске.
  Шрифты (Inter, JetBrains Mono) `@remotion/google-fonts` тянет при бандлинге -
  тоже нужна сеть при первом рендере, дальше кэш.
- **Python 3.10+**, `ffmpeg` и `ffprobe` в `PATH`: аудиомонтаж и нативные
  участки рендера используют их напрямую.
- **ОЗУ ≥ 8 ГБ** свободных; `--concurrency 4..8` подбирать по доступной памяти.
- Место: минимум 10 ГБ под исходники, аудиостемы, bundle, возобновляемые части и результат.

GPU не обязателен; исходные мастер-клипы не переписываются.
Из `runs/` (прогоны бенчмарка) внутри архива ничего не нужно: цифры уже посчитаны
в `src/data/runs.json`.

## Платформа

Проверено на macOS arm64. На Linux/Windows:

- сцены и клипы платформо-независимы (h264 mp4 + React);
- `bin/` (комплектный ffmpeg от Remotion под macOS-arm) **для рендера не нужен** -
  он только для `tools/record_*.py` (пересъёмка клипов). Chrome/ffmpeg ищутся
  сначала по macOS-пути/video/bin, затем в `PATH`: на другой ОС поставьте
  chromium и ffmpeg в `PATH`;
- шрифты рендерятся как есть, подстановок системных нет.

## Пересъёмка клипов

Клипы снимаются с живых страниц витрины в
`reports/comparison-deepseek-1021/_showcase/`. Управляемые часы headless-Chrome
двигают `requestAnimationFrame`, таймеры или `window.__step`; запись не
привязана к wall-clock и не пропускает кадры из-за длительности рендера.

Контракт каждого мастер-клипа: 1920×1080, H.264/yuv420p, 120 fps,
20 с (2400 кадров), CRF 16. Композиция выводится в 60 fps: две половины
960×1080 раскрываются шторой до полного кадра только для доступных работ.
Полный показ длится 20 секунд; движение шторы — по 0,75 с. На доступную работу
отводится ход 23 с, на сцену без полноэкранных работ — общий кадр 4 с.
Суммарная длительность рассчитывается по этому расписанию, а не как 49×46 с.
Служебные надписи про fps, способ записи и «живую сцену» в кадре не выводятся.

Увеличение отдалённых K-результатов задаёт `src/data/framing.json`.
Это только отображение: мастер-MP4, HTML, камера и графический код не
перезаписываются. После смены клипов запусти из корня репозитория
`python video/tools/build_framing.py`; он вычисляет общую область результата
по шести моментам записи. Для L остаётся центральное кадрирование.

CRF задаёт режим качества, не целевой битрейт: он меняется со сложностью сцены.
`verify_clips.py` сохраняет измеренный битрейт, размер и SHA-256 каждого файла,
а также проверяет параметры потока, полный decode и PTS каждого кадра.
Технически восстановленный клип хранится отдельно и помечается как ремонт;
исходная неудача, причина, ожидаемый результат, страница и балл остаются
видимыми и неизменными. Отсутствующую HTML-страницу не заменяют выдуманной сценой.

Пока staging не прошёл полный decode/PTS/quality report, текущие клипы и
манифесты не заменяются. При `capture_failed` публикацию сохраняют прежней;
после проверки переключают целиком набор клипов вместе с соответствующими
манифестами и запускают `preflight` перед render.

Из корня репозитория укажи путь к обычному Chrome в `--chrome`. Запись пишет в
отдельный staging; рядом с `clips/` создаются его `clips.json` и
`showcase.json`. `--force` применяется только к staging:

```bash
python video/tools/record_all.py 4 20 0.6 --out video/out/recording-20s/clips --report-root reports/comparison-deepseek-1021 --chrome "<chrome-path>" --fps 120 --crf 16 --preset medium --startup-timeout 180 --force
python video/tools/verify_clips.py video/out/recording-20s/clips --manifest video/out/recording-20s/clips.json --output video/out/recording-20s/recording-quality.json
```

`verify_clips.py` измеряет битрейт и проверяет параметры потока, полный decode
и PTS всех кадров. При ненулевом коде записи или ошибке проверки старые клипы
и манифесты сохраняются. Только после успеха владелец публикации переключает
комплект целиком; `preflight` запускается после переключения и до render.

У каждого MP4 есть одноимённый `.capture.json`: хеш входной HTML-страницы
публикации, фактическое число снятых PNG, CRF, режим часов и их начальная/
конечная отметки. Клип `ok` содержит ровно 2400 захваченных кадров; плашка
помечена своей причиной и не выдаётся за снятую работу. `verify_clips.py`
проверяет все декодированные кадры и PTS с шагом 1/120 с.
Низкий битрейт статичной работы при CRF 16 нормален; неподвижный авторский
результат не «оживляется» добавленной графикой.
Плашки генерируются явным чёрным фоном (`color=c=black`), не `nullsrc`:
нулевые YUV-компоненты дают зелёный кадр, а не чёрный. Причину и ожидаемый
результат поверх плашки показывает Remotion.

## Порядок

```bash
unzip dailybench-video.zip && cd dailybench-video
npm ci
npm run preflight                       # манифест, маркеры и decode до рендера
npm run render -- --browser-executable "<chrome-path>"
```

`npm run render` собирает аудио, визуальные сцены, музыкальную витрину и главы
из `src/data/edit.json`. `--narrative-only` рендерит только поясняющие сцены;
`--assemble-only` склеивает уже готовые части. Для общего подготовленного
bundle оба экспортёра принимают `--serve-url <absolute-bundle-path/URL>`.
После полного рендера `npm run timeline:resolve` устанавливает новый таймлайн
в открытый проект `flashVS1017`, сохраняя исходный `Timeline 1`.

Второй формат - тот же набор в `dailybench-video.tar.zst`; в git он не входит,
лежит рядом с zip в `video/` на машине записи (zstd есть в brew и в
linux-репозиториях; если нет - `zstd -d < архив.zst | tar -x`).

## Проверка

```bash
npx remotion still scene-board out/check-board.png --frame=1800      # таблица результатов
npx remotion still scene-showcase out/check-showcase.png --frame=300
```

Ожидаемо: кадр витрины показывает раскрытую полосу одного стенда, вторая полоса
сжата по краю; снизу подпись «решение <стенд> · сцена <id> · N/2 · живая
страница», кадр раскрытия - из начала клипа, а не из середины.

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
src/            144K  сцены, таймлайн (SCENES), данные - весь ролик отсюда
public/clips/   текущие MP4 и статусные маркеры; состав зависит от страниц и результатов записи
docs/           56K   текст диктора (voiceover.md), это руководство и фреймворк
                      записи (recording/: рекордер, драйвер, гайд, грабли)
tools/          24K   утилиты (для рендера не обязательны)
  build_runs_data.py    переносит цифры из runs/ в src/data/runs.json (нужен `runs/`;
                        в архиве данные уже готовы)
  format_labels.py      человеческие имена моделей
  preflight.py          - манифест/маркеры/ffprobe до Studio/render
  record_clip.py, record_all.py - пересъёмка клипов
  probe_pages.py        - пересчёт плашек: страница падает до кадра или живёт
package.json, package-lock.json, tsconfig.json
```

Сборка архивов (из каталога `video/`; в архив кладётся корень
`dailybench-video/` без `node_modules`, `out`, `dist`, рабочих кадров
`public/clips/_frames` и самого `archive/` - иначе архив дублирует себя):

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

`bin/` - комплектный ffmpeg от Remotion под macOS-arm64 (16 МБ). Для **рендера
не нужен**; пригодится только для пересъёмки клипов (`tools/record_*.py`) на macOS.
На Linux поставь любой `ffmpeg`/`ffprobe` в `PATH`.

## Что не нужно

- `runs/`, `reports/` - нужны только для пересъёмки клипов
  (`tools/record_all.py`) с исходными страницами решений.
