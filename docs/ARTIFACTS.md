# Артефакты в git: что и зачем

Репозиторий хранит не только код. Здесь — что именно опубликовано, какие
файлы исключены из публикации и как проверить целостность.

## Опубликованные артефакты

| Путь | Что это | Почему в git |
|---|---|---|
| `runs/deepseek-v4.1-flash-unc-1017/` | Прогон deepseek на стенде 1017: `report.md`, `report.html`, `visual.html`, `scores.json`, `speed.json`, `config.json`, `sizing.json` (калибровка long-context), кадры и страницы витрины (`_showcase`), `_vendor` | Публикация тестирования: отчёт открывается без прогона; кадры и живые страницы — доказательство |
| `reports/comparison-deepseek-1017/` | Сравнение старого стенда (95/100) с новым (100/100) | «До/после» одной модели на двух ingress'ах |

## Что исключено и почему

| Путь | Причина |
|---|---|
| `runs/deepseek-v4.1-flash-unc-1017/tasks.json` | 98 МБ: полные транскрипты; восстанавливается из `transcripts/` и переигрывается |
| `runs/deepseek-v4.1-flash-unc-1017/transcripts/` | 98 МБ сырых JSON агента |
| `runs/deepseek-v4.1-flash-unc-1017/hardware.json` | Машинные параметры стенда (GPU неизвестен) |
| `runs/_chat`, `runs/_logs`, `runs/_showcase`, `runs/_vendor`, старые прогоны | Внутренние и воспроизводимые |
| `reports/comparison-deepseek-1017/_vendor/` | 7 МБ копий three.js/plotly: есть в `vendor/` и `reports/comparison/_vendor/` |
| `video/public/clips/`, `video/bin/` | ~1,5 ГБ клипов (120 fps) и macOS-бандл: распакованы из архива |
| `video/archive/*.part-*` | Архив клипов частями: до 20.10 лежал в git, теперь только `RESTORE.md` и `SHA256` |
| `video/dailybench-video.zip`, `video/dailybench-video.tar.zst`, `video/dist/` | Тот же архив одним файлом (и его zstd-близнец): GitHub отклоняет >100 МБ |

## Склейка архива

Части приходят из `video/archive/` на машине записи; в клоне репозитория
рядом лежат только `RESTORE.md` и `SHA256`. Архив собирается заново, когда
меняется состав записи (например, после перезаписи клипов в 120 fps), — старые
части при этом заменяются целиком.

```bash
cd /tmp && rm -rf restore && mkdir restore && cd restore
cat <архив>/dailybench-video.zip.part-* > dailybench-video.zip
shasum -a 256 dailybench-video.zip   # сверить с SHA256
unzip -t dailybench-video.zip
```

Дальше — по `video/RENDER.md`.

## Проверка опубликованного прогона

```bash
cd runs/deepseek-v4.1-flash-unc-1017
jq '[.[].earned] | add' scores.json        # 181
jq 'length' scores.json                    # 68
which ffprobe && ffprobe -v error -show_entries format=duration speed.json 2>/dev/null
```

Баллы в `report.md`/`report.html` соответствуют `scores.json`; витрина —
`visual.html` (живые сцены требуют `python3 -m http.server` из каталога
`runs/`, как описано в README).
