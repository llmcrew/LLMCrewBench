# Видео-архив: части и склейка

`dailybench-video.zip` (828 МБ) разбит на части по 90 МБ: GitHub отклоняет
файлы больше 100 МБ (`GH001`). Архив самодостаточен — клипы уже сняты,
пересъёмка не нужна.

```bash
cd /tmp && rm -rf restore && mkdir restore && cd restore
cat <repo>/video/archive/dailybench-video.zip.part-* > dailybench-video.zip
shasum -a 256 dailybench-video.zip
# ожидается 9a673cc57f414ae5e61c93f010024a24579b1028a5f0a26da183f49e1836c5f0
unzip -t dailybench-video.zip
```

Распаковка и рендер — по `video/RENDER.md` (рядом с архивом, после распаковки
окажется в корне `dailybench-video/`).

Состав: `src/` (ролик), `docs/` (текст диктора, RENDER.md), `tools/`
(пересъёмка), `public/clips/` — 196 клипов, из них 169 живых и 27 плашек.
`node_modules`, `out`, `dist` не входят: ставятся и собираются заново.

Если архив понадобится обновить: `tools/record_all.py`, затем

```bash
cd video && rm -f archive/* && zip -r -q -X /tmp/dailybench-video.zip dailybench-video
split -b 90m /tmp/dailybench-video.zip archive/dailybench-video.zip.part-
shasum -a 256 /tmp/dailybench-video.zip   # записать в этот файл
```
