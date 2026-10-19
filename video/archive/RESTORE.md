# Видео-архив: части и склейка

`dailybench-video.zip` (828 МБ) разбит на части по 90 МБ: GitHub отклоняет
файлы больше 100 МБ (`GH001`). Архив самодостаточен — клипы уже сняты,
пересъёмка не нужна.

```bash
cd /tmp && rm -rf restore && mkdir restore && cd restore
cat <repo>/video/archive/dailybench-video.zip.part-* > dailybench-video.zip
shasum -a 256 dailybench-video.zip
# сверка с <repo>/video/archive/SHA256 (та же строка ниже)
```

Ожидаемый хеш (он же в `SHA256` рядом):

```
9063229788ac7d78aea85f9f8dc7531e10c4e4f80e28bf150f777091f5471c40
```

```bash
unzip -t dailybench-video.zip
```

Распаковка и рендер — по `video/RENDER.md` (рядом с архивом, после распаковки
окажется в корне `dailybench-video/`).

Состав: `src/` (ролик), `docs/` (текст диктора, RENDER.md, фреймворк записи
`recording/`), `tools/` (пересъёмка), `public/clips/` — 196 клипов, из них 169
живых и 27 плашек. `node_modules`, `out`, `dist`, сам `archive/` не входят.

Если архив понадобится обновить — из каталога `video/`:

```bash
rsync -a --delete --exclude node_modules --exclude out --exclude dist \
  --exclude '.DS_Store' --exclude public/clips/_frames --exclude archive \
  --exclude dailybench-video.zip --exclude dailybench-video.tar.zst \
  ./ /tmp/dailybench-video/
cp -R ../docs/recording /tmp/dailybench-video/docs/recording
cd /tmp && rm -f dailybench-video.zip && zip -r -q -X dailybench-video.zip dailybench-video
shasum -a 256 dailybench-video.zip | cut -d' ' -f1 > <repo>/video/archive/SHA256
rm -f <repo>/video/archive/dailybench-video.zip.part-*
split -b 90m /tmp/dailybench-video.zip <repo>/video/archive/dailybench-video.zip.part-
```

Хеш в `SHA256` и в этом файле обновлять одной правкой.
