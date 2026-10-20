# Видео-архив: части и склейка

`dailybench-video.zip` (1,5 ГБ, клипы 120 fps) разбит на части по 90 МБ и лежит на машине
записи, а не в git: клипы и записи не публикуются. Части получит тот, кому
передан `video/archive/`. Архив самодостаточен — клипы уже сняты, пересъёмка
не нужна.

```bash
cd /tmp && rm -rf restore && mkdir restore && cd restore
cat <repo>/video/archive/dailybench-video.zip.part-* > dailybench-video.zip
shasum -a 256 dailybench-video.zip
# сверка с <repo>/video/archive/SHA256 (та же строка ниже)
```

Ожидаемый хеш (он же в `SHA256` рядом):

```
8cd9225014c0803e1c919e110298248affd45f1938ee73fbdbad6f93683fc963
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
