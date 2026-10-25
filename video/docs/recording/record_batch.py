#!/usr/bin/env python3
"""record_batch: запись набора сцен по каталогу страниц.

Каталог - дерево ``<model>/pages/<scene>.html``. По каждому листу пишется
``<model>__<scene>.mp4`` и, если страница не ожила, маркер ``.missing`` +
тёмная плашка. Пропуск уже записанного: файл есть и маркера нет.

    python3 record_batch.py PAGES_DIR OUT_DIR [workers] [secs] [open] [warmup]
        [--force] [--repairs] [--manifest clips.json]

Драйвер браузеров: параллельные воркеры, каждый - свой профиль Chrome,
таймаут на клип, повтор на транзиентный сбой запуска Chrome. ``--repairs``
берёт починки из соседнего ``_repairs`` и не трогает опубликованные страницы.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import manifest  # noqa: E402
import timing  # noqa: E402
from record_scene import Options, record_source  # noqa: E402


def job(pair: tuple[str, str], pages: Path, out_dir: Path, options: Options,
        force: bool) -> tuple[str, str, str]:
    model, scene = pair
    out = out_dir / f"{model}__{scene}.mp4"
    marker = out_dir / f"{model}__{scene}.missing"
    if out.is_file() and not force:
        return model, scene, "missing" if marker.is_file() else "cached"
    if force and out.is_file():
        out.unlink()
    marker.unlink(missing_ok=True)
    out.with_suffix(".capture.json").unlink(missing_ok=True)
    last = "FAIL"
    for attempt in (1, 2):
        try:
            result = record_source(f"{model}/{scene}", pages, out_dir, options)
        except Exception as exc:  # noqa: BLE001 - Chrome срывается транзиентно
            last = f"FAIL {type(exc).__name__}: {exc}"
            time.sleep(1.0)
            continue
        if result.status == "ok":
            return model, scene, "ok"
        # Страница падает до кадра: честнее плашка, чем чернота.
        return model, scene, "missing"
    return model, scene, last


def discover(pages: Path) -> list[tuple[str, str]]:
    pairs = []
    for model_dir in sorted(p for p in pages.iterdir() if p.is_dir()):
        for page in sorted((model_dir / "pages").glob("*.html")):
            pairs.append((model_dir.name, page.stem))
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pages", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("workers", type=int, nargs="?", default=4)
    parser.add_argument("secs", type=float, nargs="?",
                        default=timing.SHOWCASE_TURN_SECONDS)
    parser.add_argument("open", type=float, nargs="?", default=timing.CLIP_SECONDS)
    parser.add_argument("warmup", type=float, nargs="?", default=0.6)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--repairs", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)

    pages = args.pages.resolve()
    out_dir = args.out_dir.resolve()
    try:
        options = Options(
            secs=args.secs, open=args.open, warmup=args.warmup,
            source_mode="repairs" if args.repairs else "showcase", manifest=args.manifest,
        )
        options.clock
    except ValueError as exc:
        parser.error(str(exc))
    if args.manifest:
        problems = manifest.check(manifest.load(args.manifest))
        if problems:
            parser.error("; ".join(problems[:5]))
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover(pages)
    print(f"[batch] сцен: {len(pairs)}, воркеров: {args.workers}", flush=True)
    stats: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(job, pair, pages, out_dir, options, args.force)
                   for pair in pairs]
        for done, future in enumerate(futures, 1):
            model, scene, tag = future.result()
            stats[tag.split()[0]] = stats.get(tag.split()[0], 0) + 1
            if done % 10 == 0 or tag not in ("ok", "cached", "missing"):
                print(f"[batch] {done}/{len(pairs)} {stats}"
                      + ("" if tag in ("ok", "cached", "missing")
                         else f" | {model}/{scene}: {tag}"), flush=True)
    try:
        (out_dir / "_frames").rmdir()
    except OSError:
        pass
    print(f"[batch] готово: {stats}")
    return 0 if not any(key.startswith("FAIL") for key in stats) else 1


if __name__ == "__main__":
    sys.exit(main())
