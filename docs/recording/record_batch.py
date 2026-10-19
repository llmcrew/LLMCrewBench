#!/usr/bin/env python3
"""record_batch: запись набора сцен по каталогу страниц.

Каталог — дерево ``<model>/pages/<scene>.html``. По каждому листу пишется
``<model>__<scene>.mp4`` и, если страница не ожила, маркер ``.missing`` +
тёмная плашка. Пропуск уже записанного: файл есть и маркера нет.

    python3 record_batch.py PAGES_DIR OUT_DIR [workers] [secs] [warmup] [--force]

Драйвер браузеров: параллельные воркеры, каждый — свой профиль Chrome,
таймаут на клип, повтор на транзиентный сбой запуска Chrome.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from record_scene import Options, html_server_for, placeholder, record_visited  # noqa: E402


def job(pair: tuple[str, str], pages: Path, out_dir: Path, options: Options,
        base: str, force: bool) -> tuple[str, str, str]:
    model, scene = pair
    out = out_dir / f"{model}__{scene}.mp4"
    marker = out_dir / f"{model}__{scene}.missing"
    if out.is_file() and not force:
        return model, scene, "missing" if marker.is_file() else "cached"
    if force and out.is_file():
        out.unlink()
    if marker.is_file():
        marker.unlink()
    url = f"{base}/{model}/pages/{scene}.html"
    last = "FAIL"
    for attempt in (1, 2):
        try:
            result = record_visited(f"{model}__{scene}", url, out_dir, options)
        except Exception as exc:  # noqa: BLE001 - Chrome срывается транзиентно
            last = f"FAIL {type(exc).__name__}: {exc}"
            time.sleep(1.0)
            continue
        if result.status == "ok":
            return model, scene, "ok"
        # Страница падает до кадра: честнее плашка, чем чернота.
        placeholder(out, options)
        marker.write_text("1")
        return model, scene, "missing"
    return model, scene, last


def discover(pages: Path) -> list[tuple[str, str]]:
    pairs = []
    for model_dir in sorted(p for p in pages.iterdir() if p.is_dir()):
        for page in sorted((model_dir / "pages").glob("*.html")):
            pairs.append((model_dir.name, page.stem))
    return pairs


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    pages = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    secs = float(sys.argv[4]) if len(sys.argv) > 4 else 23.0
    warmup = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6
    force = "--force" in sys.argv
    out_dir.mkdir(parents=True, exist_ok=True)

    options = Options(secs=secs, warmup=warmup)
    server, base = html_server_for(pages)
    pairs = discover(pages)
    print(f"[batch] сцен: {len(pairs)}, воркеров: {workers}", flush=True)
    stats: dict[str, int] = {}
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(job, pair, pages, out_dir, options, base, force) for pair in pairs]
            for done, future in enumerate(futures, 1):
                model, scene, tag = future.result()
                stats[tag.split()[0]] = stats.get(tag.split()[0], 0) + 1
                if done % 10 == 0 or tag not in ("ok", "cached", "missing"):
                    print(f"[batch] {done}/{len(pairs)} {stats}"
                          + ("" if tag in ("ok", "cached", "missing") else f" | {model}/{scene}: {tag}"),
                          flush=True)
    finally:
        server.shutdown()
    try:
        (out_dir / "_frames").rmdir()
    except OSError:
        pass
    print(f"[batch] готово: {stats}")
    return 0 if not any(key.startswith("FAIL") for key in stats) else 1


if __name__ == "__main__":
    sys.exit(main())
