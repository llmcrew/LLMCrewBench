#!/usr/bin/env python3
"""Записать все клипы сцен для обеих сторон сравнения: 49 сцен × 2 стенда.

Параллельно по N воркеров, каждый — свой Chrome. Пропускает уже записанное,
пишет манифест витрины и его производную clips.json.

Статусы не смешиваются: `ok` — живой клип, `not_run` — задачи нет в прогоне,
`model_failed` — страница не доставлена или падает до кадра, `capture_failed` —
страница жива, а запись сорвалась. `not_run`/`model_failed` — штатные плашки,
`capture_failed` делает код возврата ненулевым: инфраструктурный сбой нельзя
принять за успех.

  python3 record_all.py [workers] [secs] [warmup] [--force]   # --force: перезаписать всё
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import media_status  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "reports" / "comparison-deepseek-1017" / "_showcase"
OUT = ROOT / "video" / "public" / "clips"
TOOL = Path(__file__).with_name("record_clip.py")
RUNS = ROOT / "runs"

MODELS = ["deepseek-v4.1-flash-unc", "deepseek-v4.1-flash-unc-1017"]
SCENES = [f"K-{i}" for i in range(1, 21)] + [f"L-{i}" for i in range(1, 30)]


def job(pair: tuple[str, str], secs: float, warmup: float, force: bool = False) -> tuple[str, str, str]:
    model, scene = pair
    existing = media_status.clip_path(OUT, model, scene)
    marker = media_status.marker_path(OUT, model, scene)
    if existing.is_file() and not force:
        # Плашка — не кэш живой записи: статус берём из маркера, а не из файла.
        return model, scene, media_status.status_of(OUT, model, scene, RUNS)
    if force:
        existing.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
    # Chrome изредка не поднимается (Target.createTarget по таймауту) — это
    # транзиент, а не вердикт: одна повторная попытка, и только потом сбой.
    r = None
    for attempt in (1, 2):
        r = subprocess.run([sys.executable, str(TOOL), model, scene, str(OUT), str(secs), str(warmup)],
                           capture_output=True, text=True)
        if r.returncode == 0 or marker.is_file():
            break
        if attempt == 1:
            tail = (r.stderr.strip().splitlines() or ["?"])[-1][:80]
            print(f"[all] {model}/{scene}: сбой, повтор ({tail})", flush=True)
    status = media_status.status_of(OUT, model, scene, RUNS)
    if r is not None and r.returncode != 0 and status in ("media_missing", "ok"):
        # Плашку собрать не удалось: причина должна остаться в данных.
        media_status.write_marker(OUT, model, scene, "capture_failed")
        status = "capture_failed"
    if r is not None and r.returncode != 0:
        print(f"[all] {model}/{scene}: {status}\n{r.stderr[-400:]}", flush=True)
    return model, scene, status


LABELS = {
    "deepseek-v4.1-flash-unc": "DeepSeek 4.1 Flash · стенд A",
    "deepseek-v4.1-flash-unc-1017": "DeepSeek 4.1 Flash · стенд 1017",
}
ACCENTS = {
    "deepseek-v4.1-flash-unc": "#4cc2ff",
    "deepseek-v4.1-flash-unc-1017": "#ffb45c",
}


def _write_showcase() -> list[dict]:
    """Собрать src/data/showcase.json по маркерам и файлам; clips.json — из него.

    showcase.json — единственный источник статуса для витрины и preflight.
    clips.json дублирует только проекцию model/scene/status.
    """
    scenes = json.loads((ROOT / "video/src/data/scenes.json").read_text(encoding="utf-8"))
    status = media_status.scan(OUT, [sc["id"] for sc in scenes], MODELS, RUNS)
    rnd = random.Random(20261005)
    rows = []
    for sc in scenes:
        runs = [
            {
                "model": m,
                "label": LABELS[m],
                "clip": f"clips/{m}__{sc['id']}.mp4",
                "status": status[(m, sc["id"])],
                "accent": ACCENTS[m],
            }
            for m in MODELS
        ]
        avail = [i for i, r in enumerate(runs) if r["status"] == "ok"]
        rows.append({
            "scene": sc["id"],
            "title": sc.get("label") or sc["title"],
            "runs": runs,
            "open": rnd.choice(avail) if avail else 0,
        })
    (ROOT / "video/src/data/showcase.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    derived = [
        {"model": run["model"], "scene": row["scene"], "status": run["status"]}
        for row in rows
        for run in row["runs"]
    ]
    (OUT.parent / "clips.json").write_text(
        json.dumps(derived, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return derived


def main() -> int:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 23.0
    warmup = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    force = "--force" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [(m, s) for s in SCENES for m in MODELS]
    stats: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for model, scene, tag in pool.map(lambda p: job(p, secs, warmup, force), pairs):
            stats[tag] = stats.get(tag, 0) + 1
            if sum(stats.values()) % 25 == 0:
                print(f"[all] {sum(stats.values())}/{len(pairs)} {stats}", flush=True)
    _write_showcase()
    print(f"[all] готово: {stats}", flush=True)
    # Инфраструктурный сбой — не успех: перезапуск/CI должны это увидеть.
    return 1 if stats.get("capture_failed") or stats.get("FAIL") else 0


if __name__ == "__main__":
    sys.exit(main())
