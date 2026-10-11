#!/usr/bin/env python3
"""Записать все клипы сцен для всех моделей: 49 сцен × 5 моделей.

Параллельно по N воркеров, каждый — свой Chrome. Пропускает уже записанное,
пишет манифест клипов (кто/что снят, где плашка «не справилась»).

  python3 record_all.py [workers] [secs] [warmup] [--force]   # --force: перезаписать всё
"""
from __future__ import annotations
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "reports" / "comparison" / "_showcase"
OUT = ROOT / "video" / "public" / "clips"
TOOL = Path(__file__).with_name("record_clip.py")

MODELS = ["deepseek-v4.1-flash-unc", "qwen35", "qwen36", "qwen38", "minimax-m3"]
SCENES = [f"K-{i}" for i in range(1, 21)] + [f"L-{i}" for i in range(1, 30)]


def job(pair: tuple[str, str], secs: float, warmup: float, force: bool = False) -> tuple[str, str, str]:
    model, scene = pair
    existing = OUT / f"{model}__{scene}.mp4"
    if existing.is_file() and not force:
        return model, scene, "cached"
    if force and existing.is_file():
        existing.unlink()
    # Chrome изредка не поднимается (Target.createTarget по таймауту) — это
    # транзиент, а не вердикт: одна повторная попытка, и только потом FAIL.
    for attempt in (1, 2):
        r = subprocess.run([sys.executable, str(TOOL), model, scene, str(OUT), str(secs), str(warmup)],
                           capture_output=True, text=True)
        if r.returncode == 0 or (OUT / f"{model}__{scene}.missing").is_file():
            break
        if attempt == 1:
            print(f"[all] {model}/{scene}: сбой, повтор ({r.stderr.strip().splitlines()[-1][:80]})", flush=True)
    tag = "missing" if (OUT / f"{model}__{scene}.missing").is_file() else ("ok" if r.returncode == 0 else "FAIL")
    if tag == "FAIL":
        print(f"[all] {model}/{scene}: FAIL\n{r.stderr[-400:]}", flush=True)
    return model, scene, tag


LABELS = {
    "deepseek-v4.1-flash-unc": "DeepSeek 4.1 Flash",
    "qwen35": "Qwen 3.5 397B A17B",
    "qwen36": "Qwen 3.6 35B A3B",
    "qwen38": "Qwen 3.8",
    "minimax-m3": "MiniMax M3",
}
ACCENTS = {
    "deepseek-v4.1-flash-unc": "#4cc2ff",
    "qwen35": "#7ee0ff",
    "qwen36": "#3ddc97",
    "qwen38": "#b48cff",
    "minimax-m3": "#ffb454",
}


def _write_showcase() -> None:
    """Собрать src/data/showcase.json: 49 сцен x 5 моделей, плашки для пропусков."""
    import random
    scenes = json.loads((ROOT / "video/src/data/scenes.json").read_text(encoding="utf-8"))
    # Плашка — там, где записи нет: либо страницы не было (маркер), либо запись
    # сорвалась. Иначе витрина ссылается на несуществующий файл и полоса чёрная.
    missing = {p.stem for p in OUT.glob("*.missing")}
    for m in MODELS:
        for sc in scenes:
            if not (OUT / f"{m}__{sc['id']}.mp4").is_file():
                missing.add(f"{m}__{sc['id']}")
    rnd = random.Random(20261005)
    rows = []
    for sc in scenes:
        runs = [
            {
                "model": m,
                "label": LABELS[m],
                "clip": f"clips/{m}__{sc['id']}.mp4",
                "missing": f"{m}__{sc['id']}" in missing,
                "accent": ACCENTS[m],
            }
            for m in MODELS
        ]
        avail = [i for i, r in enumerate(runs) if not r["missing"]]
        rows.append({
            "scene": sc["id"],
            "title": sc.get("label") or sc["title"],
            "runs": runs,
            "open": rnd.choice(avail) if avail else 0,
        })
    (ROOT / "video/src/data/showcase.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def main() -> int:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 23.0
    warmup = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    force = "--force" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [(m, s) for s in SCENES for m in MODELS]
    stats: dict[str, int] = {}
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for model, scene, tag in pool.map(lambda p: job(p, secs, warmup, force), pairs):
            stats[tag] = stats.get(tag, 0) + 1
            results.append({"model": model, "scene": scene, "status": tag})
            if len(results) % 25 == 0:
                print(f"[all] {len(results)}/{len(pairs)} {stats}", flush=True)
    (OUT.parent / "clips.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    # Манифест витрины — по факту файлов; имена моделей форматированные.
    _write_showcase()
    print(f"[all] готово: {stats}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
