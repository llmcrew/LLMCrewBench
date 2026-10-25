#!/usr/bin/env python3
"""Записать живые клипы для сцен витрины и собрать манифесты.

Использование:
  python3 record_all.py [workers] [seconds] [warmup] [--out DIR]
      [--report-root DIR] [--models FILE] [--fps 120] [--crf 16]
      [--preset medium] [--startup-timeout 180] [--chrome PATH] [--force]

Запись каждой пары идёт в отдельный временный каталог и заменяет предыдущий
клип только после успешной записи. `capture_failed` - инфраструктурный сбой:
такой прогон не обновляет манифесты и завершает работу с ненулевым кодом.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fleet  # noqa: E402
import media_status  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "video"
DEFAULT_REPORT_ROOT = ROOT / "reports" / "comparison-deepseek-1021"
OUT = VIDEO / "public" / "clips"
TOOL = Path(__file__).with_name("record_clip.py")
RUNS = ROOT / "runs"
#: Флот и сцены - из данных; `--models` подменяет флот моделей.
MODELS = fleet.model_names()
SCENES = fleet.scene_ids()
CACHE_FILE = ".recording.json"


def source_page_path(report_root: Path, model: str, scene: str) -> Path:
    """Resolve a page under exactly the requested report root."""
    return report_root / "_showcase" / model / "pages" / f"{scene}.html"


def _cache_matches(out_dir: Path, metadata: dict) -> bool:
    try:
        cached = json.loads((out_dir / CACHE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return cached == metadata


def _recording_metadata(secs: float, warmup: float, fps: int, crf: int,
                        preset: str, source_root: Path, startup_timeout: float,
                        chrome: str | None, models: list[str] | None = None,
                        scenes: list[str] | None = None) -> dict:
    sources = {}
    for model in models or MODELS:
        for scene in scenes or SCENES:
            page = source_page_path(source_root, model, scene)
            sources[f"{model}/{scene}"] = (
                hashlib.sha256(page.read_bytes()).hexdigest() if page.is_file() else None
            )
    return {
        "seconds": secs,
        "warmup": warmup,
        "fps": fps,
        "crf": crf,
        "preset": preset,
        "source_root": str(source_root.resolve()),
        "sources": sources,
        "startup_timeout": startup_timeout,
        "chrome": chrome,
    }


def _failure(out_dir: Path, model: str, scene: str, detail: str,
             preserve_existing: bool = False) -> tuple[str, str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not preserve_existing:
        media_status.write_marker(out_dir, model, scene, "capture_failed", detail)
    print(f"[all] {model}/{scene}: capture_failed\n{detail[-400:]}", flush=True)
    return model, scene, "capture_failed"


def job(pair: tuple[str, str], secs: float, warmup: float, force: bool = False,
        *, out_dir: Path | None = None, fps: int = 120, crf: int = 16,
        preset: str = "medium", source_root: Path | None = None,
        startup_timeout: float = 180, chrome: str | None = None,
        models_file: Path | None = None,
        cache_metadata: dict | None = None) -> tuple[str, str, str]:
    model, scene = pair
    out_dir = out_dir or OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = media_status.clip_path(out_dir, model, scene)
    marker = media_status.marker_path(out_dir, model, scene)
    source_root = source_root or DEFAULT_REPORT_ROOT
    preserve_existing = (existing.is_file() and existing.stat().st_size > 0) or marker.exists()

    if not force and existing.is_file() and existing.stat().st_size > 0:
        # Direct callers without batch settings retain status-query behavior;
        # the CLI always supplies metadata and only reuses matching recordings.
        if cache_metadata is None and marker.is_file():
            return model, scene, media_status.status_of(out_dir, model, scene, RUNS)
        if cache_metadata is not None and _cache_matches(out_dir, cache_metadata):
            status = media_status.status_of(out_dir, model, scene, RUNS)
            if status not in ("capture_failed", "media_missing"):
                return model, scene, status

    # Isolate each new recording so a failed encode can never truncate a usable
    # existing clip. The temporary directory is still inside the selected video
    # output tree, satisfying record_clip's output containment contract.
    temp_dir = Path(tempfile.mkdtemp(prefix=f".record-{model}-{scene}-", dir=out_dir))
    try:
        command = [
            sys.executable, str(TOOL), model, scene, str(temp_dir),
            str(secs), str(warmup),
            "--source-root", str(source_root.resolve()),
            "--startup-timeout", str(startup_timeout),
            "--fps", str(fps), "--crf", str(crf), "--preset", preset,
        ]
        if chrome:
            command.extend(("--chrome", chrome))
        if models_file is not None:
            command.extend(("--models", str(models_file)))
        if force:
            command.append("--force")
        try:
            result = subprocess.run(command, capture_output=True, text=True)
        except OSError as exc:
            return _failure(out_dir, model, scene, f"не удалось запустить рекордер: {exc}", preserve_existing)

        if result.returncode not in (0, 3):
            detail = result.stderr.strip() or f"record_clip завершился с кодом {result.returncode}"
            return _failure(out_dir, model, scene, detail, preserve_existing)

        new_clip = media_status.clip_path(temp_dir, model, scene)
        new_marker = media_status.marker_path(temp_dir, model, scene)
        if not new_clip.is_file() or new_clip.stat().st_size == 0:
            detail = result.stderr.strip() or "record_clip не создал непустой клип"
            return _failure(out_dir, model, scene, detail, preserve_existing)

        status = media_status.status_of(temp_dir, model, scene, RUNS)
        if status in ("capture_failed", "media_missing"):
            detail = result.stderr.strip() or f"неожиданный статус рекордера: {status}"
            return _failure(out_dir, model, scene, detail, preserve_existing)
        if result.returncode == 3 and not new_marker.is_file():
            return _failure(out_dir, model, scene, "record_clip сообщил model_failed без статусного маркера", preserve_existing)
        evidence = new_clip.with_suffix(".capture.json")
        try:
            captured = json.loads(evidence.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return _failure(out_dir, model, scene, f"нет доказательств записи: {exc}", preserve_existing)
        expected = {
            "status": status, "seconds": secs, "fps": fps,
            "frames": round(secs * fps), "crf": crf, "preset": preset,
            "captured_frames": round(secs * fps) if status == "ok" else 0,
        }
        if any(captured.get(key) != value for key, value in expected.items()):
            return _failure(out_dir, model, scene, "параметры записанного клипа не совпадают с запросом", preserve_existing)
        if cache_metadata is not None and captured.get("source_sha256") != cache_metadata["sources"][f"{model}/{scene}"]:
            return _failure(out_dir, model, scene, "исходник изменился во время записи", preserve_existing)
        if status == "model_failed" and new_marker.is_file():
            reason = new_marker.read_text(encoding="utf-8").strip()
            if reason == "model_failed":
                page = source_page_path(source_root, model, scene)
                detail = "страница отсутствует" if not page.is_file() else "страница не дошла до первого кадра"
                media_status.write_marker(temp_dir, model, scene, "model_failed", detail)

        # Publish the clip only after record_clip returns a complete nonempty
        # file. Path.replace is an atomic replacement on the same filesystem.
        new_clip.replace(existing)
        evidence.replace(existing.with_suffix(".capture.json"))
        if new_marker.is_file():
            new_marker.replace(marker)
        else:
            marker.unlink(missing_ok=True)
        print(f"[all] {model}/{scene}: {status}", flush=True)
        return model, scene, status
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _write_showcase(out_dir: Path | None = None,
                    report_root: Path | None = None,
                    models: list[str] | None = None,
                    model_rows: list[dict] | None = None,
                    scenes: list[str] | None = None) -> list[dict]:
    """Write the showcase and its clips projection for a completed media tree.

    The public tree updates the site's source data as well as public/clips.json.
    A staged tree keeps both manifests beside the staged clips.
    """
    out_dir = out_dir or OUT
    report_root = report_root or DEFAULT_REPORT_ROOT
    fleet_rows = model_rows or fleet.models()
    models = models or [row["dir"] for row in fleet_rows]
    labels = {row["dir"]: row.get("label") for row in fleet_rows}
    accents = {row["dir"]: row.get("accent") for row in fleet_rows}
    scene_rows = fleet.scenes() if scenes is None else [
        {"id": scene, "title": scene, "label": scene} for scene in scenes
    ]
    status = media_status.scan(out_dir, [sc["id"] for sc in scene_rows], models, RUNS)
    for sc in scene_rows:
        for model in models:
            key = (model, sc["id"])
            if status[key] != "media_missing":
                continue
            if source_page_path(report_root, model, sc["id"]).is_file():
                continue
            reason = media_status.task_status(model, sc["id"], RUNS)
            status[key] = "model_failed" if reason in (None, "ok") else reason

    rnd = random.Random(20261005)
    rows = []
    for sc in scene_rows:
        runs = [
            {
                "model": model,
                "label": labels.get(model) or model,
                "clip": f"clips/{model}__{sc['id']}.mp4",
                "status": status[(model, sc["id"])],
                "accent": accents.get(model) or "#888888",
            }
            for model in models
        ]
        available = [index for index, run in enumerate(runs) if run["status"] == "ok"]
        rows.append({
            "scene": sc["id"],
            "title": sc.get("label") or sc["title"],
            "runs": runs,
            "open": rnd.choice(available) if available else 0,
        })

    derived = [
        {"model": run["model"], "scene": row["scene"], "status": run["status"]}
        for row in rows
        for run in row["runs"]
    ]
    clips_manifest = out_dir.parent / "clips.json"
    is_public_tree = out_dir.resolve() == (VIDEO / "public" / "clips").resolve()
    showcase_manifest = (VIDEO / "src" / "data" / "showcase.json"
                         if is_public_tree else out_dir.parent / "showcase.json")
    payload = json.dumps(rows, ensure_ascii=False, indent=1)
    projection = json.dumps(derived, ensure_ascii=False, indent=1)
    showcase_manifest.write_text(payload, encoding="utf-8")
    clips_manifest.write_text(projection, encoding="utf-8")
    return derived


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Записать клипы сцен витрины")
    parser.add_argument("workers", nargs="?", type=int, default=6)
    parser.add_argument("secs", nargs="?", type=float, default=20)
    parser.add_argument("warmup", nargs="?", type=float, default=0.6)
    parser.add_argument("--out", type=Path, default=OUT, help="каталог клипов в video/out или video/public")
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--models", type=Path, help="свой models.json вместо src/data/models.json")
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--chrome")
    parser.add_argument("--force", action="store_true")
    return parser


def _validate_out(path: Path) -> Path:
    target = path.expanduser().resolve()
    allowed_roots = (VIDEO / "out").resolve(), (VIDEO / "public").resolve()
    if not any(target != root and root in target.parents for root in allowed_roots):
        raise ValueError(f"OUT {target} должен находиться внутри {VIDEO / 'out'} или {VIDEO / 'public'}")
    return target


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.workers < 1:
        print("workers должен быть положительным", file=sys.stderr)
        return 2
    try:
        out_dir = _validate_out(args.out)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    report_root = args.report_root.expanduser().resolve()
    fleet_rows = fleet.models(args.models)
    models = [row["dir"] for row in fleet_rows]
    scenes = fleet.scene_ids()
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = _recording_metadata(args.secs, args.warmup, args.fps, args.crf,
                                   args.preset, report_root, args.startup_timeout,
                                   args.chrome, models=models, scenes=scenes)
    pairs = [(model, scene) for scene in scenes for model in models]
    stats: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = pool.map(
            lambda pair: job(
                pair, args.secs, args.warmup, args.force,
                out_dir=out_dir, fps=args.fps, crf=args.crf, preset=args.preset,
                source_root=report_root, startup_timeout=args.startup_timeout,
                chrome=args.chrome, models_file=args.models,
                cache_metadata=metadata,
            ),
            pairs,
        )
        for index, (model, scene, status) in enumerate(jobs, start=1):
            stats[status] = stats.get(status, 0) + 1
            if index % 25 == 0:
                print(f"[all] {index}/{len(pairs)} {stats}", flush=True)

    print(f"[all] готово: {stats}", flush=True)
    if stats.get("capture_failed"):
        return 1

    try:
        _write_showcase(out_dir, report_root, models=models, model_rows=fleet_rows)
        (out_dir / CACHE_FILE).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"[all] не удалось записать манифесты: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
