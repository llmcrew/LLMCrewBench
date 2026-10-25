#!/usr/bin/env python3
"""Проверить страницы витрины: кто доехал до кадра, кто падает до кадра.

Страница, упавшая до первого кадра, снимается плашкой: витрине честнее
показать причину, чем черноту. Проверка ставит/снимает маркер
`<модель>__<сцена>.missing` - тот же, что у страниц, которых нет вовсе;
манифест читает его как плашку. Причина в маркере - `model_failed` (задача
есть, страница не ожила); отсутствие задачи в прогоне даёт `not_run` ещё при
записи.

Готовность по режиму: K обязаны выставить `__ready`/`__dbscene`, свободным L
достаточно settled-разметки. `__ready` у L не требуется.

Готовность снимает маркер, но `capture_failed` без клипа переживает пробу:
живая страница без записи - всё ещё сорванный захват, а не готовая сцена.

  python3 probe_pages.py [workers] [--models FILE]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fleet  # noqa: E402
import media_status  # noqa: E402
from record_all import DEFAULT_REPORT_ROOT, OUT, _write_showcase, source_page_path  # noqa: E402
from record_clip import H, W, WS, cdp_ws, chrome, free_port, html_server, locate_source, page_ready  # noqa: E402

READY_MS = 12000


def probe(model: str, scene: str, base: str,
          report_root: Path = DEFAULT_REPORT_ROOT) -> tuple[str, str, bool, str]:
    source, server_root = locate_source(model, scene, report_root)
    if source is None:
        return model, scene, False, "нет страницы"
    port = free_port()
    profile = Path(tempfile.mkdtemp(prefix="probe-chrome-"))
    proc = subprocess.Popen(
        [str(chrome()), "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
         "--no-first-run", "--no-default-browser-check",
         "--use-angle=metal", "--disable-gpu-sandbox",
         "--hide-scrollbars", f"--window-size={W},{H}",
         "--force-device-scale-factor=1", "--use-mock-keychain", "--password-store=basic",
         "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ws = WS(cdp_ws(port))
        tgt = ws.call("Target.createTarget", {"url": "about:blank"})
        sid = ws.call("Target.attachToTarget",
                      {"targetId": tgt["result"]["targetId"], "flatten": True})["result"]["sessionId"]
        call = lambda m, p=None, **kw: ws.call(m, p, session_id=sid, **kw)  # noqa: E731
        call("Page.enable")
        call("Runtime.enable")
        call("Emulation.setDeviceMetricsOverride", {"width": W, "height": H, "deviceScaleFactor": 1, "mobile": False})
        page = source.relative_to(server_root).as_posix()
        call("Page.navigate", {"url": f"{base}/{page}"})
        ready = bool(page_ready(call, ws, scene, timeout=READY_MS / 1000))
        with ws.lock:
            errs = [e for e in ws.events if e.get("method") == "Runtime.exceptionThrown"]
        desc = (errs[0]["params"]["exceptionDetails"].get("exception", {}).get("description", "")
                .split("\n")[0] if errs else "")
        # Диалог сам по себе не приговор (он принят автоматически), но если
        # страница всё же не ожила, его текст - лучшая улика.
        note = ws.dialog_note()
        if note:
            desc = f"диалог: {note}" if not desc else f"{desc} (диалог: {note})"
        return model, scene, ready, desc
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверить readiness страниц витрины")
    parser.add_argument("workers", nargs="?", type=int, default=6)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT,
                        help="каталог отчёта; для ремонта укажите корень _repairs")
    parser.add_argument("--models", type=Path, help="свой models.json вместо src/data/models.json")
    args = parser.parse_args()
    report_root = args.report_root.expanduser().resolve()
    model_rows = fleet.models(args.models)
    models = [row["dir"] for row in model_rows]
    scenes = fleet.scene_ids()
    pairs = [(model, scene) for scene in scenes for model in models
             if source_page_path(report_root, model, scene).is_file()]
    _, server_root = locate_source(pairs[0][0], pairs[0][1], report_root) if pairs else (None, report_root)
    srv, base = html_server(server_root)
    publish_original_status = report_root == DEFAULT_REPORT_ROOT.resolve()
    broken: list[str] = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for model, scene, ready, desc in pool.map(
                    lambda pair: probe(pair[0], pair[1], base, report_root), pairs):
                if ready:
                    if publish_original_status:
                        kept = media_status.refresh_after_probe(OUT, model, scene)
                        if kept:
                            broken.append(f"{model}/{scene}: {kept} без клипа")
                            print(f"[probe] {model}/{scene}: жива, но клипа нет -> {kept}", flush=True)
                else:
                    reason = desc or "страница не дошла до первого кадра"
                    if publish_original_status:
                        media_status.write_marker(OUT, model, scene, "model_failed", reason)
                    broken.append(f"{model}/{scene}: {reason}")
                    print(f"[probe] {model}/{scene}: упала -> плашка ({reason})", flush=True)
    finally:
        srv.shutdown()
    # Repaired/custom-root probes are diagnostic only; keep the original
    # model-status markers and showcase source unchanged.
    if publish_original_status:
        _write_showcase(OUT, report_root, models=models, model_rows=model_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
