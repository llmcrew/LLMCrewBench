#!/usr/bin/env python3
"""Проверить страницы витрины: кто доехал до кадра, кто падает до кадра.

Страница, упавшая до первого кадра, снимается плашкой: витрине честнее
показать причину, чем черноту. Проверка ставит/снимает маркер
`<модель>__<сцена>.missing` — тот же, что у страниц, которых нет вовсе;
манифест читает его как плашку. Причина в маркере — `model_failed` (задача
есть, страница не ожила); отсутствие задачи в прогоне даёт `not_run` ещё при
записи.

Готовность по режиму: K обязаны выставить `__ready`/`__dbscene`, свободным L
достаточно settled-разметки. `__ready` у L не требуется.

  python3 probe_pages.py [workers]
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import media_status  # noqa: E402
from record_all import MODELS, OUT, SCENES, _write_showcase  # noqa: E402
from record_clip import ART, H, W, WS, cdp_ws, chrome, free_port, html_server, page_ready  # noqa: E402

READY_MS = 12000


def probe(model: str, scene: str, base: str) -> tuple[str, str, bool, str]:
    if not (ART / "_showcase" / model / "pages" / f"{scene}.html").is_file():
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
        call("Page.navigate", {"url": f"{base}/_showcase/{model}/pages/{scene}.html"})
        ready = bool(page_ready(call, ws, scene, timeout=READY_MS / 1000))
        with ws.lock:
            errs = [e for e in ws.events if e.get("method") == "Runtime.exceptionThrown"]
        desc = (errs[0]["params"]["exceptionDetails"].get("exception", {}).get("description", "")
                .split("\n")[0] if errs else "")
        # Диалог сам по себе не приговор (он принят автоматически), но если
        # страница всё же не ожила, его текст — лучшая улика.
        note = ws.dialog_note()
        if note:
            desc = f"диалог: {note}" if not desc else f"{desc} (диалог: {note})"
        return model, scene, ready, desc
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


def main() -> int:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    pairs = [(m, s) for s in SCENES for m in MODELS
             if (ART / "_showcase" / m / "pages" / f"{s}.html").is_file()]
    srv, base = html_server()
    broken: list[str] = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for model, scene, ready, desc in pool.map(lambda p: probe(*p, base), pairs):
                if ready:
                    media_status.clear_marker(OUT, model, scene)
                else:
                    media_status.write_marker(OUT, model, scene, "model_failed", desc)
                    broken.append(f"{model}/{scene}: {desc or 'страница не дошла до кадра'}")
                    print(f"[probe] {model}/{scene}: упала -> плашка ({desc or 'нет кадра'})", flush=True)
    finally:
        srv.shutdown()
    # Маркеры — источник, манифесты обязаны сразу следовать за ними.
    _write_showcase()
    print(f"[probe] готово: упавших {len(broken)} из {len(pairs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
