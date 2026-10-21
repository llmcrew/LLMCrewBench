"""JS-диалог в шаге страницы не должен вешать запись.

`alert()` останавливает главный поток: незакрытый, он держит `Runtime.evaluate`
до таймаута, и клип получает плашку, хотя страница жива. Клиент CDP обязан
принимать диалоги сам, а их текст — оставлять для маркера.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "video" / "tools" / "record_clip.py"

pytestmark = pytest.mark.skipif(not TOOL.is_file(), reason="рекордер вне репозитория")

ALERT_PAGE = (
    "<!DOCTYPE html><html><body><script>"
    "window.__step = () => { alert('boom'); return 1; };"
    "</script></body></html>"
)


def _module():
    spec = importlib.util.spec_from_file_location("record_clip_dialogs", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    module = _module()
    try:
        module.chrome()
    except SystemExit:
        pytest.skip("Chrome не установлен")
    return module


@pytest.fixture(scope="module")
def page(mod):
    """Открытая вкладка с проверяемой страницей; диалоги принимаются клиентом."""
    port = mod.free_port()
    profile = Path(tempfile.mkdtemp(prefix="dlg-chrome-"))
    proc = subprocess.Popen(
        [str(mod.chrome()), "--headless=new", f"--remote-debugging-port={port}",
         f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
         "--use-angle=metal", "--disable-gpu-sandbox", "--force-device-scale-factor=1",
         "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ws = None
    try:
        ws = mod.WS(mod.cdp_ws(port))
        tgt = ws.call("Target.createTarget", {"url": "about:blank"})
        sid = ws.call("Target.attachToTarget",
                      {"targetId": tgt["result"]["targetId"], "flatten": True})["result"]["sessionId"]

        def call(method, params=None, **kw):
            return ws.call(method, params, session_id=sid, **kw)

        call("Page.enable")
        call("Runtime.enable")
        call("Page.navigate", {"url": "data:text/html," + quote(ALERT_PAGE)})
        deadline = time.time() + 10
        while time.time() < deadline:
            ready = call("Runtime.evaluate", {"expression": "document.readyState", "returnByValue": True})
            value = ((ready.get("result") or {}).get("result") or {}).get("value")
            step = call("Runtime.evaluate", {"expression": "typeof window.__step", "returnByValue": True})
            if value == "complete" and ((step.get("result") or {}).get("result") or {}).get("value") == "function":
                break
            time.sleep(0.05)
        else:
            pytest.skip("страница с диалогом не загрузилась")
        yield call, ws
    finally:
        if ws is not None:
            try:
                ws.sock.close()
            except OSError:
                pass
        proc.terminate()
        proc.wait(timeout=10)
        shutil.rmtree(profile, ignore_errors=True)


def test_an_alert_does_not_block_the_step(page):
    """Без обработчика этот evaluate висит до таймаута и клип не снимается."""
    call, ws = page
    started = time.time()
    answer = call("Runtime.evaluate", {"expression": "window.__step(1 / 60)", "returnByValue": True},
                  timeout=8)
    assert time.time() - started < 8, "шаг упёрся в таймаут: диалог не принят"
    assert ((answer.get("result") or {}).get("result") or {}).get("value") == 1
    assert ws.dialogs and "boom" in ws.dialogs[-1]


def test_the_dialog_message_is_kept_for_the_marker(page):
    call, ws = page
    call("Runtime.evaluate", {"expression": "window.__step(1 / 60)", "returnByValue": True}, timeout=8)
    assert "boom" in ws.dialog_note()
