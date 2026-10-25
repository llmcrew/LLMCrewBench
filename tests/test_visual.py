"""Infrastructure tests for the visual block: extraction, predicates, scoring.

The scenarios themselves are tested next to their modules
(``tests/test_visual_<slice>.py``); here the shared machinery is pinned: what
counts as a document, how a probe payload is read, how the four checks are
turned into a score, and what happens when the browser is missing (unverified,
never zero).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from dailybench import art, browser, config, core, gallery, visual
from dailybench.tasks import visual as tasks_visual

FENCE_DOC = """Вот решение:

```html
<!DOCTYPE html>
<html><body><script type="module">
import * as THREE from './vendor/three.module.js';
window.__probe = () => ({ok: true});
window.__step = (dt) => dt;
window.__ready = true;
</script></body></html>
```

Готово.
"""


def test_extract_html_prefers_the_fenced_document():
    html = visual.extract_html(FENCE_DOC)
    assert html.startswith("<!DOCTYPE html>")
    assert "three.module.js" in html
    assert "Вот решение" not in html


def test_extract_html_takes_a_bare_document():
    bare = "<!DOCTYPE html>\n<html><body><script>1</script></body></html>"
    assert visual.extract_html(bare) == bare
    assert visual.extract_html("   <html><script>x</script></html>  ").startswith("<html>")


def test_extract_html_keeps_the_largest_block_and_ignores_prose():
    two = "```html\n<html><script>a</script></html>\n```\n```html\n<html><script>b</script><p>x</p></html>\n```"
    assert "b" in visual.extract_html(two)
    assert visual.extract_html("не знаю") == ""
    assert visual.extract_html("```python\nprint(1)\n```") == ""


def test_looks_like_document_rejects_shortcuts():
    ok, _ = visual.looks_like_document(visual.extract_html(FENCE_DOC))
    assert ok
    bad, why = visual.looks_like_document("<script>1</script>")
    assert not bad and "html" in why
    bad, why = visual.looks_like_document("<html><script>1</script></html>")
    assert not bad and "three.module.js" in why
    # A remote URL is not a static veto: the renderer refuses it and charges
    # ``runtime``, the same as a WebSocket handshake (M09).
    remote = '<html><script src="https://cdn.example/three.module.js"></script></html>'
    assert visual.looks_like_document(remote)[0]
    assert visual.remote_urls(remote) == ["https://cdn.example/three.module.js"]


def test_path_of_walks_dicts_and_lists():
    payload = {"camera": [1, 2, 3], "scene": {"balls": [{"x": 5}]}}
    assert visual.path_of(payload, "camera.1") == 2
    assert visual.path_of(payload, "scene.balls.0.x") == 5
    assert visual.path_of(payload, "scene.balls.-1.x") == 5
    assert visual.path_of(payload, "camera.9", "нет") == "нет"
    assert visual.path_of(payload, "scene.nope", None) is None


def test_predicates_read_the_probe_and_report_the_reason():
    payload = {
        "objects": ["ground", "ball"],
        "balls": [{"x": 1.0, "y": 2.0}],
        "camera": [0.0, 6.0, 14.0],
    }
    assert visual.near("balls.0.x", 1.0, 1e-3)(payload)[0]
    assert not visual.near("balls.0.x", 1.0005, 1e-6)(payload)[0]
    assert visual.between("balls.0.y", 1.0, 3.0)(payload)[0]
    assert not visual.between("balls.0.y", 3.0, 4.0)(payload)[0]
    assert visual.contains_all("objects", ["ground"])(payload)[0]
    assert not visual.contains_all("objects", ["missing"])(payload)[0]
    assert visual.size_between("objects", 2, 4)(payload)[0]
    assert not visual.size_between("objects", 3, 4)(payload)[0]
    good, detail = visual.near("balls.0.x", 9.0)(payload)
    assert not good and "balls.0.x = 1.0" in detail


def test_predicates_on_a_malformed_probe_fail_instead_of_raising():
    report = visual._predicates_report([visual.near("camera.0", 1.0)], {"camera": [None, 6, 14]})
    assert report == (False, "- camera.0: не число (None)")
    wrong_type = visual._predicates_report([visual.near("camera.0", 1.0)], {"camera": "нет"})
    assert wrong_type == (False, "- camera.0: в probe нет такого поля")
    missing = visual._predicates_report([visual.near("gyro.x", 1.0)], {"camera": []})
    assert missing == (False, "- gyro.x: в probe нет такого поля")
    boolean = visual._predicates_report([visual.between("flag", 0, 1)], {"flag": True})
    assert boolean == (False, "- flag: не число (True)")


def test_image_predicates_use_frame_statistics():
    pixels = {
        "changed_fraction": 0.51,
        "distinct_colors": 21,
        "warm": {"fraction": 0.0082, "count": 700, "centroid": [0.6, 0.24]},
        "regions": [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [44, 53, 65]] * 1
        + [[44, 53, 65], [44, 53, 65]],
    }
    assert visual.image_min("changed_fraction", 0.15)(pixels)[0]
    assert not visual.image_min("changed_fraction", 0.9)(pixels)[0]
    assert visual.image_color("warm", 0.002, "видны объекты")(pixels)[0]
    assert not visual.image_color("warm", 0.05, "видны объекты")(pixels)[0]
    assert visual.image_centroid("warm", 1, 0.0, 0.8, "в кадре")(pixels)[0]
    assert not visual.image_centroid("warm", 1, 0.9, 1.0, "в кадре")(pixels)[0]
    assert visual.image_region(8, visual.cool, "низ кадра")(pixels)[0]
    assert not visual.image_region(0, visual.warm, "верх")(pixels)[0]


def test_frame_statistics_come_from_the_png_not_the_page():
    """H17: the picture is measured outside the untrusted page realm."""
    import io

    from PIL import Image

    image = Image.new("RGB", (100, 80), (10, 16, 32))
    for x in range(10, 30):
        for y in range(10, 30):
            image.putpixel((x, y), (255, 136, 0))
    buf = io.BytesIO()
    image.save(buf, "PNG")
    stats = browser.frame_stats(buf.getvalue())
    assert stats["width"] == 100 and stats["height"] == 80
    assert 0.03 < stats["warm"]["fraction"] < 0.07
    assert stats["warm"]["centroid"][0] < 0.3 and stats["warm"]["centroid"][1] < 0.3
    assert stats["bright"]["count"] > 0

    black = io.BytesIO()
    Image.new("RGB", (100, 80), (0, 0, 0)).save(black, "PNG")
    dark = browser.frame_stats(black.getvalue())
    assert dark["warm"]["fraction"] == 0.0 and dark["mean"] == 0.0

    assert browser.frame_stats(None) == {}
    assert browser.frame_stats(b"not a png") == {}


def _task(rng: random.Random) -> core.TaskInstance:
    spec = visual.VisualSpec(
        scene=[visual.contains_all("objects", ["ground"])],
        simulation=[visual.near("t", 1.0, 1e-3)],
        image=[visual.image_min("changed_fraction", 0.02)],
        steps=60,
        dt=1 / 60,
    )
    return visual.build_task(
        task_id="K-T",
        title="проверка каркаса",
        prompt="Собери сцену.",
        spec=spec,
        golden="<html><script>1</script></html>",
    )


def test_build_task_keeps_the_contract_of_a_visual_scenario():
    task = _task(random.Random(1))
    assert task.category == "K"
    assert task.max_points == 4.0
    assert task.oracle["spec"]["steps"] == 60
    assert task.oracle["reference"].startswith("```html")
    assert task.prompt.startswith("Собери сцену.")
    assert "window.__probe()" in task.prompt  # общий контракт дописан каркасом
    assert task.meta["driver"] == "visual"


def test_garbage_answer_scores_zero_without_touching_the_browser(monkeypatch):
    task = _task(random.Random(1))
    monkeypatch.setattr(browser, "availability", lambda: (True, ""))
    transcript = core.Transcript(task_id=task.id, category="K")
    transcript.turns.append(
        core.Turn(messages=[], reply=core.ModelReply(text="не знаю"))
    )
    score = visual.evaluate(task, transcript)
    assert score.earned == 0.0
    assert score.possible == 4.0
    assert [c.name for c in score.checks] == list(visual.CHECK_NAMES)
    assert "HTML-документа" in score.checks[0].detail


def test_missing_browser_is_unverified_not_zero(monkeypatch):
    task = _task(random.Random(1))
    monkeypatch.setattr(browser, "availability", lambda: (False, "playwright не установлен"))
    transcript = core.Transcript(task_id=task.id, category="K")
    transcript.turns.append(core.Turn(messages=[], reply=core.ModelReply(text=FENCE_DOC)))
    score = visual.evaluate(task, transcript)
    assert score.skipped and "playwright" in score.skipped
    assert score.earned == 0.0
    assert score.checks == [], "непроверенное задание не должно выглядеть как проваленные проверки"


def test_a_broken_page_loses_every_check(monkeypatch):
    task = _task(random.Random(1))
    monkeypatch.setattr(browser, "availability", lambda: (True, ""))
    monkeypatch.setattr(browser, "render", lambda *a, **kw: browser.RenderResult(ok=False, reason="страница не отрисовалась"))
    transcript = core.Transcript(task_id=task.id, category="K")
    transcript.turns.append(core.Turn(messages=[], reply=core.ModelReply(text=FENCE_DOC)))
    score = visual.evaluate(task, transcript)
    assert [c.passed for c in score.checks] == [False] * len(visual.CHECK_NAMES)
    assert "рендер не удался" in score.checks[0].detail


# --------------------------------------------------------------------------- #
# Rubric separation (M09) and frame proof (H17) on real browser negatives
# --------------------------------------------------------------------------- #

#: A page that really animates one warm box and reports the simulation state.
#: ``{extra}`` receives the fault under test.
SCENE_PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<style>html,body{{margin:0;background:#0a1020;overflow:hidden}}canvas{{display:block}}</style>
</head><body>
<script type="module">
import * as THREE from './vendor/three.module.js';
const canvas = document.createElement('canvas');
canvas.width = 800; canvas.height = 600;
document.body.appendChild(canvas);
const ctx = canvas.getContext('2d');
let x = 0;
const draw = () => {{
  ctx.fillStyle = '#0a1020'; ctx.fillRect(0, 0, 800, 600);
  ctx.fillStyle = '#ff8800'; ctx.fillRect(Math.round(120 + x * 240), 240, 140, 140);
}};
const round = (value) => Math.round(value * 1e4) / 1e4;
window.__probe = () => ({{objects: ['ground', 'box'], x: round(x)}});
window.__step = (dt) => {{ x += dt; draw(); }};
window.__ready = true;
draw();
{extra}
</script></body></html>"""


def _page(extra: str = "") -> str:
    return SCENE_PAGE.format(extra=extra)


def _scene_task() -> core.TaskInstance:
    spec = visual.VisualSpec(
        scene=[visual.contains_all("objects", ["box"])],
        simulation=[visual.near("x", 1.0, 1e-3)],
        image=[visual.image_min("changed_fraction", 0.02), visual.image_color("warm", 0.002, "виден ящик")],
        steps=60,
        dt=1 / 60,
    )
    return visual.build_task(
        task_id="K-T",
        title="рубрика",
        prompt="Нарисуй ящик.",
        spec=spec,
        golden=_page(),
    )


def _score(page: str, monkeypatch, tmp_path) -> core.Score:
    monkeypatch.setenv(visual.CACHE_ENV, str(tmp_path))
    task = _scene_task()
    transcript = core.Transcript(task_id=task.id, category="K")
    transcript.turns.append(core.Turn(messages=[], reply=core.ModelReply(text=f"```html\n{page}\n```")))
    return visual.evaluate(task, transcript)


def _failed(score: core.Score) -> list[str]:
    return [check.name for check in score.checks if not check.passed]


@pytest.mark.parametrize("page_kind", ["golden", "console", "overlay", "unrelated", "protocol"])
def test_the_rubric_separates_protocol_runtime_html_and_frame(page_kind, monkeypatch, tmp_path):
    """M09/H17: each fault costs its own check and nothing else.

    Golden — all checks; one ``console.error`` — only ``runtime``; a black
    overlay over the canvas — only ``image``; a picture unrelated to the probe —
    only ``image``; a missing protocol — ``html``, ``scene`` and ``simulation``
    while the real picture still earns ``image``.
    """
    usable, reason = browser.availability()
    if not usable:
        pytest.skip(f"браузер недоступен: {reason}")

    pages = {
        "golden": _page(),
        "console": _page('console.error("fixture");'),
        "overlay": _page(
            "const overlay = document.createElement('div');"
            "overlay.style.cssText='position:fixed;inset:0;background:#000;z-index:9';"
            "document.body.appendChild(overlay);"
        ),
        # the drawing never moves while the probe reports a moving number
        "unrelated": _page("").replace(
            "window.__step = (dt) => { x += dt; draw(); };",
            "window.__step = (dt) => { x += dt; };",
        ),
        "protocol": _page()
        .replace("window.__probe = () => ({objects: ['ground', 'box'], x: round(x)});", "")
        .replace("window.__step = (dt) => { x += dt; draw(); };", "")
        .replace("window.__ready = true;", ""),
    }
    expected = {
        "golden": [],
        "console": ["runtime"],
        "overlay": ["image"],
        "unrelated": ["image"],
        "protocol": ["html", "scene", "simulation"],
    }
    score = _score(pages[page_kind], monkeypatch, tmp_path)

    assert _failed(score) == expected[page_kind]
    assert score.possible == 4.0


def test_every_check_is_one_fifth_of_the_task(monkeypatch, tmp_path):
    score = _score(_page(), monkeypatch, tmp_path)
    assert [check.name for check in score.checks] == list(visual.CHECK_NAMES)
    assert all(check.points == visual.POINTS_PER_CHECK for check in score.checks)
    assert score.earned == 4.0


def test_an_http_and_a_websocket_escape_cost_the_same_point(monkeypatch, tmp_path):
    """M09: both are refused requests; one rubric, one weight."""
    usable, reason = browser.availability()
    if not usable:
        pytest.skip(f"браузер недоступен: {reason}")

    http_page = _page("const blocked = new Image(); blocked.src = 'https://example.invalid/x.png';")
    ws_page = _page("new WebSocket('ws://127.0.0.1:9/ch');")
    http_score = _score(http_page, monkeypatch, tmp_path)
    ws_score = _score(ws_page, monkeypatch, tmp_path)

    assert _failed(http_score) == _failed(ws_score) == ["runtime"]
    assert http_score.earned == ws_score.earned
    assert "http" in http_score.checks[1].detail and "websocket" in ws_score.checks[1].detail


def test_render_without_a_document_is_refused():
    result = browser.render("")
    assert not result.ok and "пустой ответ" in result.reason


def test_live_render_screenshots_a_static_svg_without_page_hooks():
    html = """<!doctype html>
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 60">
      <rect width="100" height="60" fill="#123456"/>
      <circle cx="50" cy="30" r="18" fill="#ffcc00"/>
    </svg>"""
    result = browser.render(html, mode="live")

    assert result.ok, result.reason
    assert result.shot and result.shot.startswith(b"\x89PNG\r\n\x1a\n")
    assert result.probe is None and not result.probe_ok
    assert result.pixels is None
    assert result.shot_later_sha
    assert not result.shot_sha_again


def test_live_render_accepts_css_animation_without_reproducibility_checks():
    html = """<!doctype html>
    <style>
      @keyframes pulse { from { opacity: .2; } to { opacity: 1; } }
      #box { width: 80px; height: 80px; background: #36c;
             animation: pulse 1s linear infinite alternate; }
    </style>
    <div id="box"></div>"""
    result = browser.render(html, mode="live")

    assert result.ok, result.reason
    assert result.shot and result.shot.startswith(b"\x89PNG\r\n\x1a\n")
    assert result.shot_later_sha
    assert result.animated is None
    assert result.pixels is None


def test_live_render_keeps_a_screenshot_when_page_errors_and_external_load_fails():
    html = """<!doctype html>
    <style>body { background: #135; }</style>
    <script>throw new Error("live fixture pageerror");</script>
    <img src="https://example.invalid/unavailable.png">"""
    result = browser.render(html, mode="live")

    assert not result.ok
    assert result.shot and result.shot.startswith(b"\x89PNG\r\n\x1a\n")
    assert any("pageerror:" in error for error in result.errors)
    assert any(
        browser.external_class(url) == browser.BLOCKED_HTTP
        and url.split(": ", 1)[-1].startswith("https://example.invalid/")
        for url in result.external
    )


def test_availability_reports_a_reason_when_usable():
    usable, reason = browser.availability()
    assert usable is True, reason
    assert reason == ""


@pytest.mark.parametrize("name", ["three.module.js", "three.core.js", "PROVENANCE.md"])
def test_the_pinned_library_is_present(name):
    assert (browser.VENDOR / name).is_file()


# --------------------------------------------------------------------------- #

def test_live_render_of_a_hung_page_ends_at_the_navigation_timeout():
    """An infinite script must not stall the whole suite (V1).

    Before the guard, both screenshots each hung for their full 90 s timeout on
    a `while (true)` page — one task cost ~200 s and blocked the run. Now the
    render ends at the navigation deadline with an explicit error.
    """
    import time

    html = """<!doctype html><script>while (true) {}</script>"""
    started = time.monotonic()
    result = browser.render(html, mode="live")
    elapsed = time.monotonic() - started

    assert not result.ok
    assert any("navigation:" in error for error in result.errors)
    assert not result.shot
    assert elapsed < 60, (
        f"зависшая страница обязана упираться в таймаут навигации, а не в скриншоты ({elapsed:.0f}s)"
    )


def test_live_render_blocks_a_websocket_escape_without_touching_the_network():
    """V2: WebSocket handshakes bypass HTTP routing; they must be intercepted.

    Before ``route_web_socket``, a page could open ws:// to any local service or
    the outside without appearing in ``external``. A routed socket is never
    connected, so the listener below must stay silent while the attempt is
    recorded.
    """
    import socket
    import threading

    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(10)
    leaked = threading.Event()

    def listener() -> None:
        try:
            server.accept()
            leaked.set()
        except OSError:
            pass

    threading.Thread(target=listener, daemon=True).start()
    port = server.getsockname()[1]
    html = (
        "<!doctype html><script>const ws = new WebSocket("
        f"'ws://127.0.0.1:{port}/channel');"
        "ws.onopen = () => { document.title = 'leaked'; }</script>"
    )
    try:
        result = browser.render(html, mode="live")
    finally:
        server.close()

    assert not leaked.is_set(), "WebSocket handshake reached the network"
    assert any(url.startswith(f"websocket: ws://127.0.0.1:{port}/") for url in result.external)


# --------------------------------------------------------------------------- #
# The frame cache
# --------------------------------------------------------------------------- #


def _k_frame(**overrides):
    base = dict(
        ok=True,
        probe={"t": 0.0, "objects": ["ground"]},
        stepped={"t": 1.0, "objects": ["ground"]},
        pixels={"width": 800, "height": 600, "changed_fraction": 0.5},
        shot=b"png",
        shot_sha="a" * 64,
        shot_sha_again="a" * 64,
    )
    base.update(overrides)
    return browser.RenderResult(**base)


def test_a_failed_render_does_not_poison_a_good_frame(monkeypatch, tmp_path):
    """A retried render replaces a failed cache entry instead of keeping it.

    The screenshot can time out under load while the page is fine: the entry then
    holds no pixels, and every later run scored that page zero from the cache.
    """
    monkeypatch.setenv(visual.CACHE_ENV, str(tmp_path))
    calls = {"n": 0}

    def fake_render(html, **kwargs):
        calls["n"] += 1
        return _k_frame()

    monkeypatch.setattr(browser, "render", fake_render)
    good, hit = visual._cached_render("<html>1</html>", 30, 1 / 60)
    assert good.ok and not hit

    # A failed render lands in the cache (no frame) and is not trusted again.
    failed = browser.RenderResult(ok=False, probe=None, stepped=None)
    blob, shot = visual._entry_paths("<html>1</html>", 30, 1 / 60, visual._probe_sha(failed))
    for stale in tmp_path.glob("*.json"):
        stale.unlink()
    for stale in tmp_path.glob("*.png"):
        shutil.copy(stale, shot)
        stale.unlink()
    blob.write_text(
        json.dumps({"result": {
            "ok": False,
            "reason": "TimeoutError: Page.screenshot",
            "errors": [],
            "external": [],
            "probe": None,
            "stepped": None,
            "pixels": None,
            "pixels_again": None,
            "shot_sha": "",
            "shot_sha_again": "",
            "duration_s": 0.0,
        }}),
        encoding="utf-8",
    )
    monkeypatch.setattr(browser, "render", fake_render)
    healed, hit = visual._cached_render("<html>1</html>", 30, 1 / 60)
    assert healed.ok and healed.pixels, "битая запись обязана перерисоваться"
    assert calls["n"] == 2

    again, hit = visual._cached_render("<html>1</html>", 30, 1 / 60)
    assert calls["n"] == 2 and hit and again.ok, "исправленную запись уже можно переиспользовать"


@contextlib.contextmanager
def _vendor(monkeypatch, tmp_path, body: bytes = b"pin"):
    """Подменить vendor-каталог целиком и сбросить кеш fingerprint."""
    root = tmp_path / "vendor"
    root.mkdir(parents=True, exist_ok=True)
    for name in ("three.module.js", "three.core.js"):
        (root / name).write_bytes(body + name.encode("utf-8"))
    monkeypatch.setattr(browser, "VENDOR", root)
    visual._render_fingerprint.cache_clear()
    try:
        yield root
    finally:
        visual._render_fingerprint.cache_clear()


def test_the_frame_key_carries_the_render_version(monkeypatch, tmp_path):
    """A changed extractor must not be served a frame from the previous one.

    The art block already versioned its cache; the simulation block keyed on page
    content alone, so a bumped ``PIXEL_PROBE`` silently reused old frames.
    """
    with _vendor(monkeypatch, tmp_path):
        key = visual._frame_key("<html></html>", 10, 1 / 60)
        assert key == visual._frame_key("<html></html>", 10, 1 / 60)
        assert key != visual._frame_key("<html>other</html>", 10, 1 / 60)
        # the path and the cache entry must agree on the key
        assert visual.shot_path("<html></html>", 10, 1 / 60).stem == key


def test_vendor_bytes_change_both_frame_keys(monkeypatch, tmp_path):
    """R26: a replaced library must invalidate K and L, not reuse old frames.

    The old key hashed the page and a manual version only, so swapping the
    Three.js build kept every cached frame a "hit" and the report attributed
    pixels to a library that never produced them.
    """
    with _vendor(monkeypatch, tmp_path, b"first"):
        before_k = visual._frame_key("<html></html>", 10, 1 / 60)
        before_l = art._frame_key("<html></html>")
    with _vendor(monkeypatch, tmp_path, b"second"):
        after_k = visual._frame_key("<html></html>", 10, 1 / 60)
        after_l = art._frame_key("<html></html>")

    assert before_k != after_k, "K обязан реагировать на байты библиотеки"
    assert before_l != after_l, "L обязан реагировать на байты библиотеки"


def test_vendor_fingerprint_covers_three_core(monkeypatch, tmp_path):
    """A page may import ``three.core.js`` directly; its bytes count too."""
    with _vendor(monkeypatch, tmp_path) as root:
        before = visual._frame_key("<html></html>", 10, 1 / 60)
        (root / "three.core.js").write_bytes(b"changed core")
        visual._render_fingerprint.cache_clear()
        assert visual._frame_key("<html></html>", 10, 1 / 60) != before


def test_viewport_changes_the_frame_key(monkeypatch, tmp_path):
    with _vendor(monkeypatch, tmp_path):
        assert visual._frame_key(
            "<html></html>", 10, 1 / 60, viewport=(800, 600)
        ) != visual._frame_key("<html></html>", 10, 1 / 60, viewport=(1920, 1080))


def test_the_frame_key_is_stable_for_one_profile(monkeypatch, tmp_path):
    """Determinism: the same page and environment must reuse the same entry."""
    with _vendor(monkeypatch, tmp_path):
        first = visual._frame_key("<html></html>", 10, 1 / 60)
        visual._render_fingerprint.cache_clear()
        assert visual._frame_key("<html></html>", 10, 1 / 60) == first
        assert art._frame_key("<html></html>") == art._frame_key("<html></html>")


def _fake_browser(tmp_path, revision: int, body: bytes = b"") -> Path:
    """Фальшивый executable: ревизия в пути, версия — в ответе на ``--version``."""
    folder = tmp_path / f"chromium_headless_shell-{revision}" / "chrome-headless-shell-mac-arm64"
    folder.mkdir(parents=True, exist_ok=True)
    executable = folder / "chrome-headless-shell"
    executable.write_bytes(b"#!/bin/sh\necho 'Google Chrome for Testing 153.0.8010.12'\n" + body)
    executable.chmod(0o755)
    return executable


def test_the_fingerprint_follows_the_real_executable(monkeypatch, tmp_path):
    """M10: подмена executable меняет отпечаток даже при том же playwright.

    Ни версия пакета, ни каталоги установки не говорят, какой бинарь реально
    запустится; отпечаток обязан идти от самого executable.
    """
    with _vendor(monkeypatch, tmp_path):

        def fingerprint(executable) -> str:
            monkeypatch.setattr(visual, "_browser_executable", lambda: executable)
            visual._render_fingerprint.cache_clear()
            return visual._render_fingerprint()

        executable = _fake_browser(tmp_path, 1000, b"first")
        first = fingerprint(executable)
        assert fingerprint(executable) == first, "один executable — один отпечаток"

        executable.write_bytes(b"#!/bin/sh\necho 'Google Chrome for Testing 153.0.8010.12'\nrebuilt")
        assert fingerprint(executable) != first, "пересобранный executable обязан сменить отпечаток"

        other = _fake_browser(tmp_path, 1001, b"other")
        assert fingerprint(other) != fingerprint(executable), "другая ревизия — другой отпечаток"


def test_the_frame_record_keeps_the_fingerprint_build_and_platform(monkeypatch, tmp_path):
    """M10: executable и платформа сохраняются с кадром, а не только в хеше."""
    monkeypatch.setenv(visual.CACHE_ENV, str(tmp_path))
    executable = _fake_browser(tmp_path, 1000)
    monkeypatch.setattr(visual, "_browser_executable", lambda: executable)
    monkeypatch.setattr(browser, "render", lambda html, **kw: _k_frame())
    visual._render_fingerprint.cache_clear()
    try:
        visual._cached_render("<html>1</html>", 30, 1 / 60)
    finally:
        visual._render_fingerprint.cache_clear()

    record = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    environment = record["environment"]
    assert environment["platform"].count("/") == 1, "платформа записывается как os/arch"
    build = environment["launch_executables"][0]
    assert build["path"] == str(executable)
    assert build["revision"] == "1000"
    assert build["version"] == "Google Chrome for Testing 153.0.8010.12"
    assert build["sha"] == hashlib.sha256(executable.read_bytes()).hexdigest()


def test_the_manual_version_is_not_the_only_discriminator(monkeypatch, tmp_path):
    """Vendor identity works without bumping ``FRAME_VERSION``."""
    with _vendor(monkeypatch, tmp_path, b"a"):
        before = visual._frame_key("<html></html>", 10, 1 / 60)
        assert visual.FRAME_VERSION == 2
    with _vendor(monkeypatch, tmp_path, b"b"):
        assert visual._frame_key("<html></html>", 10, 1 / 60) != before
    # …and the manual version still counts on its own.
    with _vendor(monkeypatch, tmp_path, b"b"):
        unchanged = visual._frame_key("<html></html>", 10, 1 / 60)
        monkeypatch.setattr(visual, "FRAME_VERSION", 3)
        assert visual._frame_key("<html></html>", 10, 1 / 60) != unchanged


def _sealed_entry(tmp_path, html, frame, seal):
    """Записать запечатанную запись так, как её кладёт `_cached_render`."""
    blob, shot = visual._entry_paths(html, 30, 1 / 60, seal)
    tmp_path.mkdir(parents=True, exist_ok=True)
    shot.write_bytes(b"stored-png")
    blob.write_text(
        json.dumps({"result": {
            "ok": True,
            "shot_sha": "deadbeef",
            "pixels": {"width": 800, "height": 600, "changed_fraction": 0.5},
            "shot_sha_again": "deadbeef",
            "probe": frame.probe,
            "stepped": frame.stepped,
        }}),
        encoding="utf-8",
    )
    return blob, shot


def test_a_stored_frame_is_hashed_even_when_it_cannot_be_rerendered(monkeypatch, tmp_path):
    """The cached bytes carry their own evidence, not the stored claim."""
    monkeypatch.setenv(visual.CACHE_ENV, str(tmp_path))
    with _vendor(monkeypatch, tmp_path):
        frame = _k_frame()
        _sealed_entry(tmp_path, "<html>1</html>", frame, visual._probe_sha(frame))
        monkeypatch.setattr(browser, "render", lambda html, **kw: pytest.fail("перерисовки быть не должно"))

        from hashlib import sha256

        result, hit = visual._cached_render("<html>1</html>", 30, 1 / 60)

    assert hit and result.shot == b"stored-png"
    assert result.shot_sha == sha256(b"stored-png").hexdigest()


def test_a_different_probe_is_a_different_frame_key(monkeypatch, tmp_path):
    """P3: кадр и probe из разных рендеров не должны сойтись в одной записи."""
    with _vendor(monkeypatch, tmp_path):
        one = visual._frame_key("<html></html>", 10, 1 / 60, probe_sha="a" * 16)
        assert one != visual._frame_key("<html></html>", 10, 1 / 60, probe_sha="b" * 16)
        assert one != visual._frame_key("<html></html>", 10, 1 / 60)


def test_the_probe_hash_follows_the_payload():
    assert visual._probe_sha(_k_frame(probe={"x": 1})) != visual._probe_sha(
        _k_frame(probe={"x": 2})
    )


def test_a_hit_recomputes_the_seal_from_the_stored_probe(monkeypatch, tmp_path):
    """A frame sealed by another probe is re-rendered, not paired with the new one."""
    monkeypatch.setenv(visual.CACHE_ENV, str(tmp_path))
    with _vendor(monkeypatch, tmp_path):
        frame = _k_frame()
        # Чужая печать: кадр лежит рядом с probe, но имя подписано не им.
        _sealed_entry(tmp_path, "<html>1</html>", frame, visual._probe_sha(_k_frame(probe={"x": 9})))
        rendered = {"n": 0}

        def fake_render(html, **kwargs):
            rendered["n"] += 1
            return frame

        monkeypatch.setattr(browser, "render", fake_render)
        result, hit = visual._cached_render("<html>1</html>", 30, 1 / 60)

        assert not hit and rendered["n"] == 1
        # Новая запись запечатана собственным probe — второй раз уже попадание.
        again, hit = visual._cached_render("<html>1</html>", 30, 1 / 60)
        assert hit and rendered["n"] == 1 and again.probe == frame.probe
        assert len(list(tmp_path.glob("*.png"))) == 1, "старая печать обязана быть снята"


# --------------------------------------------------------------------------- #
# Gallery driver and standalone converter (R23/R27)
# --------------------------------------------------------------------------- #

PAGE = """<!DOCTYPE html><html><head></head><body><script type='module'>
{imports}
window.__ready = true;
</script></body></html>"""


def _seek_source() -> str:
    hook = gallery.STEP_HOOK
    start = hook.index("scene.seek = (n) => {")
    end = hook.index("    };", start) + len("    };")
    return hook[start:end].replace("scene.seek = (n) =>", "const seek = (n) =>", 1)


def test_seek_zero_does_not_step_and_seek_is_idempotent():
    """R23: the zero checkpoint is the state before any step."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("нет node для запуска драйвера")
    script = (
        "let shown = 0, t = 0, beats = 0, auto = true;\n"
        "const scene = {};\n"
        "const beat = () => { beats++; t += 1 / 60; };\n"
        "const probe = () => {};\n"
        "const draw = () => {};\n"
        + _seek_source()
        + """
const out = {};
out.zero = {steps: seek(0), shown, t, beats};
out.two = {steps: seek(2), shown, t, beats};
out.repeat = {steps: seek(2), shown, t, beats};
console.log(JSON.stringify(out));
"""
    )
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    result = json.loads(done.stdout)
    assert result["zero"] == {"steps": 0, "shown": 0, "t": 0, "beats": 0}
    assert result["two"]["steps"] == 2 and result["two"]["beats"] == 2
    assert result["two"]["t"] == pytest.approx(2 / 60)
    assert result["repeat"] == result["two"]


@pytest.mark.parametrize(
    "imports,expected",
    [
        (
            "import * as THREE from './vendor/three.module.js';",
            ["const THREE = window.THREE;"],
        ),
        (
            "import {Vector3, Scene} from './vendor/three.module.js';",
            ["const { Vector3, Scene } = window.THREE;"],
        ),
        (
            "import {Vector3 as V3} from 'three';",
            ["const { Vector3: V3 } = window.THREE;"],
        ),
        (
            "import * as THREE from './vendor/three.module.js';\n"
            "import {Vector3 as V3, Scene} from 'three';",
            ["const THREE = window.THREE;", "const { Vector3: V3, Scene } = window.THREE;"],
        ),
    ],
)
def test_the_standalone_converter_keeps_import_forms(imports: str, expected: list[str]):
    """R27: wildcard, named, alias and their combination all survive export."""
    page = gallery.standalone_html(PAGE.format(imports=imports), "X.js")
    for marker in expected:
        assert marker in page
    assert "three.module.js" not in page
    assert "<script src='X.js'></script>" in page


def test_the_standalone_converter_refuses_an_unsupported_import():
    """R27: a default import is an explicit export error, not broken HTML."""
    page = PAGE.format(imports="import THREE from './vendor/three.module.js';")
    with pytest.raises(ValueError, match="standalone-экспорт"):
        gallery.standalone_html(page)


# --------------------------------------------------------------------------- #
# Registry and profile slices
# --------------------------------------------------------------------------- #


def _profile_cfg(profile: str) -> config.RunConfig:
    return config.RunConfig(profile=profile)


def _generate(profile: str, seed: int = 3) -> list[core.TaskInstance]:
    return tasks_visual.generate(random.Random(seed), _profile_cfg(profile))


def test_the_registry_lists_every_slice_and_three_profiles() -> None:
    assert len(tasks_visual.MODULES) == len(set(tasks_visual.MODULES)) == 14
    assert tasks_visual.BASELINE_IDS == tuple(f"K-{n}" for n in range(1, 21))
    assert tasks_visual.PILOT_IDS == ("K-21", "K-23", "K-25")
    assert tasks_visual.ALL_IDS == tuple(f"K-{n}" for n in range(1, 37))
    assert tasks_visual.IDS == tasks_visual.BASELINE_IDS


def test_the_declared_profiles_are_the_three_known_ones() -> None:
    assert set(tasks_visual.PROFILE_IDS) == {"pilot", "stretch-v1", "research-v1"}
    assert set(tasks_visual.PILOT_IDS) <= set(tasks_visual.ALL_IDS)


def test_baseline_generates_the_twenty_original_scenarios_at_eighty_points() -> None:
    tasks = _generate(core.DEFAULT_PROFILE)
    assert [task.id for task in tasks] == list(tasks_visual.BASELINE_IDS)
    assert sum(task.max_points for task in tasks) == 80.0


def test_new_modules_do_not_leak_into_the_baseline() -> None:
    tasks = _generate(core.DEFAULT_PROFILE)
    assert all(int(task.id.split("-")[1]) <= 20 for task in tasks)
    assert not set(tasks_visual.PILOT_IDS) & {task.id for task in tasks}


def test_pilot_generates_only_the_three_host_scenarios_at_twelve_points() -> None:
    tasks = _generate("pilot")
    assert [task.id for task in tasks] == list(tasks_visual.PILOT_IDS)
    assert sum(task.max_points for task in tasks) == 12.0


def test_stretch_generates_everything_in_declaration_order_at_a_hundred_and_forty_four() -> None:
    tasks = _generate("stretch-v1")
    assert [task.id for task in tasks] == list(tasks_visual.ALL_IDS)
    assert sum(task.max_points for task in tasks) == 144.0


def test_the_profile_expectations_come_from_the_band_sets() -> None:
    from dailybench.tasks import visual as registered

    category = core.REGISTRY["K"]
    assert registered is tasks_visual
    assert category.expected_ids_for(core.DEFAULT_PROFILE) == tasks_visual.BASELINE_IDS
    assert category.expected_ids_for("pilot") == tasks_visual.PILOT_IDS
    assert category.expected_ids_for("stretch-v1") == tasks_visual.ALL_IDS
    assert category.points == 80.0


def test_each_profile_keeps_its_tasks_distinct_and_categorised() -> None:
    for profile in (core.DEFAULT_PROFILE, "pilot", "stretch-v1"):
        tasks = _generate(profile)
        assert len({task.id for task in tasks}) == len(tasks)
        assert all(task.category == "K" for task in tasks)


def test_the_scenarios_of_one_profile_are_deterministic() -> None:
    first, second = _generate("stretch-v1", 3), _generate("stretch-v1", 3)
    assert [task.prompt for task in first] == [task.prompt for task in second]
    assert [task.oracle["golden_html"] for task in first] == [
        task.oracle["golden_html"] for task in second
    ]


@pytest.mark.parametrize("profile", [core.DEFAULT_PROFILE, "pilot", "stretch-v1"])
def test_every_generated_scenario_rejects_garbage(profile: str) -> None:
    for task in _generate(profile):
        score = visual.evaluate(task, core.Transcript(
            task_id=task.id,
            category="K",
            turns=[core.Turn(messages=[], reply=core.ModelReply(text="не знаю"))],
        ))
        assert score.earned == 0.0, f"{task.id}: мусорный ответ набрал {score.earned}"


@pytest.mark.parametrize("profile", [core.DEFAULT_PROFILE, "pilot", "stretch-v1"])
def test_every_generated_scenario_accepts_its_own_golden(profile: str) -> None:
    usable, reason = browser.availability()
    if not usable:
        pytest.skip(f"браузер недоступен: {reason}")
    for task in _generate(profile):
        score = visual.evaluate(task, core.Transcript(
            task_id=task.id,
            category="K",
            turns=[core.Turn(messages=[], reply=core.ModelReply(text=task.oracle["final_answer"]))],
        ))
        failed = [(check.name, check.detail) for check in score.checks if not check.passed]
        assert score.earned == task.max_points, f"{task.id}: {failed}"
