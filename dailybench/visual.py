"""Shared machinery for the visual block (category K).

A visual scenario is scored from a *rendered* page, so the contract has to be
stated before the code: the model is asked for one self-contained HTML document
that exposes

``window.__ready``
    ``true`` after the first frame,
``window.__probe()``
    a JSON object describing the current state of the scene,
``window.__step(dt)``
    advances the simulation by exactly ``dt`` seconds and returns the new probe.

Everything else — how many objects, what must be where after how many steps,
which colours must be visible — is the scenario's business. This module only
owns the parts that must be identical across all scenarios: the prompt contract,
the extraction of the answer, the deterministic render, and the five checks
(``html``, ``runtime``, ``scene``, ``simulation``, ``image``) that turn a render
into a score. The checks are separate promises (M09): a page protocol, a
misbehaving page, the scene and the simulation it reports, and the frame — the
frame is measured from its own PNG bytes outside the page and at a hidden second
sample (H17), so a black overlay or a picture unrelated to the state fails
``image`` alone instead of the whole task.

The library is pinned and served locally: ``./vendor/three.module.js`` is the
only import an answer may use, and any request that leaves loopback is refused
and reported. A page that quietly needs the internet is not a working page;
HTTP loads and WebSocket handshakes share one classifier and one penalty.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import browser, core

#: Every visual scenario carries five checks; the task still costs four points,
#: so each check is worth 0.8. The split is the rubric (M09): a document and a
#: protocol, the runtime, the scene, the simulation and the frame are different
#: promises, and one of them breaking must not zero the others.
POINTS_PER_CHECK = 0.8
CHECK_NAMES = ("html", "runtime", "scene", "simulation", "image")

#: Share of ``steps`` added for the hidden variation of the initial conditions:
#: the frame is scored at the declared sample *and* at a second one the prompt
#: does not name. A picture painted once and never updated passes only the first.
VARIATION_SHARE = 0.5

#: The file an agent writes the page into. Graded from disk: an agent run is
#: judged on what it produced, not on what it said.
PAGE_FILE = "index.html"

HTML_CONTRACT = """Собери одну HTML-страницу с симуляцией на Three.js.

Требования к странице (общие для всех визуальных заданий):
1. Ответ — один самодостаточный документ `<html>` целиком, без внешних
   ресурсов. Библиотека лежит рядом: `import * as THREE from './vendor/three.module.js'`
   (никаких CDN и никаких других внешних запросов).
2. Рендерер: `new THREE.WebGLRenderer({preserveDrawingBuffer: true})`,
   `renderer.setPixelRatio(1)`, `renderer.setSize(800, 600, false)`, canvas — в
   body. Кадр 800x600 при devicePixelRatio 1.
3. Камера фиксирована (позиция и lookAt заданы в задании), сцена
   детерминирована: при двух загрузках страница обязана дать побитово
   одинаковый кадр. Никаких `Math.random()` без seed и никакой зависимости от
   `performance.now()` или реального времени.
4. Симуляция шагает только по вызову: `window.__step(dt)` продвигает состояние
   ровно на `dt` секунд и возвращает новое состояние. Своё
   `requestAnimationFrame` для физики использовать нельзя (для первой отрисовки
   — можно).
5. Состояние сцены доступно как `window.__probe()` — объект с ровно теми
   полями, которые перечислены в задании, числа округлены до 4 знаков.
6. `window.__ready = true` — после первой отрисовки, когда `__probe()` уже
   отвечает.
"""

#: Where rendered frames are cached between runs (keyed by page content hash).
CACHE_ENV = "DAILYBENCH_VISUAL_CACHE"


def cache_dir() -> Path:
    raw = os.environ.get(CACHE_ENV)
    if raw:
        return Path(raw)
    return Path.home() / ".cache" / "dailybench" / "visual"


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #

FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", re.DOTALL)


def extract_html(text: str) -> str:
    """Pull the document out of an answer: fenced block first, raw text second.

    Models wrap whole files in fences most of the time, and sometimes paste the
    document bare. Both are accepted; nothing is repaired or completed, because
    the point of the check is what the model actually produced.
    """
    body = text or ""
    blocks = [block.strip() for block in FENCE.findall(body)]
    documents = [b for b in blocks if "<html" in b.lower() or "<script" in b.lower()]
    if documents:
        return max(documents, key=len)
    stripped = body.strip()
    lowered = stripped.lower()
    if lowered.startswith("<!doctype") or lowered.startswith("<html"):
        return stripped
    return ""


def remote_urls(html: str) -> list[str]:
    """Requests the page declares against a remote host."""
    return re.findall(r"(?:src|href)\s*=\s*[\"'](https?://[^\"']+)", html)


def looks_like_document(html: str) -> tuple[bool, str]:
    """Static pre-checks: a whole document, a module script, the local library.

    Remote ``src``/``href`` are *not* vetoed here: the renderer refuses them and
    records the attempt, so an HTTP escape and a WebSocket escape — which the
    static text cannot see — cost the same ``runtime`` check instead of the
    former zeroing every check while the latter cost one.
    """
    if not html:
        return False, "в ответе нет HTML-документа"
    lowered = html.lower()
    if "<html" not in lowered:
        return False, "нет тега <html (нужен документ целиком)"
    if "<script" not in lowered:
        return False, "нет ни одного <script>"
    if "three.module.js" not in html:
        return False, "страница не подключает ./vendor/three.module.js"
    return True, ""


# --------------------------------------------------------------------------- #
# Predicates
# --------------------------------------------------------------------------- #

Predicate = Callable[[dict[str, Any]], tuple[bool, str]]


MISSING = object()
"""Distinguishes "the probe has no such field" from "the field is null"."""


def path_of(payload: dict[str, Any], path: str, default: Any = MISSING) -> Any:
    """Read ``"camera.0"`` / ``"params.total"`` out of a probe payload."""
    node: Any = payload
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.lstrip("-").isdigit():
            index = int(part)
            if -len(node) <= index < len(node):
                node = node[index]
            else:
                return default
        else:
            return default
    return node


def _label(check: Predicate, text: str) -> Predicate:
    """Tag a predicate so published oracles stay readable and deterministic."""
    setattr(check, "check_label", text)
    return check


def _number(payload: dict[str, Any], path: str) -> tuple[float | None, str]:
    """The number at ``path``, or a message explaining why there is none."""
    actual = path_of(payload, path)
    if actual is MISSING:
        return None, f"{path}: в probe нет такого поля"
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return None, f"{path}: не число ({actual!r})"
    return float(actual), ""


def near(path: str, value: float, tol: float = 1e-3) -> Predicate:
    def check(payload: dict[str, Any]) -> tuple[bool, str]:
        actual, problem = _number(payload, path)
        if problem:
            return False, problem
        good = abs(actual - float(value)) <= tol
        return good, f"{path} = {actual} (ожидалось {value} ± {tol})"

    return _label(check, f"{path} ≈ {value} ± {tol}")


def between(path: str, low: float, high: float) -> Predicate:
    def check(payload: dict[str, Any]) -> tuple[bool, str]:
        actual, problem = _number(payload, path)
        if problem:
            return False, problem
        good = low <= actual <= high
        return good, f"{path} = {actual} (ожидалось {low}..{high})"

    return _label(check, f"{path} в {low}..{high}")


def contains_all(path: str, values: Iterable[Any]) -> Predicate:
    wanted = list(values)

    def check(payload: dict[str, Any]) -> tuple[bool, str]:
        actual = path_of(payload, path)
        if not isinstance(actual, list):
            return False, f"{path}: не список ({actual!r})"
        missing = [v for v in wanted if v not in actual]
        return not missing, f"{path}: отсутствуют {missing}" if missing else f"{path}: все {len(wanted)} на месте"

    return _label(check, f"{path} содержит {wanted}")


def size_between(path: str, low: int, high: int) -> Predicate:
    def check(payload: dict[str, Any]) -> tuple[bool, str]:
        actual = path_of(payload, path)
        if not isinstance(actual, list):
            return False, f"{path}: не список ({actual!r})"
        good = low <= len(actual) <= high
        return good, f"{path}: {len(actual)} элементов (ожидалось {low}..{high})"

    return _label(check, f"размер {path} в {low}..{high}")


def image_min(key: str, low: float) -> Predicate:
    def check(pixels: dict[str, Any]) -> tuple[bool, str]:
        actual = pixels.get(key)
        if not isinstance(actual, (int, float)):
            return False, f"кадр: нет метрики {key}"
        good = float(actual) >= low
        return good, f"кадр {key} = {actual} (нужно >= {low})"

    return _label(check, f"кадр: {key} >= {low}")


def image_region(index: int, test: Callable[[list[float]], bool], label: str) -> Predicate:
    def check(pixels: dict[str, Any]) -> tuple[bool, str]:
        regions = pixels.get("regions") or []
        if index >= len(regions):
            return False, "кадр: нет областей"
        color = list(regions[index])
        return test(color), f"область {index}: {color} ({label})"

    return _label(check, f"область {index}: {label}")


def image_color(name: str, min_fraction: float, label: str) -> Predicate:
    """At least ``min_fraction`` of the frame belongs to a colour class."""

    def check(pixels: dict[str, Any]) -> tuple[bool, str]:
        stats = pixels.get(name)
        if not isinstance(stats, dict):
            return False, f"кадр: нет класса {name}"
        fraction = float(stats.get("fraction") or 0.0)
        return fraction >= min_fraction, f"кадр: {label} — {fraction:.4f} кадра (нужно >= {min_fraction})"

    return _label(check, f"кадр: {label} (доля >= {min_fraction})")


def image_centroid(name: str, axis: int, low: float, high: float, label: str) -> Predicate:
    """Where the pixels of a colour class sit: 0 is top/left, 1 bottom/right."""

    def check(pixels: dict[str, Any]) -> tuple[bool, str]:
        stats = pixels.get(name)
        centroid = (stats or {}).get("centroid")
        if not isinstance(centroid, list) or len(centroid) < 2:
            return False, f"кадр: {label} — объект не найден"
        value = float(centroid[axis])
        return low <= value <= high, f"кадр: {label} — центр {['x', 'y'][axis]} = {value:.3f} (нужно {low}..{high})"

    return _label(check, f"кадр: {label} (центр {['x', 'y'][axis]} в {low}..{high})")


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def _rgb_to_hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    mx, mn = max(r, g, b), min(r, g, b)
    if mx == mn:
        return 0.0, 0.0, mx / 255
    span = mx - mn
    hue = (
        60 * ((g - b) / span % 6) if mx == r else 60 * ((b - r) / span + 2) if mx == g else 60 * ((r - g) / span + 4)
    )
    return hue % 360, span / mx, mx / 255


def _colour_distance(want: tuple[float, float, float], have: tuple[float, float, float]) -> float:
    """Hue-first distance between two HSV colours, brightness weighing least."""
    dh = min(abs(want[0] - have[0]), 360 - abs(want[0] - have[0])) / 180.0
    return 0.80 * dh + 0.15 * abs(want[1] - have[1]) + 0.05 * abs(want[2] - have[2])


def colour_error(hex_value: str, rgb: Sequence[float]) -> float:
    """Distance between a declared colour and an RGB sample."""
    return _colour_distance(_rgb_to_hsv(*_hex_to_rgb(hex_value)), _rgb_to_hsv(*(float(v) for v in rgb)))


def hsv_error(hex_value: str, hue: float, sat: float, val: float) -> float:
    """Distance between a declared colour and an HSV sample (hue in degrees)."""
    return _colour_distance(_rgb_to_hsv(*_hex_to_rgb(hex_value)), (float(hue), float(sat), float(val)))


def palette_colour(hex_value: str, min_share: float, label: str, tolerance: float = 0.22) -> Predicate:
    """At least ``min_share`` of the frame sits on that declared colour.

    The share is summed over frame buckets that are close to the colour by hue
    (see :func:`colour_error`): a shaded or gradient surface counts in full, a
    scene painted in the wrong hue counts zero. Brightness is nearly free, so a
    dusk gradient still counts as its own colour.
    """
    target = _hex_to_rgb(hex_value)

    def check(pixels: dict[str, Any]) -> tuple[bool, str]:
        histogram = pixels.get("hue_histogram") or []
        if not histogram:
            return False, "кадр: нет гистограммы"
        share = 0.0
        for hue, sat, val, fraction in histogram:
            if hsv_error(hex_value, hue, sat, val) <= tolerance:
                share += float(fraction)
        good = share >= min_share
        return good, f"кадр: {label} ({hex_value}) занимает {share:.4f} (нужно >= {min_share})"

    return _label(check, f"кадр: цвет {hex_value} не меньше {min_share} ({label})")


def gradient(path: str, low: float, high: float, label: str) -> Predicate:
    """A monotone ramp along one image band profile: sky brightens or darkens."""

    def check(pixels: dict[str, Any]) -> tuple[bool, str]:
        profile = pixels.get(path) or []
        if len(profile) < 4:
            return False, f"кадр: нет профиля {path}"
        values = [entry["lum"] for entry in profile[:4]] if "top" in label else [entry["lum"] for entry in profile[4:]]
        if max(values) - min(values) < 1e-9:
            return False, f"кадр: {label} — ровный профиль {values}"
        step = 1 if values[-1] > values[0] else -1
        strict = sum(1 for a, b in zip(values, values[1:]) if step * (b - a) >= -1.5)
        good = strict >= len(values) - 1 and (high - low) * (max(values) - min(values)) >= 0
        return good, f"кадр: {label} — профиль {values} (убыв." if step > 0 else f"кадр: {label} — профиль {values}"

    return _label(check, f"кадр: градиент {label}")


def mirror(left_right: bool, max_difference: float, label: str) -> Predicate:
    """The frame is (nearly) symmetric about its vertical axis."""

    def check(pixels: dict[str, Any]) -> tuple[bool, str]:
        difference = pixels.get("mirror_difference")
        edge = pixels.get("edges")
        if not isinstance(difference, (int, float)):
            return False, "кадр: нет метрики зеркала"
        good = float(difference) <= max_difference
        return good, f"кадр: {label} — зеркальная разница {difference} (нужно <= {max_difference}, границы {edge})"

    return _label(check, f"кадр: зеркальная симметрия ({label})")


def rich_edges(min_edges: float, label: str) -> Predicate:
    def check(pixels: dict[str, Any]) -> tuple[bool, str]:
        edge = pixels.get("edges")
        if not isinstance(edge, (int, float)):
            return False, "кадр: нет метрики границ"
        good = float(edge) >= min_edges
        return good, f"кадр: {label} — плотность границ {edge} (нужно >= {min_edges})"

    return _label(check, f"кадр: детализация {label} (границы >= {min_edges})")


def animated(want: bool, label: str) -> Predicate:
    """Predicate over a *render*, not over pixels: does the scene move?"""

    def check(_pixels: dict[str, Any]) -> tuple[bool, str]:
        return True, "проверяется отдельно"

    check.animated = want  # type: ignore[attr-defined]
    return _label(check, f"кадр: {label}")


def warm(color: Sequence[float]) -> bool:
    """A lit object: red channel clearly above the other two."""
    return color[0] > color[2] + 25 and color[0] > 60


def cool(color: Sequence[float]) -> bool:
    return color[2] > color[0] + 15


def bright(color: Sequence[float]) -> bool:
    return sum(color) / 3 > 110


# --------------------------------------------------------------------------- #
# The generic evaluator
# --------------------------------------------------------------------------- #


@dataclass
class VisualSpec:
    """Everything one visual scenario is scored against.

    ``scene`` and ``simulation`` are predicates over the probe payload; ``image``
    holds predicates over the frame statistics. ``steps``/``dt`` say when the
    simulation is sampled, so "after 2 seconds" means exactly that.
    """

    scene: list[Predicate]
    simulation: list[Predicate]
    image: list[Predicate]
    steps: int
    dt: float = 1 / 60
    need_same_frame: bool = True
    variation_steps: int = 0
    """Extra steps for the hidden second sample; ``0`` derives it from ``steps``."""

    def variation(self) -> int:
        """Steps of the unnamed second sample the frame is also scored at."""
        if self.steps <= 0:
            return 0
        return self.variation_steps or max(1, round(self.steps * VARIATION_SHARE))

    def as_dict(self) -> dict[str, Any]:
        return {
            "checks": list(CHECK_NAMES),
            "steps": self.steps,
            "dt": self.dt,
            "need_same_frame": self.need_same_frame,
            "variation_steps": self.variation_steps,
        }


def _predicates_report(predicates: Sequence[Predicate], payload: dict[str, Any]) -> tuple[bool, str]:
    details: list[str] = []
    passed = True
    for predicate in predicates:
        try:
            good, detail = predicate(payload)
        except Exception as exc:  # a malformed probe must not crash the run
            good, detail = False, f"{exc.__class__.__name__}: {exc}"
        details.append(("+ " if good else "- ") + detail)
        passed &= bool(good)
    return passed, "; ".join(details)


#: Frame-format version for the cache key. Bumped when the *scoring contract*
#: changes; the render environment is fingerprinted separately below, so a new
#: browser or library invalidates the cache without a manual bump.
FRAME_VERSION = 2

#: Schema of the render fingerprint. Bumped when the fingerprint *composition*
#: changes, not when one of its inputs does — inputs hash themselves.
RENDER_FINGERPRINT_VERSION = 2


def _sha_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "missing"


def _sha_big(path: Path) -> str:
    """Hash of a large file, in chunks: a browser binary is hundreds of MiB."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return "missing"
    return digest.hexdigest()


#: Install directory of a browser build: ``chromium-1243``, and its headless
#: shell twin ``chromium_headless_shell-1243``.
_REVISION = re.compile(r"^chromium(?:_headless_shell)?-(\d+)$")

#: Name of the binary ``headless=True`` starts inside a shell build.
_SHELL_NAME = "chrome-headless-shell"


def _platform_tag() -> str:
    """``<os>/<arch>``: pixels differ between platforms for the same build."""
    return f"{platform.system() or '?'}/{platform.machine() or '?'}"


@functools.lru_cache(maxsize=1)
def _browser_executable() -> Path | None:
    """The browser binary playwright resolves, without launching it.

    ``None`` when playwright or its browser is missing: the identity then stays
    an explicit ``?`` instead of guessing from install directories.
    """
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            return Path(pw.chromium.executable_path)
    except Exception:  # pragma: no cover - depends on the machine
        return None


def _headless_shell(executable: Path) -> Path | None:
    """The build a headless launch really starts, next to the full browser.

    ``launch(headless=True)`` runs ``chrome-headless-shell``: the full Chromium
    beside it never paints a frame and must not stand in for it.
    """
    for parent in executable.parents:
        match = _REVISION.match(parent.name)
        if not match:
            continue
        shell_root = parent.with_name(f"chromium_headless_shell-{match.group(1)}")
        for candidate in sorted(shell_root.glob(f"**/{_SHELL_NAME}*")):
            if candidate.is_file() and candidate.suffix != ".dSYM":
                return candidate
    return None


def _revision_of(executable: Path) -> str:
    """Build revision from the install directory, the only place it is written."""
    for part in executable.parts:
        match = _REVISION.match(part)
        if match:
            return match.group(1)
    return "?"


@functools.lru_cache(maxsize=8)
def _executable_version(path: str) -> str:
    """Version the binary reports itself, ``?`` when it cannot answer."""
    commands = [[path, "--version"]]
    if sys.platform == "win32":
        try:
            with Path(path).open("rb") as handle:
                if handle.read(2) != b"MZ":
                    commands.append([sys.executable, path, "--version"])
        except OSError:
            pass
    for command in commands:
        try:
            done = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            continue
        lines = (done.stdout or done.stderr or "").strip().splitlines()
        if lines and done.returncode == 0:
            return lines[0][:80]
    return "?"


def _executable_identity(executable: Path) -> dict[str, str]:
    """Who renders: path, build revision, self-reported version and bytes."""
    try:
        info = executable.stat()
    except OSError:
        info = None
    return _identity_of(
        str(executable),
        info.st_size if info else -1,
        info.st_mtime_ns if info else -1,
    )


@functools.lru_cache(maxsize=8)
def _identity_of(path: str, size: int, mtime_ns: int) -> dict[str, str]:
    """One executable's identity, reused until its bytes on disk move on."""
    executable = Path(path)
    return {
        "path": path,
        "revision": _revision_of(executable),
        "version": _executable_version(path),
        "sha": _sha_big(executable),
    }


def _render_environment() -> dict[str, Any]:
    """The rendering build and platform, recorded beside every frame.

    Page content alone is not enough: the same HTML renders differently after a
    rebuild or a swap of the browser. The playwright version only names the
    driver — it is unchanged when the executable under it is replaced, so it is
    auxiliary here, not identification.
    """
    executable = _browser_executable()
    builds: list[Path] = []
    if executable is not None:
        builds.append(executable)
        shell = _headless_shell(executable)
        if shell is not None and shell != executable:
            builds.append(shell)
    return {
        "platform": _platform_tag(),
        "playwright": _package_version("playwright"),
        "launch_executables": [_executable_identity(build) for build in builds] or "?",
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "?"


@functools.lru_cache(maxsize=1)
def _render_fingerprint() -> str:
    """Hash of everything that decides what a frame looks like.

    Page content alone is not enough: the same HTML renders differently after a
    vendor bump, another browser executable, other GL flags, a changed extractor
    or capture profile. Each of those was able to serve a stale frame as a hit.
    """
    payload = {
        "schema": RENDER_FINGERPRINT_VERSION,
        # оба файла библиотеки: страница может импортировать core напрямую
        "vendor": {
            name: _sha_file(browser.VENDOR / name) for name in ("three.module.js", "three.core.js")
        },
        # executable, его сборка и платформа: кадр принадлежит им, а не версии пакета
        "environment": _render_environment(),
        # драйвер и extractor: правка browser.py меняет способ снятия кадра
        "driver": _sha_file(Path(browser.__file__)),
        "extractor": hashlib.sha256(browser.PIXEL_PROBE.encode("utf-8")).hexdigest(),
        "renderer": {
            "launch_args": list(browser.LAUNCH_ARGS),
            "headless": True,
            "dpr": 1,
            "app": browser.APP_NAME,
        },
        "capture": {
            "timeout_ms": browser.DEFAULT_TIMEOUT_MS,
            "screenshot_timeout_ms": browser.SCREENSHOT_TIMEOUT_MS,
            "live_settle_ms": browser.LIVE_SETTLE_MS,
            "live_interval_ms": browser.LIVE_SAMPLE_INTERVAL_MS,
            "reduced_motion_fixed": "reduce",
            "reduced_motion_live": "no-preference",
            "again_for_live": False,
        },
        "python": platform.python_version(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _frame_key(
    html: str,
    steps: int,
    dt: float,
    viewport: tuple[int, int] = browser.DEFAULT_VIEWPORT,
    mode: str = "fixed",
    kind: str = "fixed",
    probe_sha: str = "",
    probe: bool = True,
) -> str:
    """Cache identity of one render: page, sampling and the whole environment.

    ``probe_sha`` — отпечаток probe, которым кадр был измерен. Он есть только
    после рендера, поэтому имя записи складывается из базового ключа (без
    probe) и ключа с ним: кадр нельзя выдать за снятый с другим probe.
    """
    material = json.dumps(
        {
            "v": FRAME_VERSION,
            "kind": kind,
            "render": _render_fingerprint(),
            "viewport": list(viewport),
            "dpr": 1,
            "steps": steps,
            "dt": dt,
            "mode": mode,
            "probe": probe_sha,
            "pixel_probe": probe,
            "html": html,
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _probe_sha(result: browser.RenderResult) -> str:
    """Отпечаток probe-полезной нагрузки: чем кадр был измерен."""
    material = json.dumps(
        {"probe": result.probe, "stepped": result.stepped}, sort_keys=True, default=repr
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _entry_paths(html: str, steps: int, dt: float, probe_sha: str, *, probe: bool = True) -> tuple[Path, Path]:
    """Пара файлов одной записи: blob, затем кадр."""
    folder = cache_dir()
    base = _frame_key(html, steps, dt, probe=probe)
    key = _frame_key(html, steps, dt, probe_sha=probe_sha, probe=probe)
    return folder / f"{base}__{key}.json", folder / f"{base}__{key}.png"


def _cached_render(
    html: str, steps: int, dt: float, *, again: bool = True, probe: bool = True,
) -> tuple[browser.RenderResult, bool]:
    """Render once per unique page; reuse the verdict and the frame afterwards.

    A failed render is retried instead of being trusted forever: the screenshot
    can time out under load, and the entry then holds ``ok=False`` with no
    pixels, while a fresh render of the same page succeeds. `_cached_render`
    still *writes* such a result — a page that really is broken must fail
    quickly — but the next call re-renders it. The frame file is required too:
    without it the entry has no picture to score.

    An entry is only reused when its stored probe hashes to the seal in its own
    name: the numbers in the blob and the picture beside it must come from one
    render, or the checks would score a fresh run with someone else's probe.
    """
    folder = cache_dir()
    base = _frame_key(html, steps, dt, probe=probe)
    for blob in folder.glob(f"{base}__*.json"):
        shot = blob.with_suffix(".png")
        if not shot.is_file():
            continue
        try:
            data = json.loads(blob.read_text(encoding="utf-8"))
            result = browser.RenderResult(**data["result"])
        except Exception:
            continue
        # Хеш сохранённого probe пересчитывается, а не берётся на веру.
        if _frame_key(html, steps, dt, probe_sha=_probe_sha(result), probe=probe) != blob.stem.split("__", 1)[1]:
            continue
        result.shot = shot.read_bytes()
        # Хеш сохранённого кадра считается от самих байт, а не берётся из
        # блоба: он обязан оставаться валидным доказательством, даже если
        # страницу больше нельзя перерисовать (нет браузера, стенд уехал).
        stored_sha = result.shot_sha
        stored_again = result.shot_sha_again
        # An entry without the second load cannot prove reproducibility:
        # ``same_frame`` would read as False forever. Such entries were
        # written by flows that render once (the recording pipeline); the
        # evaluator needs the second load, so it re-renders instead.
        if not stored_again or (stored_sha and stored_again != stored_sha):
            continue
        if result.shot:
            result.shot_sha = hashlib.sha256(result.shot).hexdigest()
        if result.ok:
            return result, True
    result = browser.render(html, steps=steps, dt=dt, again=again, probe=probe)
    unstable = bool(result.shot_sha_again) and result.shot_sha_again != result.shot_sha
    if unstable:
        return result, False
    folder.mkdir(parents=True, exist_ok=True)
    blob, shot = _entry_paths(html, steps, dt, _probe_sha(result), probe=probe)
    # Прочие запечатки этого кадра (другой probe, другая среда) уже не годятся.
    for stale in folder.glob(f"{base}__*"):
        if stale not in (blob, shot):
            stale.unlink(missing_ok=True)
    # Кадр — до блоба и своим try: probe, который не сериализуется (или blob,
    # упавший по любой причине), не должен оставлять галерею без картинки.
    if result.shot:
        shot.write_bytes(result.shot)
    try:
        payload = result.as_dict()
        # Каким браузером и на какой платформе снят кадр: доказательство,
        # которое нельзя вывести из одного заголовка файла.
        payload["environment"] = _render_environment()
        payload["result"] = {
            "ok": result.ok,
            "reason": result.reason,
            "errors": result.errors,
            "external": result.external,
            "probe": result.probe,
            "stepped": result.stepped,
            "pixels": result.pixels,
            "pixels_again": result.pixels_again,
            "probe_ok": result.probe_ok,
            "ready_ok": result.ready_ok,
            "step_ok": result.step_ok,
            "shot_sha": result.shot_sha,
            "shot_sha_again": result.shot_sha_again,
            "duration_s": result.duration_s,
        }
        blob.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return result, False


def shot_path(html: str, steps: int, dt: float) -> Path:
    """Where the frame of this page is cached, if it rendered.

    Записи запечатаны отпечатком probe, поэтому кадр ищется по базовому ключу:
    годится самый свежий снимок этой страницы, а при пустом кэше — путь под
    будущую запись.
    """
    folder = cache_dir()
    base = _frame_key(html, steps, dt)
    sealed = sorted(folder.glob(f"{base}__*.png"), key=lambda p: p.stat().st_mtime)
    return sealed[-1] if sealed else folder / f"{base}.png"


def page_of(task: core.TaskInstance, transcript: core.Transcript) -> str:
    """The page the run produced: the agent's file on disk, else its printed text.

    The agent harness writes `index.html` in the working copy and the checks read
    it; a chat run has no copy and falls back to the fenced block in the reply.
    """
    from . import agent

    return agent.answer_text(task, transcript, transcript.final_text)


def measure(task: core.TaskInstance, transcript: core.Transcript) -> tuple[browser.RenderResult | None, str]:
    """Deterministic render of the answer, with the cache keyed by its content."""
    spec = VisualSpec(**task.oracle["spec"])
    html = extract_html(page_of(task, transcript))
    static_ok, static_detail = looks_like_document(html)
    if not static_ok:
        return None, static_detail
    result, _ = _cached_render(html, spec.steps, spec.dt)
    return result, spec.as_dict().__repr__()


#: Screen axes: probe ``x`` grows to the right, probe ``y`` grows upwards, so a
#: rising ``y`` moves the picture *up* (negative screen y).
_PROBE_AXES = {"x": (0, 1.0), "y": (1, -1.0)}
#: Smallest centroid shift (share of the frame) that counts as "the picture moved".
#: Large enough that scattered objects (particles, boids) cannot contradict the
#: probe by their own jitter: a diffuse cloud's centroid is a weak witness.
MOTION_MIN_SHIFT = 0.05


def _numeric_leaves(payload: Any, prefix: str = "") -> dict[str, float]:
    """Flatten a probe payload to ``{"balls.0.x": 2.0, ...}``."""
    found: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            found.update(_numeric_leaves(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.update(_numeric_leaves(value, f"{prefix}.{index}"))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        found[prefix] = float(payload)
    return found


def _axis_motion(base_probe: dict[str, Any], varied_probe: dict[str, Any]) -> tuple[str, float] | None:
    """Largest ``x``/``y`` change between two probes, if the probe moved at all."""
    base = _numeric_leaves(base_probe)
    varied = _numeric_leaves(varied_probe)
    moves = [
        (abs(varied[path] - value), path, varied[path] - value)
        for path, value in base.items()
        if path in varied and path.rsplit(".", 1)[-1] in _PROBE_AXES and varied[path] != value
    ]
    if not moves:
        return None
    _, path, delta = max(moves)
    return path, delta


def _motion_check(
    base_probe: dict[str, Any],
    varied_probe: dict[str, Any],
    base_stats: dict[str, Any],
    varied_stats: dict[str, Any],
) -> tuple[bool | None, str]:
    """Does the picture move the way the probe's own numbers say it does?

    ``None`` when the probe reports no ``x``/``y`` movement to compare: the
    frame is then only required to differ. Otherwise the largest declared shift
    picks the axis and a colour class that is on screen in both frames must
    follow it — a drawing unrelated to the reported state fails here even
    though both frames render.
    """
    motion = _axis_motion(base_probe, varied_probe)
    if motion is None:
        return None, "probe не сообщил сдвига по x/y"
    path, delta = motion
    axis, screen_sign = _PROBE_AXES[path.rsplit(".", 1)[-1]]
    wanted = screen_sign * (1.0 if delta > 0 else -1.0)
    seen: list[str] = []
    agreeing = False
    contradicting = False
    for name in ("warm", "cool", "bright", "dark"):
        before = base_stats.get(name) or {}
        after = varied_stats.get(name) or {}
        if not before.get("centroid") or not after.get("centroid"):
            continue
        if float(before.get("fraction") or 0) < 0.0005:
            continue
        shift = float(after["centroid"][axis]) - float(before["centroid"][axis])
        seen.append(f"{name} {shift:+.3f}")
        if shift * wanted >= MOTION_MIN_SHIFT:
            agreeing = True
        elif shift * wanted <= -MOTION_MIN_SHIFT:
            contradicting = True
    detail = f"{path} {delta:+.3f} → пиксели {', '.join(seen) or 'объект не виден'}"
    if not seen:
        return False, detail
    if agreeing or not contradicting:
        # Маленькие и многочисленные объекты (стая, частицы) сдвигают общий
        # центроид класса слабо: согласие — плюс, противоречие — минус.
        return True, detail
    return False, f"{path} {delta:+.3f}, но пиксели ушли против probe: {', '.join(seen)}"


def evaluate(task: core.TaskInstance, transcript: core.Transcript) -> core.Score:
    """The five checks every visual scenario shares.

    The checks are independent promises (M09): ``html`` is the document and the
    page protocol, ``runtime`` is a page that misbehaved (console error, page
    error, a refused request of either transport), ``scene``/``simulation`` are
    the probe against the Python recomputation, and ``image`` is the frame
    measured from its own PNG bytes (H17) at the declared sample and at a hidden
    second one.
    """
    score = core.Score(task_id=task.id, category=task.category)
    spec = VisualSpec(**task.oracle["spec"])

    usable, unavailable = browser.availability()
    if not usable:
        # Same rule as a missing Go toolchain: a machine that cannot render has
        # not verified anything, and an unverified task is not a zero.
        score.skipped = f"визуальный блок не проверялся: {unavailable}"
        score.notes.append(unavailable)
        return score

    def all_failed(detail: str) -> core.Score:
        for name in CHECK_NAMES:
            score.add(name, False, POINTS_PER_CHECK, detail)
        return score

    html = extract_html(page_of(task, transcript))

    static_ok, static_detail = looks_like_document(html)
    if not static_ok:
        score.notes.append("Ответ не содержит исполняемого документа")
        return all_failed(static_detail)

    result, _ = _cached_render(html, spec.steps, spec.dt, probe=False)
    if not result.ok or not result.shot:
        reason = result.reason or "; ".join(result.errors[:3]) or "страница не отрисовалась"
        score.notes.append(reason)
        if result.failure:
            # A worker that never started or died says nothing about the model;
            # a page that exhausted the declared deadline is the candidate's
            # outcome. Both stay in the denominator, only the status differs.
            score.error = f"{core.failure_label(result.failure)}: {reason}"
            score.failure = result.failure
        return all_failed(f"рендер не удался: {reason}")

    stats = browser.frame_stats(result.shot)
    detail = f"кадр {stats.get('width')}x{stats.get('height')}" if stats.get("width") else "кадр снят"

    # html — документ и протокол страницы (``__ready``/``__probe``/``__step``).
    # Потерянный probe больше не обнуляет кадр, но и html без протокола не
    # засчитывается: страницу нельзя ни прочитать, ни продвинуть.
    missing = [
        label
        for label, ok in (
            ("нет window.__ready", result.ready_ok),
            ("нет window.__probe()", result.probe_ok),
            ("нет window.__step(dt)" if spec.steps else "", result.step_ok),
        )
        if not ok and label
    ]
    score.add(
        "html",
        not missing,
        POINTS_PER_CHECK,
        f"документ и протокол в порядке; {detail}" if not missing else f"протокол: {'; '.join(missing)}",
    )

    # runtime — страница отработала, но с ошибками. Обе транспорта-побега
    # (HTTP-загрузка и WebSocket-рукопожатие) идут одним классификатором и
    # стоят одинаково: это один и тот же отказ.
    externals = browser.external_kinds(result.external)
    runtime_ok = not result.errors and not any(externals.values())
    runtime_parts: list[str] = []
    if result.errors:
        runtime_parts.append(f"ошибки страницы: {'; '.join(result.errors[:3])}")
    if any(externals.values()):
        runtime_parts.append(browser.describe_externals(result.external))
    score.add(
        "runtime",
        runtime_ok,
        POINTS_PER_CHECK,
        "; ".join(runtime_parts) if runtime_parts else "страница отработала без ошибок и без внешних запросов",
    )

    probe = browser.probe_of(result)
    stepped = browser.probe_of(result, stepped=True)
    scene_ok, scene_detail = _predicates_report(spec.scene, probe)
    if not result.probe_ok:
        scene_ok, scene_detail = False, f"нет probe: {scene_detail}"
    score.add("scene", scene_ok, POINTS_PER_CHECK, scene_detail)

    sim_ok, sim_detail = _predicates_report(spec.simulation, stepped)
    if not result.step_ok or not result.probe_ok:
        sim_ok, sim_detail = False, f"шаг недоступен: {sim_detail}"
    score.add("simulation", sim_ok, POINTS_PER_CHECK, f"{spec.steps} шагов по {spec.dt:g} с: {sim_detail}")

    # image — кадр из PNG, а не из canvas недоверенного realm. Кроме объявленных
    # предикатов проверяются: воспроизводимость, второй (скрытый) момент и то,
    # что картинка идёт за числами probe.
    image_ok, image_detail = _image_check(html, spec, result, stats, stepped)
    score.add("image", image_ok, POINTS_PER_CHECK, image_detail)

    path = shot_path(html, spec.steps, spec.dt)
    score.notes.append(f"кадр сохранён: {path}")
    # The report copies the frame next to the run and builds the gallery from
    # this: the picture is the deliverable of the visual block.
    transcript.extra["visual"] = {
        "shot": str(path) if path.is_file() else "",
        "steps": spec.steps,
        "dt": spec.dt,
        "pixels": result.pixels if isinstance(result.pixels, dict) else {},
        "frame": stats,
        "same_frame": result.same_frame,
        "duration_s": round(result.duration_s, 3),
    }
    return score


def _image_check(
    html: str,
    spec: VisualSpec,
    result: browser.RenderResult,
    stats: dict[str, Any],
    stepped: dict[str, Any],
) -> tuple[bool, str]:
    """Score the frame: declared predicates, reproducibility and the hidden sample."""
    if not stats:
        return False, "кадр не прочитан"
    ok, detail = _predicates_report(spec.image, stats)
    if spec.need_same_frame and not result.same_frame:
        return False, f"кадр не воспроизводится; {detail}"
    if not result.step_ok or not result.probe_ok:
        # Нечем варьировать: второй момент и согласие с probe требуют протокола.
        # Кадр при этом остаётся оценённым по объявленным предикатам.
        return ok, f"{detail} (протокола шага нет: кадр только по предикатам)"

    variation = spec.variation()
    if not variation:
        return ok, detail
    # Две пробы: объявленная точка плюс половина шагов, и — если сцена уже в
    # установившемся режиме — ранняя точка (регулятор, клапан, транзиент).
    # Объявленные предикаты относятся к объявленной точке, поэтому на пробах
    # проверяется только связь: кадр сменился или сцена законно неподвижна.
    tried: list[str] = []
    static = 0
    followed = True
    for extra in dict.fromkeys((spec.steps + variation, max(1, spec.steps // 8))):
        varied, _ = _cached_render(html, extra, spec.dt, again=False, probe=False)
        if not varied.ok or not varied.shot:
            followed = False
            tried.append(f"{extra}: не отрисовалось")
            continue
        varied_stats = browser.frame_stats(varied.shot)
        if not varied_stats:
            followed = False
            tried.append(f"{extra}: кадр не прочитан")
            continue
        varied_probe = browser.probe_of(varied, stepped=True)
        if varied.shot_sha == result.shot_sha:
            if _axis_motion(stepped, varied_probe) is None:
                static += 1
                tried.append(f"{extra}: сцена статична и probe не двигает x/y")
            else:
                followed = False
                tried.append(f"{extra}: кадр тот же, хотя probe сдвинул x/y")
            continue
        motion, motion_detail = _motion_check(stepped, varied_probe, stats, varied_stats)
        if motion is False:
            followed = False
            tried.append(f"{extra}: {motion_detail}")
            continue
        tried.append(f"{extra} шагов: {motion_detail}")
    if not followed:
        return False, f"кадр не следует за симуляцией: {'; '.join(tried)}"
    if static and static == len(tried):
        return ok, f"{detail}; {'; '.join(tried)}"
    return ok, f"{detail}; {'; '.join(tried)}"


# --------------------------------------------------------------------------- #
# Task construction
# --------------------------------------------------------------------------- #


def build_task(
    *,
    task_id: str,
    title: str,
    prompt: str,
    spec: VisualSpec,
    golden: str,
    rng_meta: dict[str, Any] | None = None,
) -> core.TaskInstance:
    """One visual scenario, with its reference page as the oracle answer.

    The reference is a real page: ``selfcheck`` renders it in the same browser
    and requires all four checks to pass, so a scenario cannot be shipped with
    checks that its own author's page does not satisfy.
    """
    answer = f"```html\n{golden.strip()}\n```"
    from .providers import estimate_tokens

    prompt_text = f"{prompt.strip()}\n\n{HTML_CONTRACT}"
    meta = dict(rng_meta or {})
    meta.update({"driver": "visual", "browser": True, "steps": spec.steps})
    meta["estimated_tokens"] = estimate_tokens(prompt_text)
    meta["answer_tokens_estimate"] = estimate_tokens(answer)
    return core.TaskInstance(
        id=task_id,
        category="K",
        title=title,
        prompt=prompt_text,
        max_points=POINTS_PER_CHECK * len(CHECK_NAMES),
        oracle={
            "spec": {
                "scene": spec.scene,
                "simulation": spec.simulation,
                "image": spec.image,
                "steps": spec.steps,
                "dt": spec.dt,
                "need_same_frame": spec.need_same_frame,
                "variation_steps": spec.variation_steps,
            },
            "reference": answer,
            "final_answer": answer,
            "golden_html": golden,
            "label": title,
            "meta": meta,
            # An agent writes the page to this file and the pinned library is
            # staged next to it; a chat run ignores both.
            "answer_file": PAGE_FILE,
            "assets": ["vendor/three.module.js"],
        },
        meta=meta,
    )


def spec_of(task: core.TaskInstance) -> VisualSpec:
    return VisualSpec(**task.oracle["spec"])
