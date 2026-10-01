"""Галерея ответов витрины: кадр, живая сцена и сырой HTML рядом.

Кадр — доказательство того, что модель нарисовала; но кадр нельзя потрогать.
Поэтому отчёт умеет три вещи на одну задачу:

* показать снимок (быстро, работает всегда);
* открыть **живую страницу** ответа в iframe — ту самую, что сняла кадр, с той
  же библиотекой, так что видно, как сцена двигается;
* отдать сырой HTML в новую вкладку, чтобы сравнить двух моделей рядом.

Плюс режим съёмки: кадры расставляются с фиксированным размером и порядком,
чтобы запись видео шла по одному клику.
"""

from __future__ import annotations

import html
import re
import shutil
from pathlib import Path
from typing import Any

from . import core
from .visual import extract_html

#: Vendored library is served to the gallery pages from this relative folder.
ASSET_DIR = "_vendor"


def _cell(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


STEP_HOOK = """
<script>
// Сцена играет сама: фиксированный шаг 1/60, чтобы кадры шли ровно и запись
// видео не зависела от загрузки машины. Отчёт может вмешаться сообщением
// 'step' (один шаг), 'pause' и 'play' — так работает режим съёмки.
(() => {
  const DT = 1 / 60;
  const step = () => (typeof window.__step === 'function' ? window.__step(DT) : null);
  let acc = 0;
  let last = 0;
  window.__auto = true;

  window.addEventListener('load', () => {
    // Нулевой шаг даёт первый кадр: некоторые сцены рисуют только по шагу.
    if (typeof window.__step === 'function') { try { window.__step(0); } catch (e) {} }
    last = performance.now();
    requestAnimationFrame(loop);
  });

  function loop(now) {
    const elapsed = Math.min(0.2, (now - last) / 1000);
    last = now;
    if (window.__auto) {
      acc += elapsed;
      while (acc >= DT) { acc -= DT; step(); }
    }
    requestAnimationFrame(loop);
  }

  window.addEventListener('message', (event) => {
    const cmd = event.data;
    if (cmd === 'step') { window.__auto = false; acc = 0; step(); }
    else if (cmd === 'pause') { window.__auto = false; }
    else if (cmd === 'play') { window.__auto = true; acc = 0; last = performance.now(); }
    else if (cmd === 'reset') { window.location.reload(); }
    else return;
    window.dispatchEvent(new Event('dailybench-step'));
  });

  window.addEventListener('keydown', (event) => {
    if (event.target && /INPUT|TEXTAREA/.test(event.target.tagName || '')) return;
    if (event.key === ' ') { event.preventDefault(); window.__auto = !window.__auto; acc = 0; last = performance.now(); }
    if (event.key === 'ArrowRight') { event.preventDefault(); window.__auto = false; acc = 0; step(); }
    if (event.key === 'r' || event.key === 'R' || event.key === 'к' || event.key === 'К') window.location.reload();
  });
})();
</script>
"""


def standalone_html(page: str, bundle: str = "./three.global.js") -> str:
    """Ответ, который запускается прямо с диска (``file://``), без сервера.

    Страница ответа импортирует ``./vendor/three.module.js`` — это ES-модуль, а
    под ``file://`` Chrome не грузит внешние модули (политика CORS для origin
    ``null``). Поэтому: классический бандл ``three.global.js`` кладётся рядом и
    подключается первым, а сам импорт заменяется на ``window.THREE`` — модульный
    скрипт остаётся, но библиотеку уже не тянет.
    """
    import_alias = re.search(
        r"""import\s+\*\s+as\s+([A-Za-z_$][\w$]*)\s+from\s+['"]\./vendor/three\.module\.js['"];?""",
        page,
    )
    if not import_alias:
        return page
    alias = import_alias.group(1)
    inject = f"<script src='{bundle}'></script>\n"
    if "</head>" in page:
        page = page.replace("</head>", inject + "</head>", 1)
    else:
        page = inject + page
    return page.replace(
        import_alias.group(0), f"const {alias} = window.THREE;"
    )


def three_bundle(vendor: Path) -> str:
    """Собирает ``vendor/three/*.js`` в один классический скрипт под ``file://``.

    Оба файла — ES-модули: ``three.core.js`` экспортирует имена, ``three.module.js``
    импортирует их и реэкспортирует ядро. Снимаем ``import``/``export``, оба файла
    оборачиваем в IIFE и кладём результат в ``window.THREE`` — получается тот же
    объект, что давал `import * as THREE`.
    """
    core = (vendor / "three.core.js").read_text(encoding="utf-8")
    module = (vendor / "three.module.js").read_text(encoding="utf-8")

    def parts(stmt: str) -> list[str]:
        body = stmt[stmt.index("{") + 1: stmt.rindex("}")]
        return [p.strip() for p in body.split(",") if p.strip()]

    def bindings(stmt: str) -> str:
        out = []
        for p in parts(stmt):
            out.append(f"{p.split(' as ')[0].strip()}: {p.split(' as ')[-1].strip()}" if " as " in p else p)
        return ", ".join(out)

    def exported(stmt: str) -> list[str]:
        return [p.split(" as ")[-1].strip() for p in parts(stmt)]

    core_names: list[str] = []

    def cut_core(match: "re.Match[str]") -> str:
        core_names.extend(exported(match.group(0)))
        return ""

    core_code = re.sub(r"^export \{[^}]*\};?$", cut_core, core, flags=re.M)
    # Реэкспорт и импорт ядра одним и тем же способом превращаем в деструктуризацию.
    module = re.sub(
        r"^(?:export|import) \{([^}]*)\} from '\./three\.core\.js';?$",
        lambda m: f"const {{ {bindings(m.group(0))} }} = __CORE__;",
        module,
        flags=re.M,
    )
    module_names: list[str] = []

    def cut_module(match: "re.Match[str]") -> str:
        module_names.extend(exported(match.group(0)))
        return ""

    module_code = re.sub(r"^export \{[^}]*\};?$", cut_module, module, flags=re.M)
    return (
        "/* three.js offline-бандл для file://: снят import/export, THREE в window. */\n"
        "(function () {\n'use strict';\n"
        f"const __CORE__ = (() => {{\n{core_code}\nreturn {{ {', '.join(core_names)} }};\n}})();\n"
        f"const __MOD__ = (() => {{\n{module_code}\nreturn {{ {', '.join(module_names)} }};\n}})();\n"
        "window.THREE = Object.assign({}, __CORE__, __MOD__);\n"
        "})();\n"
    )


def answer_html(entry: dict[str, Any]) -> str:
    """The page a model produced for this task, straight from the transcript."""
    turns = (entry.get("transcript") or {}).get("turns") or []
    if not turns:
        return ""
    text = (turns[-1].get("reply") or {}).get("text") or ""
    page = extract_html(text)
    if page and "</body>" in page:
        return page.replace("</body>", STEP_HOOK + "</body>", 1)
    return page + STEP_HOOK if page else ""


def scene_page(item: dict[str, Any], target: Path, task_id: str) -> str:
    """Страница живой сцены: standalone-версия, путь к бандлу — от каталога отчёта."""
    if not item.get("html"):
        return ""
    depth = "../../../"  # pages/ -> <model>/ -> _showcase/ -> каталог отчёта
    caption = f"{item.get('model') or target.name} · {task_id}"
    banner = (
        "<div style='position:fixed;z-index:9;left:8px;top:8px;background:#000000b3;"
        "color:#c9d1d9;font:12px/1.4 ui-monospace,monospace;padding:4px 8px;border-radius:6px;"
        "pointer-events:none'>"
        + html.escape(caption)
        + " · пробел — пауза, → — шаг, R — сброс</div>"
    )
    page = standalone_html(item["html"], f"{depth}{ASSET_DIR}/three.global.js")
    if "</body>" in page:
        page = page.replace("</body>", banner + "</body>", 1)
    else:
        page += banner
    return page


def showcase_entries(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Every showcase task of a run: id, block, title, page, frame, spend."""
    out: list[dict[str, Any]] = []
    for entry in run.get("tasks") or []:
        task = entry.get("task") or {}
        category = task.get("category", "")
        if category not in core.VISUAL_CATEGORIES:
            continue
        transcript = entry.get("transcript") or {}
        visual = ((transcript.get("extra") or {}).get("visual") or {})
        out.append(
            {
                "id": task.get("id", "?"),
                "model": (run.get("config") or {}).get("model", ""),
                "category": category,
                "title": task.get("title", ""),
                "html": answer_html(entry),
                "shot": str(visual.get("shot") or ""),
                "score": entry.get("score") or {},
                "spend": transcript.get("spend") or {},
                "atmosphere": visual.get("atmosphere") or [],
                "steps": visual.get("steps"),
                "dt": visual.get("dt"),
            }
        )
    return out


def materialise(run_dir: Path, target: Path) -> dict[str, Any]:
    """Copy frames and pages next to the gallery; returns per-model index.

    The library is copied once per report: an answer page imports
    ``./vendor/three.module.js``, so the copied page must sit next to a folder
    with the same name — otherwise the live view would need the network.
    """
    target.mkdir(parents=True, exist_ok=True)
    pages = target / "pages"
    frames = target / "frames"
    pages.mkdir(exist_ok=True)
    frames.mkdir(exist_ok=True)
    from .report import load_run

    data = load_run(run_dir)
    ensure_vendor(target.parent.parent)  # <report>/_showcase/<model> -> <report>
    index: list[dict[str, Any]] = []
    for item in showcase_entries(data):
        task_id = item["id"]
        record = {"id": task_id, "category": item["category"], "title": item["title"],
                  "score": item["score"], "spend": item["spend"],
                  "atmosphere": item.get("atmosphere") or []}
        if item["html"]:
            page = pages / f"{task_id}.html"
            page.write_text(scene_page(item, target, task_id), encoding="utf-8")
            record["page"] = f"pages/{task_id}.html"
        shot = Path(item["shot"]) if item["shot"] else None
        if shot and shot.is_file():
            frame = frames / f"{task_id}.png"
            shutil.copyfile(shot, frame)
            record["frame"] = f"frames/{task_id}.png"
        index.append(record)
    return {"model": (data.get("config") or {}).get("model", run_dir.name), "dir": run_dir.name, "tasks": index}


def ensure_vendor(report_dir: Path) -> None:
    """Положить рядом с отчётом всё, что нужно его страницам.

    Страницы сцен собираются standalone-версией (см. :func:`scene_page`): они
    грузят классический бандл ``_vendor/three.global.js``, собранный из
    пришпиленной ``vendor/three/``. Поэтому копии ES-модулей рядом с каждой
    моделью не нужны — это экономит около двух мегабайт на модель.
    """
    _write_three_bundle(report_dir)


def _write_three_bundle(report_dir: Path) -> None:
    """Один бандл three.js на отчёт: страницы сцен ссылаются на него."""
    from .browser import VENDOR

    bundle = report_dir / ASSET_DIR / "three.global.js"
    if bundle.is_file():
        return
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_text(three_bundle(VENDOR), encoding="utf-8")
    # Один классический скрипт на весь отчёт: под file:// внешний ES-модуль не
    # грузится (CORS для origin null), и без него живая сцена не запускается.

