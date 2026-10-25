"""Start a local runtime for a reference configuration and wait until it serves.

One place owns the "bring the model up" step, so a single-model run
(``dailybench bench``) and the multi-model sweep share the same code path: the
studio server is started once, later models are loaded through its own API, and
the loaded configuration is the one the application itself would produce —
including the memory strategy and the KV-cache type, which a CLI passthrough
does not reproduce identically.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONTEXT_WINDOW

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "runs" / "_servers"
STUDIO_LOG = LOGS / "studio.log"
API_KEY_RE = re.compile(r"API Key:\s*(sk-\S+)")

UNSLOTH = shutil.which("unsloth") or str(Path.home() / ".unsloth" / "studio" / "bin" / "unsloth.exe")


def api(base_url: str, path: str, timeout: float = 10.0) -> dict[str, Any] | None:
    """GET one OpenAI path (``/models``) under the base URL; ``None`` if silent."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def api_root(base_url: str) -> str:
    """The studio's management API lives at the server root, not under /v1."""
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def props(base_url: str, timeout: float = 10.0) -> dict[str, Any] | None:
    """``GET /props`` at the root — llama.cpp's report of what it loaded.

    Joining it to the base URL asked ``/v1/props`` and silently 404'd, which the
    load check then read as "nothing to verify".
    """
    try:
        with urllib.request.urlopen(api_root(base_url) + "/props", timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def studio_api(
    base_url: str, path: str, key: str, payload: dict | None = None, timeout: float = 900
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        api_root(base_url) + path,
        data=data,
        method="POST" if payload is not None else "GET",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        return {"error": f"HTTP {exc.code}: {body}"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def server_reachable(base_url: str) -> bool:
    return api(base_url, "/models") is not None


def _reuse_key() -> str:
    """The key printed once at server startup, recovered from any log we started."""
    for candidate in sorted(LOGS.glob("*.log")):
        found = API_KEY_RE.search(candidate.read_text(encoding="utf-8", errors="replace"))
        if found:
            return found.group(1)
    return ""


def ensure_server(base_url: str, entry: dict[str, Any]) -> str:
    """Start the studio server once; later models are loaded through its API."""
    if server_reachable(base_url):
        key = _reuse_key()
        if key:
            return key
        # A reachable server whose key we do not have: restart it ourselves.
        subprocess.run([UNSLOTH, "studio", "stop"], capture_output=True, timeout=180)
        deadline = time.time() + 60
        while time.time() < deadline and server_reachable(base_url):
            time.sleep(2)

    LOGS.mkdir(parents=True, exist_ok=True)
    command = [
        UNSLOTH, "studio", "run",
        "--model", str(entry["model"]),
        "--gguf-variant", str(entry["quant"]),
        "--max-seq-length", str(entry["context_window"]),
    ]
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    with STUDIO_LOG.open("w", encoding="utf-8") as log:
        subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), creationflags=flags)

    deadline = time.time() + 900
    while time.time() < deadline:
        if server_reachable(base_url):
            found = API_KEY_RE.search(STUDIO_LOG.read_text(encoding="utf-8", errors="replace"))
            if found:
                return found.group(1)
        time.sleep(2)
    raise RuntimeError("сервер studio не поднялся за 15 минут")


def _loaded_mismatch(props_payload: dict[str, Any] | None, entry: dict[str, Any]) -> str:
    """Why the server does not serve the declared configuration ("" when it does).

    A reachable server still holding the previous model would otherwise be
    measured under the new entry's name: the model ID is shared across quants,
    so only the loaded file and the window can tell them apart.
    """
    declared_window = int(entry["context_window"])
    if not props_payload:
        return (
            f"runtime не подтвердил загрузку: {api_root(str(entry.get('base_url') or ''))}/props молчит — "
            "проверить окно и файл весов нечем"
        )
    generation = props_payload.get("default_generation_settings") or {}
    n_ctx = generation.get("n_ctx") or props_payload.get("n_ctx")
    if n_ctx is None:
        return "runtime не сообщил n_ctx: окно не подтверждено"
    if int(n_ctx) != declared_window:
        return (
            f"окно сервера n_ctx={int(n_ctx)} не совпадает с объявленным {declared_window}: "
            "запись и сервер описывают разные условия"
        )
    model_path = str(props_payload.get("model_path") or "")
    quant = str(entry.get("quant") or "")
    if not model_path:
        return "runtime не сообщил model_path: какой файл загружен, неизвестно"
    if quant and quant.casefold() not in model_path.casefold():
        return f"загружен {model_path}, объявлен квант {quant}: сервер держит другой файл"
    return ""


def load_request(entry: dict[str, Any]) -> dict[str, Any]:
    """The exact body ``/api/inference/load`` is sent — the executed lane.

    Kept separate from ``ReferenceConfig.launch_line``: that one is a template
    for reproduction, this one is what the launcher really asks the runtime for,
    including the KV type and window the server ends up with.
    """
    return {
        "model_path": str(entry["model"]),
        "gguf_variant": str(entry["quant"]),
        "max_seq_length": int(entry["context_window"]),
        "n_parallel": 1,
        "gpu_memory_mode": "auto",
        "cache_type_kv": str(entry.get("kv_type") or "q8_0"),
        "force_reload": True,
        "llama_extra_args": [
            "-fa", "on",
            "--jinja",
            "--reasoning-format", "deepseek",
            # Without this the thinking phase eats the whole answer budget: the
            # coding tasks returned `finish_reason=length` with empty content
            # until the cap was declared.
            "--reasoning-budget", str(int(_policy().get("reasoning_budget", 2048))),
            "--no-context-shift",
            "--temp", "0",
            "--top-p", "1",
        ],
    }


def load_entry(base_url: str, key: str, entry: dict[str, Any], timeout_s: float = 900) -> tuple[bool, str]:
    """Load one configuration through the studio's own API and wait for it.

    The request actually sent is stored on ``entry["executed"]`` so the caller
    can put it in the report beside the declared template instead of passing the
    template off as the executed command.
    """
    model = str(entry["model"])
    studio_api(base_url, "/api/inference/unload", key, {"model_path": model})
    payload = load_request(entry)
    entry["executed"] = {
        "method": "studio-api POST /api/inference/load",
        "request": payload,
    }
    started = time.time()
    result = studio_api(base_url, "/api/inference/load", key, payload)
    if result.get("error"):
        return False, f"загрузка отклонена: {result['error']}"

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        models = api(base_url, "/models") or {}
        row = next((m for m in models.get("data", []) if m.get("id") == model), None)
        if row and row.get("loaded"):
            payload_props = props(base_url)
            mismatch = _loaded_mismatch(payload_props, entry)
            if mismatch:
                return False, f"загрузка за {int(time.time() - started)} с, но {mismatch}"
            n_ctx = ((payload_props or {}).get("default_generation_settings") or {}).get("n_ctx")
            return True, f"загружена за {int(time.time() - started)} с, n_ctx={n_ctx}, файл подтверждён"
        time.sleep(5)
    return False, "модель не загрузилась за отведённое время"


def _policy() -> dict[str, Any]:
    from . import reference

    return dict(reference.POLICY)


def stop() -> None:
    """Stop the studio server this module started."""
    subprocess.run([UNSLOTH, "studio", "stop"], capture_output=True, timeout=180)


def up(entry: dict[str, Any]) -> tuple[bool, str, str]:
    """Server up and model loaded. Returns ``(ok, note, api_key)``.

    ``entry`` is expected to be the *effective* view (``ReferenceConfig.effective``).
    A raw entry is filled from the policy window once, so no later step has to
    guess a different fallback. The dict is filled in place: after the call it
    carries ``executed`` — the load request that was really sent — which the
    caller records next to the declared template.
    """
    if not entry.get("context_window"):
        entry["context_window"] = int(_policy().get("context_window") or DEFAULT_CONTEXT_WINDOW)
    base_url = str(entry.get("base_url") or "http://127.0.0.1:8888/v1")
    key = ensure_server(base_url, entry)
    ok, note = load_entry(base_url, key, entry)
    return ok, note, key
