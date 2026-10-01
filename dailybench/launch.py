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

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "runs" / "_servers"
STUDIO_LOG = LOGS / "studio.log"
API_KEY_RE = re.compile(r"API Key:\s*(sk-\S+)")

UNSLOTH = shutil.which("unsloth") or str(Path.home() / ".unsloth" / "studio" / "bin" / "unsloth.exe")


def api(base_url: str, path: str, timeout: float = 10.0) -> dict[str, Any] | None:
    """GET one management path; ``None`` when it does not answer."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def api_root(base_url: str) -> str:
    """The studio's management API lives at the server root, not under /v1."""
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


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
        "--max-seq-length", str(entry.get("context_window") or 32768),
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


def load_entry(base_url: str, key: str, entry: dict[str, Any], timeout_s: float = 900) -> tuple[bool, str]:
    """Load one configuration through the studio's own API and wait for it."""
    model = str(entry["model"])
    studio_api(base_url, "/api/inference/unload", key, {"model_path": model})
    payload = {
        "model_path": model,
        "gguf_variant": str(entry["quant"]),
        "max_seq_length": int(entry.get("context_window") or 32768),
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
    started = time.time()
    result = studio_api(base_url, "/api/inference/load", key, payload)
    if result.get("error"):
        return False, f"загрузка отклонена: {result['error']}"

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        models = api(base_url, "/models") or {}
        row = next((m for m in models.get("data", []) if m.get("id") == model), None)
        if row and row.get("loaded"):
            props = api(base_url, "/props")
            n_ctx = ((props or {}).get("default_generation_settings") or {}).get("n_ctx")
            note = f"загружена за {int(time.time() - started)} с, n_ctx={n_ctx}"
            if n_ctx and int(n_ctx) < int(entry.get("context_window") or 0):
                note += " — окно урезано по VRAM (auto), допуск long context будет честно провален"
            return True, note
        time.sleep(5)
    return False, "модель не загрузилась за отведённое время"


def _policy() -> dict[str, Any]:
    from . import reference

    return dict(reference.POLICY)


def stop() -> None:
    """Stop the studio server this module started."""
    subprocess.run([UNSLOTH, "studio", "stop"], capture_output=True, timeout=180)


def up(entry: dict[str, Any]) -> tuple[bool, str, str]:
    """Server up and model loaded. Returns ``(ok, note, api_key)``."""
    base_url = str(entry.get("base_url") or "http://127.0.0.1:8888/v1")
    key = ensure_server(base_url, entry)
    ok, note = load_entry(base_url, key, entry)
    return ok, note, key
