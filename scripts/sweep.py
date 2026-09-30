"""Load every reference configuration in turn and run the full suite on it.

One GPU, one model at a time: stop the previous server, start the next with the
entry's own launch settings, verify it with ``preflight --full``, then run all
29 tasks. Failures are recorded and the sweep continues — a model that does not
fit in 16 GiB is a result, not an interruption.

    python scripts/sweep.py                       # every entry except the template
    python scripts/sweep.py --only qwen3.8-27b-ud-q3-k-xl
    python scripts/sweep.py --skip lc --limit 12  # быстрый прогон подмножества
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dailybench import reference  # noqa: E402
from dailybench.cli import main as cli_main  # noqa: E402
from dailybench.measure import used_mib  # noqa: E402

UNSLOTH = shutil.which("unsloth") or str(Path.home() / ".unsloth" / "studio" / "bin" / "unsloth.exe")
LOGS = ROOT / "runs" / "_servers"


def api(base_url: str, path: str, timeout: float = 10.0):
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


STUDIO_LOG = LOGS / "studio.log"
API_KEY_RE = re.compile(r"API Key:\s*(sk-\S+)")


def server_reachable(base_url: str) -> bool:
    return api(base_url, "/models") is not None


def ensure_server(base_url: str, entry: dict) -> str:
    """Start the studio server once; later models are loaded through its API.

    Switching models through ``POST /api/inference/load`` uses the same code path
    as the application itself, so the loaded configuration is the one the app
    would produce — including the memory strategy and the KV cache type, which
    the CLI passthrough does not reproduce identically.
    """
    if server_reachable(base_url):
        # The key is printed once at startup; find it in any log we started.
        for candidate in sorted(LOGS.glob("*.log")):
            found = API_KEY_RE.search(candidate.read_text(encoding="utf-8", errors="replace"))
            if found:
                return found.group(1)
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
            text = STUDIO_LOG.read_text(encoding="utf-8", errors="replace")
            found = API_KEY_RE.search(text)
            if found:
                return found.group(1)
        time.sleep(2)
    raise RuntimeError("сервер studio не поднялся за 15 минут")


def api_root(base_url: str) -> str:
    """The studio's management API lives at the server root, not under /v1."""
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def studio_api(base_url: str, path: str, key: str, payload: dict | None = None, timeout: float = 900):
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


def load_entry(base_url: str, key: str, entry: dict, timeout_s: float = 900) -> tuple[bool, str]:
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
            "--reasoning-budget", str(int(reference.POLICY.get("reasoning_budget", 2048))),
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
        models = api(base_url, "/models")
        row = next((m for m in (models or {}).get("data", []) if m.get("id") == model), None)
        if row and row.get("loaded"):
            props = api(base_url, "/props")
            n_ctx = ((props or {}).get("default_generation_settings") or {}).get("n_ctx")
            note = f"загружена за {int(time.time() - started)} с, n_ctx={n_ctx}"
            if n_ctx and int(n_ctx) < int(entry.get("context_window") or 0):
                note += " — окно урезано по VRAM (auto), допуск long context будет честно провален"
            return True, note
        time.sleep(5)
    return False, "модель не загрузилась за отведённое время"


def run_one(entry: dict, args: argparse.Namespace) -> dict:
    cfg_id = entry["id"]
    base_url = str(entry.get("base_url") or "http://127.0.0.1:8888/v1")
    result: dict = {"id": cfg_id, "label": entry.get("label"), "vram_before_mib": used_mib()}

    key = ensure_server(base_url, entry)
    ok, note = load_entry(base_url, key, entry)
    result["loaded"] = ok
    result["load_note"] = note
    if not ok:
        result["verdict"] = f"не загрузилась: {note}"
        return result
    result["vram_after_load_mib"] = used_mib()

    resolved = reference.resolve(cfg_id)
    report = reference.preflight(resolved, full=not args.no_full)
    preflight_path = ROOT / "runs" / f"preflight-{cfg_id}.md"
    preflight_path.write_text(report.render() + "\n", encoding="utf-8")
    result["preflight"] = report.verdict
    result["preflight_probes"] = {probe.name: probe.status for probe in report.probes}
    result["preflight_metrics"] = report.metrics

    if report.verdict == "rejected":
        result["verdict"] = "отклонена preflight'ом"
        return result

    argv = ["run", "--config", cfg_id, "--name", cfg_id, "--speed", "--speed-long", "4096", "--speed-repeats", "2"]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.tasks:
        argv += ["--only", args.tasks]
    if args.skip_calibration:
        argv += ["--no-calibrate"]
    started = time.time()
    code = cli_main(argv)
    result["run_seconds"] = round(time.time() - started, 1)
    result["run_exit_code"] = code

    run_dir = ROOT / "runs" / cfg_id
    scores_path = run_dir / "scores.json"
    if scores_path.is_file():
        from dailybench.report import summarise

        scores = json.loads(scores_path.read_text(encoding="utf-8"))
        summary = summarise(scores, 29)
        result["verdict"] = summary["verdict"]
        result["earned"] = summary["headline_earned"]
        result["possible"] = summary["headline_possible"]
        result["per_category"] = {
            key: round(100 * row["earned"] / row["possible"], 1) if row["possible"] else None
            for key, row in summary["per_category"].items()
            if row["tasks"]
        }
        speed_path = run_dir / "speed.json"
        if speed_path.is_file():
            result["speed"] = json.loads(speed_path.read_text(encoding="utf-8"))
    else:
        result["verdict"] = "прогон не дал scores.json"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Загрузить и протестировать все эталонные конфигурации")
    parser.add_argument("--only", default="", help="id конфигураций через запятую")
    parser.add_argument("--skip", default="", help="id, которые пропустить (например generic-gguf)")
    parser.add_argument("--limit", type=int, default=0, help="ограничить число заданий в прогоне")
    parser.add_argument("--tasks", default="", help="ограничить сьют заданиями (id через запятую)")
    parser.add_argument("--no-full", action="store_true", help="не заполнять окно в preflight")
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    wanted = {part.strip() for part in args.only.split(",") if part.strip()}
    skipped = {part.strip() for part in args.skip.split(",") if part.strip()} or {"generic-gguf"}

    entries = [entry.entry for entry in reference.model_entries()]
    entries = [e for e in entries if e["id"] not in skipped and (not wanted or e["id"] in wanted)]
    print(f"К прогону: {[e['id'] for e in entries]}", flush=True)

    summary_path = ROOT / "runs" / "sweep-summary.json"
    existing: list[dict] = []
    if summary_path.is_file() and args.only:
        existing = json.loads(summary_path.read_text(encoding="utf-8")).get("results", [])
    results: list[dict] = list(existing)
    for entry in entries:
        print(f"\n=== {entry['id']} — {entry.get('label')}", flush=True)
        try:
            row = run_one(entry, args)
        except Exception as exc:  # pragma: no cover - a sweep must not die on one model
            row = {"id": entry["id"], "verdict": f"ошибка драйвера: {type(exc).__name__}: {exc}"}
        results = [row if item.get("id") == row.get("id") else item for item in results] \
            if any(item.get("id") == row.get("id") for item in results) else results + [row]
        print(f"--- {entry['id']}: {row.get('verdict')}", flush=True)
        summary_path.write_text(
            json.dumps({"updated": datetime.now().isoformat(timespec="seconds"), "results": results},
                       ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    print("\nИтоговая таблица:")
    for row in results:
        print(
            f"  {row['id']:28} {str(row.get('verdict')):40} "
            f"VRAM после загрузки: {row.get('vram_after_load_mib', 'n/a')} MiB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
