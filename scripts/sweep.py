"""Load every reference configuration in turn and run the full suite on it.

One GPU, one model at a time: stop the previous server, start the next with the
entry's own launch settings, verify it with ``preflight --full``, then run the
whole suite. Failures are recorded and the sweep continues — a model that does not
fit in 16 GiB is a result, not an interruption.

    python scripts/sweep.py                       # every entry except the template
    python scripts/sweep.py --only qwen3.8-27b-ud-q3-k-xl
    python scripts/sweep.py --skip lc --limit 12  # быстрый прогон подмножества
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dailybench import launch, reference  # noqa: E402
from dailybench.cli import main as cli_main  # noqa: E402
from dailybench.measure import used_mib  # noqa: E402


def _suite_size(cfg_id: str) -> int:
    """How many tasks the current suite generates for a reference entry."""
    from dailybench.config import RunConfig
    from dailybench.suite import build_suite

    return len(build_suite(reference.apply_to_run_config(RunConfig(), reference.resolve(cfg_id))))


def run_one(entry: dict, args: argparse.Namespace) -> dict:
    cfg_id = entry["id"]
    base_url = str(entry.get("base_url") or "http://127.0.0.1:8888/v1")
    result: dict = {"id": cfg_id, "label": entry.get("label"), "vram_before_mib": used_mib()}

    key = launch.ensure_server(base_url, entry)
    ok, note = launch.load_entry(base_url, key, entry)
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
        summary = summarise(scores, _suite_size(cfg_id))
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
