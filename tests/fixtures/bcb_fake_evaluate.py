"""Stand-in for ``bigcodebench.evaluate``: records argv and writes results.

The real CLI resolves ``<samples>_eval_results.json`` next to the sample file
and stores one entry per task with ``status`` ``pass``/``fail``/``timeout``.
The fake mirrors that shape and marks a sample as passed when its solution
contains the literal ``PASS``, so tests control the verdict without a model.
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    argv = sys.argv[1:]
    log = os.environ.get("FAKE_BCB_LOG")
    if log:
        with open(log, "w", encoding="utf-8") as handle:
            json.dump(argv, handle)
    samples = argv[argv.index("--samples") + 1]
    with open(samples, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    results = {
        "date": "2026-10-25 00:00",
        "eval": {
            row["task_id"]: [
                {
                    "task_id": row["task_id"],
                    "solution": row["solution"],
                    "status": "pass" if "PASS" in row["solution"] else "fail",
                    "details": {},
                }
            ]
            for row in rows
        },
    }
    with open(samples.replace(".jsonl", "_eval_results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("pass@1:\t0.500")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
